/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Unit tests for the CTE GpuMetadataCache projection.
 *
 * Each TEST_CASE shares one Chimaera + CTE bring-up via EnsureInit():
 *   - kServer Chimaera, gpu_metadata_cache enabled in CreateParams.
 *   - One RAM bdev target (GPU-visible) + one FILE bdev target (NOT
 *     GPU-visible) registered against the CTE pool.
 *   - A unique pool id (513, 0) keeps us from being shadowed by the
 *     auto-composed default pool at (512, 0).
 *
 * The kernel-side lookup pattern (inline open-addressing walk that
 * eagerly materializes slot fields) is the workaround documented in
 * test_sycl_chimod_to_cpu.cc — without it DPC++'s device-side
 * optimizer DCE's the slot loads when the only sink is a conditional
 * write the compiler can't prove fires.
 *
 * Tests:
 *  1. PutBlob (RAM) -> kernel finds the blob projection.            (Step 4 baseline)
 *  2. GetOrCreateTag -> kernel finds the tag entry.                 (Step 5.1)
 *  3. PutBlob -> kernel sees blob; DelBlob -> kernel no longer sees it. (Step 5.2)
 *  4. PutBlob N blobs under a tag; DelTag -> kernel sees zero of them. (Step 5.3)
 *  5. PutBlob (FILE) -> kernel does NOT see it (non-DRAM eviction).  (Step 5.4)
 */

#if HSHM_ENABLE_SYCL && !(HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM)

#include "simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/gpu_metadata_cache.h>

#include <hermes_shm/util/gpu_api.h>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

class chi_sycl_gpu_meta_cache_kernel;
class chi_sycl_gpu_meta_cache_tag_kernel;
class chi_sycl_gpu_meta_cache_delblob_kernel;
class chi_sycl_gpu_meta_cache_deltag_kernel;
class chi_sycl_gpu_meta_cache_filetier_kernel;

// ---------- Shared bring-up (idempotent across TEST_CASEs) ----------
bool g_initialized = false;
chi::PoolId g_test_pool_id(513, 0);
chi::PoolId g_ram_bdev_pool_id(905, 0);
chi::PoolId g_file_bdev_pool_id(906, 0);
const std::string g_ram_target_name = "gpu_meta_cache_ram_target";
std::string g_file_target_path;
wrp_cte::core::Client *g_cte_client = nullptr;
wrp_cte::core::GpuMetadataCacheHeader *g_gpu_cache = nullptr;

void EnsureInit() {
  if (g_initialized) return;

  REQUIRE(chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer));

  REQUIRE(wrp_cte::core::WRP_CTE_CLIENT_INIT());
  g_cte_client = WRP_CTE_CLIENT;
  REQUIRE(g_cte_client != nullptr);

  // Unique pool id to avoid the auto-composed default pool at (512, 0)
  // shadowing our AsyncCreate; without this our CreateParams (and the
  // gpu_metadata_cache flag) never reach Runtime::Create.
  g_cte_client->Init(g_test_pool_id);

  wrp_cte::core::CreateParams params;
  params.config_.gpu_metadata_cache_.enabled_ = true;
  params.config_.gpu_metadata_cache_.capacity_bytes_ = 256ULL * 1024;
  params.config_.gpu_metadata_cache_.max_tags_ = 64;
  params.config_.gpu_metadata_cache_.max_blobs_ = 256;

  auto pool_task = g_cte_client->AsyncCreate(
      chi::PoolQuery::Dynamic(),
      std::string("gpu_meta_cache_test_pool"),
      g_test_pool_id, params);
  pool_task.Wait();
  REQUIRE(pool_task->GetReturnCode() == 0);

  wrp_cte::core::CreateParams out = pool_task->GetParams();
  REQUIRE(out.gpu_cache_ptr_ != 0);
  g_gpu_cache =
      reinterpret_cast<wrp_cte::core::GpuMetadataCacheHeader *>(
          out.gpu_cache_ptr_);
  REQUIRE(g_gpu_cache != nullptr);
  REQUIRE(g_gpu_cache->ValidMagic());
  REQUIRE(g_gpu_cache->max_tags_ > 0);
  REQUIRE(g_gpu_cache->max_blobs_ > 0);
  std::fprintf(stderr,
               "[TRACE] gpu_cache=%p max_tags=%u max_blobs=%u\n",
               static_cast<void *>(g_gpu_cache),
               g_gpu_cache->max_tags_, g_gpu_cache->max_blobs_);

  // RAM-tier target — GPU-visible storage class (kStorageRam).
  size_t ram_size = 4ULL * 1024 * 1024;
  chimaera::bdev::Client ram_bdev_client(g_ram_bdev_pool_id);
  auto ram_bdev_create = ram_bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      g_ram_target_name, g_ram_bdev_pool_id,
      chimaera::bdev::BdevType::kRam, ram_size);
  ram_bdev_create.Wait();
  REQUIRE(ram_bdev_create->GetReturnCode() == 0);

  auto ram_reg = g_cte_client->AsyncRegisterTarget(
      g_ram_target_name, chimaera::bdev::BdevType::kRam,
      ram_size, chi::PoolQuery::Local(), g_ram_bdev_pool_id);
  ram_reg.Wait();
  REQUIRE(ram_reg->GetReturnCode() == 0);

  // FILE-tier target — NOT GPU-visible. Used by the file-tier exclusion
  // test to confirm GpuCacheOnPutBlob evicts/refuses non-DRAM blobs.
  const char *home = std::getenv("HOME");
  REQUIRE(home != nullptr);
  g_file_target_path = std::string(home) + "/gpu_meta_cache_file.dat";
  if (fs::exists(g_file_target_path)) fs::remove(g_file_target_path);
  size_t file_size = 4ULL * 1024 * 1024;
  chimaera::bdev::Client file_bdev_client(g_file_bdev_pool_id);
  auto file_bdev_create = file_bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      g_file_target_path, g_file_bdev_pool_id,
      chimaera::bdev::BdevType::kFile, file_size);
  file_bdev_create.Wait();
  REQUIRE(file_bdev_create->GetReturnCode() == 0);

  auto file_reg = g_cte_client->AsyncRegisterTarget(
      g_file_target_path, chimaera::bdev::BdevType::kFile,
      file_size, chi::PoolQuery::Local(), g_file_bdev_pool_id);
  file_reg.Wait();
  REQUIRE(file_reg->GetReturnCode() == 0);

  g_initialized = true;
}

// Allocate an IPC buffer, fill it with a deterministic pattern, and
// run AsyncPutBlob. Returns the task return code so callers can REQUIRE.
int HostPutBlob(const wrp_cte::core::TagId &tag_id,
                const std::string &blob_name, size_t bytes) {
  std::vector<char> pattern(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    pattern[i] = static_cast<char>(0x33 ^ (i & 0xFF));
  }
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> buf = ipc->AllocateBuffer(bytes);
  if (buf.IsNull()) return -1;
  std::memcpy(buf.ptr_, pattern.data(), bytes);
  hipc::ShmPtr<> shm(buf.shm_);
  auto task = g_cte_client->AsyncPutBlob(
      tag_id, blob_name, /*offset=*/0, bytes, shm, /*score=*/-1.0f);
  task.Wait();
  int rc = task->GetReturnCode();
  ipc->FreeBuffer(buf);
  return rc;
}

}  // namespace

// ---------------------------------------------------------------------
// TEST 1 — Step 4 baseline: PutBlob (RAM) -> kernel sees the projection.
// ---------------------------------------------------------------------
TEST_CASE("GPU metadata cache: PutBlob -> GPU kernel sees the projection",
          "[sycl][gpu][metadata_cache]") {
  EnsureInit();
  auto *gpu_cache = g_gpu_cache;

  auto tag_task = g_cte_client->AsyncGetOrCreateTag("gpu_meta_cache_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  const std::string blob_name = "alpha_blob";
  const size_t kBlobBytes = 512;
  REQUIRE(HostPutBlob(tag_id, blob_name, kBlobBytes) == 0);

  REQUIRE(gpu_cache->num_blobs_ >= 1u);
  const auto *host_lookup = wrp_cte::core::GpuCacheFindBlob(
      gpu_cache, tag_id.major_, tag_id.minor_, blob_name.c_str());
  REQUIRE(host_lookup != nullptr);
  REQUIRE(host_lookup->size_ == kBlobBytes);
  REQUIRE(wrp_cte::core::gpu_cache::IsGpuVisible(host_lookup->storage_class_));

  sycl::queue &q = hshm::GpuApi::SyclQueue();
  // Single-pointer kernel arg pattern — see KernelBlobVisible's
  // comment for the rationale. We bundle the cache pointer, the tag
  // id, and the result buffer into one shared-USM struct so the lambda
  // captures only one pointer.
  struct KernelCtx {
    void *cache;
    chi::u32 cap_tag_major;
    chi::u32 cap_tag_minor;
    chi::u32 _pad;
    chi::u32 results[4];
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  REQUIRE(ctx != nullptr);
  ctx->cache = static_cast<void *>(gpu_cache);
  ctx->cap_tag_major = tag_id.major_;
  ctx->cap_tag_minor = tag_id.minor_;
  ctx->_pad = 0;
  for (int i = 0; i < 4; ++i) ctx->results[i] = 0;
  KernelCtx *ctx_ptr = ctx;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_gpu_meta_cache_kernel>([=]() {
      ctx_ptr->results[0] = 0;
#if HSHM_IS_DEVICE_PASS
      auto *cache =
          reinterpret_cast<wrp_cte::core::GpuMetadataCacheHeader *>(
              ctx_ptr->cache);
      if (!cache) return;
      ctx_ptr->results[0] = (cache->magic_ ==
                              wrp_cte::core::GpuMetadataCacheHeader::kMagic)
                                ? 1u : 0u;
      const auto *slots = reinterpret_cast<const wrp_cte::core::gpu_cache::GpuBlobEntry *>(
          reinterpret_cast<const char *>(cache) +
          sizeof(wrp_cte::core::GpuMetadataCacheHeader) +
          static_cast<size_t>(cache->max_tags_) *
              sizeof(wrp_cte::core::gpu_cache::GpuTagEntry));
      chi::u32 cap = cache->max_blobs_;
      chi::u32 cap_major = ctx_ptr->cap_tag_major;
      chi::u32 cap_minor = ctx_ptr->cap_tag_minor;
      for (chi::u32 i = 0; i < cap; ++i) {
        const auto &s = slots[i];
        chi::u32 s_state = s.state_;
        chi::u32 s_major = s.tag_major_;
        chi::u32 s_minor = s.tag_minor_;
        chi::u32 name_word =
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[0])) << 24) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[1])) << 16) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[2])) << 8) |
             static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[3]));
        // XOR sink keeps DPC++ from DCE'ing per-slot loads.
        ctx_ptr->results[2] ^= s_state + s_major + (s_minor << 16) + name_word;
        if (s_state != wrp_cte::core::gpu_cache::kOccupied) continue;
        if (s_major == cap_major && s_minor == cap_minor) {
          ctx_ptr->results[1] = 1u;
          ctx_ptr->results[2] = static_cast<chi::u32>(s.size_);
          ctx_ptr->results[3] = s.storage_class_;
        }
      }
#else
      (void)ctx_ptr;
#endif
    });
  }).wait_and_throw();

  std::fprintf(stderr,
               "[TRACE] kernel result: valid_magic=%u found=%u size=%u "
               "storage_class=%u\n",
               ctx->results[0], ctx->results[1], ctx->results[2], ctx->results[3]);

  REQUIRE(ctx->results[0] == 1u);
  REQUIRE(ctx->results[1] == 1u);
  REQUIRE(ctx->results[2] == static_cast<chi::u32>(kBlobBytes));
  REQUIRE(ctx->results[3] == wrp_cte::core::gpu_cache::kStorageRam);

  sycl::free(ctx, q);
  (void)blob_name;
}

// ---------------------------------------------------------------------
// TEST 2 — Step 5.1: GetOrCreateTag projects a tag entry the kernel
// can find via the tag-slot array.
// ---------------------------------------------------------------------
TEST_CASE("GPU metadata cache: GetOrCreateTag -> kernel sees tag entry",
          "[sycl][gpu][metadata_cache][tag]") {
  EnsureInit();
  auto *gpu_cache = g_gpu_cache;

  const std::string tag_name = "gpu_meta_cache_tag_test";
  auto tag_task = g_cte_client->AsyncGetOrCreateTag(tag_name);
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  // Host-side sanity check.
  const auto *host_tag = wrp_cte::core::GpuCacheFindTag(
      gpu_cache, tag_id.major_, tag_id.minor_);
  REQUIRE(host_tag != nullptr);
  std::fprintf(stderr,
               "[TRACE] tag2 host_tag=%p tag_id=(%u,%u) host_name='%s'\n",
               static_cast<const void *>(host_tag),
               tag_id.major_, tag_id.minor_,
               host_tag ? host_tag->tag_name_ : "");

  sycl::queue &q = hshm::GpuApi::SyclQueue();
  // Single-pointer kernel arg pattern (see KernelBlobVisible's comment).
  struct KernelCtx {
    void *cache;
    chi::u32 cap_tag_major;
    chi::u32 cap_tag_minor;
    chi::u32 _pad;
    chi::u32 results[4];
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  REQUIRE(ctx != nullptr);
  ctx->cache = static_cast<void *>(gpu_cache);
  ctx->cap_tag_major = tag_id.major_;
  ctx->cap_tag_minor = tag_id.minor_;
  ctx->_pad = 0;
  for (int i = 0; i < 4; ++i) ctx->results[i] = 0;
  KernelCtx *ctx_ptr = ctx;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_gpu_meta_cache_tag_kernel>([=]() {
      ctx_ptr->results[0] = 0;
#if HSHM_IS_DEVICE_PASS
      auto *cache =
          reinterpret_cast<wrp_cte::core::GpuMetadataCacheHeader *>(
              ctx_ptr->cache);
      if (!cache) return;
      ctx_ptr->results[0] = (cache->magic_ ==
                              wrp_cte::core::GpuMetadataCacheHeader::kMagic)
                                ? 1u : 0u;
      const auto *tag_slots = reinterpret_cast<const wrp_cte::core::gpu_cache::GpuTagEntry *>(
          reinterpret_cast<const char *>(cache) +
          sizeof(wrp_cte::core::GpuMetadataCacheHeader));
      chi::u32 cap = cache->max_tags_;
      chi::u32 count = 0;
      chi::u32 occupied = 0;
      chi::u32 cap_major = ctx_ptr->cap_tag_major;
      chi::u32 cap_minor = ctx_ptr->cap_tag_minor;
      for (chi::u32 i = 0; i < cap; ++i) {
        const auto &s = tag_slots[i];
        chi::u32 s_state = s.state_;
        chi::u32 s_major = s.tag_major_;
        chi::u32 s_minor = s.tag_minor_;
        chi::u32 name_word =
            (static_cast<chi::u32>(static_cast<unsigned char>(s.tag_name_[0])) << 24) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.tag_name_[1])) << 16) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.tag_name_[2])) << 8) |
             static_cast<chi::u32>(static_cast<unsigned char>(s.tag_name_[3]));
        ctx_ptr->results[2] ^= s_state + s_major + (s_minor << 16) + name_word;
        if (s_state != wrp_cte::core::gpu_cache::kOccupied) continue;
        ++occupied;
        if (s_major == cap_major && s_minor == cap_minor) ++count;
      }
      ctx_ptr->results[1] = (count > 0) ? 1u : 0u;
      ctx_ptr->results[3] = occupied;
#else
      (void)ctx_ptr;
#endif
    });
  }).wait_and_throw();

  std::fprintf(stderr,
               "[TRACE] tag kernel result: valid_magic=%u found=%u "
               "xor_sink=0x%x occupied=%u\n",
               ctx->results[0], ctx->results[1], ctx->results[2], ctx->results[3]);

  REQUIRE(ctx->results[0] == 1u);
  REQUIRE(ctx->results[1] == 1u);

  sycl::free(ctx, q);
}

// ---------------------------------------------------------------------
// Helper: kernel that returns 1 if (tag, blob_name) is in the cache,
// 0 otherwise. Reused by DelBlob / DelTag / file-tier tests.
// ---------------------------------------------------------------------
namespace {

template <class KernelTag>
chi::u32 KernelBlobVisible(wrp_cte::core::GpuMetadataCacheHeader *gpu_cache,
                           chi::u32 tag_major, chi::u32 tag_minor,
                           const std::string &blob_name) {
  sycl::queue &q = hshm::GpuApi::SyclQueue();
  // Bundle every kernel parameter into a single shared-USM struct and
  // capture only the struct pointer in the lambda. This bypasses
  // DPC++'s host-vs-device kernel-arg layout pitfalls (the static_assert
  // at sycl/handler.hpp:669 trips when the host compiler and device
  // compiler infer different lambda capture layouts — typically when
  // mixing pointers and chi::u32 captures in templated kernels).
  struct KernelCtx {
    void *cache;
    chi::u32 cap_tag_major;
    chi::u32 cap_tag_minor;
    chi::u32 _pad;
    chi::u32 results[8];
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  if (!ctx) return 0xFFFFFFFFu;
  ctx->cache = static_cast<void *>(gpu_cache);
  ctx->cap_tag_major = tag_major;
  ctx->cap_tag_minor = tag_minor;
  ctx->_pad = 0;
  for (int i = 0; i < 8; ++i) ctx->results[i] = 0;
  KernelCtx *ctx_ptr = ctx;
  (void)blob_name;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<KernelTag>([=]() {
      ctx_ptr->results[0] = 0;
#if HSHM_IS_DEVICE_PASS
      auto *cache =
          reinterpret_cast<wrp_cte::core::GpuMetadataCacheHeader *>(
              ctx_ptr->cache);
      if (!cache) return;
      ctx_ptr->results[0] = (cache->magic_ ==
                              wrp_cte::core::GpuMetadataCacheHeader::kMagic)
                                ? 1u : 0u;
      const auto *slots = reinterpret_cast<const wrp_cte::core::gpu_cache::GpuBlobEntry *>(
          reinterpret_cast<const char *>(cache) +
          sizeof(wrp_cte::core::GpuMetadataCacheHeader) +
          static_cast<size_t>(cache->max_tags_) *
              sizeof(wrp_cte::core::gpu_cache::GpuTagEntry));
      chi::u32 cap = cache->max_blobs_;
      ctx_ptr->results[6] = ctx_ptr->cap_tag_major;
      ctx_ptr->results[7] = ctx_ptr->cap_tag_minor;
      chi::u32 count = 0;
      chi::u32 occupied = 0;
      chi::u32 last_major = 0xFFFFFFFFu;
      chi::u32 last_minor = 0xFFFFFFFFu;
      chi::u32 cap_major = ctx_ptr->cap_tag_major;
      chi::u32 cap_minor = ctx_ptr->cap_tag_minor;
      for (chi::u32 i = 0; i < cap; ++i) {
        const auto &s = slots[i];
        chi::u32 s_state = s.state_;
        chi::u32 s_major = s.tag_major_;
        chi::u32 s_minor = s.tag_minor_;
        chi::u32 name_word =
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[0])) << 24) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[1])) << 16) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[2])) << 8) |
             static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[3]));
        ctx_ptr->results[2] ^= s_state + s_major + (s_minor << 16) + name_word;
        if (s_state != wrp_cte::core::gpu_cache::kOccupied) continue;
        ++occupied;
        last_major = s_major;
        last_minor = s_minor;
        if (s_major == cap_major && s_minor == cap_minor) ++count;
      }
      ctx_ptr->results[1] = (count > 0) ? 1u : 0u;
      ctx_ptr->results[3] = occupied;
      ctx_ptr->results[4] = last_major;
      ctx_ptr->results[5] = last_minor;
#else
      (void)ctx_ptr;
#endif
    });
  }).wait_and_throw();

  chi::u32 visible = ctx->results[1];
  std::fprintf(stderr,
               "[TRACE] visible(%s)=%u valid_magic=%u occupied=%u "
               "last_seen=(%u,%u) kernel_cap=(%u,%u) host_cap=(%u,%u)\n",
               blob_name.c_str(), visible, ctx->results[0], ctx->results[3],
               ctx->results[4], ctx->results[5],
               ctx->results[6], ctx->results[7],
               tag_major, tag_minor);
  sycl::free(ctx, q);
  return visible;
}

// Count blobs the kernel sees with a given tag id.
template <class KernelTag>
chi::u32 KernelCountBlobsForTag(
    wrp_cte::core::GpuMetadataCacheHeader *gpu_cache,
    chi::u32 tag_major, chi::u32 tag_minor) {
  sycl::queue &q = hshm::GpuApi::SyclQueue();
  // See KernelBlobVisible for why we route everything through a single
  // shared-USM struct rather than capturing chi::u32 / pointers
  // separately in the lambda.
  struct KernelCtx {
    void *cache;
    chi::u32 cap_tag_major;
    chi::u32 cap_tag_minor;
    chi::u32 _pad;
    chi::u32 results[4];
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  if (!ctx) return 0xFFFFFFFFu;
  ctx->cache = static_cast<void *>(gpu_cache);
  ctx->cap_tag_major = tag_major;
  ctx->cap_tag_minor = tag_minor;
  ctx->_pad = 0;
  for (int i = 0; i < 4; ++i) ctx->results[i] = 0;
  KernelCtx *ctx_ptr = ctx;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<KernelTag>([=]() {
      ctx_ptr->results[0] = 0;
#if HSHM_IS_DEVICE_PASS
      auto *cache =
          reinterpret_cast<wrp_cte::core::GpuMetadataCacheHeader *>(
              ctx_ptr->cache);
      if (!cache) return;
      ctx_ptr->results[0] = (cache->magic_ ==
                              wrp_cte::core::GpuMetadataCacheHeader::kMagic)
                                ? 1u : 0u;
      const auto *slots = reinterpret_cast<const wrp_cte::core::gpu_cache::GpuBlobEntry *>(
          reinterpret_cast<const char *>(cache) +
          sizeof(wrp_cte::core::GpuMetadataCacheHeader) +
          static_cast<size_t>(cache->max_tags_) *
              sizeof(wrp_cte::core::gpu_cache::GpuTagEntry));
      chi::u32 cap = cache->max_blobs_;
      chi::u32 count = 0;
      chi::u32 occupied = 0;
      chi::u32 cap_major = ctx_ptr->cap_tag_major;
      chi::u32 cap_minor = ctx_ptr->cap_tag_minor;
      for (chi::u32 i = 0; i < cap; ++i) {
        const auto &s = slots[i];
        chi::u32 s_state = s.state_;
        chi::u32 s_major = s.tag_major_;
        chi::u32 s_minor = s.tag_minor_;
        chi::u32 name_word =
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[0])) << 24) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[1])) << 16) |
            (static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[2])) << 8) |
             static_cast<chi::u32>(static_cast<unsigned char>(s.blob_name_[3]));
        ctx_ptr->results[3] ^= s_state + s_major + (s_minor << 16) + name_word;
        if (s_state != wrp_cte::core::gpu_cache::kOccupied) continue;
        ++occupied;
        if (s_major == cap_major && s_minor == cap_minor) ++count;
      }
      ctx_ptr->results[1] = count;
      ctx_ptr->results[2] = occupied;
#else
      (void)ctx_ptr;
#endif
    });
  }).wait_and_throw();

  chi::u32 count = ctx->results[1];
  std::fprintf(stderr,
               "[TRACE] count_for_tag(%u,%u)=%u valid_magic=%u "
               "total_occupied=%u\n",
               tag_major, tag_minor, count, ctx->results[0], ctx->results[2]);
  sycl::free(ctx, q);
  return count;
}

}  // namespace

// ---------------------------------------------------------------------
// TEST 3 — Step 5.2: PutBlob -> kernel sees blob; DelBlob -> kernel no
// longer sees it.
// ---------------------------------------------------------------------
TEST_CASE("GPU metadata cache: DelBlob -> kernel no longer sees the blob",
          "[sycl][gpu][metadata_cache][delblob]") {
  EnsureInit();
  auto *gpu_cache = g_gpu_cache;

  auto tag_task = g_cte_client->AsyncGetOrCreateTag("gpu_meta_cache_tag_delblob");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  const std::string blob_name = "delblob_target";
  REQUIRE(HostPutBlob(tag_id, blob_name, /*bytes=*/256) == 0);

  // Host-side sanity: the cache must contain the just-put blob.
  {
    const auto *host_lookup = wrp_cte::core::GpuCacheFindBlob(
        gpu_cache, tag_id.major_, tag_id.minor_, blob_name.c_str());
    std::fprintf(stderr,
                 "[TRACE] DelBlob pre-host: host_lookup=%p tag=(%u,%u)\n",
                 static_cast<const void *>(host_lookup),
                 tag_id.major_, tag_id.minor_);
    REQUIRE(host_lookup != nullptr);
    REQUIRE(host_lookup->size_ == 256u);
  }

  // Pre-condition: kernel sees the blob.
  REQUIRE(KernelBlobVisible<chi_sycl_gpu_meta_cache_delblob_kernel>(
              gpu_cache, tag_id.major_, tag_id.minor_, blob_name) == 1u);

  // Delete the blob.
  auto del = g_cte_client->AsyncDelBlob(tag_id, blob_name);
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);

  // Host-side: cache should report the entry as gone.
  const auto *host_lookup = wrp_cte::core::GpuCacheFindBlob(
      gpu_cache, tag_id.major_, tag_id.minor_, blob_name.c_str());
  REQUIRE(host_lookup == nullptr);

  // Post-condition: kernel no longer sees the blob.
  REQUIRE(KernelBlobVisible<chi_sycl_gpu_meta_cache_delblob_kernel>(
              gpu_cache, tag_id.major_, tag_id.minor_, blob_name) == 0u);
}

// ---------------------------------------------------------------------
// TEST 4 — Step 5.3: PutBlob N blobs under a tag; DelTag -> kernel
// sees zero of them.
// ---------------------------------------------------------------------
TEST_CASE("GPU metadata cache: DelTag -> kernel sees zero blobs for tag",
          "[sycl][gpu][metadata_cache][deltag]") {
  EnsureInit();
  auto *gpu_cache = g_gpu_cache;

  auto tag_task = g_cte_client->AsyncGetOrCreateTag("gpu_meta_cache_tag_deltag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  // Put 3 blobs under this tag.
  const std::vector<std::string> blob_names = {
      "deltag_blob_a", "deltag_blob_b", "deltag_blob_c"};
  for (const auto &n : blob_names) {
    REQUIRE(HostPutBlob(tag_id, n, /*bytes=*/256) == 0);
  }

  REQUIRE(KernelCountBlobsForTag<chi_sycl_gpu_meta_cache_deltag_kernel>(
              gpu_cache, tag_id.major_, tag_id.minor_) ==
          static_cast<chi::u32>(blob_names.size()));

  // Delete the tag.
  auto del = g_cte_client->AsyncDelTag(tag_id);
  del.Wait();
  REQUIRE(del->GetReturnCode() == 0);

  // Host-side: tag and all owned blobs should be tombstoned.
  const auto *host_tag = wrp_cte::core::GpuCacheFindTag(
      gpu_cache, tag_id.major_, tag_id.minor_);
  REQUIRE(host_tag == nullptr);
  for (const auto &n : blob_names) {
    const auto *host_blob = wrp_cte::core::GpuCacheFindBlob(
        gpu_cache, tag_id.major_, tag_id.minor_, n.c_str());
    REQUIRE(host_blob == nullptr);
  }

  REQUIRE(KernelCountBlobsForTag<chi_sycl_gpu_meta_cache_deltag_kernel>(
              gpu_cache, tag_id.major_, tag_id.minor_) == 0u);
}

// ---------------------------------------------------------------------
// TEST 5 — Step 5.4: PutBlob to a kFile target -> kernel does NOT see
// the blob (non-DRAM exclusion).
// ---------------------------------------------------------------------
TEST_CASE("GPU metadata cache: kFile blobs are NOT projected into the cache",
          "[sycl][gpu][metadata_cache][filetier]") {
  EnsureInit();
  auto *gpu_cache = g_gpu_cache;

  auto tag_task = g_cte_client->AsyncGetOrCreateTag("gpu_meta_cache_tag_filetier");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  wrp_cte::core::TagId tag_id = tag_task->tag_id_;

  // Force placement on the FILE target by giving the put a score that
  // disqualifies the RAM target. CTE's DPE places the lowest-score blobs
  // on the highest-persistence target available; FILE has higher
  // persistence than RAM, so a score near 0.0 lands on FILE.
  //
  // (DPE selection is policy-dependent; if this proves fragile, the
  // alternative is to ONLY register the FILE target for this test —
  // but we share targets via EnsureInit, so we use score steering.)
  const std::string blob_name = "filetier_blob";
  const size_t kBlobBytes = 256;
  std::vector<char> pattern(kBlobBytes, static_cast<char>(0x5C));
  auto *ipc = CHI_IPC;
  hipc::FullPtr<char> buf = ipc->AllocateBuffer(kBlobBytes);
  REQUIRE(!buf.IsNull());
  std::memcpy(buf.ptr_, pattern.data(), kBlobBytes);
  hipc::ShmPtr<> shm(buf.shm_);
  auto put_task = g_cte_client->AsyncPutBlob(
      tag_id, blob_name, /*offset=*/0, kBlobBytes, shm,
      /*score=*/0.0f);
  put_task.Wait();
  REQUIRE(put_task->GetReturnCode() == 0);
  ipc->FreeBuffer(buf);

  // Host-side check: the cache must NOT contain the blob (or, if DPE
  // happened to place it on RAM regardless, the StorageClass must NOT
  // be Ram/Hbm/Pinned — but the helper evicts non-DRAM blobs, so
  // GpuCacheFindBlob should return nullptr).
  const auto *host_lookup = wrp_cte::core::GpuCacheFindBlob(
      gpu_cache, tag_id.major_, tag_id.minor_, blob_name.c_str());
  if (host_lookup != nullptr) {
    // Blob landed on a GPU-visible tier despite our score steering;
    // log and skip the kernel-side check.
    std::fprintf(
        stderr,
        "[INFO] file-tier test: blob landed on storage_class=%u "
        "(DPE chose GPU-visible tier); skipping exclusion assertion.\n",
        host_lookup->storage_class_);
    REQUIRE(wrp_cte::core::gpu_cache::IsGpuVisible(host_lookup->storage_class_));
    return;
  }

  // Kernel-side: visible?
  REQUIRE(KernelBlobVisible<chi_sycl_gpu_meta_cache_filetier_kernel>(
              gpu_cache, tag_id.major_, tag_id.minor_, blob_name) == 0u);
}

SIMPLE_TEST_MAIN()

#else  // !HSHM_ENABLE_SYCL or CUDA/ROCm path

int main() { return 0; }

#endif  // HSHM_ENABLE_SYCL && !(CUDA||ROCM)
