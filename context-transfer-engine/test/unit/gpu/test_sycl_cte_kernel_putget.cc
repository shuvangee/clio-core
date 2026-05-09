/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * SYCL kernel-side PutBlob + GetBlob round-trip test.
 *
 * What this validates (Phases 1-2 of the kernel-side CTE plan):
 *
 *   A SYCL single_task running on the GPU device:
 *     1. CHIMAERA_GPU_INIT — bind kernel-scope IpcManager (gpu_alloc_
 *        backed by gpu2cpu_backend, the pinned-host SYCL malloc_host
 *        region the runtime sets up in ServerInit).
 *     2. CHI_IPC->AllocateBuffer(N) — carve a host-pinned data buffer
 *        out of gpu_alloc_; fill it from the device with a deterministic
 *        pattern.
 *     3. CHI_IPC->NewTask<PutBlobTask>(...) — placement-new a PutBlobTask
 *        and a trailing FutureShm in the same backend region (NewTask
 *        appends sizeof(FutureShm) past the task). Routes via
 *        PoolQuery::ToLocalCpu() so the gpu2cpu_queue carries it to a
 *        CPU worker, which dispatches wrp_cte::core::Runtime::PutBlob.
 *     4. future.Wait() — spin on FUTURE_COMPLETE.
 *     5. AllocateBuffer + NewTask<GetBlobTask>(...) — same shape; CTE
 *        Runtime::GetBlob memcpys the bdev-stored bytes into the kernel-
 *        owned ShmPtr.
 *     6. After GetBlob's future completes, the kernel byte-compares the
 *        receive buffer against the pattern it originally wrote.
 *
 * Why each step matters:
 *
 *   - Step 3 is the first time a CTE task (rather than the trivial
 *     MOD_NAME::GpuSubmitTask used in test_sycl_chimod_to_cpu.cc) rides
 *     the gpu2cpu_queue. PutBlobTask carries a chi::priv::string
 *     blob_name_ — which fits in its 32-byte SSO buffer for short names
 *     and therefore does not invoke the allocator at construction time
 *     (verified in string.h:781-833). The struct's sizeof is identical
 *     across host and device passes because the AllocT* template
 *     parameter is held only as a pointer.
 *   - Step 5 closes the loop on the GPU side: a kernel that produced
 *     data can also consume it back through the same CPU service,
 *     proving the path is symmetric and that the bdev stores bytes the
 *     CTE server can hand back faithfully.
 *
 * Layout discipline (shared with test_sycl_gpu_metadata_cache.cc):
 *
 *   All kernel parameters are bundled into a single shared-USM
 *   KernelCtx struct; the lambda captures only the struct pointer. This
 *   avoids the DPC++ static_assert "Unexpected kernel lambda size" that
 *   trips when the host pass and device pass infer different layouts
 *   for a multi-capture kernel lambda — and the related kernel-arg
 *   aliasing where chi::u32 captures end up at offsets that overlap an
 *   adjacent char* / void* capture's bytes.
 */

#if HSHM_ENABLE_SYCL && !(HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM)

#include "simple_test.h"

#include <chimaera/chimaera.h>
#include <chimaera/singletons.h>
#include <chimaera/types.h>
#include <chimaera/pool_query.h>
#include <chimaera/gpu/future.h>
#include <chimaera/gpu/gpu_info.h>
#include <chimaera/gpu/gpu_ipc_manager.h>
#include <chimaera/bdev/bdev_client.h>
#include <chimaera/bdev/bdev_tasks.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cte/core/core_tasks.h>

#include <hermes_shm/util/gpu_api.h>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

// ---------- Shared bring-up ----------
bool g_initialized = false;
chi::PoolId g_bdev_pool_id(907, 0);
wrp_cte::core::Client *g_cte_client = nullptr;
wrp_cte::core::TagId g_tag_id;
const std::string g_target_name = "kernel_putget_ram_target";
const size_t kRamTargetBytes = 4ULL * 1024 * 1024;

void EnsureInit() {
  if (g_initialized) return;

  std::fprintf(stderr, "[INIT] Starting Chimaera server (SYCL backend)...\n");
  REQUIRE(chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer));

  auto *ipc = CHI_CPU_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  REQUIRE(ipc->GetGpuQueueCount() == 1u);

  // CTE pool — default config is fine; we only need PutBlob/GetBlob.
  REQUIRE(wrp_cte::core::WRP_CTE_CLIENT_INIT());
  g_cte_client = WRP_CTE_CLIENT;
  REQUIRE(g_cte_client != nullptr);
  g_cte_client->Init(wrp_cte::core::kCtePoolId);

  wrp_cte::core::CreateParams params;
  auto pool_task = g_cte_client->AsyncCreate(
      chi::PoolQuery::Dynamic(),
      wrp_cte::core::kCtePoolName,
      wrp_cte::core::kCtePoolId, params);
  pool_task.Wait();
  REQUIRE(pool_task->GetReturnCode() == 0);

  // RAM bdev target — keeps the test independent of the filesystem.
  chimaera::bdev::Client bdev_client(g_bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      g_target_name, g_bdev_pool_id,
      chimaera::bdev::BdevType::kRam, kRamTargetBytes);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);

  auto reg_task = g_cte_client->AsyncRegisterTarget(
      g_target_name, chimaera::bdev::BdevType::kRam,
      kRamTargetBytes, chi::PoolQuery::Local(), g_bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);

  // Pre-create the tag from the host so the kernel only has to do
  // PutBlob / GetBlob (the kernel-side CTE path doesn't include
  // GetOrCreateTag yet — separate task).
  auto tag_task = g_cte_client->AsyncGetOrCreateTag("kernel_putget_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  g_tag_id = tag_task->tag_id_;

  g_initialized = true;
  std::fprintf(stderr, "[INIT] Ready (tag_id=(%u,%u))\n",
               g_tag_id.major_, g_tag_id.minor_);
}

// Opaque kernel-name types — each SYCL submission needs a distinct one.
class chi_sycl_cte_kernel_putget_bare;
class chi_sycl_cte_kernel_putget_main;
class chi_sycl_cte_kernel_putget_hbm_fill;
class chi_sycl_cte_kernel_putget_hbm_io;

}  // namespace

// ---------------------------------------------------------------------
// TEST 1 — bare SYCL sanity, no Chimaera. Same shape as
// test_sycl_chimod_to_cpu.cc's first TEST_CASE so a failure here points
// at SYCL/USM, not at the runtime.
// ---------------------------------------------------------------------
TEST_CASE("SYCL bare kernel sanity (no chimaera init)",
          "[sycl][gpu2cpu][cte][bare]") {
  sycl::queue bare_q{sycl::gpu_selector_v};
  int *bare_done = sycl::malloc_shared<int>(1, bare_q);
  REQUIRE(bare_done != nullptr);
  *bare_done = 0;
  bare_q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_cte_kernel_putget_bare>([=]() {
      *bare_done = 0xBEEF;
    });
  }).wait_and_throw();
  REQUIRE(*bare_done == 0xBEEF);
  sycl::free(bare_done, bare_q);
}

// ---------------------------------------------------------------------
// TEST 2 — kernel-side PutBlob followed by kernel-side GetBlob.
// Verifies a single SYCL kernel can issue both halves of the CTE
// round trip and observe byte-identical data on return.
// ---------------------------------------------------------------------
TEST_CASE("SYCL kernel: PutBlob + GetBlob round trip via gpu2cpu_queue",
          "[sycl][gpu2cpu][cte][kernel_putget]") {
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  // Pull the populated IpcManagerGpuInfo (gpu2cpu_queue + gpu2cpu_backend).
  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(/*size=*/0, /*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
  REQUIRE(gpu_info.gpu2cpu_backend.data_ != nullptr);

  // Fresh GPU queue, NOT hshm::GpuApi::SyclQueue() — see
  // test_sycl_chimod_to_cpu.cc:171-176 for the no-op-on-resubmit issue
  // the singleton has after CHIMAERA_INIT consumed it.
  sycl::queue q{sycl::gpu_selector_v};

  // Stage non-trivially-copyable IpcManagerGpuInfo + a kernel-scope
  // gpu::IpcManager in shared USM (memcpy in, capture pointers).
  auto *info_storage = sycl::malloc_shared<chi::IpcManagerGpuInfo>(1, q);
  REQUIRE(info_storage != nullptr);
  std::memcpy(static_cast<void *>(info_storage), &gpu_info,
              sizeof(chi::IpcManagerGpuInfo));

  auto *ipc_storage = sycl::malloc_shared<chi::gpu::IpcManager>(1, q);
  REQUIRE(ipc_storage != nullptr);
  new (ipc_storage) chi::gpu::IpcManager();

  // Fixed-size name buffer (well under priv::string's 32-byte SSO).
  static constexpr chi::u32 kBlobNameMax = 32;
  static const char kBlobName[] = "krn_blob";  // length 8, fits SSO

  // KernelCtx bundles every parameter the kernel reads and every
  // result it writes. Capturing only ctx_ptr in the lambda sidesteps
  // the DPC++ host vs device kernel-arg layout mismatch documented in
  // test_sycl_gpu_metadata_cache.cc.
  struct KernelCtx {
    // IN
    void *info_storage;
    void *ipc_storage;
    chi::PoolId cte_pool_id;
    wrp_cte::core::TagId tag_id;
    chi::u32 gpu_id;
    chi::u32 blob_size_bytes;
    chi::u32 pattern_seed;       // kernel writes byte i = (seed ^ i) & 0xFF
    char blob_name[kBlobNameMax];
    // OUT — every stage stamps a distinct value so a failure points
    // at the exact step that stalled.
    chi::u32 stage;              // 1..10
    chi::u32 put_rc;             // PutBlobTask return_code_
    chi::u32 get_rc;             // GetBlobTask return_code_
    chi::u32 alloc_put_ok;       // 1 if AllocateBuffer for put succeeded
    chi::u32 alloc_get_ok;       // 1 if AllocateBuffer for get succeeded
    chi::u32 newtask_put_ok;     // 1 if NewTask<PutBlobTask> non-null
    chi::u32 newtask_get_ok;     // 1 if NewTask<GetBlobTask> non-null
    chi::u32 first_mismatch_idx; // 0xFFFFFFFF if no mismatch
    chi::u32 mismatch_got;       // byte we read at first mismatch
    chi::u32 mismatch_want;      // byte we expected
    chi::u32 bytes_match;        // 1 if all kBlobBytes match
    chi::u32 done;               // 1 when the kernel finished cleanly
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  REQUIRE(ctx != nullptr);
  std::memset(ctx, 0, sizeof(KernelCtx));
  ctx->info_storage = static_cast<void *>(info_storage);
  ctx->ipc_storage = static_cast<void *>(ipc_storage);
  ctx->cte_pool_id = wrp_cte::core::kCtePoolId;
  ctx->tag_id = g_tag_id;
  ctx->gpu_id = 0;
  ctx->blob_size_bytes = 256;          // small enough to fit gpu2cpu_backend
  ctx->pattern_seed = 0x5Au;
  std::strncpy(ctx->blob_name, kBlobName, kBlobNameMax - 1);
  ctx->blob_name[kBlobNameMax - 1] = '\0';
  ctx->first_mismatch_idx = 0xFFFFFFFFu;

  KernelCtx *ctx_ptr = ctx;

  std::fprintf(stderr,
               "[TRACE] Launching kernel-side PutBlob+GetBlob "
               "(blob_size=%u, name='%s')\n",
               ctx->blob_size_bytes, ctx->blob_name);

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_cte_kernel_putget_main>([=]() {
      // Unconditional write to keep DPC++ from eliding the kernel
      // (same trick as test_sycl_chimod_to_cpu.cc).
      ctx_ptr->stage = 1;
#if HSHM_IS_DEVICE_PASS
      // Force the device pass to load info_storage->backend.data_
      // before CHIMAERA_GPU_INIT, otherwise DPC++'s optimizer leaves
      // gpu_alloc_ unset (observed in test_sycl_chimod_to_cpu.cc).
      auto *info = reinterpret_cast<chi::IpcManagerGpuInfo *>(
          ctx_ptr->info_storage);
      auto *ipc_obj = reinterpret_cast<chi::gpu::IpcManager *>(
          ctx_ptr->ipc_storage);
      if (info->backend.data_ != nullptr) {
        auto *pre_alloc = reinterpret_cast<hipc::RoundRobinAllocator *>(
            info->backend.data_);
        (void)pre_alloc->heap_ready_.load();
      }
      CHIMAERA_GPU_INIT(*info, ipc_obj);
      ctx_ptr->stage = 2;

      // ---- PutBlob ----
      chi::u32 size = ctx_ptr->blob_size_bytes;
      chi::u32 seed = ctx_ptr->pattern_seed;

      auto put_buf = CHI_IPC->AllocateBuffer(size);
      if (put_buf.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->alloc_put_ok = 1;
      ctx_ptr->stage = 3;

      // Fill put_buf with deterministic pattern from device.
      char *put_bytes = put_buf.ptr_;
      for (chi::u32 i = 0; i < size; ++i) {
        put_bytes[i] = static_cast<char>((seed ^ i) & 0xFFu);
      }
      ctx_ptr->stage = 4;

      // The gpu_alloc_ that backs CHI_IPC->AllocateBuffer is a kernel-side
      // PrivateBuddyAllocator carved out of the gpu2cpu_backend. Its
      // AllocatorId is NOT registered in the CPU IpcManager's
      // gpu_alloc_map_ under SYCL (the ToFullPtr GPU-allocator branch
      // is gated on CUDA/ROCm only — see ipc_manager.h:949). So the
      // CPU bdev would fail to resolve a ShmPtr that carries the
      // gpu_alloc_'s id. Workaround: set alloc_id_=null and stash the
      // raw host VA in off_. ToFullPtr's null-alloc_id branch (line 925)
      // returns FullPtr(raw_ptr) directly. The buffer lives in pinned
      // host memory (sycl::malloc_host), so the CPU dereferences the
      // same VA the kernel sees.
      hipc::ShmPtr<> put_shm;
      put_shm.alloc_id_ = hipc::AllocatorId::GetNull();
      put_shm.off_ = reinterpret_cast<size_t>(put_bytes);
      auto put_task = CHI_IPC->NewTask<wrp_cte::core::PutBlobTask>(
          chi::CreateTaskId(), ctx_ptr->cte_pool_id,
          chi::PoolQuery::ToLocalCpu(),
          ctx_ptr->tag_id, ctx_ptr->blob_name,
          /*offset=*/static_cast<chi::u64>(0),
          static_cast<chi::u64>(size),
          put_shm,
          /*score=*/-1.0f,
          wrp_cte::core::Context(),
          /*flags=*/0u);
      if (put_task.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->newtask_put_ok = 1;
      ctx_ptr->stage = 5;

      auto put_future = CHI_IPC->Send(put_task);
      ctx_ptr->stage = 6;
      put_future.Wait();
      ctx_ptr->put_rc = put_task->return_code_.load();
      ctx_ptr->stage = 7;

      // ---- GetBlob ----
      auto get_buf = CHI_IPC->AllocateBuffer(size);
      if (get_buf.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->alloc_get_ok = 1;
      // Zero the receive buffer so any partial fill is detectable.
      char *get_bytes = get_buf.ptr_;
      for (chi::u32 i = 0; i < size; ++i) get_bytes[i] = 0;
      ctx_ptr->stage = 8;

      // Same null-alloc_id workaround as the put_shm (see comment above).
      hipc::ShmPtr<> get_shm;
      get_shm.alloc_id_ = hipc::AllocatorId::GetNull();
      get_shm.off_ = reinterpret_cast<size_t>(get_bytes);
      auto get_task = CHI_IPC->NewTask<wrp_cte::core::GetBlobTask>(
          chi::CreateTaskId(), ctx_ptr->cte_pool_id,
          chi::PoolQuery::ToLocalCpu(),
          ctx_ptr->tag_id, ctx_ptr->blob_name,
          /*offset=*/static_cast<chi::u64>(0),
          static_cast<chi::u64>(size),
          /*flags=*/0u,
          get_shm);
      if (get_task.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->newtask_get_ok = 1;
      ctx_ptr->stage = 9;

      auto get_future = CHI_IPC->Send(get_task);
      get_future.Wait();
      ctx_ptr->get_rc = get_task->return_code_.load();
      ctx_ptr->stage = 10;

      // Byte-compare get_buf against the same pattern we wrote.
      chi::u32 first_bad = 0xFFFFFFFFu;
      chi::u32 got = 0;
      chi::u32 want = 0;
      for (chi::u32 i = 0; i < size; ++i) {
        unsigned char want_b = static_cast<unsigned char>((seed ^ i) & 0xFFu);
        unsigned char got_b = static_cast<unsigned char>(get_bytes[i]);
        if (got_b != want_b) {
          first_bad = i;
          got = got_b;
          want = want_b;
          break;
        }
      }
      ctx_ptr->first_mismatch_idx = first_bad;
      ctx_ptr->mismatch_got = got;
      ctx_ptr->mismatch_want = want;
      ctx_ptr->bytes_match = (first_bad == 0xFFFFFFFFu) ? 1u : 0u;

      // Cleanup: free buffers and tasks the kernel allocated.
      // (Tasks: CPU runtime is the canonical free-er for gpu2cpu tasks
      //  per the comment in IpcGpu2Cpu::ClientRecv. Buffers we own.)
      CHI_IPC->FreeBuffer(put_buf);
      CHI_IPC->FreeBuffer(get_buf);
#else
      (void)ctx_ptr;
#endif
      ctx_ptr->done = 1;
    });
  }).wait_and_throw();

  // Poll for done after wait_and_throw so the host sees the writes
  // through USM-host coherence.
  auto t0 = std::chrono::steady_clock::now();
  while (ctx->done == 0u) {
    std::this_thread::sleep_for(100us);
    if (std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t0).count() > 10.0f) break;
  }
  float ms = std::chrono::duration<float, std::milli>(
                 std::chrono::steady_clock::now() - t0).count();

  std::fprintf(stderr,
               "[TRACE] kernel done=%u stage=%u\n"
               "        alloc_put_ok=%u alloc_get_ok=%u\n"
               "        newtask_put_ok=%u newtask_get_ok=%u\n"
               "        put_rc=%u get_rc=%u\n"
               "        bytes_match=%u first_mismatch_idx=0x%x "
               "(got=0x%x want=0x%x)\n"
               "        elapsed=%.2f ms\n",
               ctx->done, ctx->stage,
               ctx->alloc_put_ok, ctx->alloc_get_ok,
               ctx->newtask_put_ok, ctx->newtask_get_ok,
               ctx->put_rc, ctx->get_rc,
               ctx->bytes_match, ctx->first_mismatch_idx,
               ctx->mismatch_got, ctx->mismatch_want, ms);

  REQUIRE(ctx->done == 1u);
  REQUIRE(ctx->stage == 10u);
  REQUIRE(ctx->alloc_put_ok == 1u);
  REQUIRE(ctx->newtask_put_ok == 1u);
  REQUIRE(ctx->put_rc == 0u);
  REQUIRE(ctx->alloc_get_ok == 1u);
  REQUIRE(ctx->newtask_get_ok == 1u);
  REQUIRE(ctx->get_rc == 0u);
  REQUIRE(ctx->bytes_match == 1u);

  ipc_storage->~IpcManager();
  sycl::free(static_cast<void *>(ipc_storage), q);
  sycl::free(static_cast<void *>(info_storage), q);
  sycl::free(ctx, q);
}

// ---------------------------------------------------------------------
// TEST 3 (Phase 3) — kernel-side PutBlob / GetBlob with HBM (device
// USM) data buffers, exercising bdev's device-aware memcpy hook.
//
// Flow:
//   1. Host allocates two HBM buffers via sycl::malloc_device on a
//      fresh queue (PUT-side and GET-side).
//   2. A first kernel writes a deterministic pattern into the PUT-side
//      HBM buffer (proves the buffer is real device USM and the kernel
//      can fill it from device code — host can't dereference it).
//   3. The IO kernel calls PutBlob with a ShmPtr whose off_ is the
//      PUT-side HBM device address (alloc_id_=null for ToFullPtr's
//      raw-VA branch). The CTE server -> bdev's WriteToRam is
//      compiled WITHOUT -fsycl, so it can't call SYCL APIs directly —
//      instead it calls chi::DeviceAwareMemcpy (chimaera/device_memcpy.h)
//      whose hook was installed by ServerInitGpuQueuesSycl. The hook
//      routes through sycl::queue::memcpy which dispatches based on
//      USM pointer kind, so the device-USM source is read correctly.
//   4. Same kernel calls GetBlob with the GET-side HBM device pointer.
//      ReadFromRam similarly hits the hook for host-RAM-buffer ->
//      device-USM-dest.
//   5. Host reads the GET-side HBM buffer back to host (sycl::queue::memcpy
//      device -> host) and byte-compares against the pattern.
// ---------------------------------------------------------------------
TEST_CASE("SYCL kernel: PutBlob/GetBlob with HBM device USM data buffers",
          "[sycl][gpu2cpu][cte][kernel_putget][hbm_data]") {
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(/*size=*/0, /*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
  REQUIRE(gpu_info.gpu2cpu_backend.data_ != nullptr);

  sycl::queue q{sycl::gpu_selector_v};

  const chi::u32 kBlobBytes = 256;
  const chi::u32 kPatternSeed = 0xC3u;

  // ---- Step 1: allocate HBM buffers on the host. ----
  char *put_hbm = sycl::malloc_device<char>(kBlobBytes, q);
  REQUIRE(put_hbm != nullptr);
  char *get_hbm = sycl::malloc_device<char>(kBlobBytes, q);
  REQUIRE(get_hbm != nullptr);

  // Zero the GET buffer up front so any byte left untouched after
  // GetBlob is detectable as a mismatch.
  q.memset(get_hbm, 0, kBlobBytes).wait_and_throw();

  // ---- Step 2: fill put_hbm from a SYCL kernel (proves device USM). ----
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_cte_kernel_putget_hbm_fill>([=]() {
      // Each work-item is the whole task (single_task). Fill the entire
      // buffer with the deterministic pattern.
      for (chi::u32 i = 0; i < kBlobBytes; ++i) {
        put_hbm[i] = static_cast<char>((kPatternSeed ^ i) & 0xFFu);
      }
    });
  }).wait_and_throw();

  // ---- Stage non-trivially-copyable IpcManagerGpuInfo + kernel-scope
  // gpu::IpcManager + KernelCtx in shared USM (same pattern as TEST 2). ----
  auto *info_storage = sycl::malloc_shared<chi::IpcManagerGpuInfo>(1, q);
  REQUIRE(info_storage != nullptr);
  std::memcpy(static_cast<void *>(info_storage), &gpu_info,
              sizeof(chi::IpcManagerGpuInfo));

  auto *ipc_storage = sycl::malloc_shared<chi::gpu::IpcManager>(1, q);
  REQUIRE(ipc_storage != nullptr);
  new (ipc_storage) chi::gpu::IpcManager();

  static const char kHbmBlobName[] = "hbm_blob";

  struct KernelCtx {
    void *info_storage;
    void *ipc_storage;
    chi::PoolId cte_pool_id;
    wrp_cte::core::TagId tag_id;
    void *put_hbm;
    void *get_hbm;
    chi::u32 blob_size_bytes;
    char blob_name[32];
    chi::u32 stage;
    chi::u32 put_rc;
    chi::u32 get_rc;
    chi::u32 newtask_put_ok;
    chi::u32 newtask_get_ok;
    chi::u32 done;
  };
  auto *ctx = sycl::malloc_shared<KernelCtx>(1, q);
  REQUIRE(ctx != nullptr);
  std::memset(ctx, 0, sizeof(KernelCtx));
  ctx->info_storage = static_cast<void *>(info_storage);
  ctx->ipc_storage = static_cast<void *>(ipc_storage);
  ctx->cte_pool_id = wrp_cte::core::kCtePoolId;
  ctx->tag_id = g_tag_id;
  ctx->put_hbm = static_cast<void *>(put_hbm);
  ctx->get_hbm = static_cast<void *>(get_hbm);
  ctx->blob_size_bytes = kBlobBytes;
  std::strncpy(ctx->blob_name, kHbmBlobName, sizeof(ctx->blob_name) - 1);
  ctx->blob_name[sizeof(ctx->blob_name) - 1] = '\0';
  KernelCtx *ctx_ptr = ctx;

  std::fprintf(stderr,
               "[TRACE] HBM-data PutBlob+GetBlob: put_hbm=%p get_hbm=%p "
               "size=%u\n",
               static_cast<void *>(put_hbm),
               static_cast<void *>(get_hbm), kBlobBytes);

  // ---- Step 3-4: kernel issues PutBlob then GetBlob with HBM ShmPtrs. ----
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<chi_sycl_cte_kernel_putget_hbm_io>([=]() {
      ctx_ptr->stage = 1;
#if HSHM_IS_DEVICE_PASS
      auto *info = reinterpret_cast<chi::IpcManagerGpuInfo *>(
          ctx_ptr->info_storage);
      auto *ipc_obj = reinterpret_cast<chi::gpu::IpcManager *>(
          ctx_ptr->ipc_storage);
      if (info->backend.data_ != nullptr) {
        auto *pre_alloc = reinterpret_cast<hipc::RoundRobinAllocator *>(
            info->backend.data_);
        (void)pre_alloc->heap_ready_.load();
      }
      CHIMAERA_GPU_INIT(*info, ipc_obj);
      ctx_ptr->stage = 2;

      // PutBlob: ShmPtr points directly at the HBM device USM buffer.
      // The CPU bdev's WriteToRam will resolve this via ToFullPtr's
      // null-alloc_id branch (off_ is the raw VA) and then hit
      // chi::DeviceAwareMemcpy, which is wired to sycl::queue::memcpy.
      hipc::ShmPtr<> put_shm;
      put_shm.alloc_id_ = hipc::AllocatorId::GetNull();
      put_shm.off_ = reinterpret_cast<size_t>(ctx_ptr->put_hbm);

      auto put_task = CHI_IPC->NewTask<wrp_cte::core::PutBlobTask>(
          chi::CreateTaskId(), ctx_ptr->cte_pool_id,
          chi::PoolQuery::ToLocalCpu(),
          ctx_ptr->tag_id, ctx_ptr->blob_name,
          /*offset=*/static_cast<chi::u64>(0),
          static_cast<chi::u64>(ctx_ptr->blob_size_bytes),
          put_shm,
          /*score=*/-1.0f,
          wrp_cte::core::Context(),
          /*flags=*/0u);
      if (put_task.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->newtask_put_ok = 1;
      ctx_ptr->stage = 3;

      auto put_future = CHI_IPC->Send(put_task);
      put_future.Wait();
      ctx_ptr->put_rc = put_task->return_code_.load();
      ctx_ptr->stage = 4;

      // GetBlob into a different HBM buffer.
      hipc::ShmPtr<> get_shm;
      get_shm.alloc_id_ = hipc::AllocatorId::GetNull();
      get_shm.off_ = reinterpret_cast<size_t>(ctx_ptr->get_hbm);

      auto get_task = CHI_IPC->NewTask<wrp_cte::core::GetBlobTask>(
          chi::CreateTaskId(), ctx_ptr->cte_pool_id,
          chi::PoolQuery::ToLocalCpu(),
          ctx_ptr->tag_id, ctx_ptr->blob_name,
          /*offset=*/static_cast<chi::u64>(0),
          static_cast<chi::u64>(ctx_ptr->blob_size_bytes),
          /*flags=*/0u,
          get_shm);
      if (get_task.IsNull()) { ctx_ptr->done = 1; return; }
      ctx_ptr->newtask_get_ok = 1;
      ctx_ptr->stage = 5;

      auto get_future = CHI_IPC->Send(get_task);
      get_future.Wait();
      ctx_ptr->get_rc = get_task->return_code_.load();
      ctx_ptr->stage = 6;
#else
      (void)ctx_ptr;
#endif
      ctx_ptr->done = 1;
    });
  }).wait_and_throw();

  // ---- Step 5: copy GET-side HBM back to host and byte-compare. ----
  std::vector<char> host_check(kBlobBytes, 0);
  q.memcpy(host_check.data(), get_hbm, kBlobBytes).wait_and_throw();

  chi::u32 first_bad = 0xFFFFFFFFu;
  unsigned char got = 0, want = 0;
  for (chi::u32 i = 0; i < kBlobBytes; ++i) {
    unsigned char want_b = static_cast<unsigned char>((kPatternSeed ^ i) & 0xFFu);
    unsigned char got_b = static_cast<unsigned char>(host_check[i]);
    if (got_b != want_b) {
      first_bad = i;
      got = got_b;
      want = want_b;
      break;
    }
  }
  std::fprintf(stderr,
               "[TRACE] HBM-data result: stage=%u put_rc=%u get_rc=%u "
               "newtask_put=%u newtask_get=%u "
               "first_mismatch=0x%x (got=0x%x want=0x%x)\n",
               ctx->stage, ctx->put_rc, ctx->get_rc,
               ctx->newtask_put_ok, ctx->newtask_get_ok,
               first_bad, got, want);

  REQUIRE(ctx->done == 1u);
  REQUIRE(ctx->stage == 6u);
  REQUIRE(ctx->newtask_put_ok == 1u);
  REQUIRE(ctx->put_rc == 0u);
  REQUIRE(ctx->newtask_get_ok == 1u);
  REQUIRE(ctx->get_rc == 0u);
  REQUIRE(first_bad == 0xFFFFFFFFu);

  ipc_storage->~IpcManager();
  sycl::free(static_cast<void *>(ipc_storage), q);
  sycl::free(static_cast<void *>(info_storage), q);
  sycl::free(ctx, q);
  sycl::free(put_hbm, q);
  sycl::free(get_hbm, q);
}

SIMPLE_TEST_MAIN()

#else  // !HSHM_ENABLE_SYCL or CUDA/ROCm

int main() { return 0; }

#endif  // HSHM_ENABLE_SYCL && !(CUDA||ROCM)
