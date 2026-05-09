/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 * This file is part of IOWarp Core. BSD 3-Clause License.
 */

/**
 * CUDA / ROCm kernel-side PutBlob + GetBlob round-trip test.
 *
 * Direct analog of test_sycl_cte_kernel_putget.cc, written using HSHM
 * portability macros + hshm::GpuApi so the same source compiles for
 * BOTH CUDA (via nvcc) and ROCm (via hipcc). The two backends share
 * the `<<<>>>` kernel launch syntax and HIP's runtime API mirrors
 * CUDA's, so the only build-time choice is which compiler to invoke;
 * the source itself doesn't fork on HSHM_ENABLE_CUDA vs HSHM_ENABLE_ROCM.
 *
 * Extension is `.cc` per project policy. The CMake target that builds
 * this file should mark it as CUDA / HIP language (typically via
 * add_cuda_executable or set_source_files_properties LANGUAGE CUDA)
 * so nvcc / hipcc compiles it — the `<<<>>>` launch syntax and
 * HSHM_GPU_KERNEL macro require a GPU compiler even though the
 * extension would normally suggest plain C++.
 *
 * What this validates:
 *
 *   A single `__global__` kernel (HSHM_GPU_KERNEL) running with one
 *   thread:
 *     1. CHIMAERA_GPU_INIT — bind kernel-scope IpcManager.
 *     2. CHI_IPC->AllocateBuffer / write pattern.
 *     3. CHI_IPC->NewTask<PutBlobTask>(...) + Send + future.Wait().
 *     4. CHI_IPC->NewTask<GetBlobTask>(...) + Send + future.Wait().
 *     5. Byte-compare the GET buffer against the pattern in-kernel.
 *
 * The CPU-side flow is identical to the SYCL test: gpu2cpu_queue
 * carries the task to a CPU worker which dispatches into the standard
 * chi::Container path; the bdev's WriteToRam/ReadFromRam use
 * chi::DeviceAwareMemcpy (hooked by ServerInitGpuQueuesHip in
 * gpu2cpu_init_hip.cc to dispatch through cudaMemcpy / hipMemcpy).
 *
 * A second TEST_CASE (when implemented) should mirror the SYCL HBM
 * test: host pre-allocates a device USM buffer (cudaMalloc / hipMalloc),
 * a fill kernel populates it, then the IO kernel calls Put/Get with
 * the device pointer in the ShmPtr. The bdev's device-aware memcpy
 * hook handles the device-source / device-dest copies. Stubbed below.
 */

#if (HSHM_ENABLE_CUDA || HSHM_ENABLE_ROCM) && !HSHM_ENABLE_SYCL

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

// hshm::GpuApi handles cudaXxx / hipXxx runtime calls — no direct
// cuda_runtime.h / hip_runtime.h includes needed in this TU. The
// `<<<>>>` launch syntax and HSHM_GPU_KERNEL macro are picked up
// from whichever GPU compiler (nvcc / hipcc) is invoking the build.

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
#if HSHM_IS_HOST
wrp_cte::core::Client *g_cte_client = nullptr;
#endif
wrp_cte::core::TagId g_tag_id;
const size_t kRamTargetBytes = 4ULL * 1024 * 1024;

#if HSHM_IS_HOST
const std::string g_target_name = "kernel_putget_ram_target";

void EnsureInit() {
  if (g_initialized) return;

  std::fprintf(stderr, "[INIT] Starting Chimaera server (CUDA/HIP backend)...\n");
  REQUIRE(chi::CHIMAERA_INIT(chi::ChimaeraMode::kServer));
  std::fprintf(stderr, "[STAGE] CHIMAERA_INIT ok\n");

  auto *ipc = CHI_CPU_IPC;
  REQUIRE(ipc != nullptr);
  std::fprintf(stderr, "[STAGE] CHI_CPU_IPC ok\n");
  REQUIRE(ipc->GetGpuIpcManager() != nullptr);
  std::fprintf(stderr, "[STAGE] GetGpuIpcManager ok\n");

  REQUIRE(wrp_cte::core::WRP_CTE_CLIENT_INIT());
  std::fprintf(stderr, "[STAGE] WRP_CTE_CLIENT_INIT ok\n");
  g_cte_client = WRP_CTE_CLIENT;
  REQUIRE(g_cte_client != nullptr);
  g_cte_client->Init(wrp_cte::core::kCtePoolId);
  std::fprintf(stderr, "[STAGE] cte_client->Init ok\n");

  // The CTE pool is auto-composed at CHIMAERA_INIT (visible in the log
  // as "Creating pool 'wrp_cte_core'") so we skip the explicit
  // AsyncCreate that the SYCL test does. Calling it again on the CUDA
  // path crashes during future processing — likely a stale call site
  // that depended on the GPU-runtime-era client init we removed.
  // Track as follow-up: harden Future<CreateTask>::Wait against
  // duplicate-pool server responses on the CUDA path.
  std::fprintf(stderr, "[STAGE] using auto-composed CTE pool\n");

  chimaera::bdev::Client bdev_client(g_bdev_pool_id);
  auto bdev_create = bdev_client.AsyncCreate(
      chi::PoolQuery::Dynamic(),
      g_target_name, g_bdev_pool_id,
      chimaera::bdev::BdevType::kRam, kRamTargetBytes);
  bdev_create.Wait();
  REQUIRE(bdev_create->GetReturnCode() == 0);
  std::fprintf(stderr, "[STAGE] bdev create done\n");

  auto reg_task = g_cte_client->AsyncRegisterTarget(
      g_target_name, chimaera::bdev::BdevType::kRam,
      kRamTargetBytes, chi::PoolQuery::Local(), g_bdev_pool_id);
  reg_task.Wait();
  REQUIRE(reg_task->GetReturnCode() == 0);
  std::fprintf(stderr, "[STAGE] RegisterTarget ok\n");

  auto tag_task = g_cte_client->AsyncGetOrCreateTag("kernel_putget_tag");
  tag_task.Wait();
  REQUIRE(tag_task->GetReturnCode() == 0);
  g_tag_id = tag_task->tag_id_;

  g_initialized = true;
  std::fprintf(stderr, "[INIT] Ready (tag_id=(%u,%u))\n",
               g_tag_id.major_, g_tag_id.minor_);
}
#endif  // HSHM_IS_HOST

}  // namespace

// KernelCtx — same struct as the SYCL test. Defined at file scope
// (NOT in an anonymous namespace) because nvcc with
// CUDA_SEPARABLE_COMPILATION sometimes drops kernel symbols whose
// argument types come from anonymous namespaces during device link
// (the host translation unit and the device translation unit see
// different `(anonymous namespace)::T` types).
//
// Data buffers (put_buf, get_buf) are pre-allocated by the host (via
// hshm::GpuApi::MallocHost or sycl::malloc_*) and passed to the
// kernel through this struct. The kernel does NOT call
// CHI_IPC->AllocateBuffer for user data; the user owns data-buffer
// lifetime and only the Task + FutureShm allocation (small, fixed
// size) goes through the kernel-side gpu_alloc_ via NewTask.
struct KernelCtx {
  // IN
  void *ipc_gpu_info;
  void *ipc_storage;
  chi::PoolId cte_pool_id;
  wrp_cte::core::TagId tag_id;
  chi::u32 gpu_id;
  chi::u32 blob_size_bytes;
  chi::u32 pattern_seed;
  char blob_name[32];
  // Host-allocated data buffers (pinned host memory) the kernel
  // writes into / reads from. Not allocated by the kernel.
  void *put_buf;
  void *get_buf;
  // OUT
  chi::u32 stage;
  chi::u32 put_rc;
  chi::u32 get_rc;
  chi::u32 newtask_put_ok;
  chi::u32 newtask_get_ok;
  chi::u32 first_mismatch_idx;
  chi::u32 mismatch_got;
  chi::u32 mismatch_want;
  chi::u32 bytes_match;
  chi::u32 done;
};

// ---------------------------------------------------------------------
// The IO kernel itself. HSHM_GPU_KERNEL expands to __global__ on both
// CUDA (nvcc) and ROCm (hipcc). The body is portable C++ that uses
// chi:: APIs through the kernel-side IpcManager. No CUDA/ROCm
// intrinsics — same logic as the SYCL test's inner lambda.
// ---------------------------------------------------------------------
HSHM_GPU_KERNEL void chi_cuda_cte_kernel_putget_main(KernelCtx *ctx_ptr) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  // Single-thread kernel: launched <<<1, 1>>>. If we ever lift to a
  // multi-thread launch the prologue would need a `threadIdx.x == 0`
  // guard around Send to keep ClientSend's per-block invariant.
  if (threadIdx.x != 0) return;

  ctx_ptr->stage = 1;

  auto *info = reinterpret_cast<chi::IpcManagerGpuInfo *>(ctx_ptr->ipc_gpu_info);
  auto *ipc_obj = reinterpret_cast<chi::gpu::IpcManager *>(ctx_ptr->ipc_storage);
  CHIMAERA_GPU_INIT(*info, ipc_obj);
  ctx_ptr->stage = 2;

  // ---- PutBlob ----
  chi::u32 size = ctx_ptr->blob_size_bytes;
  chi::u32 seed = ctx_ptr->pattern_seed;

  // Data buffers were pre-allocated by the host (see TC2 body) and
  // passed in via ctx_ptr->put_buf / get_buf. The kernel only writes
  // the pattern; it does NOT call CHI_IPC->AllocateBuffer for data —
  // that API is reserved for task allocation under NewTask, not for
  // user-managed data buffers.
  char *put_bytes = static_cast<char *>(ctx_ptr->put_buf);
  for (chi::u32 i = 0; i < size; ++i) {
    put_bytes[i] = static_cast<char>((seed ^ i) & 0xFFu);
  }
  ctx_ptr->stage = 3;

  // Same null-alloc_id workaround as the SYCL test: kernel-side
  // allocator IDs aren't registered in the CPU's gpu_alloc_map_ on
  // the slim path, so we encode the raw host VA in off_ and let
  // ToFullPtr's null-alloc_id branch resolve it.
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
  if (put_task.IsNull()) {
    ctx_ptr->first_mismatch_idx = 0xAB000001u;  // newtask_put failure
    ctx_ptr->done = 1;
    return;
  }
  ctx_ptr->newtask_put_ok = 1;
  ctx_ptr->stage = 4;

  auto put_future = CHI_IPC->Send(put_task);
  put_future.Wait();
  ctx_ptr->put_rc = put_task->return_code_.load();
  ctx_ptr->stage = 5;

  // ---- GetBlob ----
  // Host-allocated get buffer; zero it from device-side first so any
  // partial fill is detectable.
  char *get_bytes = static_cast<char *>(ctx_ptr->get_buf);
  for (chi::u32 i = 0; i < size; ++i) get_bytes[i] = 0;
  ctx_ptr->stage = 6;

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
  if (get_task.IsNull()) {
    ctx_ptr->first_mismatch_idx = 0xAB000002u;  // newtask_get failure
    ctx_ptr->done = 1;
    return;
  }
  ctx_ptr->newtask_get_ok = 1;
  ctx_ptr->stage = 7;

  auto get_future = CHI_IPC->Send(get_task);
  get_future.Wait();
  ctx_ptr->get_rc = get_task->return_code_.load();
  ctx_ptr->stage = 8;

  // Byte-compare on device.
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
  ctx_ptr->stage = 9;
  // Buffer cleanup (FreeHost on put_buf / get_buf) happens host-side
  // after the kernel returns — kernel doesn't own them.
#else
  (void)ctx_ptr;
#endif
  ctx_ptr->done = 1;
}

// ---------------------------------------------------------------------
// TEST 1 — bare kernel sanity, no Chimaera. Mirror of the SYCL
// equivalent: writes a sentinel from a device kernel into pinned host
// memory, reads it back. If this fails the toolchain / runtime is
// broken, not anything Chimaera-side.
// ---------------------------------------------------------------------
HSHM_GPU_KERNEL void chi_cuda_cte_kernel_putget_bare(int *out) {
  *out = 0xBEEF;
}

#if HSHM_IS_HOST
// Host-only TEST_CASE bodies — the simple_test framework, REQUIRE,
// std::vector, and the wrp_cte::core::Client methods are all host-
// only (most are gated `#if HSHM_IS_HOST` in their headers). The
// device pass of nvcc/hipcc only needs to see the HSHM_GPU_KERNEL
// definitions above; the launch sites below + TEST_CASE registration
// stay host-only.

TEST_CASE("CUDA/HIP bare kernel sanity (no chimaera init)",
          "[cuda][hip][gpu2cpu][cte][bare]") {
  int *bare = hshm::GpuApi::MallocHost<int>(sizeof(int));
  REQUIRE(bare != nullptr);
  *bare = 0;
  chi_cuda_cte_kernel_putget_bare<<<1, 1>>>(bare);
  hshm::GpuApi::Synchronize();
  REQUIRE(*bare == 0xBEEF);
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(bare));
}

// ---------------------------------------------------------------------
// TEST 2 — kernel-side PutBlob + GetBlob round trip via gpu2cpu_queue.
// ---------------------------------------------------------------------
TEST_CASE("CUDA/HIP kernel: PutBlob + GetBlob round trip via gpu2cpu_queue",
          "[cuda][hip][gpu2cpu][cte][kernel_putget]") {
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(/*size=*/0, /*gpu_id=*/0);
  std::fprintf(stderr,
               "[TRACE] gpu_info: queue=%p backend.data=%p backend.cap=%lu "
               "gpu_queue_depth=%u\n",
               static_cast<void *>(gpu_info.gpu2cpu_queue),
               static_cast<void *>(gpu_info.gpu2cpu_backend.data_),
               (unsigned long)gpu_info.gpu2cpu_backend.data_capacity_,
               gpu_info.gpu_queue_depth);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
  REQUIRE(gpu_info.gpu2cpu_backend.data_ != nullptr);

  // Stage IpcManagerGpuInfo + a kernel-scope gpu::IpcManager in pinned
  // host memory so the kernel can read them through UVA. (Same
  // staging pattern as the SYCL test, just using cuda/hipHostMalloc
  // via hshm::GpuApi::MallocHost.)
  auto *info_storage = hshm::GpuApi::MallocHost<chi::IpcManagerGpuInfo>(
      sizeof(chi::IpcManagerGpuInfo));
  REQUIRE(info_storage != nullptr);
  std::memcpy(static_cast<void *>(info_storage), &gpu_info,
              sizeof(chi::IpcManagerGpuInfo));

  auto *ipc_storage = hshm::GpuApi::MallocHost<chi::gpu::IpcManager>(
      sizeof(chi::gpu::IpcManager));
  REQUIRE(ipc_storage != nullptr);
  new (ipc_storage) chi::gpu::IpcManager();

  static const char kBlobName[] = "krn_blob";
  const chi::u32 kBlobSize = 256;

  // Pre-allocate the data buffers on the host so the kernel doesn't
  // need to call CHI_IPC->AllocateBuffer for user data (kernel-side
  // AllocateBuffer is reserved for task allocation under NewTask).
  // pinned host memory is reachable from the device via UVA and
  // host-readable on the CPU bdev side (chi::DeviceAwareMemcpy
  // handles either).
  char *put_buf = hshm::GpuApi::MallocHost<char>(kBlobSize);
  REQUIRE(put_buf != nullptr);
  char *get_buf = hshm::GpuApi::MallocHost<char>(kBlobSize);
  REQUIRE(get_buf != nullptr);
  std::memset(put_buf, 0, kBlobSize);
  std::memset(get_buf, 0, kBlobSize);

  auto *ctx = hshm::GpuApi::MallocHost<KernelCtx>(sizeof(KernelCtx));
  REQUIRE(ctx != nullptr);
  std::memset(ctx, 0, sizeof(KernelCtx));
  ctx->ipc_gpu_info = static_cast<void *>(info_storage);
  ctx->ipc_storage = static_cast<void *>(ipc_storage);
  ctx->cte_pool_id = wrp_cte::core::kCtePoolId;
  ctx->tag_id = g_tag_id;
  ctx->gpu_id = 0;
  ctx->blob_size_bytes = kBlobSize;
  ctx->pattern_seed = 0x5Au;
  std::strncpy(ctx->blob_name, kBlobName, sizeof(ctx->blob_name) - 1);
  ctx->blob_name[sizeof(ctx->blob_name) - 1] = '\0';
  ctx->put_buf = static_cast<void *>(put_buf);
  ctx->get_buf = static_cast<void *>(get_buf);
  ctx->first_mismatch_idx = 0xFFFFFFFFu;

  std::fprintf(stderr,
               "[TRACE] Launching kernel-side PutBlob+GetBlob "
               "(blob_size=%u, name='%s')\n",
               ctx->blob_size_bytes, ctx->blob_name);

  chi_cuda_cte_kernel_putget_main<<<1, 1>>>(ctx);
  hshm::GpuApi::Synchronize();

  // Poll for done in case the kernel uses asynchronous writes the
  // launch synchronize doesn't fully observe.
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
               "        newtask_put_ok=%u newtask_get_ok=%u\n"
               "        put_rc=%u get_rc=%u\n"
               "        bytes_match=%u first_mismatch_idx=0x%x "
               "(got=0x%x want=0x%x)\n"
               "        elapsed=%.2f ms\n",
               ctx->done, ctx->stage,
               ctx->newtask_put_ok, ctx->newtask_get_ok,
               ctx->put_rc, ctx->get_rc,
               ctx->bytes_match, ctx->first_mismatch_idx,
               ctx->mismatch_got, ctx->mismatch_want, ms);

  REQUIRE(ctx->done == 1u);
  REQUIRE(ctx->stage == 9u);
  REQUIRE(ctx->newtask_put_ok == 1u);
  REQUIRE(ctx->put_rc == 0u);
  REQUIRE(ctx->newtask_get_ok == 1u);
  REQUIRE(ctx->get_rc == 0u);
  REQUIRE(ctx->bytes_match == 1u);

  hshm::GpuApi::FreeHost(put_buf);
  hshm::GpuApi::FreeHost(get_buf);
  ipc_storage->~IpcManager();
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(ipc_storage));
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(info_storage));
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(ctx));
}

// ---------------------------------------------------------------------
// TEST 3 — kernel-side PutBlob/GetBlob with HBM (device USM) data buffers.
// Mirror of the SYCL test's `hbm_data` TEST_CASE, written using
// hshm::GpuApi so the same source compiles for CUDA and ROCm.
//
// Flow:
//   1. Host allocates two HBM buffers via hshm::GpuApi::Malloc
//      (cudaMalloc / hipMalloc under the hood).
//   2. A first kernel fills the PUT-side HBM buffer with a deterministic
//      pattern from device-side stores (host can't dereference these).
//   3. The IO kernel calls PutBlob with a ShmPtr whose off_ is the
//      PUT-side HBM device address; CTE's bdev WriteToRam routes
//      through chi::DeviceAwareMemcpy which the HIP init hooked to
//      hshm::GpuApi::Memcpy (cudaMemcpyDefault / hipMemcpyDefault).
//   4. Same kernel calls GetBlob with the GET-side HBM buffer.
//   5. Host copies the GET-side HBM buffer back to host (GpuApi::Memcpy
//      device -> host) and byte-compares against the pattern.
// ---------------------------------------------------------------------
#endif  // HSHM_IS_HOST  (close host block; HBM struct + kernels need
        // to be visible to the device pass)

// File-scope HbmKernelCtx (see KernelCtx comment above re: anon
// namespace + nvcc separable compilation device-link drops).
struct HbmKernelCtx {
  void *ipc_gpu_info;
  void *ipc_storage;
  chi::PoolId cte_pool_id;
  wrp_cte::core::TagId tag_id;
  void *put_hbm;
  void *get_hbm;
  chi::u32 blob_size_bytes;
  char blob_name[32];
  // OUT
  chi::u32 stage;
  chi::u32 put_rc;
  chi::u32 get_rc;
  chi::u32 newtask_put_ok;
  chi::u32 newtask_get_ok;
  chi::u32 done;
};

HSHM_GPU_KERNEL void chi_cuda_cte_kernel_putget_hbm_fill(
    char *put_hbm, chi::u32 size, chi::u32 seed) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  chi::u32 tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid != 0) return;  // launched <<<1, 1>>>
  for (chi::u32 i = 0; i < size; ++i) {
    put_hbm[i] = static_cast<char>((seed ^ i) & 0xFFu);
  }
#else
  (void)put_hbm; (void)size; (void)seed;
#endif
}

HSHM_GPU_KERNEL void chi_cuda_cte_kernel_putget_hbm_io(HbmKernelCtx *ctx_ptr) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if (threadIdx.x != 0) return;
  ctx_ptr->stage = 1;

  auto *info = reinterpret_cast<chi::IpcManagerGpuInfo *>(ctx_ptr->ipc_gpu_info);
  auto *ipc_obj = reinterpret_cast<chi::gpu::IpcManager *>(ctx_ptr->ipc_storage);
  CHIMAERA_GPU_INIT(*info, ipc_obj);
  ctx_ptr->stage = 2;

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
}

#if HSHM_IS_HOST  // Re-enter host-only block for the HBM TEST_CASE.

TEST_CASE("CUDA/HIP kernel: PutBlob/GetBlob with HBM device USM data buffers",
          "[cuda][hip][gpu2cpu][cte][kernel_putget][hbm_data]") {
  EnsureInit();
  auto *ipc = CHI_CPU_IPC;

  chi::IpcManagerGpuInfo gpu_info =
      ipc->GetGpuIpcManager()->CreateGpuAllocator(/*size=*/0, /*gpu_id=*/0);
  REQUIRE(gpu_info.gpu2cpu_queue != nullptr);
  REQUIRE(gpu_info.gpu2cpu_backend.data_ != nullptr);

  const chi::u32 kBlobBytes = 256;
  const chi::u32 kPatternSeed = 0xC3u;

  // Allocate HBM put / get buffers via the unified GpuApi wrapper —
  // resolves to cudaMalloc on CUDA and hipMalloc on ROCm.
  char *put_hbm = hshm::GpuApi::Malloc<char>(kBlobBytes);
  REQUIRE(put_hbm != nullptr);
  char *get_hbm = hshm::GpuApi::Malloc<char>(kBlobBytes);
  REQUIRE(get_hbm != nullptr);
  // Zero the GET buffer so untouched bytes after GetBlob are
  // detectable as a mismatch.
  std::vector<char> zeros(kBlobBytes, 0);
  hshm::GpuApi::Memcpy(get_hbm, zeros.data(), kBlobBytes);

  // Fill put_hbm from a device kernel.
  chi_cuda_cte_kernel_putget_hbm_fill<<<1, 1>>>(put_hbm, kBlobBytes,
                                                kPatternSeed);
  hshm::GpuApi::Synchronize();

  // Stage IpcManagerGpuInfo + kernel-scope IpcManager + KernelCtx in
  // pinned host memory (UVA).
  auto *info_storage = hshm::GpuApi::MallocHost<chi::IpcManagerGpuInfo>(
      sizeof(chi::IpcManagerGpuInfo));
  REQUIRE(info_storage != nullptr);
  std::memcpy(static_cast<void *>(info_storage), &gpu_info,
              sizeof(chi::IpcManagerGpuInfo));

  auto *ipc_storage = hshm::GpuApi::MallocHost<chi::gpu::IpcManager>(
      sizeof(chi::gpu::IpcManager));
  REQUIRE(ipc_storage != nullptr);
  new (ipc_storage) chi::gpu::IpcManager();

  static const char kHbmBlobName[] = "hbm_blob";
  auto *ctx = hshm::GpuApi::MallocHost<HbmKernelCtx>(sizeof(HbmKernelCtx));
  REQUIRE(ctx != nullptr);
  std::memset(ctx, 0, sizeof(HbmKernelCtx));
  ctx->ipc_gpu_info = static_cast<void *>(info_storage);
  ctx->ipc_storage = static_cast<void *>(ipc_storage);
  ctx->cte_pool_id = wrp_cte::core::kCtePoolId;
  ctx->tag_id = g_tag_id;
  ctx->put_hbm = static_cast<void *>(put_hbm);
  ctx->get_hbm = static_cast<void *>(get_hbm);
  ctx->blob_size_bytes = kBlobBytes;
  std::strncpy(ctx->blob_name, kHbmBlobName, sizeof(ctx->blob_name) - 1);
  ctx->blob_name[sizeof(ctx->blob_name) - 1] = '\0';

  std::fprintf(stderr,
               "[TRACE] HBM-data PutBlob+GetBlob: put_hbm=%p get_hbm=%p "
               "size=%u\n",
               static_cast<void *>(put_hbm), static_cast<void *>(get_hbm),
               kBlobBytes);

  chi_cuda_cte_kernel_putget_hbm_io<<<1, 1>>>(ctx);
  hshm::GpuApi::Synchronize();

  // Pull GET buffer back to host and byte-compare.
  std::vector<char> host_check(kBlobBytes, 0);
  hshm::GpuApi::Memcpy(host_check.data(), get_hbm, kBlobBytes);

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
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(ipc_storage));
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(info_storage));
  hshm::GpuApi::FreeHost(reinterpret_cast<char *>(ctx));
  hshm::GpuApi::Free(put_hbm);   // device USM (cudaMalloc)
  hshm::GpuApi::Free(get_hbm);
}

SIMPLE_TEST_MAIN()
#endif  // HSHM_IS_HOST

#else  // !((CUDA||ROCM) && !SYCL)

int main() { return 0; }

#endif
