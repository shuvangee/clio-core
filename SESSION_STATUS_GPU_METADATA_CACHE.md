# Session status — CTE Core GPU Metadata Cache

Date: 2026-05-06
Branch: `sycl-gpu2cpu` (work-in-progress; not yet committed)

## What was requested

Add an optional GPU-resident projection of CTE Core metadata so GPU
kernels can directly observe which blobs / tags the CTE has placed in
DRAM-tier (GPU-reachable) storage.

User constraints recorded:
- Helper functions must hold the cache logic; `PutBlob`, `DelBlob`,
  `GetOrCreateTag`, `DelTag` must each call into a single helper, no
  inlined cache logic.
- New configuration controls cache memory size.
- Create allocates GPU memory and exposes the pointer back through the
  CreateTask params.
- Cache mutations should ideally run via GPU kernels; reads happen from
  GPU kernels too.
- Cross-GPU sharing on Intel needs Level Zero (called out by user, not
  tackled in this session).
- Build a basic unit test: host PutBlob -> GPU kernel reads the cache
  and confirms the blob is visible.

## What is done

### Foundation (Steps 1-3, all complete)

1. **Config** — `wrp_cte::core::GpuMetadataCacheConfig` added to
   `core_config.h` with `enabled_`, `capacity_bytes_`,
   `max_blobs_`, `max_tags_`. YAML parser in `core_config.cc` reads a
   `gpu_metadata_cache:` section. Disabled by default.

2. **GPU-friendly entry types** — new header
   `core/include/wrp_cte/core/gpu_metadata_cache.h`. POD `GpuTagEntry`
   and `GpuBlobEntry` structs with fixed-size `char[64]` names (no
   `chi::priv::string` — that's the type whose layout differs between
   SYCL host/device passes). Open-addressing hash map carved out of a
   single managed-USM region with `GpuMetadataCacheHeader::Layout()`.
   Inline `HSHM_CROSS_FUN` helpers `GpuCacheUpsertTag`,
   `GpuCacheUpsertBlob`, `GpuCacheRemoveBlob`, `GpuCacheRemoveTag`,
   `GpuCacheFindBlob`, `GpuCacheFindTag` that work from both host and
   device.

3. **Allocation in Create** — `Runtime::GpuCacheCreate()` /
   `Runtime::GpuCacheDestroy()` use `hshm::GpuApi::MallocManaged` /
   `hshm::GpuApi::Free`. Managed USM lets the CPU server call the
   inline upsert/remove helpers directly (still GPU-readable). Cache
   header pointer is reinterpreted as `chi::u64` and re-serialized into
   `chimod_params_` during `Create` so the client sees it via
   `pool_task->GetParams().gpu_cache_ptr_`.

4. **Helpers wired into PutBlob** —
   `Runtime::GpuCacheOnPutBlob(tag_id, blob_name, blob_info)` is called
   from `Runtime::PutBlob` as a single line. The helper resolves the
   blob's tier via a new `Runtime::GetBdevTypeForBlob()` (which reads
   the `bdev_type_` field newly stored on `TargetInfo` at
   `RegisterTarget` time) and either upserts or evicts the entry
   depending on whether the tier is `ram`/`hbm`/`pinned` or
   `file`/`noop`. The remaining helper signatures
   (`GpuCacheOnGetOrCreateTag`, `GpuCacheOnDelBlob`, `GpuCacheOnDelTag`)
   are declared and implemented but not yet called from the
   corresponding methods (Step 5).

5. **OUT path** — `CreateParams` now serializes the `gpu_metadata_cache`
   subset of Config (so the client can opt in without compose YAML)
   plus an OUT `gpu_cache_ptr_`. Server overwrites `chimod_params_`
   with the populated CreateParams at the end of `Create`. Client's
   `GetParams()` returns the cache pointer.

### Build wiring

- `context-transfer-engine/core/CMakeLists.txt`: surface
  `HSHM_ENABLE_SYCL=1` (or CUDA / ROCm) on `wrp_cte_core_runtime` so
  the runtime sees `hshm::GpuApi::MallocManaged`'s SYCL branch. Adds
  the matching SYCL include directories.
- Test linked against `wrp_cte_core_runtime` (only needed because
  `CreateParams::LoadConfig` template-instantiation references
  `Config::LoadFromString`; runtime path never calls it).

### Test (Step 4 — passes)

`context-transfer-engine/test/unit/gpu/test_sycl_gpu_metadata_cache.cc`:

- `CHIMAERA_INIT(kServer)`.
- Init `wrp_cte::core::Client` on a unique pool id `(513, 0)` to avoid
  the auto-composed default pool at `(512, 0)` shadowing our params.
- `AsyncCreate` with `gpu_metadata_cache_.enabled_ = true`,
  capacity = 256 KiB, 64 tag slots, 256 blob slots.
- Deserialize OUT `gpu_cache_ptr_` from `pool_task->GetParams()`.
- Register a `BdevType::kRam` target, create a tag, `AsyncPutBlob` of
  512 bytes.
- Sanity-check from the host: `GpuCacheFindBlob` returns non-null with
  the correct size and storage class.
- SYCL kernel (`hshm::GpuApi::SyclQueue()` — same context as the
  managed-USM allocation) walks the slot array and confirms:
  - cache magic = `0xCAFEC7E0`
  - blob entry is present
  - `size_ == 512`
  - `storage_class_ == kStorageRam`
- Passes 3 of 3 consecutive runs.

CMake test target: `cte_sycl_gpu_metadata_cache_tests` (labels:
`gpu;sycl;cte;metadata_cache`).

### DPC++ workarounds discovered

The kernel needs at least one read of every slot field that flows into
the result (or another visible side-effect) before the conditional
"found" write. Without it, DPC++'s device-side optimizer hoists/DCEs
the slot loads and `GpuCacheFindBlob` misses entries that the host
clearly sees. The test's lookup body uses an inline open-addressing
walk that materializes name/tag reads up front — see comment in the
test header.

The cache USM was allocated by `GpuApi::SyclQueue()` (the singleton
queue inside `hshm::GpuApi`); the test's kernel **must** use the same
queue, otherwise the kernel-side dereference faults
(CUDA_ERROR_ILLEGAL_ADDRESS). Cross-context USM sharing isn't a thing
for `sycl::malloc_shared` on the DPC++/CUDA backend.

## What is remaining

Step 5 is complete. All four helper call sites now wire into their
operations, and each has a kernel-side test passing 3-of-3 consecutive
runs:

1. **`GetOrCreateTag`** — calls `GpuCacheOnGetOrCreateTag(tag_id_,
   tag_name)` on both the local-canonical and remote-tag success paths
   (Runtime::GetOrCreateTag in core_runtime.cc).
2. **`DelBlob`** — calls `GpuCacheOnDelBlob(tag_id_, blob_name)`
   immediately before `task->return_code_ = 0` on the success path.
3. **`DelTag`** — calls `GpuCacheOnDelTag(tag_id)` after the tag-info
   erase, on the success path.
4. **PutBlob non-DRAM eviction** — covered by the existing helper logic
   (file/noop targets call `GpuCacheRemoveBlob`); a regression test
   exercises this end-to-end with a `BdevType::kFile` target.

Test file `test/unit/gpu/test_sycl_gpu_metadata_cache.cc` now contains
5 TEST_CASEs sharing one `EnsureInit()` setup — passes 5/5 across 3+
consecutive runs.

### DPC++ kernel-arg layout pitfall (lessons learned)

While adding the new tests we hit two distinct DPC++ issues that the
test code now codifies:

1. **`Unexpected kernel lambda size` static_assert in
   sycl/handler.hpp:669.** Triggered when the host compiler and the
   device compiler infer different lambda capture layouts. Workaround
   used here: bundle EVERY kernel parameter (cache pointer, tag id,
   result buffer) into a single `KernelCtx` struct allocated in
   shared USM, and capture only the struct pointer. Lambdas that
   capture multiple disparate types (especially `chi::u32` alongside
   `void *` or `char *`) seem fragile across templated kernel
   instantiations.

2. **Garbage `cap_tag_*` reads inside the kernel.** When the lambda
   *did* compile, captures of `chi::u32 cap_tag_major` / `chi::u32
   cap_tag_minor` could end up overlapping the bytes of an adjacent
   `char *kernel_blob_name` capture (verified via in-kernel diagnostic
   writes — kernel saw `(838892544, 32512) = (0x32004A40, 0x7F00)`
   matching upper bytes of the `kernel_blob_name` shared-USM
   allocation). Same fix: route all parameters through one struct
   pointer so captures don't alias.

The new tests no longer rely on per-character name comparison either —
each TEST_CASE uses a unique tag id, so `(tag_major, tag_minor)`
matching is sufficient.

### Open follow-ups (not in any task)

- **Pure device memory + per-update kernels.** The current
  implementation uses managed/shared USM and has the CPU server call
  the inline upsert helpers directly. To match the user's "update via
  GPU kernel" wording strictly, switch to `GpuApi::Malloc`
  (`sycl::malloc_device`) for the cache region and have each
  `GpuCacheOn*` helper enqueue a one-WI kernel that mutates the cache.
  This requires giving `wrp_cte_core_runtime` (or a sibling
  `_gpu.cc` companion library) `-fsycl` so kernels can be submitted
  from CPU code. The `add_chimod_runtime` build pattern already
  supports this via `_gpu.cc` files; see how the existing
  `core_runtime_gpu.cc` is handled for a reference pattern.

- **Cross-process IPC.** Returning `gpu_cache_ptr_` as a raw
  `chi::u64` works only when the caller and the CTE Core server are in
  the same process (e.g. `CHIMAERA_INIT(kServer)` mode). Real
  cross-process sharing needs an IPC mem handle — `cudaIpcGetMemHandle`
  on CUDA, `hipIpcGetMemHandle` on ROCm, and Level-Zero
  `zeMemGetIpcHandle` on SYCL/Intel. `hshm::GpuApi` already has CUDA /
  ROCm IPC entry points (`GetIpcMemHandle` / `OpenIpcMemHandle`); they
  need a SYCL implementation behind a Level-Zero backend dispatch.
  Surface the chosen handle through the OUT serialization path next
  to / instead of `gpu_cache_ptr_`.

- **Lock-free updates from concurrent worker coroutines.** The
  inline `GpuCacheUpsert*` / `GpuCacheRemove*` helpers in
  `gpu_metadata_cache.h` mutate `state_` and `num_blobs_` without
  atomic operations. Today CTE PutBlob/DelBlob serialize through
  `tag_map_lock_` / `blob_map_lock_` so single-server access is
  fine, but bringing the cache into a multi-writer regime (or
  splitting writers per worker) needs `std::atomic<u32>` slot
  states with proper acquire/release semantics. The header layout
  is already POD-compatible with that change.

- **Eviction policy.** The cache is fixed-capacity with no eviction
  beyond `RemoveBlob` / `RemoveTag` calls. If a steady-stream PutBlob
  pattern fills the cache, new entries return nullptr and silently
  drop on the GPU side. Two options: (a) make `GpuCacheUpsertBlob`
  evict the lowest-scored entry on a full-table miss; (b) emit a host
  HLOG warning when the cache fill ratio exceeds some threshold and
  let the CTE config bump capacity.

## Files touched in this session

```
context-transfer-engine/core/include/wrp_cte/core/core_config.h
context-transfer-engine/core/include/wrp_cte/core/core_runtime.h
context-transfer-engine/core/include/wrp_cte/core/core_tasks.h
context-transfer-engine/core/include/wrp_cte/core/gpu_metadata_cache.h   (new)
context-transfer-engine/core/src/core_config.cc
context-transfer-engine/core/src/core_runtime.cc
context-transfer-engine/core/CMakeLists.txt
context-transfer-engine/test/unit/CMakeLists.txt
context-transfer-engine/test/unit/gpu/test_sycl_gpu_metadata_cache.cc   (new)
```

Nothing has been committed; everything is in the working tree on
branch `sycl-gpu2cpu`.

## How to resume

1. `cd /workspace/build_gpu`
2. `cmake --build . --target test_sycl_gpu_metadata_cache --parallel 4`
3. `rm -f /tmp/chimaera_iowarp/* && ./bin/test_sycl_gpu_metadata_cache`
4. Expected: `Total tests: 5  Passed: 5  Failed: 0`.

Next-up items are the open follow-ups in this doc:
pure-device-memory + per-update kernels, cross-process IPC handles,
lock-free updates, and an eviction policy. None of them are
required for the basic CPU-mediated cache to be useful from a single
GPU kernel in the same process.
