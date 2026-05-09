/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef WRP_CTE_GPU_VECTOR_VIEW_H_
#define WRP_CTE_GPU_VECTOR_VIEW_H_

#include <chimaera/types.h>
#include <chimaera/gpu/future.h>
#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/gpu_vector/gpu_vector_page.h>

namespace wrp_cte::gpu_vector {

/**
 * Stride between adjacent Block structs in the meta backend (the
 * Page[pages_per_block] flexible array makes sizeof(Block) too small).
 * Computed once at ctor time and stored on DeviceView so kernels can
 * index `((char*)blocks) + block_stride_bytes * block_idx`.
 */
struct DeviceViewBase {
  Block *blocks;             /**< device pointer (kDeviceMem). */
  chi::u32 block_stride_bytes;
  void *pages_base;          /**< device pointer (kDeviceMem). */
  /**
   * Pre-allocated PutBlob/GetBlob task slots in pinned host memory.
   * One Task+FutureShm pair per (block, page) — task at offset
   * `slot * (sizeof(TaskT) + sizeof(gpu::FutureShm))`.
   */
  char *put_pool_base;       /**< pinned host. */
  char *get_pool_base;       /**< pinned host. */
  chi::u32 put_slot_stride;  /**< sizeof(PutBlobTask)+sizeof(gpu::FutureShm). */
  chi::u32 get_slot_stride;
  /**
   * Allocator id of pages_backend so the kernel can stamp blob_data_
   * ShmPtrs with the right `alloc_id_` before Send. The bdev runtime
   * resolves these via chi::g_device_aware_memcpy.
   */
  hipc::AllocatorId pages_alloc_id;
  hipc::AllocatorId put_pool_alloc_id;
  hipc::AllocatorId get_pool_alloc_id;
  /** Tag the kernel writes blobs against. Set once at Vector ctor. */
  wrp_cte::core::TagId tag_id;
  chi::u32 nblocks;
  chi::u32 pages_per_block;
  chi::u64 page_size_bytes;
};

/**
 * Strongly-typed view captured by user kernels. Trivially copyable POD.
 * The kernel side macro WRP_GPU_VECTOR_KERNEL_INIT(view) sets up the
 * per-warp last-page cache so the operator[]() fast path works.
 */
template <typename T>
struct DeviceView {
  DeviceViewBase base;
  /**
   * Number of T-elements per page. Computed as
   * `page_size_bytes / sizeof(T)` at ctor time.
   */
  chi::u64 page_capacity_t;
};

/** Per-block resolution helper — handles the variable-size Page array. */
HSHM_INLINE_CROSS_FUN Block *GetBlock(const DeviceViewBase &v,
                                             chi::u32 block_idx) {
  return reinterpret_cast<Block *>(
      reinterpret_cast<char *>(v.blocks) +
      static_cast<chi::u64>(v.block_stride_bytes) * block_idx);
}

/** Resolve the i-th task in the put pool. */
HSHM_INLINE_CROSS_FUN wrp_cte::core::PutBlobTask *GetPutTask(
    const DeviceViewBase &v, chi::u32 block_idx, chi::u32 slot) {
  chi::u64 off =
      (static_cast<chi::u64>(block_idx) * v.pages_per_block + slot) *
      v.put_slot_stride;
  return reinterpret_cast<wrp_cte::core::PutBlobTask *>(v.put_pool_base + off);
}

/** Resolve the i-th task in the get pool. */
HSHM_INLINE_CROSS_FUN wrp_cte::core::GetBlobTask *GetGetTask(
    const DeviceViewBase &v, chi::u32 block_idx, chi::u32 slot) {
  chi::u64 off =
      (static_cast<chi::u64>(block_idx) * v.pages_per_block + slot) *
      v.get_slot_stride;
  return reinterpret_cast<wrp_cte::core::GetBlobTask *>(v.get_pool_base + off);
}

/** Co-located gpu::FutureShm for a put task. */
HSHM_INLINE_CROSS_FUN chi::gpu::FutureShm *GetPutFutureShm(
    const DeviceViewBase &v, chi::u32 block_idx, chi::u32 slot) {
  return reinterpret_cast<chi::gpu::FutureShm *>(
      reinterpret_cast<char *>(GetPutTask(v, block_idx, slot)) +
      sizeof(wrp_cte::core::PutBlobTask));
}

/** Co-located gpu::FutureShm for a get task. */
HSHM_INLINE_CROSS_FUN chi::gpu::FutureShm *GetGetFutureShm(
    const DeviceViewBase &v, chi::u32 block_idx, chi::u32 slot) {
  return reinterpret_cast<chi::gpu::FutureShm *>(
      reinterpret_cast<char *>(GetGetTask(v, block_idx, slot)) +
      sizeof(wrp_cte::core::GetBlobTask));
}

}  // namespace wrp_cte::gpu_vector

#endif  // WRP_CTE_GPU_VECTOR_VIEW_H_
