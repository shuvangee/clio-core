/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 * BSD 3-Clause License. See LICENSE file.
 */

#ifndef WRP_CTE_GPU_VECTOR_PAGE_H_
#define WRP_CTE_GPU_VECTOR_PAGE_H_

#include <chimaera/types.h>
#include <chimaera/gpu/future.h>
#include <wrp_cte/core/core_tasks.h>

namespace wrp_cte::gpu_vector {

/**
 * Per-page metadata. Lives in device memory only — the CPU never reads or
 * writes Page fields directly. Bookkeeping changes flow through the
 * cache-management kernel and the user kernel via atomic ops.
 *
 * `modify_min` / `modify_max` are element offsets within the page (in T
 * units). They are atomically swapped to -1 by the kernel that picks up
 * the page for flush, so the user kernel and the cache-management kernel
 * never disagree about which range is in flight.
 *
 * `active_put` / `active_get` are the in-flight Send futures. Empty
 * (`IsNull()`) means the slot is idle. The next access boundary that
 * needs the page calls `Wait()` and then clears the future.
 */
struct Page {
  void *device_ptr;        /**< Base of this page inside pages_backend. */
  int32_t page_idx;       /**< -1 if slot is empty. */
  int32_t modify_min;     /**< -1 = clean. */
  int32_t modify_max;     /**< -1 = clean. */
  chi::u32 flags;          /**< bit 0 = put-in-flight CAS lock. */
  chi::u64 lru_clock;      /**< clock64() at last access (for LRU). */
  chi::gpu::Future<wrp_cte::core::PutBlobTask> active_put;
  chi::gpu::Future<wrp_cte::core::GetBlobTask> active_get;
};

/**
 * Per-block control structure: a fixed-size Page table plus a counter of
 * dirty pages. The blob naming scheme is `<tag_name>_b<block>_p<slot>`
 * (slot is the index in `pages[]`, not the global blob page index).
 */
struct Block {
  chi::u32 block_idx;
  chi::u32 num_modified;     /**< atomic counter, bumped by writers. */
  chi::u32 pages_per_block;
  chi::u32 _pad;
  /** `pages[i]` is the i-th cached slot for this block. The blob page
   *  currently held in slot i is `pages[i].page_idx`. */
  Page pages[];              /**< flexible: sized by ctor. */
};

/** Page::flags bits. */
inline constexpr chi::u32 kPagePutInFlight = 1u << 0;

}  // namespace wrp_cte::gpu_vector

#endif  // WRP_CTE_GPU_VECTOR_PAGE_H_
