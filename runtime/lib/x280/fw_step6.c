// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Step 6 firmware: generic multi-kernel dispatch on the X280.
 *
 * Unlike step 5 (single hardcoded kernel, one-shot), this firmware runs a
 * persistent task loop and dispatches arbitrary CPU-hoisted kernels whose
 * compiled code is loaded separately into L2CPU0 DRAM by the host.
 *
 * Protocol (sequence-number handshake):
 *   Host writes the task body (excluding `done`), then writes
 *   kick = <sequence number>. The firmware spins until kick differs from
 *   the previous value, processes the task, and echoes done = kick.
 *   The host polls done until it equals the expected sequence number.
 *   (Host MUST never write the `done` field — silicon quirk.)
 *
 * For each task the firmware:
 *   1. Configures 128 GiB TLB windows for the DRAM banks.
 *   2. Bump-allocates contiguous staging buffers from STAGING_BASE for
 *      every tensor in the task.
 *   3. Copies input tensors from bank-interleaved Tensix DRAM into their
 *      staging buffers (same bank walk as step 5).
 *   4. Builds the helper descriptor array that the compiler-generated
 *      helper functions expect: per-tensor {alloc, aligned, offset,
 *      sizes_strides_ptr} laid out as 4 consecutive uint64_t values.
 *   5. Calls the code blob's dispatch function (at CODE_LOAD_ADDR) with
 *      (func_id, descriptor_array_ptr).
 *   6. Copies output tensors from staging buffers back to DRAM banks.
 *   7. Signals done = kick sequence number.
 */
#include <stdint.h>

/* ----- Layout constants (must match common.hpp) ----- */
#define MAX_BANKS 8
#define MAX_TENSORS 8
#define MAX_RANK 4

#define TASK_ADDR 0x400030100000ULL      /* kTaskAddr      (+1 MiB)  */
#define CODE_LOAD_ADDR 0x400030400000ULL /* kCodeLoadAddr  (+4 MiB)  */
#define STAGING_BASE 0x400030800000ULL   /* kStagingBase   (+8 MiB)  */

/* ----- Task structs (byte-for-byte match with common.hpp) ----- */

struct Step6TensorMeta {
  uint64_t bank_base[MAX_BANKS];
  uint32_t aligned_page_size;
  uint32_t page_size;
  uint64_t num_pages;
  uint64_t total_size_bytes;
  int64_t sizes_and_strides[MAX_RANK * 2];
  uint32_t rank;
  uint8_t is_input;
  uint8_t is_output;
  uint8_t pad[2];
};

struct Step6Task {
  volatile uint32_t kick;
  uint32_t num_tensors;
  uint32_t num_banks;
  uint32_t func_id;
  uint32_t bank_x[MAX_BANKS];
  uint32_t bank_y[MAX_BANKS];
  struct Step6TensorMeta tensors[MAX_TENSORS];
  uint32_t done;
  uint32_t pad_done;
};

/* ----- NOC TLB helpers (same as step 4/5) ----- */

#define TLB_2M_CONFIG_BASE 0x2FF00000U
#define TLB_2M_COUNT 224U
#define TLB_128G_CONFIG_STRIDE 0x0CU
#define TLB_128G_CONFIG_BASE (TLB_2M_CONFIG_BASE + 0x10U * TLB_2M_COUNT)

#define SYSTEM_PORT_BASE 0x30000000ULL
#define WINDOW_128G_BASE (SYSTEM_PORT_BASE + 0x80400000000ULL)
#define WINDOW_128G_SHIFT 37
#define WINDOW_128G_SIZE (1ULL << WINDOW_128G_SHIFT)

void *memcpy(void *dest, const void *src, uint64_t count) {
  uint8_t *d = (uint8_t *)dest;
  const uint8_t *s = (const uint8_t *)src;
  while (count--) {
    *d++ = *s++;
  }
  return dest;
}

static inline void configure_128g_window(uint32_t window_idx, uint32_t noc_x,
                                         uint32_t noc_y) {
  volatile uint32_t *cfg =
      (volatile uint32_t *)(uintptr_t)(TLB_128G_CONFIG_BASE +
                                       window_idx * TLB_128G_CONFIG_STRIDE);
  cfg[0] = 0;
  cfg[1] = (noc_x & 0x3Fu) | ((noc_y & 0x3Fu) << 6);
  cfg[2] = 0;
}

/* ----- Staging helpers ----- */

static void stage_in(const struct Step6Task *task,
                     const struct Step6TensorMeta *tm, void *local_buf) {
  for (uint64_t p = 0; p < tm->num_pages; p++) {
    uint32_t bank_id = (uint32_t)(p % task->num_banks);
    uint64_t page_in_bank = p / task->num_banks;
    uint64_t addr_in_bank =
        tm->bank_base[bank_id] + page_in_bank * tm->aligned_page_size;
    uint64_t window_addr =
        WINDOW_128G_BASE + (uint64_t)bank_id * WINDOW_128G_SIZE + addr_in_bank;
    uintptr_t dst = (uintptr_t)local_buf + p * tm->page_size;
    memcpy((void *)dst, (void *)window_addr, tm->page_size);
  }
}

static void stage_out(const struct Step6Task *task,
                      const struct Step6TensorMeta *tm, void *local_buf) {
  for (uint64_t p = 0; p < tm->num_pages; p++) {
    uint32_t bank_id = (uint32_t)(p % task->num_banks);
    uint64_t page_in_bank = p / task->num_banks;
    uint64_t addr_in_bank =
        tm->bank_base[bank_id] + page_in_bank * tm->aligned_page_size;
    uint64_t window_addr =
        WINDOW_128G_BASE + (uint64_t)bank_id * WINDOW_128G_SIZE + addr_in_bank;
    uintptr_t src = (uintptr_t)local_buf + p * tm->page_size;
    memcpy((void *)window_addr, (void *)src, tm->page_size);
  }
}

/* ----- Dispatch function type (entry point of the code blob) ----- */
typedef void (*dispatch_fn_t)(uint32_t func_id, void *args);

/* ----- Main loop ----- */

__attribute__((noreturn)) void fw_main(void) {
  struct Step6Task *task = (struct Step6Task *)TASK_ADDR;

  /* Clear control fields so the host can rely on a known initial state.
   * The host never writes `done`, so host reads will see our value. */
  task->done = 0;
  __asm__ volatile("fence rw, rw" ::: "memory");

  uint32_t last_kick = 0;

  for (;;) {
    /* Wait for the host to publish a new sequence number in kick. */
    uint32_t k;
    while ((k = __atomic_load_n(&task->kick, __ATOMIC_ACQUIRE)) == last_kick) {
      __asm__ volatile("nop");
    }
    last_kick = k;

    uint32_t num_tensors = task->num_tensors;
    uint32_t num_banks = task->num_banks;

    /* Configure TLB windows for the DRAM banks. */
    for (uint32_t b = 0; b < num_banks; b++) {
      configure_128g_window(b, task->bank_x[b], task->bank_y[b]);
    }
    __asm__ volatile("fence rw, rw" ::: "memory");

    /* Bump-allocate staging buffers from the staging region. */
    void *local_bufs[MAX_TENSORS];
    uint8_t *cursor = (uint8_t *)STAGING_BASE;
    for (uint32_t t = 0; t < num_tensors; t++) {
      /* Align to 64 bytes for cache-line alignment. */
      cursor = (uint8_t *)(((uintptr_t)cursor + 63u) & ~(uintptr_t)63u);
      local_bufs[t] = cursor;
      cursor += task->tensors[t].total_size_bytes;
    }

    /* Stage input tensors from DRAM banks into local staging buffers. */
    for (uint32_t t = 0; t < num_tensors; t++) {
      if (task->tensors[t].is_input) {
        stage_in(task, &task->tensors[t], local_bufs[t]);
      }
    }

    /* Build the helper descriptor array.
     *
     * Each helper expects a pointer to an array of 4-element groups
     * (one per tensor): { alloc_ptr, aligned_ptr, offset, ss_ptr }.
     * All elements are pointer-sized (uint64_t on rv64).
     *
     * The sizes_strides arrays live in a separate block. */
    uint64_t desc_array[MAX_TENSORS * 4];
    int64_t ss_storage[MAX_TENSORS][MAX_RANK * 2];

    for (uint32_t t = 0; t < num_tensors; t++) {
      uint32_t rank = task->tensors[t].rank;
      for (uint32_t r = 0; r < rank * 2 && r < MAX_RANK * 2; r++) {
        ss_storage[t][r] = task->tensors[t].sizes_and_strides[r];
      }
      desc_array[t * 4 + 0] = (uint64_t)(uintptr_t)local_bufs[t];
      desc_array[t * 4 + 1] = (uint64_t)(uintptr_t)local_bufs[t];
      desc_array[t * 4 + 2] = 0; /* offset */
      desc_array[t * 4 + 3] = (uint64_t)(uintptr_t)&ss_storage[t][0];
    }

    /* Call the code blob's dispatch function. */
    dispatch_fn_t dispatch = (dispatch_fn_t)CODE_LOAD_ADDR;
    dispatch(task->func_id, (void *)desc_array);

    /* Stage output tensors from local staging buffers back to DRAM. */
    for (uint32_t t = 0; t < num_tensors; t++) {
      if (task->tensors[t].is_output) {
        stage_out(task, &task->tensors[t], local_bufs[t]);
      }
    }

    /* Signal completion by echoing the sequence number. */
    __asm__ volatile("fence rw, rw" ::: "memory");
    __atomic_store_n(&task->done, k, __ATOMIC_RELEASE);
  }
}
