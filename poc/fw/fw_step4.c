// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Step 4 firmware: increment every float in a TTNN-interleaved DRAM tensor
 * in-place, by reaching into Tensix DRAM directly via the X280's own NOC
 * TLB windows. No host staging.
 *
 * The host hands us:
 *   - num_banks
 *   - per-bank NOC0 coords + buffer base address
 *   - aligned_page_size (how far apart consecutive same-bank pages live)
 *   - page_size (how many bytes of actual float data per page)
 *   - num_pages
 *
 * We allocate one 128GB TLB window per bank, configured to point at that
 * bank's NOC coords with addr=0. Accessing
 *   WINDOW_128G_BASE + bank_id * 128GB + addr_in_bank
 * then translates to a NOC unicast read/write at (bank_x, bank_y, addr).
 *
 * We use the System Port window base because System Port is uncached on
 * the X280 — writes go straight out via the NOC TLB without sitting in
 * L3 (cbo.flush hangs on this silicon, and we don't want to fight cache
 * coherency on tensor data).
 *
 * Page-to-bank mapping matches tt-metal's interleaved allocator:
 *   bank_id      = page_idx % num_banks
 *   page_in_bank = page_idx / num_banks
 *   addr_in_bank = bank_base[bank_id] + page_in_bank * aligned_page_size
 *
 * Element layout within a TILE-layout page is face-major, but we just
 * increment every float in the page so order doesn't matter.
 */
#include <stdint.h>

#define MAX_BANKS 8

struct Step4Task {
  volatile uint32_t kick;     /* host -> fw */
  uint32_t num_banks;
  uint32_t aligned_page_size; /* bytes between same-bank consecutive pages */
  uint32_t page_size;         /* bytes of float data per page */
  uint64_t num_pages;
  uint64_t bank_base[MAX_BANKS]; /* per-bank base addr of buffer in tile DRAM */
  uint32_t bank_x[MAX_BANKS];    /* NOC0 X of each bank's DRAM core */
  uint32_t bank_y[MAX_BANKS];    /* NOC0 Y of each bank's DRAM core */
  uint32_t done;                 /* fw -> host */
  uint32_t pad_done;
};

#define TASK_ADDR 0x400030100000ULL
#define KICK 0x1u
#define DONE 0xC0FFEE03u

/* X280 NOC TLB control registers (see tt-bh-linux/docs/addressing.md). */
#define TLB_2M_CONFIG_BASE  0x2FF00000U
#define TLB_2M_COUNT        224U
#define TLB_128G_CONFIG_STRIDE 0x0CU                        /* 3 x uint32 */
#define TLB_128G_CONFIG_BASE                                                   \
  (TLB_2M_CONFIG_BASE + 0x10U * TLB_2M_COUNT)               /* = 0x2FF00E00 */

#define SYSTEM_PORT_BASE     0x30000000ULL
#define WINDOW_128G_BASE     (SYSTEM_PORT_BASE + 0x80400000000ULL)
#define WINDOW_128G_SHIFT    37
#define WINDOW_128G_SIZE     (1ULL << WINDOW_128G_SHIFT)

static inline void configure_128g_window(uint32_t window_idx, uint32_t noc_x,
                                         uint32_t noc_y) {
  /* 96-bit register, 3 x uint32. Layout (from addressing.md TLB_128G_REG):
   *   word 0: addr[26:0]                               -> 0 (whole bank)
   *   word 1: x_end[5:0] | y_end[5:0]<<6 | ...          -> unicast coords
   *   word 2: strided / exclude / num_destinations      -> 0
   */
  volatile uint32_t *cfg =
      (volatile uint32_t *)(uintptr_t)(TLB_128G_CONFIG_BASE +
                                       window_idx * TLB_128G_CONFIG_STRIDE);
  cfg[0] = 0;
  cfg[1] = (noc_x & 0x3Fu) | ((noc_y & 0x3Fu) << 6);
  cfg[2] = 0;
}

__attribute__((noreturn)) void fw_main(void) {
  struct Step4Task *task = (struct Step4Task *)TASK_ADDR;

  while (__atomic_load_n(&task->kick, __ATOMIC_ACQUIRE) != KICK) {
    __asm__ volatile("nop");
  }

  uint32_t num_banks = task->num_banks;
  uint32_t aligned_page_size = task->aligned_page_size;
  uint32_t page_size = task->page_size;
  uint64_t num_pages = task->num_pages;

  /* One TLB window per bank, addr=0, x_end/y_end = bank's NOC0 coords. */
  for (uint32_t b = 0; b < num_banks; b++) {
    configure_128g_window(b, task->bank_x[b], task->bank_y[b]);
  }
  __asm__ volatile("fence rw, rw" ::: "memory");

  const uint32_t floats_per_page = page_size / (uint32_t)sizeof(float);

  for (uint64_t p = 0; p < num_pages; p++) {
    uint32_t bank_id = (uint32_t)(p % num_banks);
    uint64_t page_in_bank = p / num_banks;
    uint64_t addr_in_bank =
        task->bank_base[bank_id] + page_in_bank * aligned_page_size;

    uint64_t window_addr = WINDOW_128G_BASE +
                           (uint64_t)bank_id * WINDOW_128G_SIZE + addr_in_bank;
    volatile float *page = (volatile float *)(uintptr_t)window_addr;

    for (uint32_t i = 0; i < floats_per_page; i++) {
      page[i] = page[i] + 1.0f;
    }
  }

  __asm__ volatile("fence rw, rw" ::: "memory");
  __atomic_store_n(&task->done, DONE, __ATOMIC_RELEASE);

  for (;;) {
    __asm__ volatile("wfi");
  }
}
