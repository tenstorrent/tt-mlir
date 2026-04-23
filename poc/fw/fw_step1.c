// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Step 1 firmware: write a magic value to the mailbox and park.
 *
 * The host polls the mailbox via a NOC read into L2CPU DRAM. The NOC path
 * into the L2CPU tile is coherent with the X280's L3 cache, but we issue
 * an explicit `fence rw, rw` after the store so there's no ordering doubt.
 */
#include <stdint.h>

#define MAILBOX_ADDR 0x400030100000ULL

__attribute__((noreturn)) void fw_main(void) {
  volatile uint32_t *mailbox = (volatile uint32_t *)MAILBOX_ADDR;
  *mailbox = 0xDEADBEEFu;
  __asm__ volatile("fence rw, rw" ::: "memory");
  for (;;) {
    __asm__ volatile("wfi");
  }
}
