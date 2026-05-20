// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Dispatch entry point for the dynamically-loaded kernel code blob.
 *
 * The firmware calls this function (at CODE_LOAD_ADDR, which is the start
 * of the .text.start section) with a func_id and a pointer to the helper
 * descriptor array. This function routes to the appropriate helper.
 *
 * This file also provides memcpy, which the LLVM-lowered kernels reference
 * via R_RISCV_CALL_PLT relocations against `memcpy` (from llvm.memcpy
 * intrinsics). The blob is independently linked, so it needs its own copy.
 */
#include <stdint.h>

extern void cpu_hoisted_ttir_abs_713b937c_helper(void *);
extern void cpu_hoisted_ttir_matmul_37790696_helper(void *);

void *memcpy(void *dest, const void *src, uint64_t count) {
  uint8_t *d = (uint8_t *)dest;
  const uint8_t *s = (const uint8_t *)src;
  while (count--) {
    *d++ = *s++;
  }
  return dest;
}

__attribute__((section(".text.start"))) void cpu2_dispatch(uint32_t func_id,
                                                           void *args) {
  switch (func_id) {
  case 0:
    cpu_hoisted_ttir_abs_713b937c_helper(args);
    break;
  case 1:
    cpu_hoisted_ttir_matmul_37790696_helper(args);
    break;
  }
}
