// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Freestanding runtime support for the X280 kernel code blob.
// Provides memcpy, which the LLVM-lowered kernels reference via
// R_RISCV_CALL_PLT relocations (from llvm.memcpy intrinsics).
#include <stdint.h>

void *memcpy(void *dest, const void *src, uint64_t count) {
  uint8_t *d = (uint8_t *)dest;
  const uint8_t *s = (const uint8_t *)src;
  while (count--) {
    *d++ = *s++;
  }
  return dest;
}
