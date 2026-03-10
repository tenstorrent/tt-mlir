// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H

namespace experimental {

// Stalls the UNPACK thread until the previous PACK cycle's write is
// committed to L1. Call this before unpacking data that was just packed
// by the previous linalg.generic to guarantee L1 read-after-write ordering.
ALWI void unpack_stall_on_pack() {
  tile_regs_acquire();
  tile_regs_commit();
  tile_regs_wait();
  tile_regs_release();
  UNPACK(TTI_SEMWAIT(
      p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
      semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX));
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
