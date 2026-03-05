// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H

namespace experimental {

ALWI void tile_regs_acquire() {
  UNPACK(TTI_SEMWAIT(
      p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
      semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX));
  MATH((llk_math_wait_for_dest_available()));
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
