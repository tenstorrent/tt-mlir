// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H

namespace experimental {

// Initializes the PACK_DONE semaphore used by unpack_stall_on_pack. This should
// run once at compute kernel startup, not at every synchronization point.
ALWI void unpack_stall_on_pack_init() {
  PACK(t6_semaphore_init(semaphore::PACK_DONE, 0, 1));
}

// Stalls the UNPACK thread until the previous PACK cycle's write is
// committed to L1. Call this before unpacking data that was just packed
// by the previous linalg.generic to guarantee L1 read-after-write ordering.
ALWI void unpack_stall_on_pack() {
  PACK(t6_semaphore_post<>(semaphore::PACK_DONE));
  UNPACK(t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE));
  UNPACK(t6_semaphore_get<>(semaphore::PACK_DONE));
}

} // namespace experimental

#endif // TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_REG_API_H
