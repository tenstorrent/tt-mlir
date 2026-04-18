// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SEMAPHORE_H
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_SEMAPHORE_H

namespace experimental {

FORCE_INLINE
void semaphore_wait(volatile tt_l1_ptr uint32_t *sem_addr, uint32_t val) {
  WAYPOINT("NSW");
  do {
    invalidate_l1_cache();
  } while (*sem_addr < val);
  WAYPOINT("NSD");
}

FORCE_INLINE
void semaphore_wait_min(volatile tt_l1_ptr uint32_t *sem_addr, uint32_t val) {
  WAYPOINT("NSMW");
  do {
    invalidate_l1_cache();
  } while (*sem_addr < val);
  WAYPOINT("NSMD");
}

} // namespace experimental

#endif
