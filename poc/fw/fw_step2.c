// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Step 2 firmware: wait for the host's kick at TASK_ADDR + 0, then
 * increment each float in the host-supplied data buffer by 1.0f, then
 * publish a "done" magic at TASK_ADDR + offsetof(done) so the host can
 * poll for completion.
 *
 * Why kick and done are at different offsets: writes from the host to a
 * DRAM word create a per-address read shadow on this Blackhole silicon —
 * subsequent host reads of that word return the host's last value, even
 * after the X280 overwrites it. Putting `done` at an offset the host
 * never writes is the workaround. (The data buffer at +2 MiB seems to
 * sidestep this — bulk NocWrite + bulk read of the float array works.)
 */
#include <stdint.h>

struct Task {
  volatile uint32_t kick; /* host -> fw */
  uint32_t pad0;
  uint64_t data_addr;
  uint64_t num_elems;
  uint32_t done; /* fw -> host */
  uint32_t pad1;
};

#define TASK_ADDR 0x400030100000ULL
#define KICK 0x1u
#define DONE 0xC0FFEE03u

__attribute__((noreturn)) void fw_main(void) {
  struct Task *task = (struct Task *)TASK_ADDR;

  while (__atomic_load_n(&task->kick, __ATOMIC_ACQUIRE) != KICK) {
    __asm__ volatile("nop");
  }

  float *data = (float *)task->data_addr;
  uint64_t n = task->num_elems;
  for (uint64_t i = 0; i < n; i++) {
    data[i] += 1.0f;
  }

  __asm__ volatile("fence rw, rw" ::: "memory");
  __atomic_store_n(&task->done, DONE, __ATOMIC_RELEASE);

  for (;;) {
    __asm__ volatile("wfi");
  }
}
