// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for the X280 PoC host programs.
//
// All constants come from tt-bh-linux (boot.py, clock.py, console/l2cpu.cpp)
// and the tt-umd blackhole_implementation.hpp. We only target L2CPU0 because
// that's what the tt-bh-linux `make boot` default drives and it keeps the
// PoC simple.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tt::umd {
class TTDevice;
} // namespace tt::umd

namespace poc {

// ---------------------------------------------------------------------------
// L2CPU0 NOC0 coordinates and memory layout (from tt-bh-linux).
// ---------------------------------------------------------------------------
constexpr uint16_t kL2cpu0NocX = 8;
constexpr uint16_t kL2cpu0NocY = 3;

constexpr uint64_t kL2cpu0DramBase = 0x400030000000ULL;

// Layout inside L2CPU0 DRAM. These MUST match the hard-coded addresses in
// fw_step1.c / fw_step2.c — if you change one, change both. All addresses
// are on the Memory Port (data-cacheable); NOC reads from the host see the
// X280's stores via the same path with no explicit CMO needed (cbo.flush
// hangs on this silicon — see README "Silicon quirks worth knowing").
constexpr uint64_t kFwAddr = kL2cpu0DramBase;
constexpr uint64_t kMailboxAddr = kL2cpu0DramBase + 0x00100000ULL; // +1 MiB
constexpr uint64_t kTaskAddr =
    kL2cpu0DramBase + 0x00100000ULL; // +1 MiB (alias of mailbox)
constexpr uint64_t kDataAddr = kL2cpu0DramBase + 0x00200000ULL; // +2 MiB

// ---------------------------------------------------------------------------
// NOC-addressable registers inside the L2CPU tile (from boot.py).
// ---------------------------------------------------------------------------
constexpr uint64_t kResetVectorBase = 0xfffff7fefff10000ULL;
constexpr uint64_t kL3CacheEnableReg = 0x02010008ULL;
constexpr uint32_t kL3CacheEnableVal = 0xfu;
constexpr uint64_t kL2PrefetchBase = 0x02030000ULL;

// ARC APB offsets — relative to the ARC APB peripheral region, NOT to the
// full AXI peripheral space. pyluwen's axi_write32(0x80030014) and UMD's
// write_to_arc_apb(0x30014) target the same register; UMD strips the
// 0x80000000 AXI base internally (see blackhole_implementation.hpp:
// ARC_RESET_UNIT_OFFSET = 0x30000, ARC_APB_BAR0_XBAR_OFFSET_START = 0x1FF00000,
// and BlackholeTTDevice::write_to_arc_apb at blackhole_tt_device.cpp:250).
// Earlier versions used the pyluwen address space (0x80030000 / 0x80020500)
// which silently writes ~0x9FF3xxxx in BAR0 — outside the ARC mapping —
// so the L2CPU reset never deasserted and PLL4 never moved.
constexpr uint64_t kArcResetUnitBase = 0x30000ULL;
constexpr uint64_t kArcL2cpuResetOffset = 0x14ULL; // L2CPU_RESET
constexpr uint64_t kArcPll4Base = 0x20500ULL;
constexpr uint64_t kArcPllCntl1Offset = 0x4ULL;  // refdiv/postdiv/fbdiv
constexpr uint64_t kArcPllCntl5Offset = 0x14ULL; // 4x postdiv (byte each)

// ---------------------------------------------------------------------------
// Task descriptor shared with fw_step2.c. Layout must match byte-for-byte.
// ---------------------------------------------------------------------------
//
// `kick` is host -> fw and `done` is fw -> host; they live at distinct
// offsets on purpose. On this Blackhole silicon, any DRAM word the host
// has written via NOC returns the host's last write on subsequent host
// reads — even after the X280 overwrites it. Writes by the X280 to
// addresses the host has not touched ARE visible normally. So if both
// signals shared a word, host polling for kDone would never see it (it
// would always read back its own kick value). The host MUST never write
// to `done`. data_addr / num_elems are part of the body the host
// publishes; the X280 only reads them after observing kick == kKick.
struct Task {
  uint32_t kick; // host -> fw start signal (0 / kKick)
  uint32_t pad0;
  uint64_t data_addr; // host -> fw
  uint64_t num_elems; // host -> fw
  uint32_t done;      // fw -> host (host MUST never write)
  uint32_t pad1;
};
static_assert(sizeof(Task) == 32, "Task layout must match firmware");

constexpr uint32_t kKick = 0x1u;
constexpr uint32_t kDone = 0xC0FFEE03u;

// Task descriptor for step4 (in-place increment of a TTNN-interleaved DRAM
// tensor via the X280's NOC TLBs). Layout must match fw_step4.c byte-for-byte.
constexpr int kStep4MaxBanks = 8;
struct Step4Task {
  uint32_t kick; // host -> fw start signal (0 / kKick)
  uint32_t num_banks;
  uint32_t aligned_page_size; // bytes between same-bank consecutive pages
  uint32_t page_size;         // bytes of float data per page
  uint64_t num_pages;
  uint64_t bank_base[kStep4MaxBanks]; // per-bank base addr of the buffer
  uint32_t bank_x[kStep4MaxBanks];    // NOC0 X of each bank's DRAM core
  uint32_t bank_y[kStep4MaxBanks];    // NOC0 Y of each bank's DRAM core
  uint32_t done;                      // fw -> host (host MUST never write)
  uint32_t pad_done;
};
static_assert(sizeof(Step4Task) == 24 + 8 * 8 + 8 * 4 * 2 + 8,
              "Step4Task layout must match firmware");

// Task descriptor for step5 (X280 copies a TTNN-interleaved DRAM tensor into
// a contiguous local-DRAM staging buffer, runs an MLIR-lowered LLVM-dialect
// kernel linked into the firmware via poc/fw/cpu.o, and writes the result
// back to a separate TTNN output tensor). Layout must match fw_step5.c
// byte-for-byte. Input and output tensors are assumed to share the same
// allocator configuration (num_banks / page_size / aligned_page_size /
// num_pages and the same bank DRAM cores), so we keep one bank_x/bank_y
// table and only duplicate the per-bank base offsets.
struct Step5Task {
  uint32_t kick; // host -> fw start signal (0 / kKick)
  uint32_t num_banks;
  uint32_t aligned_page_size;
  uint32_t page_size;
  uint64_t num_pages;
  uint64_t input_bank_base[kStep4MaxBanks];
  uint64_t output_bank_base[kStep4MaxBanks];
  uint32_t bank_x[kStep4MaxBanks];
  uint32_t bank_y[kStep4MaxBanks];
  uint32_t done; // fw -> host (host MUST never write)
  uint32_t pad_done;
};
static_assert(sizeof(Step5Task) == 24 + 8 * 8 * 2 + 8 * 4 * 2 + 8,
              "Step5Task layout must match firmware");

// ---------------------------------------------------------------------------
// Host helpers.
// ---------------------------------------------------------------------------

// Read an entire file into memory. Throws on failure.
std::vector<uint8_t> ReadFile(const std::string &path);

// Issue a warm reset of /dev/tenstorrent/<device_index> and wait for the
// chip to come back. This is the equivalent of `tt-smi -r 0` /
// `pci_board_reset` from tt-bh-linux/boot.py — the L2CPU PLL/reset dance
// in BootL2cpu0() *requires* the chip to start from a freshly-reset state,
// otherwise stepping PLL4 to 200 MHz on a running L2CPU wedges its AXI
// fabric (and can hang the host). Must be called BEFORE TTDevice::create.
// Throws on failure.
void ResetCard(int device_index);

// Program the L2CPU PLL (PLL4) to the target frequency. Only 200 and 1750 MHz
// are supported — those are the two settings boot.py cycles through.
void SetL2cpuPll(tt::umd::TTDevice *dev, int target_mhz);

// Deassert reset on L2CPU `index` (0..3) by setting bit (index + 4) in the
// ARC reset unit's L2CPU_RESET register. Must be preceded by SetL2cpuPll(200).
void DeassertL2cpuReset(tt::umd::TTDevice *dev, int index);

// Full boot flow for L2CPU0:
//   1. write firmware image to DRAM base
//   2. enable L3 cache
//   3. program reset vectors for all 4 X280 harts to point at firmware
//   4. drop PLL to 200 MHz, deassert reset, raise PLL to 1750 MHz
//   5. configure L2 prefetchers (matches boot.py order exactly)
// Throws if L2CPU0 is harvested or any NOC/APB access fails.
void BootL2cpu0(tt::umd::TTDevice *dev, const std::vector<uint8_t> &firmware);

// NOC helpers that hide the tt_xy_pair / cast boilerplate.
void NocWrite32(tt::umd::TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr,
                uint32_t value);
uint32_t NocRead32(tt::umd::TTDevice *dev, uint16_t x, uint16_t y,
                   uint64_t addr);
void NocWrite(tt::umd::TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr,
              const void *data, size_t size);
void NocRead(tt::umd::TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr,
             void *out, size_t size);

} // namespace poc
