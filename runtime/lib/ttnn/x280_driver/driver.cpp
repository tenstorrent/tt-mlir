// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "driver.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "umd/device/tt_device/tt_device.hpp"
#include "umd/device/types/xy_pair.hpp"
#include "umd/device/warm_reset.hpp"

namespace poc {

using tt::umd::TTDevice;

namespace {

// ARC APB helpers. TTDevice::{read,write}_to_arc_apb takes the absolute ARC
// XBAR offset — same value you'd pass to pyluwen's axi_read32/axi_write32.
uint32_t ArcRead32(TTDevice *dev, uint64_t arc_offset) {
  uint32_t v = 0;
  dev->read_from_arc_apb(&v, arc_offset, sizeof(v));
  return v;
}

void ArcWrite32(TTDevice *dev, uint64_t arc_offset, uint32_t value) {
  dev->write_to_arc_apb(&value, arc_offset, sizeof(value));
}

// clock.py / l2cpu.cpp step the PLL one unit at a time with a small delay to
// avoid jitter. The delay only needs to be long enough for the PLL to settle
// between writes; 1 us is generous compared to the 1 ns in clock.py.
void PllStepDelay() {
  std::this_thread::sleep_for(std::chrono::microseconds(1));
}

struct PllSolution {
  uint16_t fbdiv;
  uint8_t postdiv[4];
};

// Solutions copied verbatim from tt-bh-linux/clock.py.
PllSolution SolutionFor(int mhz) {
  switch (mhz) {
  case 200:
    return {128, {15, 15, 15, 15}};
  case 1750:
    return {140, {1, 1, 1, 1}};
  default:
    throw std::runtime_error(
        "poc::SetL2cpuPll: only 200 and 1750 MHz supported, got " +
        std::to_string(mhz));
  }
}

// Step each postdiv byte one unit at a time toward `target`.
void StepPostdiv(TTDevice *dev, uint8_t postdiv[4], int index, uint8_t target) {
  const int8_t delta = (target > postdiv[index]) ? +1 : -1;
  while (postdiv[index] != target) {
    postdiv[index] = static_cast<uint8_t>(postdiv[index] + delta);
    uint32_t packed;
    std::memcpy(&packed, postdiv, 4);
    ArcWrite32(dev, kArcPll4Base + kArcPllCntl5Offset, packed);
    PllStepDelay();
  }
}

// Step fbdiv one unit at a time. PLL_CNTL_1 layout: refdiv[0], postdiv[1],
// fbdiv[2..3].
void StepFbdiv(TTDevice *dev, uint32_t &cntl1, uint16_t target) {
  uint16_t fbdiv = static_cast<uint16_t>(cntl1 >> 16);
  const int16_t delta = (target > fbdiv) ? +1 : -1;
  while (fbdiv != target) {
    fbdiv = static_cast<uint16_t>(fbdiv + delta);
    cntl1 = (cntl1 & 0x0000ffffu) | (static_cast<uint32_t>(fbdiv) << 16);
    ArcWrite32(dev, kArcPll4Base + kArcPllCntl1Offset, cntl1);
    PllStepDelay();
  }
}

} // namespace

void ResetCard(int device_index) {
  // Mirrors tt-bh-linux/boot.py:98 (`pci_board_reset(...)`). On Blackhole
  // this drops to a config-space SBR via the kernel driver. Must be done
  // before any TTDevice is created so we don't hold a stale mapping across
  // the reset.
  if (!tt::umd::WarmReset::warm_reset_chip_id({device_index})) {
    throw std::runtime_error("ResetCard: WarmReset::warm_reset_chip_id failed "
                             "for /dev/tenstorrent/" +
                             std::to_string(device_index) +
                             ". Try `tt-smi -r " +
                             std::to_string(device_index) + "` manually.");
  }
  // boot.py:104 sleeps 5 s after reset because telemetry isn't immediately
  // available. WarmReset already does a 2 s POST_RESET_WAIT internally, so
  // 3 s extra here keeps total wait at ~5 s.
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

std::vector<uint8_t> ReadFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open firmware file: " + path);
  }
  in.seekg(0, std::ios::end);
  auto size = in.tellg();
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (!in.read(reinterpret_cast<char *>(buf.data()), size)) {
    throw std::runtime_error("failed to read firmware file: " + path);
  }
  // TTDevice::write_to_device requires 4-byte-aligned sizes.
  while ((buf.size() % 4) != 0) {
    buf.push_back(0);
  }
  return buf;
}

void SetL2cpuPll(TTDevice *dev, int target_mhz) {
  const PllSolution sol = SolutionFor(target_mhz);

  uint32_t cntl5 = ArcRead32(dev, kArcPll4Base + kArcPllCntl5Offset);
  uint32_t cntl1 = ArcRead32(dev, kArcPll4Base + kArcPllCntl1Offset);

  uint8_t postdiv[4];
  std::memcpy(postdiv, &cntl5, 4);

  // Step 1: increase any postdivs that need to go up (slower first).
  for (int i = 0; i < 4; i++) {
    if (sol.postdiv[i] > postdiv[i]) {
      StepPostdiv(dev, postdiv, i, sol.postdiv[i]);
    }
  }

  // Step 2: adjust fbdiv.
  StepFbdiv(dev, cntl1, sol.fbdiv);

  // Step 3: decrease any postdivs that need to go down (faster last).
  for (int i = 0; i < 4; i++) {
    if (sol.postdiv[i] < postdiv[i]) {
      StepPostdiv(dev, postdiv, i, sol.postdiv[i]);
    }
  }
}

void DeassertL2cpuReset(TTDevice *dev, int index) {
  if (index < 0 || index > 3) {
    throw std::runtime_error("DeassertL2cpuReset: index must be 0..3");
  }
  const uint64_t reg = kArcResetUnitBase + kArcL2cpuResetOffset;
  uint32_t val = ArcRead32(dev, reg);
  val |= (1u << (index + 4));
  ArcWrite32(dev, reg, val);
  // boot.py does a read after the write to flush the APB transaction.
  (void)ArcRead32(dev, reg);
}

void NocWrite(TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr,
              const void *data, size_t size) {
  dev->write_to_device(data, tt_xy_pair(x, y), addr,
                       static_cast<uint32_t>(size));
}

void NocRead(TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr, void *out,
             size_t size) {
  dev->read_from_device(out, tt_xy_pair(x, y), addr,
                        static_cast<uint32_t>(size));
}

void NocWrite32(TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr,
                uint32_t value) {
  NocWrite(dev, x, y, addr, &value, sizeof(value));
}

uint32_t NocRead32(TTDevice *dev, uint16_t x, uint16_t y, uint64_t addr) {
  uint32_t v = 0;
  NocRead(dev, x, y, addr, &v, sizeof(v));
  return v;
}

void BootL2cpu0(TTDevice *dev, const std::vector<uint8_t> &firmware) {
  // 1. Load firmware into DRAM at the NOC tile address.
  NocWrite(dev, kL2cpu0NocX, kL2cpu0NocY, kFwAddr, firmware.data(),
           firmware.size());

  // 2. Zero the mailbox / task descriptor area. sizeof(Task) covers both
  // the Step 1 mailbox (its first uint32) and the full Step 2 Task struct.
  uint32_t zeros[sizeof(Task) / sizeof(uint32_t)] = {};
  NocWrite(dev, kL2cpu0NocX, kL2cpu0NocY, kMailboxAddr, zeros, sizeof(zeros));

  // 3. Enable the whole L3 cache (see boot.py, "Enable the whole cache when
  //    using DRAM"). Without this, instruction fetches from DRAM may stall.
  NocWrite32(dev, kL2cpu0NocX, kL2cpu0NocY, kL3CacheEnableReg,
             kL3CacheEnableVal);

  // 4. Program reset vectors for all 4 X280 harts (core_0..3, each with a
  //    lo/hi uint32 pair at +0x0/+0x4, +0x8/+0xC, ..., +0x18/+0x1C).
  const uint32_t reset_lo = static_cast<uint32_t>(kFwAddr & 0xffffffffu);
  const uint32_t reset_hi = static_cast<uint32_t>(kFwAddr >> 32);
  for (int core = 0; core < 4; core++) {
    NocWrite32(dev, kL2cpu0NocX, kL2cpu0NocY, kResetVectorBase + core * 8 + 0,
               reset_lo);
    NocWrite32(dev, kL2cpu0NocX, kL2cpu0NocY, kResetVectorBase + core * 8 + 4,
               reset_hi);
  }

  // 5. Canonical X280 reset dance: PLL down → deassert reset → PLL up.
  SetL2cpuPll(dev, 200);
  DeassertL2cpuReset(dev, /*index=*/0);
  SetL2cpuPll(dev, 1750);

  // 6. Configure L2 prefetchers (boot.py always does this after reset,
  //    even without --boot, so we replicate).
  const uint64_t prefetch_offsets[] = {0x0000, 0x2000, 0x4000, 0x6000};
  for (uint64_t off : prefetch_offsets) {
    NocWrite32(dev, kL2cpu0NocX, kL2cpu0NocY, kL2PrefetchBase + off + 0,
               0x15811);
    NocWrite32(dev, kL2cpu0NocX, kL2cpu0NocY, kL2PrefetchBase + off + 4,
               0x38c84e);
  }
}

} // namespace poc
