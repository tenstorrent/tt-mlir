// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 2: boot the X280 inside L2CPU0, hand it a 16-element float array via
// a task descriptor in L2CPU DRAM, and verify the firmware adds 1.0 to each
// element.
//
// Lifecycle mirrors step1: ResetCard at entry and at exit (success or
// failure), no init_tt_device() (it blocks waiting for ARC telemetry on a
// freshly-reset card), pre-boot NOC read of L2CPU0 DRAM as the harvest
// sanity check.
//
// Signaling: the host publishes the Task body (data_addr, num_elems) with
// kick=0, then writes kick=kKick as a separate NocWrite32. The firmware
// publishes completion by writing kDone to the `done` field, which the
// host polls. The two signals MUST live at different offsets — see the
// shadow note in common.hpp.
//
// Usage: step2_task <path/to/fw_step2.bin>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "umd/device/tt_device/tt_device.hpp"

using namespace std::chrono_literals;
using tt::umd::TTDevice;

namespace {

constexpr size_t kNumElems = 16;
constexpr float kInitialValue = 5.0f;
constexpr float kExpectedValue = 6.0f;

uint32_t PollWord(TTDevice *dev, uint64_t addr, uint32_t target,
                  std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  uint32_t value = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    value = poc::NocRead32(dev, poc::kL2cpu0NocX, poc::kL2cpu0NocY, addr);
    if (value == target) {
      return value;
    }
    std::this_thread::sleep_for(1ms);
  }
  return value;
}

} // namespace

static int run(const std::string &fw_path) {
  auto firmware = poc::ReadFile(fw_path);
  std::printf("[step2] loaded %zu bytes from %s\n", firmware.size(),
              fw_path.c_str());

  poc::ResetCard(0);
  auto dev = TTDevice::create(0);

  uint32_t pre = poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                                poc::kL2cpu0DramBase);
  if (pre == 0xffffffffu) {
    std::fprintf(stderr,
                 "error: pre-boot NOC read returned 0xffffffff — L2CPU0 "
                 "GDDR likely harvested or chip wedged.\n");
    return 4;
  }

  poc::BootL2cpu0(dev.get(), firmware);

  std::vector<float> data(kNumElems, kInitialValue);
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kDataAddr,
                data.data(), data.size() * sizeof(float));

  poc::Task task{};
  task.kick = 0;
  task.data_addr = poc::kDataAddr;
  task.num_elems = kNumElems;
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                &task, sizeof(task));
  // Publish the kick as a separate write so the firmware can't observe
  // kick=kKick before the body fields land in DRAM (NOC packetisation can
  // reorder bytes within a single NocWrite).
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  poc::kKick);

  const uint64_t done_addr = poc::kTaskAddr + offsetof(poc::Task, done);
  const uint32_t final_done = PollWord(dev.get(), done_addr, poc::kDone, 5s);
  if (final_done != poc::kDone) {
    std::fprintf(stderr, "TIMEOUT: done = 0x%08x (expected 0x%08x)\n",
                 final_done, poc::kDone);
    return 2;
  }

  std::vector<float> result(kNumElems, 0.0f);
  poc::NocRead(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kDataAddr,
               result.data(), result.size() * sizeof(float));

  size_t mismatches = 0;
  for (size_t i = 0; i < kNumElems; i++) {
    if (result[i] != kExpectedValue) {
      mismatches++;
    }
  }

  std::printf("Result:");
  for (size_t i = 0; i < kNumElems; i++) {
    std::printf(" %.1f", result[i]);
  }
  std::printf("\n");

  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL: %zu / %zu elements != %.1f\n", mismatches,
                 kNumElems, kExpectedValue);
    return 3;
  }
  std::printf("SUCCESS: X280 incremented all %zu floats 5.0 -> 6.0\n",
              kNumElems);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fw_step2.bin>\n", argv[0]);
    return 1;
  }

  int rc = 4;
  try {
    rc = run(argv[1]);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 4;
  }

  try {
    poc::ResetCard(0);
  } catch (const std::exception &e) {
    std::fprintf(stderr,
                 "warning: cleanup warm-reset failed: %s\n"
                 "         the chip may be in a bad state — `tt-smi -r 0` "
                 "before retrying.\n",
                 e.what());
  }
  return rc;
}
