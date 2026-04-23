// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 1: boot the X280 inside L2CPU0 with a trivial firmware and wait for
// it to write 0xDEADBEEF to a mailbox in L2CPU DRAM.
//
// We do not call init_tt_device() — on a freshly-reset card it blocks
// waiting for ARC telemetry that isn't ready yet. The pre-boot NOC read
// of L2CPU0 DRAM is our harvest sanity check instead: if it returns
// 0xffffffff the L2CPU0 GDDR slice is harvested or the chip is wedged.
//
// On exit (success or failure) we issue a warm reset so the X280 doesn't
// keep running after the process ends — leaving it running causes the host
// to crash ~1 minute later from runaway NOC traffic.
//
// Usage: step1_boot <path/to/fw_step1.bin>

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>

#include "common.hpp"
#include "umd/device/tt_device/tt_device.hpp"

using namespace std::chrono_literals;
using tt::umd::TTDevice;

// `dev` must be destroyed before main() issues the cleanup warm reset, so
// keep the chip-touching code in its own scope.
static int run(const std::string &fw_path) {
  auto firmware = poc::ReadFile(fw_path);
  std::printf("[step1] loaded %zu bytes from %s\n", firmware.size(),
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

  const auto deadline = std::chrono::steady_clock::now() + 5s;
  uint32_t mailbox = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    mailbox = poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                             poc::kMailboxAddr);
    if (mailbox == 0xDEADBEEFu) {
      break;
    }
    std::this_thread::sleep_for(1ms);
  }

  if (mailbox != 0xDEADBEEFu) {
    std::fprintf(stderr, "TIMEOUT: mailbox = 0x%08x (expected 0xDEADBEEF)\n",
                 mailbox);
    return 2;
  }

  std::printf("SUCCESS: X280 booted, mailbox = 0x%08x\n", mailbox);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fw_step1.bin>\n", argv[0]);
    return 1;
  }

  int rc = 3;
  try {
    rc = run(argv[1]);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    rc = 3;
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
