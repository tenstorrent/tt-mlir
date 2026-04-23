// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 3: full PoC. TTNN runs ttnn::add on Tensix to produce a tensor of
// 5.0s; the host stages that to L2CPU0 DRAM; the X280 increments each
// float to 6.0; the host reads back and verifies. TTNN keeps the device
// open the whole time. We open a parallel UMD TTDevice on the same
// /dev/tenstorrent/0 to drive the L2CPU0 tile (NOC + ARC APB). UMD allows
// concurrent opens; TTNN only touches Tensix, so the two never conflict
// on a register or DRAM bank.
//
// (MetalContext::instance().get_cluster().get_driver() would let us share
// TTNN's TTDevice instead, but that header isn't part of the public
// install. The parallel-open approach keeps step3 dependency-light.)
//
// At exit we close the MeshDevice first so its mappings drop, THEN warm
// reset the card to stop the X280 from running our firmware in the
// background (otherwise the host crashes ~1 minute later).
//
// Usage: step3_ttnn_x280 <path/to/fw_step2.bin>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"

#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/distributed.hpp>
#include "umd/device/tt_device/tt_device.hpp"

#ifndef POC_TT_METAL_HOME
#error "POC_TT_METAL_HOME must be set by CMake to the tt-metal source path"
#endif

using namespace std::chrono_literals;
using tt::umd::TTDevice;

namespace {

// 32x32 = 1024 floats = 4096 bytes — same shape step3a uses, comfortably
// fits in L2CPU0 DRAM and in a single Tensix tile.
constexpr uint32_t kH = 32;
constexpr uint32_t kW = 32;

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

void PrintFirst(const char *tag, const std::vector<float> &v, size_t n = 8) {
  std::printf("[step3] %s (first %zu): ", tag, n);
  for (size_t i = 0; i < n && i < v.size(); i++) {
    std::printf("%.1f ", v[i]);
  }
  std::printf("\n");
}

} // namespace

static int run(const std::string &fw_path) {
  ::setenv("TT_METAL_RUNTIME_ROOT", POC_TT_METAL_HOME, /*overwrite=*/1);
  ::setenv("TT_METAL_HOME", POC_TT_METAL_HOME, /*overwrite=*/1);

  auto firmware = poc::ReadFile(fw_path);
  std::printf("[step3] loaded %zu bytes from %s\n", firmware.size(),
              fw_path.c_str());

  // ---- Phase A: TTNN add on Tensix ---------------------------------------
  auto mesh_device =
      tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

  ttnn::Shape shape({kH, kW});
  auto a = ttnn::full(shape, 2.0f, ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, *mesh_device);
  auto b = ttnn::full(shape, 3.0f, ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, *mesh_device);
  auto c = ttnn::add(a, b);
  auto host_data = c.to_vector<float>();
  PrintFirst("ttnn::add result", host_data);

  // ---- Phase B: X280 hop, sharing the chip with TTNN ---------------------
  // Open a parallel TTDevice on the same chardev. TTNN only drives Tensix,
  // we only drive L2CPU0 — the two don't share state. No init_tt_device()
  // because TTNN already initialised the chip.
  auto dev = TTDevice::create(0);

  // Pre-boot harvest sanity check on L2CPU0 DRAM (TTNN's init touches Tensix
  // DRAM, not the L2CPU tile).
  uint32_t pre = poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                                poc::kL2cpu0DramBase);
  if (pre == 0xffffffffu) {
    std::fprintf(stderr,
                 "error: pre-boot NOC read returned 0xffffffff — L2CPU0 "
                 "GDDR likely harvested.\n");
    return 4;
  }

  poc::BootL2cpu0(dev.get(), firmware);

  // Stage host_data into L2CPU0 DRAM.
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kDataAddr,
                host_data.data(), host_data.size() * sizeof(float));

  // Publish the Task body, then kick separately to avoid the reorder race
  // documented in poc/host/common.hpp.
  poc::Task task{};
  task.kick = 0;
  task.data_addr = poc::kDataAddr;
  task.num_elems = host_data.size();
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                &task, sizeof(task));
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  poc::kKick);

  const uint64_t done_addr = poc::kTaskAddr + offsetof(poc::Task, done);
  const uint32_t final_done = PollWord(dev.get(), done_addr, poc::kDone, 5s);
  if (final_done != poc::kDone) {
    std::fprintf(stderr, "TIMEOUT: done = 0x%08x (expected 0x%08x)\n",
                 final_done, poc::kDone);
    return 2;
  }

  std::vector<float> result(host_data.size(), 0.0f);
  poc::NocRead(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kDataAddr,
               result.data(), result.size() * sizeof(float));
  PrintFirst("after X280", result);

  // Verify every element bumped by 1.
  size_t mismatches = 0;
  for (size_t i = 0; i < result.size(); i++) {
    if (result[i] != host_data[i] + 1.0f) {
      mismatches++;
    }
  }

  // ---- Phase C: shutdown TTNN before resetting the card ------------------
  // Drop our parallel TTDevice's BAR mappings BEFORE TTNN's, so MeshDevice
  // closes from a clean state. close() then frees TTNN's mappings, after
  // which the warm reset in main() is safe.
  dev.reset();
  bool closed = mesh_device->close();
  if (!closed) {
    std::fprintf(stderr, "warning: MeshDevice::close() returned false\n");
  }

  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL: %zu / %zu elements not incremented by 1\n",
                 mismatches, result.size());
    return 3;
  }
  std::printf("SUCCESS: TTNN add (5.0) -> X280 increment (6.0) across %zu "
              "elements\n",
              result.size());
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

  // Stop the X280 before we leave so it doesn't keep making bad NOC fetches
  // and crash the host a minute later. TTNN was closed inside run().
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
