// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 4: X280 modifies a TTNN-resident DRAM tensor IN PLACE, no host
// staging hop. TTNN computes a 5.0-tensor on Tensix and leaves it
// interleaved across DRAM banks; we tell the X280 the bank-info table and
// it walks every page through its own NOC TLB windows, adding 1.0 to each
// float; TTNN reads back through the normal Tensix-DRAM path and sees the
// X280's 6.0s.
//
// This validates the actual goal of the PoC: the X280 can act on TTNN
// tensor data without round-tripping through PCIe.
//
// Usage: step4_inplace <path/to/fw_step4.bin>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"

#include "tt-metalium/allocator.hpp"
#include "tt-metalium/buffer.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/mesh_buffer.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "umd/device/tt_device/tt_device.hpp"

#ifndef POC_TT_METAL_HOME
#error "POC_TT_METAL_HOME must be set by CMake to the tt-metal source path"
#endif

using namespace std::chrono_literals;
using tt::umd::TTDevice;

namespace {

// 512x32 = 16 tiles in TILE layout = 16 pages. With ~7 active DRAM banks on
// this card the tensor wraps multiple times across banks, exercising the
// firmware's bank/page-in-bank arithmetic.
constexpr uint32_t kH = 512;
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

} // namespace

static int run(const std::string &fw_path) {
  ::setenv("TT_METAL_RUNTIME_ROOT", POC_TT_METAL_HOME, /*overwrite=*/1);
  ::setenv("TT_METAL_HOME", POC_TT_METAL_HOME, /*overwrite=*/1);

  auto firmware = poc::ReadFile(fw_path);
  std::printf("[step4] loaded %zu bytes from %s\n", firmware.size(),
              fw_path.c_str());

  // ---- Phase A: TTNN — compute 5.0 in interleaved Tensix DRAM ------------
  auto mesh_device =
      tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

  ttnn::Shape shape({kH, kW});
  auto a = ttnn::full(shape, 2.0f, ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, *mesh_device);
  auto b = ttnn::full(shape, 3.0f, ttnn::DataType::FLOAT32,
                      ttnn::Layout::TILE, *mesh_device);
  auto c = ttnn::add(a, b);

  // Pull bank/buffer info from the TTNN result.
  const auto &mesh_buf = c.mesh_buffer();
  tt::tt_metal::Buffer *buf = mesh_buf.get_reference_buffer();
  if (!buf || buf->buffer_type() != tt::tt_metal::BufferType::DRAM) {
    std::fprintf(stderr,
                 "error: result buffer missing or not in DRAM (type=%d)\n",
                 buf ? static_cast<int>(buf->buffer_type()) : -1);
    return 5;
  }

  const auto &allocator = mesh_device->allocator();
  const uint32_t num_banks =
      allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
  if (num_banks == 0 || num_banks > poc::kStep4MaxBanks) {
    std::fprintf(stderr,
                 "error: num_banks=%u out of range [1, %d]\n", num_banks,
                 poc::kStep4MaxBanks);
    return 5;
  }

  const uint32_t page_size = buf->page_size();
  const uint32_t aligned_page_size = buf->aligned_page_size();
  const uint64_t num_pages = mesh_buf.num_pages();
  const uint64_t buf_base = mesh_buf.address();

  std::printf("[step4] num_banks=%u page_size=%u aligned=%u num_pages=%lu "
              "buf_base=0x%lx\n",
              num_banks, page_size, aligned_page_size,
              static_cast<unsigned long>(num_pages),
              static_cast<unsigned long>(buf_base));

  if (page_size % sizeof(float) != 0) {
    std::fprintf(stderr,
                 "error: page_size=%u not a multiple of sizeof(float)\n",
                 page_size);
    return 5;
  }

  // Build the per-bank info: NOC0 coords + buffer base in that bank.
  poc::Step4Task task{};
  task.kick = 0;
  task.num_banks = num_banks;
  task.aligned_page_size = aligned_page_size;
  task.page_size = page_size;
  task.num_pages = num_pages;
  // The allocator's get_logical_core_from_bank_id returns a Tensix-style
  // coord, not a DRAM channel — which then errors out of the
  // virtual_core_from_logical_core(..., CoreType::DRAM) bounds check. The
  // public API for DRAM channel -> NOC core goes through MeshDevice's
  // logical_core_from_dram_channel + virtual_core_from_logical_core. For
  // the default Blackhole DRAM allocator there's one bank per channel, so
  // bank_id == dram_channel.
  const int num_dram_channels = mesh_device->num_dram_channels();
  if (num_dram_channels != static_cast<int>(num_banks)) {
    std::fprintf(stderr,
                 "warning: num_dram_channels=%d != allocator num_banks=%u; "
                 "step4 assumes 1:1, results may be wrong\n",
                 num_dram_channels, num_banks);
  }
  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    const auto logical = mesh_device->logical_core_from_dram_channel(bank_id);
    const auto noc = mesh_device->virtual_core_from_logical_core(
        logical, tt::CoreType::DRAM);
    task.bank_x[bank_id] = static_cast<uint32_t>(noc.x);
    task.bank_y[bank_id] = static_cast<uint32_t>(noc.y);
    const int32_t bank_offset =
        allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bank_id);
    task.bank_base[bank_id] =
        static_cast<uint64_t>(static_cast<int64_t>(buf_base) + bank_offset);
    std::printf("[step4]   bank[%u]: noc=(%u,%u) base=0x%lx\n", bank_id,
                task.bank_x[bank_id], task.bank_y[bank_id],
                static_cast<unsigned long>(task.bank_base[bank_id]));
  }

  // ---- Phase B: boot X280 alongside TTNN ---------------------------------
  auto dev = TTDevice::create(0);

  uint32_t pre = poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                                poc::kL2cpu0DramBase);
  if (pre == 0xffffffffu) {
    std::fprintf(stderr,
                 "error: pre-boot NOC read returned 0xffffffff — L2CPU0 "
                 "GDDR likely harvested.\n");
    return 4;
  }

  poc::BootL2cpu0(dev.get(), firmware);

  // Publish the Task body, then kick separately to avoid the offset-0
  // reorder race.
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                &task, sizeof(task));
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  poc::kKick);

  const uint64_t done_addr = poc::kTaskAddr + offsetof(poc::Step4Task, done);
  // Larger tensors take longer; 10s gives plenty of margin for 16 pages of
  // 1024 floats with current TLB-window throughput.
  const uint32_t final_done = PollWord(dev.get(), done_addr, poc::kDone, 10s);
  if (final_done != poc::kDone) {
    std::fprintf(stderr, "TIMEOUT: done = 0x%08x (expected 0x%08x)\n",
                 final_done, poc::kDone);
    return 2;
  }

  // ---- Phase C: read tensor through TTNN, verify in-place 5.0 -> 6.0 -----
  auto result = c.to_vector<float>();
  std::printf("[step4] result size=%zu, first 8: ", result.size());
  for (size_t i = 0; i < 8 && i < result.size(); i++) {
    std::printf("%.1f ", result[i]);
  }
  std::printf("\n");

  size_t mismatches = 0;
  for (float v : result) {
    if (v != 6.0f) {
      mismatches++;
    }
  }

  // Drop the parallel TTDevice before TTNN's MeshDevice so close() runs on
  // a clean state.
  dev.reset();
  bool closed = mesh_device->close();
  if (!closed) {
    std::fprintf(stderr, "warning: MeshDevice::close() returned false\n");
  }

  if (mismatches != 0) {
    std::fprintf(stderr, "FAIL: %zu / %zu elements != 6.0\n", mismatches,
                 result.size());
    return 3;
  }
  std::printf("SUCCESS: TTNN tensor (5.0) modified in-place by X280 -> 6.0 "
              "across %zu elements (%lu pages, %u banks)\n",
              result.size(), static_cast<unsigned long>(num_pages),
              num_banks);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fw_step4.bin>\n", argv[0]);
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
