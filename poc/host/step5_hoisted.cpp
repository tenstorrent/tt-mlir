// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 5: end-to-end CPU-hoisting on the X280, no PCIe round-trip during the
// modify phase.
//
// TTNN allocates two ROW_MAJOR FLOAT32 tensors in interleaved DRAM:
//   - input  filled with -5.0f
//   - output filled with  0.0f (placeholder)
// The X280 firmware (built from fw_step5.c, linked against poc/fw/cpu.o) is
// booted, handed both tensors' bank tables, copies the input into a
// contiguous local-DRAM buffer, calls the MLIR-lowered abs kernel from
// cpu.o, and writes the result back to the output tensor's banks. The host
// then reads `output` via TTNN's normal Tensix-DRAM read path and verifies
// every element is +5.0f.
//
// This is the first step where the firmware actually executes compiler-
// generated code rather than hand-written kernels — closing the loop on the
// substrate the eventual tt-mlir CPU-hoisting integration will rest on.
//
// Usage: step5_hoisted <path/to/fw_step5.bin>

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

// 4000 floats: this is the exact element count baked into the cpu.o kernel
// (see the alloca [4000 x float] in the kernel's LLVM IR). Changing it
// would require regenerating poc/fw/cpu.o from a matching MLIR source.
constexpr uint32_t kNumElems = 4000;

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

bool BuildBankTable(tt::tt_metal::distributed::MeshDevice &mesh_device,
                    tt::tt_metal::Buffer *buf, uint32_t num_banks,
                    uint64_t *bank_base_out) {
  const auto &allocator = mesh_device.allocator();
  const uint64_t buf_base = buf->address();
  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    const int32_t bank_offset =
        allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bank_id);
    bank_base_out[bank_id] =
        static_cast<uint64_t>(static_cast<int64_t>(buf_base) + bank_offset);
  }
  return true;
}

} // namespace

static int run(const std::string &fw_path) {
  ::setenv("TT_METAL_RUNTIME_ROOT", POC_TT_METAL_HOME, /*overwrite=*/1);
  ::setenv("TT_METAL_HOME", POC_TT_METAL_HOME, /*overwrite=*/1);

  auto firmware = poc::ReadFile(fw_path);
  std::printf("[step5] loaded %zu bytes from %s\n", firmware.size(),
              fw_path.c_str());

  // ---- Phase A: TTNN — allocate input(-5.0) and output(0.0) tensors ------
  auto mesh_device =
      tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

  ttnn::Shape shape({kNumElems});
  auto input = ttnn::full(shape, -5.0f, ttnn::DataType::FLOAT32,
                          ttnn::Layout::ROW_MAJOR, *mesh_device);
  auto output = ttnn::full(shape, 0.0f, ttnn::DataType::FLOAT32,
                           ttnn::Layout::ROW_MAJOR, *mesh_device);

  const auto &in_mesh_buf = input.mesh_buffer();
  const auto &out_mesh_buf = output.mesh_buffer();
  tt::tt_metal::Buffer *in_buf = in_mesh_buf.get_reference_buffer();
  tt::tt_metal::Buffer *out_buf = out_mesh_buf.get_reference_buffer();
  if (!in_buf || in_buf->buffer_type() != tt::tt_metal::BufferType::DRAM ||
      !out_buf || out_buf->buffer_type() != tt::tt_metal::BufferType::DRAM) {
    std::fprintf(stderr, "error: input/output tensors must be DRAM-resident\n");
    return 5;
  }

  const auto &allocator = mesh_device->allocator();
  const uint32_t num_banks =
      allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
  if (num_banks == 0 || num_banks > poc::kStep4MaxBanks) {
    std::fprintf(stderr, "error: num_banks=%u out of range [1, %d]\n",
                 num_banks, poc::kStep4MaxBanks);
    return 5;
  }

  // Step 5 assumes input and output share allocator-determined geometry. They
  // are both DRAM, ROW_MAJOR, FLOAT32, same shape — TTNN should pick the same
  // page_size / aligned_page_size / num_pages for both. Bail loudly if not.
  const uint32_t page_size = in_buf->page_size();
  const uint32_t aligned_page_size = in_buf->aligned_page_size();
  const uint64_t num_pages = in_mesh_buf.num_pages();
  if (out_buf->page_size() != page_size ||
      out_buf->aligned_page_size() != aligned_page_size ||
      out_mesh_buf.num_pages() != num_pages) {
    std::fprintf(
        stderr,
        "error: input/output buffer geometry mismatch — "
        "in=(page=%u aligned=%u pages=%lu) "
        "out=(page=%lu aligned=%lu pages=%lu)\n",
        page_size, aligned_page_size, static_cast<unsigned long>(num_pages),
        static_cast<unsigned long>(out_buf->page_size()),
        static_cast<unsigned long>(out_buf->aligned_page_size()),
        static_cast<unsigned long>(out_mesh_buf.num_pages()));
    return 5;
  }
  if (page_size % sizeof(float) != 0) {
    std::fprintf(stderr,
                 "error: page_size=%u not a multiple of sizeof(float)\n",
                 page_size);
    return 5;
  }

  std::printf("[step5] num_banks=%u page_size=%u aligned=%u num_pages=%lu "
              "in_base=0x%lx out_base=0x%lx\n",
              num_banks, page_size, aligned_page_size,
              static_cast<unsigned long>(num_pages),
              static_cast<unsigned long>(in_mesh_buf.address()),
              static_cast<unsigned long>(out_mesh_buf.address()));

  // Build the per-bank table. Bank coords are shared (both buffers live on
  // the same allocator's DRAM cores); only the per-bank base offset differs.
  poc::Step5Task task{};
  task.kick = 0;
  task.num_banks = num_banks;
  task.aligned_page_size = aligned_page_size;
  task.page_size = page_size;
  task.num_pages = num_pages;
  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    const auto logical = mesh_device->logical_core_from_dram_channel(bank_id);
    const auto noc = mesh_device->virtual_core_from_logical_core(
        logical, tt::CoreType::DRAM);
    task.bank_x[bank_id] = static_cast<uint32_t>(noc.x);
    task.bank_y[bank_id] = static_cast<uint32_t>(noc.y);
  }
  BuildBankTable(*mesh_device, in_buf, num_banks, task.input_bank_base);
  BuildBankTable(*mesh_device, out_buf, num_banks, task.output_bank_base);

  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    std::printf("[step5]   bank[%u]: noc=(%u,%u) in=0x%lx out=0x%lx\n", bank_id,
                task.bank_x[bank_id], task.bank_y[bank_id],
                static_cast<unsigned long>(task.input_bank_base[bank_id]),
                static_cast<unsigned long>(task.output_bank_base[bank_id]));
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

  // Publish task body, then kick separately to avoid the offset-0 race.
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                &task, sizeof(task));
  poc::NocWrite32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr,
                  poc::kKick);

  const uint64_t done_addr = poc::kTaskAddr + offsetof(poc::Step5Task, done);
  // Generous timeout: byte-by-byte memcpy across the System-Port window is
  // slow (every byte is a NOC transaction). 30s is plenty for 16 KiB.
  const uint32_t final_done = PollWord(dev.get(), done_addr, poc::kDone, 30s);
  if (final_done != poc::kDone) {
    std::fprintf(stderr, "TIMEOUT: done = 0x%08x (expected 0x%08x)\n",
                 final_done, poc::kDone);
    return 2;
  }

  // ---- Phase C: read output through TTNN, verify abs(-5.0) == 5.0 --------
  auto result = output.cpu().to_vector<float>();
  std::printf("[step5] result size=%zu, first 8: ", result.size());
  for (size_t i = 0; i < 8 && i < result.size(); i++) {
    std::printf("%.1f ", result[i]);
  }
  std::printf("\n");

  size_t mismatches = 0;
  for (float v : result) {
    if (v != 5.0f) {
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
    std::fprintf(stderr, "FAIL: %zu / %zu elements != 5.0\n", mismatches,
                 result.size());
    return 3;
  }
  std::printf("SUCCESS: cpu.o on X280 mapped TTNN(-5.0) to TTNN(5.0) across "
              "%zu elements (%lu pages, %u banks)\n",
              result.size(), static_cast<unsigned long>(num_pages), num_banks);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <fw_step5.bin>\n", argv[0]);
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
