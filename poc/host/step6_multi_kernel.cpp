// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Step 6: generic multi-kernel dispatch on the X280.
//
// Demonstrates dispatching two different CPU-hoisted kernels (abs and matmul)
// through a single persistent firmware loop, each with different numbers of
// tensor arguments and different tensor shapes.
//
// The kernel code (compiled from cpu2.ll) is loaded as a separate blob into
// L2CPU0 DRAM at runtime, not statically linked into the firmware. The
// firmware calls into the blob via a dispatch function at a known address.
//
// Test scenario:
//   Task 1 (abs):    input(4000x1, -5.0) → output(4000x1, 5.0)
//   Task 2 (matmul): A(1x4000, 3.0) × B(4000x1, 2.0) → C(1x1, 24000.0)
//
// Usage: step6_multi_kernel <fw_step6.bin> <cpu2_blob.bin>

#include <chrono>
#include <cmath>
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

constexpr uint32_t kAbsFuncId = 0;
constexpr uint32_t kMatmulFuncId = 1;

// Fill the shared bank-coordinate tables in a Step6Task.
void FillBankCoords(tt::tt_metal::distributed::MeshDevice &mesh_device,
                    poc::Step6Task &task, uint32_t num_banks) {
  task.num_banks = num_banks;
  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    auto logical = mesh_device.logical_core_from_dram_channel(bank_id);
    auto noc =
        mesh_device.virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
    task.bank_x[bank_id] = static_cast<uint32_t>(noc.x);
    task.bank_y[bank_id] = static_cast<uint32_t>(noc.y);
  }
}

// Fill a Step6TensorMeta from a TTNN buffer's metadata.
void FillTensorMeta(tt::tt_metal::distributed::MeshDevice &mesh_device,
                    const ::ttnn::Tensor &tensor, poc::Step6TensorMeta &tm,
                    bool is_input, bool is_output, uint32_t num_banks,
                    uint32_t rank, const int64_t *sizes_and_strides) {
  const auto &mesh_buf = tensor.mesh_buffer();
  tt::tt_metal::Buffer *buf = mesh_buf.get_reference_buffer();
  const auto &allocator = mesh_device.allocator();

  tm.aligned_page_size = buf->aligned_page_size();
  tm.page_size = buf->page_size();
  tm.num_pages = mesh_buf.num_pages();
  tm.total_size_bytes = mesh_buf.num_pages() * buf->page_size();
  tm.rank = rank;
  tm.is_input = is_input ? 1 : 0;
  tm.is_output = is_output ? 1 : 0;
  std::memset(tm.pad, 0, sizeof(tm.pad));

  std::memset(tm.sizes_and_strides, 0, sizeof(tm.sizes_and_strides));
  std::memcpy(tm.sizes_and_strides, sizes_and_strides,
              rank * 2 * sizeof(int64_t));

  std::memset(tm.bank_base, 0, sizeof(tm.bank_base));
  for (uint32_t bank_id = 0; bank_id < num_banks; bank_id++) {
    int32_t bank_offset =
        allocator->get_bank_offset(tt::tt_metal::BufferType::DRAM, bank_id);
    tm.bank_base[bank_id] = static_cast<uint64_t>(
        static_cast<int64_t>(mesh_buf.address()) + bank_offset);
  }
}

// Send a task to the firmware and wait for completion.
uint32_t DispatchTask(TTDevice *dev, poc::Step6Task &task, uint32_t seq) {
  task.kick = 0; // will be overwritten by the separate kick write

  // Write the body starting after kick, up to (not including) done.
  const size_t body_offset = sizeof(uint32_t); // skip kick
  const size_t body_size = offsetof(poc::Step6Task, done) - body_offset;
  poc::NocWrite(
      dev, poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr + body_offset,
      reinterpret_cast<const uint8_t *>(&task) + body_offset, body_size);

  // Write kick = seq to trigger the firmware.
  poc::NocWrite32(dev, poc::kL2cpu0NocX, poc::kL2cpu0NocY, poc::kTaskAddr, seq);

  // Poll for done == seq.
  const uint64_t done_addr = poc::kTaskAddr + offsetof(poc::Step6Task, done);
  const auto deadline = std::chrono::steady_clock::now() + 30s;
  uint32_t done_val = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    done_val =
        poc::NocRead32(dev, poc::kL2cpu0NocX, poc::kL2cpu0NocY, done_addr);
    if (done_val == seq) {
      return done_val;
    }
    std::this_thread::sleep_for(1ms);
  }
  return done_val;
}

} // namespace

static int run(const std::string &fw_path, const std::string &blob_path) {
  ::setenv("TT_METAL_RUNTIME_ROOT", POC_TT_METAL_HOME, /*overwrite=*/1);
  ::setenv("TT_METAL_HOME", POC_TT_METAL_HOME, /*overwrite=*/1);

  auto firmware = poc::ReadFile(fw_path);
  auto code_blob = poc::ReadFile(blob_path);
  std::printf("[step6] firmware: %zu bytes from %s\n", firmware.size(),
              fw_path.c_str());
  std::printf("[step6] code blob: %zu bytes from %s\n", code_blob.size(),
              blob_path.c_str());

  // ---- Phase A: TTNN — allocate tensors ----------------------------------
  auto mesh_device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);

  const auto &allocator = mesh_device->allocator();
  uint32_t num_banks = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
  if (num_banks == 0 || num_banks > poc::kStep4MaxBanks) {
    std::fprintf(stderr, "error: num_banks=%u out of range\n", num_banks);
    return 5;
  }

  // Task 1 tensors: abs(4000x1)
  constexpr uint32_t kAbsElems = 4000;
  ttnn::Shape abs_shape({kAbsElems, 1});
  auto abs_input = ttnn::full(abs_shape, -5.0f, ttnn::DataType::FLOAT32,
                              ttnn::Layout::ROW_MAJOR, *mesh_device);
  auto abs_output = ttnn::full(abs_shape, 0.0f, ttnn::DataType::FLOAT32,
                               ttnn::Layout::ROW_MAJOR, *mesh_device);

  // Task 2 tensors: matmul(1x4000, 4000x1) → 1x1
  ttnn::Shape mat_a_shape({1, kAbsElems});
  ttnn::Shape mat_b_shape({kAbsElems, 1});
  ttnn::Shape mat_c_shape({1, 1});
  auto mat_a = ttnn::full(mat_a_shape, 3.0f, ttnn::DataType::FLOAT32,
                          ttnn::Layout::ROW_MAJOR, *mesh_device);
  auto mat_b = ttnn::full(mat_b_shape, 2.0f, ttnn::DataType::FLOAT32,
                          ttnn::Layout::ROW_MAJOR, *mesh_device);
  auto mat_c = ttnn::full(mat_c_shape, 0.0f, ttnn::DataType::FLOAT32,
                          ttnn::Layout::ROW_MAJOR, *mesh_device);

  std::printf("[step6] allocated tensors, num_banks=%u\n", num_banks);

  // ---- Phase B: boot X280 and load code blob -----------------------------
  auto dev = TTDevice::create(0);

  uint32_t pre = poc::NocRead32(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                                poc::kL2cpu0DramBase);
  if (pre == 0xffffffffu) {
    std::fprintf(stderr, "error: L2CPU0 GDDR likely harvested\n");
    return 4;
  }

  poc::BootL2cpu0(dev.get(), firmware);
  std::printf("[step6] firmware booted\n");

  // Load the kernel code blob to CODE_LOAD_ADDR.
  poc::NocWrite(dev.get(), poc::kL2cpu0NocX, poc::kL2cpu0NocY,
                poc::kCodeLoadAddr, code_blob.data(), code_blob.size());
  std::printf("[step6] code blob loaded at 0x%lx (%zu bytes)\n",
              static_cast<unsigned long>(poc::kCodeLoadAddr), code_blob.size());

  // ---- Phase C: Task 1 — abs(-5.0) → 5.0 --------------------------------
  {
    poc::Step6Task task{};
    FillBankCoords(*mesh_device, task, num_banks);
    task.func_id = kAbsFuncId;
    task.num_tensors = 2;

    // memref<4000x1xf32>: sizes = [4000, 1], strides = [1, 1]
    int64_t abs_ss[] = {4000, 1, 1, 1};
    FillTensorMeta(*mesh_device, abs_input, task.tensors[0],
                   /*is_input=*/true, /*is_output=*/false, num_banks, 2,
                   abs_ss);
    FillTensorMeta(*mesh_device, abs_output, task.tensors[1],
                   /*is_input=*/false, /*is_output=*/true, num_banks, 2,
                   abs_ss);

    uint32_t seq = 1;
    std::printf("[step6] dispatching task 1 (abs)...\n");
    uint32_t done = DispatchTask(dev.get(), task, seq);
    if (done != seq) {
      std::fprintf(stderr, "TIMEOUT on abs: done=0x%x expected=%u\n", done,
                   seq);
      return 2;
    }

    // Verify abs result.
    auto abs_result = abs_output.cpu().to_vector<float>();
    std::printf("[step6] abs result: size=%zu, first 4: ", abs_result.size());
    for (size_t i = 0; i < 4 && i < abs_result.size(); i++) {
      std::printf("%.1f ", abs_result[i]);
    }
    std::printf("\n");

    size_t mismatches = 0;
    for (float v : abs_result) {
      if (v != 5.0f) {
        mismatches++;
      }
    }
    if (mismatches != 0) {
      std::fprintf(stderr, "FAIL (abs): %zu / %zu elements != 5.0\n",
                   mismatches, abs_result.size());
      return 3;
    }
    std::printf("[step6] abs PASSED: %zu elements correct\n",
                abs_result.size());
  }

  // ---- Phase D: Task 2 — matmul(3.0, 2.0) → 24000.0 --------------------
  {
    poc::Step6Task task{};
    FillBankCoords(*mesh_device, task, num_banks);
    task.func_id = kMatmulFuncId;
    task.num_tensors = 3;

    // tensor 0: memref<1x4000xf32> — sizes = [1, 4000], strides = [4000, 1]
    int64_t mat_a_ss[] = {1, 4000, 4000, 1};
    FillTensorMeta(*mesh_device, mat_a, task.tensors[0],
                   /*is_input=*/true, /*is_output=*/false, num_banks, 2,
                   mat_a_ss);

    // tensor 1: memref<4000x1xf32> — sizes = [4000, 1], strides = [1, 1]
    int64_t mat_b_ss[] = {4000, 1, 1, 1};
    FillTensorMeta(*mesh_device, mat_b, task.tensors[1],
                   /*is_input=*/true, /*is_output=*/false, num_banks, 2,
                   mat_b_ss);

    // tensor 2: memref<1x1xf32> — sizes = [1, 1], strides = [1, 1]
    int64_t mat_c_ss[] = {1, 1, 1, 1};
    FillTensorMeta(*mesh_device, mat_c, task.tensors[2],
                   /*is_input=*/false, /*is_output=*/true, num_banks, 2,
                   mat_c_ss);

    uint32_t seq = 2;
    std::printf("[step6] dispatching task 2 (matmul)...\n");
    uint32_t done = DispatchTask(dev.get(), task, seq);
    if (done != seq) {
      std::fprintf(stderr, "TIMEOUT on matmul: done=0x%x expected=%u\n", done,
                   seq);
      return 2;
    }

    // Verify matmul result: dot(3.0 * [1x4000], 2.0 * [4000x1]) = 3*2*4000
    auto mat_result = mat_c.cpu().to_vector<float>();
    std::printf("[step6] matmul result: ");
    for (size_t i = 0; i < mat_result.size(); i++) {
      std::printf("%.1f ", mat_result[i]);
    }
    std::printf("\n");

    const float expected = 3.0f * 2.0f * static_cast<float>(kAbsElems);
    if (mat_result.size() != 1 || std::fabs(mat_result[0] - expected) > 1.0f) {
      std::fprintf(stderr, "FAIL (matmul): got %.1f, expected %.1f\n",
                   mat_result.empty() ? 0.0f : mat_result[0], expected);
      return 3;
    }
    std::printf("[step6] matmul PASSED: %.1f == %.1f\n", mat_result[0],
                expected);
  }

  // ---- Cleanup -----------------------------------------------------------
  dev.reset();
  bool closed = mesh_device->close();
  if (!closed) {
    std::fprintf(stderr, "warning: MeshDevice::close() returned false\n");
  }

  std::printf("SUCCESS: step6 dispatched abs + matmul on X280 via generic "
              "firmware loop\n");
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <fw_step6.bin> <cpu2_blob.bin>\n", argv[0]);
    return 1;
  }

  int rc = 4;
  try {
    rc = run(argv[1], argv[2]);
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
