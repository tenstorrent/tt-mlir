// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#pragma clang diagnostic pop

// These paths are set by CMake
#ifndef TT_METAL_HOME_PATH
#define TT_METAL_HOME_PATH ""
#endif
#ifndef TTNN_PYTHON_PATH
#define TTNN_PYTHON_PATH ""
#endif
#ifndef TTMETAL_BUILD_LIB_PATH
#define TTMETAL_BUILD_LIB_PATH ""
#endif
#ifndef TEST_DIR_PATH
#define TEST_DIR_PATH ""
#endif

int main() {
  std::cout << "[C++] Starting PythonModelRunner test..." << std::endl;

  // Set TT_METAL_HOME if not already set (required before importing ttnn)
  if (std::getenv("TT_METAL_HOME") == nullptr) {
    setenv("TT_METAL_HOME", TT_METAL_HOME_PATH, 1);
  }

  tt::alchemist::PythonModelRunner runner;
  std::cout << "[C++] Created PythonModelRunner" << std::endl;

  // Add paths to Python sys.path for ttnn and test modules
  runner.addToPath(TEST_DIR_PATH);
  runner.addToPath(TTNN_PYTHON_PATH);
  runner.addToPath(TTMETAL_BUILD_LIB_PATH);

  runner.loadModule("test_model", "forward");
  std::cout << "[C++] Loaded test_model module" << std::endl;

  // Create a device
  auto device = ttnn::MeshDevice::create_unit_mesh(0);
  std::cout << "[C++] Created device" << std::endl;

  // Create input tensors
  ttnn::Tensor input1 =
      ttnn::ones(ttnn::Shape({32, 32}), ttnn::DataType::BFLOAT16,
                 ttnn::Layout::TILE, std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                    ttnn::BufferType::DRAM});
  ttnn::Tensor input2 =
      ttnn::ones(ttnn::Shape({32, 32}), ttnn::DataType::BFLOAT16,
                 ttnn::Layout::TILE, std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                    ttnn::BufferType::DRAM});

  // Move to device
  input1 =
      ttnn::to_device(input1, device.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  input2 =
      ttnn::to_device(input2, device.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  std::cout << "[C++] Created and moved input tensors to device" << std::endl;

  // Run forward
  std::vector<ttnn::Tensor> inputs = {input1, input2};
  std::cout << "[C++] Calling forward..." << std::endl;

  auto outputs = runner.forward(inputs, device.get());
  std::cout << "[C++] Got " << outputs.size() << " output tensor(s)"
            << std::endl;

  if (!outputs.empty()) {
    std::cout << "[C++] Output logical shape: " << outputs[0].logical_shape()
              << std::endl;
  }

  std::cout << "[C++] Test PASSED!" << std::endl;
  return 0;
}
