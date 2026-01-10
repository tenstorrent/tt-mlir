// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
  std::cout << "[C++] Starting PythonModelRunner test..." << std::endl;

  // Set TT_METAL_HOME if not already set (required before importing ttnn)
  if (std::getenv("TT_METAL_HOME") == nullptr) {
    setenv("TT_METAL_HOME", TT_METAL_HOME_PATH, 1);
  }

  tt::alchemist::PythonModelRunner runner;
  std::cout << "[C++] Created PythonModelRunner" << std::endl;

  // Add paths to Python sys.path for ttnn and test modules
  runner.addToSysPath(TEST_DIR_PATH);
  runner.addToSysPath(TTNN_PYTHON_PATH);
  runner.addToSysPath(TTMETAL_BUILD_LIB_PATH);

  runner.loadModule("test_model", "forward");
  std::cout << "[C++] Loaded test_model module" << std::endl;

  // Create a device
  auto meshDevice = ttnn::MeshDevice::create_unit_mesh(0);
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
      ttnn::to_device(input1, meshDevice.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  input2 =
      ttnn::to_device(input2, meshDevice.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  std::cout << "[C++] Created and moved input tensors to device" << std::endl;

  // Convert TTNN inputs/device to runtime types for the runner API.
  std::vector<tt::runtime::Tensor> inputs = {
      tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(input1),
      tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(input2),
  };
  tt::runtime::Device device =
      tt::runtime::ttnn::utils::createRuntimeDeviceFromTTNN(meshDevice.get());
  std::cout << "[C++] Calling forward..." << std::endl;

  auto outputs = runner.forward(inputs, device);
  std::cout << "[C++] Got " << outputs.size() << " output tensor(s)"
            << std::endl;

  if (!outputs.empty()) {
    std::cout << "[C++] Output shape: ";
    for (auto dim : tt::runtime::getTensorShape(outputs[0])) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "[C++] Test PASSED!" << std::endl;
  return 0;
}
