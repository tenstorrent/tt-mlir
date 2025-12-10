// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "python_runner.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#pragma clang diagnostic pop

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

  // Create a TTNN device and wrap it in a runtime Device
  auto ttnnDevice = ttnn::MeshDevice::create_unit_mesh(0);
  tt::runtime::Device device =
      tt::runtime::ttnn::utils::createRuntimeDeviceFromTTNN(ttnnDevice.get());
  std::cout << "[C++] Created device" << std::endl;

  // Create input tensors using TTNN API
  ttnn::Tensor ttnnInput1 =
      ttnn::ones(ttnn::Shape({32, 32}), ttnn::DataType::BFLOAT16,
                 ttnn::Layout::TILE, std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                    ttnn::BufferType::DRAM});
  ttnn::Tensor ttnnInput2 =
      ttnn::ones(ttnn::Shape({32, 32}), ttnn::DataType::BFLOAT16,
                 ttnn::Layout::TILE, std::nullopt,
                 ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                    ttnn::BufferType::DRAM});

  // Move to device
  ttnnInput1 =
      ttnn::to_device(ttnnInput1, ttnnDevice.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  ttnnInput2 =
      ttnn::to_device(ttnnInput2, ttnnDevice.get(),
                      ttnn::MemoryConfig{ttnn::TensorMemoryLayout::INTERLEAVED,
                                         ttnn::BufferType::DRAM});
  std::cout << "[C++] Created and moved input tensors to device" << std::endl;

  // Convert TTNN tensors to runtime tensors
  tt::runtime::Tensor input1 =
      tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(ttnnInput1);
  tt::runtime::Tensor input2 =
      tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(ttnnInput2);

  // Run forward with runtime types
  std::vector<tt::runtime::Tensor> inputs = {input1, input2};
  std::cout << "[C++] Calling forward..." << std::endl;

  auto outputs = runner.forward(inputs, device);
  std::cout << "[C++] Got " << outputs.size() << " output tensor(s)"
            << std::endl;

  if (!outputs.empty()) {
    // Get the underlying TTNN tensor to access shape info
    ttnn::Tensor &outputTensor =
        tt::runtime::ttnn::utils::getTTNNTensorFromRuntimeTensor(outputs[0]);
    std::cout << "[C++] Output logical shape: " << outputTensor.logical_shape()
              << std::endl;
  }

  std::cout << "[C++] Test PASSED!" << std::endl;
  return 0;
}
