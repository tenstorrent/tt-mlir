// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test program for CUDA ProgramExecutor
#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/types.h"
#include <iostream>
#include <memory>
#include <vector>

::tt::runtime::Tensor createCudaTensor(const void *data, size_t sizeBytes) {
  // Allocate memory for tensor data and copy the input data
  auto tensorData = std::shared_ptr<void>(malloc(sizeBytes), free);
  std::memcpy(tensorData.get(), data, sizeBytes);

  return ::tt::runtime::Tensor(nullptr, tensorData,
                               ::tt::runtime::DeviceRuntime::Cuda);
}

int main(int argc, char *argv[]) {
  std::cout << "=== CUDA ProgramExecutor Test ===" << std::endl;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <flatbuffer_file>" << std::endl;
    std::cout << "Example: " << argv[0]
              << " test/ttmlir/Conversion/TTIRToNVVM/ttir_to_nvvm.fb"
              << std::endl;
    return 1;
  }

  std::cout << "Loading flatbuffer: " << argv[1] << std::endl;

  // Load the binary from file
  ::tt::runtime::Binary binary = ::tt::runtime::Binary::loadFromPath(argv[1]);
  std::cout << "✓ Binary loaded successfully!" << std::endl;
  std::cout << "  Binary ID: " << binary.id() << std::endl;

  std::vector<::tt::runtime::Tensor> programInputs;

  std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  ::tt::runtime::Tensor tensor1 =
      createCudaTensor(data1.data(), data1.size() * sizeof(float));
  programInputs.push_back(tensor1);
  std::cout << "✓ Created Float32 tensor arg0" << std::endl;

  std::vector<float> data2 = {2.0f, 3.0f, -4.0f, 5.0f,   6.0f,
                              7.0f, 8.0f, -9.0f, -10.0f, 11.0f};
  ::tt::runtime::Tensor tensor2 =
      createCudaTensor(data2.data(), data2.size() * sizeof(float));
  programInputs.push_back(tensor2);
  std::cout << "✓ Created Float32 tensor arg1" << std::endl;

  std::cout << "Creating CUDA ProgramExecutor..." << std::endl;
  ::tt::runtime::cuda::ProgramExecutor executor(binary, programInputs);

  std::cout << "Executing program..." << std::endl;
  executor.execute();

  std::cout << "✓ Program execution completed!" << std::endl;
  return 0;
}
