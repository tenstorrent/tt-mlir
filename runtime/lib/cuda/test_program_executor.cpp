// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test program for CUDA ProgramExecutor
#include "tt/runtime/detail/cuda/program_executor.h"
#include "tt/runtime/types.h"
#include <iostream>
#include <vector>

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

  // Create empty input tensors (replace with actual tensors as needed)
  std::vector<::tt::runtime::Tensor> programInputs;

  std::cout << "Creating CUDA ProgramExecutor..." << std::endl;
  ::tt::runtime::cuda::ProgramExecutor executor(binary, programInputs);

  std::cout << "Executing program..." << std::endl;
  executor.execute();

  std::cout << "✓ Program execution completed!" << std::endl;
  return 0;
}
