// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Simple test for PythonModelRunner that doesn't require device/tensor

#include "python_runner.hpp"

#include <iostream>

// Path set by CMake
#ifndef TEST_DIR_PATH
#define TEST_DIR_PATH "."
#endif

int main() {
  std::cout << "[C++] Starting simple PythonModelRunner test..." << std::endl;

  tt::alchemist::PythonModelRunner runner;
  std::cout << "[C++] Created PythonModelRunner" << std::endl;

  runner.addToPath(TEST_DIR_PATH);
  std::cout << "[C++] Added " << TEST_DIR_PATH << " to Python path"
            << std::endl;

  runner.loadModule("simple_test_model", "forward");
  std::cout << "[C++] Loaded simple_test_model module" << std::endl;

  std::cout << "[C++] Test PASSED!" << std::endl;
  return 0;
}
