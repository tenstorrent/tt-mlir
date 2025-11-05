// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist_c_api.hpp"
#include <iostream>

int main() {
  std::cout << "Testing tt-alchemist from external project..." << std::endl;

  try {
    // Test: Get TTAlchemist singleton instance
    void *instance = tt_alchemist_TTAlchemist_getInstance();

    if (instance == nullptr) {
      std::cerr << "Failed to get TTAlchemist instance!" << std::endl;
      return 1;
    }

    std::cout << "Successfully obtained TTAlchemist instance at: " << instance
              << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
