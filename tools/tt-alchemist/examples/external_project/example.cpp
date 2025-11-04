// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.h"
#include "utils.hpp"
#include <filesystem>
#include <iostream>

int main() {
  std::cout << "Testing tt-alchemist from external project..." << std::endl;

  try {
    // Test 1: Get templates directory
    auto templates_dir = tt::alchemist::utils::get_templates_dir();
    std::cout << "Templates directory: " << templates_dir << std::endl;

    // Test 2: Verify templates exist
    if (!std::filesystem::exists(templates_dir)) {
      std::cerr << "Templates directory doesn't exist!" << std::endl;
      return 1;
    }

    // Test 3: List some templates
    std::cout << "\nAvailable templates:" << std::endl;
    for (const auto &entry :
         std::filesystem::directory_iterator(templates_dir)) {
      if (entry.is_directory()) {
        std::cout << "  - " << entry.path().filename() << "/" << std::endl;
      }
    }

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
