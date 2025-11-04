// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_alchemist.h"
#include "utils.hpp"
#include <iostream>

int main() {
  try {
    auto templates_dir = tt::alchemist::utils::get_templates_dir();
    std::cout << "Found templates directory: " << templates_dir << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
