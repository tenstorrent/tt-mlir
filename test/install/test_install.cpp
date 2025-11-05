// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/RegisterAll.h"

int main() {
  // Test that the header can be included and the library links correctly.
  // This verifies the installation is working correctly.
  mlir::tt::registerAllPasses();

  return 0;
}
