// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYL1SPILLMANAGEMENT_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYL1SPILLMANAGEMENT_H

#include "mlir/Pass/PassRegistry.h"

namespace mlir::tt::ttnn {

//===----------------------------------------------------------------------===//
// TTNNGreedyL1SpillManagement
//===----------------------------------------------------------------------===//
struct TTNNGreedyL1SpillManagementOptions {
  // Future: spill strategy selection, L1 cap override, etc.
  TTNNGreedyL1SpillManagementOptions() = default;
};

std::unique_ptr<::mlir::Pass> createTTNNGreedyL1SpillManagement();
std::unique_ptr<::mlir::Pass>
createTTNNGreedyL1SpillManagement(TTNNGreedyL1SpillManagementOptions options);

//===----------------------------------------------------------------------===//
// TTNNGreedyL1SpillManagement Registration
//===----------------------------------------------------------------------===//
inline void registerTTNNGreedyL1SpillManagement() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createTTNNGreedyL1SpillManagement();
  });
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_GREEDYL1SPILLMANAGEMENT_H
