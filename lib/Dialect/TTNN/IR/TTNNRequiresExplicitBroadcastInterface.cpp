// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/IR/TTNNRequiresExplicitBroadcastInterface.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttnn {
#include "ttmlir/Dialect/TTNN/IR/TTNNRequiresExplicitBroadcastInterface.h.inc"

// Verifier function for TTNN RequiresExplicitBroadcast Interface.
LogicalResult verifyTTNNRequiresExplicitBroadcastInterface(Operation *op) {
  if (op->getOperands().size() <= 1) {
    return op->emitOpError()
           << "RequiresExplicitBroadcastInterface can only be attached to the "
              "operations that have more than one operand";
  }

  return success();
}

} // namespace mlir::tt::ttnn
