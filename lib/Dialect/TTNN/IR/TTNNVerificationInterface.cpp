// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/IR/TTNNVerificationInterface.h"

namespace mlir::tt::ttnn {
#include "ttmlir/Dialect/TTNN/IR/TTNNVerificationInterface.cpp.inc"

// Verifier function for TTNN verification interface.
mlir::LogicalResult verifyTTNNVerificationInterface(mlir::Operation *op) {
  return mlir::LogicalResult::success();
}
} // namespace mlir::tt::ttnn
