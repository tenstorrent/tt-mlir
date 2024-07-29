// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include <mlir/Support/LogicalResult.h>

mlir::LogicalResult
mlir::tt::ttir::detail::verifyElementwiseOp(mlir::Operation *op) {
  // Currently, elementwise trait already performs the basic verification.
  // Let this be a placeholder for future extensions.

  return success();
}
