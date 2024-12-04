// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_TRANSFORMUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::tt::ttnn::utils {
// Get or insert device for the given operation.
mlir::Value getOrInsertDevice(mlir::PatternRewriter &rewriter,
                              mlir::Operation *op);
} // namespace mlir::tt::ttnn::utils

#endif
