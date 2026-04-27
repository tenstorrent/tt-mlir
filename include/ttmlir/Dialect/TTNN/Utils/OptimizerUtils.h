// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <vector>

namespace mlir::tt::ttnn::optimizer_utils {

// Returns unique op-specific attributes from a list of OpConfigs.
// Deduplicates by comparing OpConfig::OpSpecificAttrs values.
std::vector<mlir::tt::ttnn::OpConfig::OpSpecificAttrs>
getUniqueOpSpecificAttrs(const std::vector<mlir::tt::ttnn::OpConfig> &configs);

// Returns unique test configs for Matmul/Linear ops.
// Generates Cartesian product of unique (bufferType, memLayout) pairs
// with unique op-specific attrs, using ignorePhysicalLayout.
llvm::SmallVector<mlir::tt::ttnn::OpConfig> getUniqueTestConfigsForMatmulLinear(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs);

// Returns unique test configs for validation.
// - For non-Matmul/Linear ops: Only unique op-specific attrs (no output layout
//   needed).
// - For Matmul/Linear ops: Cartesian product of unique (bufferType, memLayout)
//   pairs with unique op-specific attrs, using ignorePhysicalLayout
llvm::SmallVector<mlir::tt::ttnn::OpConfig> getUniqueTestConfigs(
    const std::vector<mlir::tt::ttnn::OpConfig> &consumerConfigs,
    bool isMatmulOrLinear);

} // namespace mlir::tt::ttnn::optimizer_utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_OPTIMIZERUTILS_H
