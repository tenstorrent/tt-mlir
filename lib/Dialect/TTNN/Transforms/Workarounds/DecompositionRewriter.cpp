// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/DecompositionRewriter.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult DecompositionRewriter::matchAndRewrite(
    DecompositionWorkaroundInterface op,
    PatternRewriter &rewriter) const {
  // Get all workarounds for this op
  auto workarounds = op.getDecompositionWorkarounds();

  // Try each workaround in order
  for (auto &workaround : workarounds) {
    if (workaround->matchAndRewrite(op.getOperation(), rewriter).succeeded()) {
      return success();
    }
  }

  return failure();
}

} // namespace mlir::tt::ttnn::decomposition