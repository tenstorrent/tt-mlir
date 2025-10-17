// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dSliceConfigRewritePattern.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

mlir::LogicalResult Conv2dSliceConfigRewritePattern::matchAndRewrite(
    ttnn::Conv2dOp srcOp, PatternRewriter &rewriter) const {
  if (srcOp.getConv2dSliceConfig()) {
    // Conv2dSliceConfig is already set, no need to apply the workaround.
    return failure();
  }


  return failure();
}
} // namespace mlir::tt::ttnn::workarounds::decomposition
