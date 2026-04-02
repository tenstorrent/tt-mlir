// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult
SDPADecompositionPattern::matchAndRewrite(ScaledDotProductAttentionOp op,
                                          PatternRewriter &rewriter) const {
  (void)forceDecompose;
  (void)validationConfig;
  return failure();
}

} // namespace mlir::tt::ttnn::decomposition
