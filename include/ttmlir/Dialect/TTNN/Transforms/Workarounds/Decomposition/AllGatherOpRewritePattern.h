// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLGATHEROPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLGATHEROPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// AllGatherOp in TTNN currently does not support tensors with rank < 4
// correctly. As a temporary workaround, we insert reshape ops front and back
// to make the tensor as four dimensional tensor.
// Related tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/25143
class TTNNAllGatherWorkarounds : public OpRewritePattern<ttnn::AllGatherOp> {
public:
  using OpRewritePattern<ttnn::AllGatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllGatherOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_ALLGATHEROPREWRITEPATTERN_H
