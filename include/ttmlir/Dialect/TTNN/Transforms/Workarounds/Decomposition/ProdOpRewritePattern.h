// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PRODOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PRODOPREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Prod op creates incorrect results on non tile-aligned shapes.
// Add padding to the input tensor to make it tile-aligned.
// Issue: https://github.com/tenstorrent/tt-metal/issues/40168

class ProdOpRewritePattern : public mlir::OpRewritePattern<ProdOp> {
public:
  using mlir::OpRewritePattern<ProdOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ProdOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PRODOPREWRITEPATTERN_H
