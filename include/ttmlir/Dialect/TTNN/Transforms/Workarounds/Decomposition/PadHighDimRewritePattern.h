// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// ttnn::pad only supports padding on the lowest 3 dimensions for tensors with
// rank > 4. This pattern decomposes a 5D PadOp [N,D,H,W,C] with padding on
// D/H/W into:
//   permute [4,0,1,2,3] → reshape [C*N,D,H,W] → pad 4D → reshape
//   [C,N,D',H',W'] → permute [1,2,3,4,0]
// This pattern is used to handle padding for 3d convolutions.
// Issue: https://github.com/tenstorrent/tt-metal/issues/38144
class PadHighDimRewritePattern : public mlir::OpRewritePattern<PadOp> {
public:
  using mlir::OpRewritePattern<PadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(PadOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H
