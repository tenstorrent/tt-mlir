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
// rank > 4. When padding is requested on higher dimensions, this pattern
// decomposes the pad into: permute (move high dims to low) -> pad -> permute
// (restore original order).tt
// Issue: https://github.com/tenstorrent/tt-metal/issues/38144
class PadHighDimRewritePattern : public mlir::OpRewritePattern<PadOp> {
public:
  using mlir::OpRewritePattern<PadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(PadOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_PADHIGHDIMREWRITEPATTERN_H
