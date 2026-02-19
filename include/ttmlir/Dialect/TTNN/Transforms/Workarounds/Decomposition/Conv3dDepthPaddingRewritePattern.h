// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DDEPTHPADDINGREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DDEPTHPADDINGREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// TT-Metal's conv3d requires depth padding to be zero (padding[0] == 0).
// When non-zero depth padding is requested, this pattern explicitly pads the
// input tensor along the depth dimension using a PadOp, then runs conv3d with
// depth padding set to zero.
// Issue: https://github.com/tenstorrent/tt-metal/issues/38143
class Conv3dDepthPaddingRewritePattern
    : public mlir::OpRewritePattern<Conv3dOp> {
public:
  using mlir::OpRewritePattern<Conv3dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(Conv3dOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DDEPTHPADDINGREWRITEPATTERN_H
