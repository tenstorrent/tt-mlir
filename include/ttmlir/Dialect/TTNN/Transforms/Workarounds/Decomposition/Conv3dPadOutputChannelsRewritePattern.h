// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DPADOUTPUTCHANNELSREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DPADOUTPUTCHANNELSREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Conv3d requires output channels to be a multiple of TILE_WIDTH (32).
// When output channels aren't aligned, this pattern pads the weight (and bias)
// tensors, runs conv3d with padded output channels, then slices the result back
// to the original size.
class Conv3dPadOutputChannelsRewritePattern
    : public mlir::OpRewritePattern<Conv3dOp> {
public:
  using mlir::OpRewritePattern<Conv3dOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(Conv3dOp srcOp,
                                      PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_CONV3DPADOUTPUTCHANNELSREWRITEPATTERN_H
