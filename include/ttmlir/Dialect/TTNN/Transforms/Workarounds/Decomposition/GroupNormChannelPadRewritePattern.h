// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMCHANNELPADREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMCHANNELPADREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Pad the channel dimension (last dim) of GroupNormOp inputs to the nearest
// multiple of TILE_WIDTH (32) when it is not already aligned. The num_groups
// attribute is scaled proportionally so that channels_per_group remains
// unchanged, preserving mathematical correctness. Weight and bias operands
// are similarly padded. A SliceStaticOp is inserted after the padded
// GroupNormOp to restore the original channel size.
class GroupNormChannelPadRewritePattern
    : public OpRewritePattern<ttnn::GroupNormOp> {
public:
  using OpRewritePattern<ttnn::GroupNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::GroupNormOp srcOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_GROUPNORMCHANNELPADREWRITEPATTERN_H
