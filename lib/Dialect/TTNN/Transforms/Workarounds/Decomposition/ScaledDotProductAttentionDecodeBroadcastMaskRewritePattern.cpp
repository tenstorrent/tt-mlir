// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionDecodeOp srcOp,
    PatternRewriter &rewriter) const {

  Value mask = srcOp.getAttentionMask();
  if (!mask) {
    return failure();
  }

  auto maskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType || maskType.getRank() != 4) {
    return failure();
  }

  // Decode query shape: [1, batch, num_heads, head_dim]
  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  if (!queryType || queryType.getRank() != 4) {
    return failure();
  }

  // Mask layout (per ScaledDotProductAttentionDecodeOp::verify):
  //   [batch_or_1, 1, num_heads_or_1, kv_seq_len].
  // tt-metal handles batch broadcasting natively, so only the heads dimension
  // still requires this workaround.
  // See https://github.com/tenstorrent/tt-metal/issues/39910.
  int64_t numHeads = queryType.getShape()[2];
  int64_t maskHeads = maskType.getShape()[2];

  bool needHeadBroadcast = (maskHeads == 1 && numHeads > 1);

  if (!needHeadBroadcast) {
    return failure();
  }

  SmallVector<int64_t> targetShape(maskType.getShape());
  targetShape[2] = numHeads;

  auto broadcastType =
      utils::RankedTensorTypeFactory::create(maskType, targetShape);
  auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
      maskType.getShape(), targetShape);
  auto shapeAttr = ShapeAttr::get(rewriter.getContext(), broadcastDims);

  Value broadcastedMask =
      rewriter.create<RepeatOp>(srcOp.getLoc(), broadcastType, mask, shapeAttr);

  rewriter.replaceOpWithNewOp<ScaledDotProductAttentionDecodeOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getQuery(), srcOp.getKey(),
      srcOp.getValue(), srcOp.getIsCausal(), broadcastedMask,
      srcOp.getCurPosTensor(), srcOp.getAttentionSink(), srcOp.getScaleAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getProgramConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
