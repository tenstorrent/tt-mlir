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

  // Query layout: [1, batch, num_heads, head_dim].
  // Mask layout (per ScaledDotProductAttentionDecodeOp::verify):
  //   [batch_or_1, 1, num_heads_or_1, kv_seq_len]
  // so dim 0 is the batch dim and dim 2 is the heads dim.
  int64_t batch = queryType.getShape()[1];
  int64_t numHeads = queryType.getShape()[2];
  int64_t maskBatch = maskType.getShape()[0];
  int64_t maskHeads = maskType.getShape()[2];

  bool needBatchBroadcast = (maskBatch == 1 && batch > 1);
  bool needHeadBroadcast = (maskHeads == 1 && numHeads > 1);
  if (!needBatchBroadcast && !needHeadBroadcast) {
    return failure();
  }

  SmallVector<int64_t> targetShape(maskType.getShape());
  if (needBatchBroadcast) {
    targetShape[0] = batch;
  }
  if (needHeadBroadcast) {
    targetShape[2] = numHeads;
  }

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
