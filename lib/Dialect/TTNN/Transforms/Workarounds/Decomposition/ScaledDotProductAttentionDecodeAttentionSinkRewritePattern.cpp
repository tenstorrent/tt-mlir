// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeAttentionSinkRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ScaledDotProductAttentionDecodeAttentionSinkRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionDecodeOp srcOp,
    PatternRewriter &rewriter) const {

  Value sink = srcOp.getAttentionSink();
  if (!sink) {
    return failure();
  }

  auto sinkType = mlir::dyn_cast<RankedTensorType>(sink.getType());
  if (!sinkType) {
    return failure();
  }

  // Already 2D — nothing to do.
  if (sinkType.getRank() == 2) {
    return failure();
  }

  // Expect 4D sink with shape [*, num_heads, 1, 1].
  if (sinkType.getRank() != 4 || sinkType.getShape()[2] != 1 ||
      sinkType.getShape()[3] != 1) {
    return failure();
  }

  int64_t batch = sinkType.getShape()[0];
  int64_t numHeads = sinkType.getShape()[1];
  Location loc = srcOp.getLoc();

  // If batch > 1, slice to first batch entry. The sink is identical across
  // batches so any entry is valid.
  if (batch != 1) {
    auto slicedType =
        utils::RankedTensorTypeFactory::create(sinkType, {1, numHeads, 1, 1});
    sink =
        rewriter
            .create<SliceStaticOp>(
                loc, slicedType, sink, rewriter.getI32ArrayAttr({0, 0, 0, 0}),
                rewriter.getI32ArrayAttr(
                    {1, static_cast<int32_t>(numHeads), 1, 1}),
                rewriter.getI32ArrayAttr({1, 1, 1, 1}))
            .getResult();
    sinkType = mlir::cast<RankedTensorType>(sink.getType());
  }

  // Reshape from 4D [1, num_heads, 1, 1] to 2D [num_heads, 1].
  auto reshapedType =
      utils::RankedTensorTypeFactory::create(sinkType, {numHeads, 1});
  Value reshapedSink =
      rewriter
          .create<ReshapeOp>(
              loc, reshapedType, sink,
              rewriter.getI32ArrayAttr(
                  {static_cast<int32_t>(numHeads), static_cast<int32_t>(1)}),
              /*memory_config=*/MemoryConfigAttr())
          .getResult();

  // Pad last dim from 1 to TILE_WIDTH: [num_heads, 1] -> [num_heads, 32].
  auto paddedType = utils::RankedTensorTypeFactory::create(
      reshapedType, {numHeads, static_cast<int64_t>(TILE_WIDTH)});
  SmallVector<int32_t> padding = {0, 0, 0,
                                  static_cast<int32_t>(TILE_WIDTH - 1)};
  Value paddedSink =
      rewriter
          .create<PadOp>(loc, paddedType, reshapedSink,
                         rewriter.getDenseI32ArrayAttr(padding),
                         rewriter.getF32FloatAttr(0.0f),
                         /*use_multicore=*/rewriter.getBoolAttr(true),
                         /*memory_config=*/nullptr)
          .getResult();

  rewriter.replaceOpWithNewOp<ScaledDotProductAttentionDecodeOp>(
      srcOp, srcOp.getResult().getType(), srcOp.getQuery(), srcOp.getKey(),
      srcOp.getValue(), srcOp.getIsCausal(), srcOp.getAttentionMask(),
      srcOp.getCurPosTensor(), paddedSink, srcOp.getScaleAttr(),
      srcOp.getMemoryConfigAttr(), srcOp.getProgramConfigAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
