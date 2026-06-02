// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/TopKRouterGptDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult TopKRouterGptDecompositionRewritePattern::matchAndRewrite(
    TopKRouterGptOp srcOp, PatternRewriter &rewriter) const {
  // tt-metal requires the topk output dim to be aligned to 8 elements.
  constexpr int64_t kHwAlignment = 8;
  int64_t k = static_cast<int64_t>(srcOp.getK());
  int64_t kPadded = ((k + kHwAlignment - 1) / kHwAlignment) * kHwAlignment;

  // No padding needed — k is already a multiple of 8.
  if (k == kPadded) {
    return failure();
  }

  auto indicesType =
      mlir::cast<RankedTensorType>(srcOp.getExpertIndices().getType());
  auto weightsType =
      mlir::cast<RankedTensorType>(srcOp.getExpertWeights().getType());
  int64_t B = indicesType.getDimSize(0);

  // Build [B, k_padded] output types for the hardware-facing op.
  auto paddedIndicesType = RankedTensorType::get(
      {B, kPadded}, indicesType.getElementType(), indicesType.getEncoding());
  auto paddedWeightsType = RankedTensorType::get(
      {B, kPadded}, weightsType.getElementType(), weightsType.getEncoding());

  // Create a new TopKRouterGptOp with [B, k_padded] outputs (what tt-metal
  // actually produces).  This op intentionally has k_padded output dims and
  // is immediately consumed by the slices below.
  auto newTopkOp = TopKRouterGptOp::create(
      rewriter, srcOp.getLoc(), TypeRange{paddedIndicesType, paddedWeightsType},
      srcOp.getInput(), srcOp.getWeight(), srcOp.getBias(),
      rewriter.getI32IntegerAttr(static_cast<int32_t>(kPadded)),
      srcOp.getNumExpertsAttr());

  // Slice from [B, k_padded] back to [B, k].
  llvm::SmallVector<int32_t, 2> begins = {0, 0};
  llvm::SmallVector<int32_t, 2> ends = {static_cast<int32_t>(B),
                                        static_cast<int32_t>(k)};
  llvm::SmallVector<int32_t, 2> step = {1, 1};

  auto sliceIndices = SliceStaticOp::create(
      rewriter, srcOp.getLoc(), indicesType, newTopkOp.getExpertIndices(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(step));

  auto sliceWeights = SliceStaticOp::create(
      rewriter, srcOp.getLoc(), weightsType, newTopkOp.getExpertWeights(),
      rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
      rewriter.getI32ArrayAttr(step));

  rewriter.replaceOp(
      srcOp, ValueRange{sliceIndices.getResult(), sliceWeights.getResult()});
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
