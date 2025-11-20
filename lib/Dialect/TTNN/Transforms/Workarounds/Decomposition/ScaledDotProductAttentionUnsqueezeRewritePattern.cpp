// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionUnsqueezeRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

// Workaround which converts 3D ScaledDotProductAttention inputs to 4D by
// prepending dimensions of size 1, then squeezes the output back to 3D.
// TTNN ScaledDotProductAttention requires 4D tensors.
// Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/32503
namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

Value unsqueezeTo4D(Value tensor, PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  int64_t rank = tensorType.getRank();
  if (rank >= 4) {
    return tensor;
  }

  SmallVector<int64_t> newShape(4 - rank, 1);
  ArrayRef<int64_t> origShape = tensorType.getShape();
  newShape.append(origShape.begin(), origShape.end());

  auto newType = utils::RankedTensorTypeFactory::create(tensorType, newShape);

  return rewriter
      .create<ReshapeOp>(loc, newType, tensor,
                         rewriter.getI32ArrayAttr(SmallVector<int32_t>(
                             newShape.begin(), newShape.end())),
                         /*memory_config=*/nullptr)
      .getResult();
}

Value squeezeTo3D(Value tensor, PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  int64_t rank = tensorType.getRank();
  if (rank != 4) {
    return tensor;
  }

  ArrayRef<int64_t> origShape = tensorType.getShape();
  SmallVector<int64_t> newShape(origShape.begin() + 1, origShape.end());

  auto newType = utils::RankedTensorTypeFactory::create(tensorType, newShape);

  return rewriter
      .create<ReshapeOp>(loc, newType, tensor,
                         rewriter.getI32ArrayAttr(SmallVector<int32_t>(
                             newShape.begin(), newShape.end())),
                         /*memory_config=*/nullptr)
      .getResult();
}

} // namespace

LogicalResult ScaledDotProductAttentionUnsqueezeRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionOp srcOp, PatternRewriter &rewriter) const {
  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  auto keyType = mlir::dyn_cast<RankedTensorType>(srcOp.getKey().getType());
  auto valueType = mlir::dyn_cast<RankedTensorType>(srcOp.getValue().getType());
  if (!queryType || !keyType || !valueType) {
    return failure();
  }

  if (queryType.getRank() == 4 && keyType.getRank() == 4 &&
      valueType.getRank() == 4) {
    return failure();
  }

  Value query = unsqueezeTo4D(srcOp.getQuery(), rewriter, srcOp.getLoc());
  Value key = unsqueezeTo4D(srcOp.getKey(), rewriter, srcOp.getLoc());
  Value value = unsqueezeTo4D(srcOp.getValue(), rewriter, srcOp.getLoc());

  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      srcOp.getLoc(), query.getType(), query, key, value,
      srcOp.getAttentionMask(), srcOp.getIsCausal(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr());

  Value result = sdpaOp.getResult();

  // Squeeze output back to 3D if query was originally 3D
  if (queryType.getRank() == 3) {
    result = squeezeTo3D(result, rewriter, srcOp.getLoc());
  }

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
