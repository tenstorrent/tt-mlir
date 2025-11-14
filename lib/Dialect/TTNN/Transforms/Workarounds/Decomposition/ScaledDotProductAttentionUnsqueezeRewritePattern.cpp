// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionUnsqueezeRewritePattern.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

// Unsqueeze a tensor to 4D by prepending dimensions of size 1.
Value unsqueezeTo4D(Value tensor, mlir::PatternRewriter &rewriter,
                    Location loc) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  int64_t rank = tensorType.getRank();
  if (rank >= 4) {
    return tensor;
  }

  llvm::SmallVector<int64_t> newShape(4 - rank, 1);
  llvm::ArrayRef<int64_t> origShape = tensorType.getShape();
  newShape.append(origShape.begin(), origShape.end());

  auto newType = RankedTensorType::get(newShape, tensorType.getElementType(),
                                       tensorType.getEncoding());

  return rewriter
      .create<ReshapeOp>(loc, newType, tensor,
                         rewriter.getI32ArrayAttr(SmallVector<int32_t>(
                             newShape.begin(), newShape.end())),
                         /*memory_config=*/nullptr)
      .getResult();
}

// Squeeze a 4D tensor to 3D by removing the leading dimension.
Value squeezeTo3D(Value tensor, mlir::PatternRewriter &rewriter, Location loc) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(tensor.getType());
  if (!tensorType) {
    return tensor;
  }

  int64_t rank = tensorType.getRank();
  if (rank != 4) {
    return tensor;
  }

  llvm::ArrayRef<int64_t> origShape = tensorType.getShape();
  // Remove leading dimension (assumed to be 1)
  llvm::SmallVector<int64_t> newShape(origShape.begin() + 1, origShape.end());

  auto newType = RankedTensorType::get(newShape, tensorType.getElementType(),
                                       tensorType.getEncoding());

  return rewriter
      .create<ReshapeOp>(loc, newType, tensor,
                         rewriter.getI32ArrayAttr(SmallVector<int32_t>(
                             newShape.begin(), newShape.end())),
                         /*memory_config=*/nullptr)
      .getResult();
}

} // namespace

LogicalResult ScaledDotProductAttentionUnsqueezeRewritePattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionOp srcOp, PatternRewriter &rewriter) const {

  auto queryType = mlir::dyn_cast<RankedTensorType>(srcOp.getQuery().getType());
  auto keyType = mlir::dyn_cast<RankedTensorType>(srcOp.getKey().getType());
  auto valueType = mlir::dyn_cast<RankedTensorType>(srcOp.getValue().getType());

  if (!queryType || !keyType || !valueType) {
    return failure();
  }

  // Only apply workaround if inputs are 3D
  if (queryType.getRank() != 3 || keyType.getRank() != 3 ||
      valueType.getRank() != 3) {
    return failure();
  }

  // Unsqueeze Q, K, V to 4D
  Value query = unsqueezeTo4D(srcOp.getQuery(), rewriter, srcOp.getLoc());
  Value key = unsqueezeTo4D(srcOp.getKey(), rewriter, srcOp.getLoc());
  Value value = unsqueezeTo4D(srcOp.getValue(), rewriter, srcOp.getLoc());

  // Handle attention mask if present
  Value mask = srcOp.getAttentionMask();
  if (mask) {
    auto maskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
    if (maskType && maskType.getRank() == 3) {
      mask = unsqueezeTo4D(mask, rewriter, srcOp.getLoc());
    }
  }

  // Create new SDPA op with 4D inputs
  auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
      srcOp.getLoc(), query.getType(), query, key, value, mask,
      srcOp.getIsCausal(), srcOp.getScaleAttr(),
      srcOp.getSlidingWindowSizeAttr(), srcOp.getMemoryConfigAttr());

  Value result = sdpaOp.getResult();

  // Squeeze output back to 3D
  result = squeezeTo3D(result, rewriter, srcOp.getLoc());

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
