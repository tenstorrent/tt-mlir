// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult NLPConcatHeadsDecodeInputRewritePattern::matchAndRewrite(
    ttnn::NLPConcatHeadsDecodeOp op, PatternRewriter &rewriter) const {
  auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
  TTNNLayoutAttr inputLayout = utils::getLayoutAttrFromTensor(inputType);

  // Skip if input is already height-sharded in L1.
  if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout().getValue() ==
                                           TensorMemoryLayout::HeightSharded) {
    return failure();
  }

  // Input shape is [S, B, num_heads, head_dim].
  int64_t batchSize = inputType.getShape()[1];

  rewriter.setInsertionPoint(op);
  auto shardedInput = utils::createHeightShardedToLayout(
      op.getInput(), inputType, batchSize, rewriter, op.getLoc());

  rewriter.modifyOpInPlace(
      op, [&]() { op.getInputMutable().assign(shardedInput); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
