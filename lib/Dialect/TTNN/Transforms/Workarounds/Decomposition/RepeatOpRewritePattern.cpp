// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include <algorithm>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
TTNNRepeatFoldingWorkaround::matchAndRewrite(ttnn::RepeatOp op,
                                             PatternRewriter &rewriter) const {
  Value device;
  if (op.getOperand().getDefiningOp()) {
    device = ttnn::utils::getOrInsertDevice(rewriter,
                                            op.getOperand().getDefiningOp());
  } else {
    device = ttnn::utils::getOrInsertDevice(rewriter, op);
  }
  float fillValue = 0;
  ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);

  // Create a zero Full Op to be used with AddOp
  ttnn::FullOp zeroOp = rewriter.create<ttnn::FullOp>(
      op->getLoc(), op.getResult().getType(), device, fillValueAttr);

  SmallVector<Value> addInputs;
  addInputs.push_back(op.getOperand());
  addInputs.push_back(zeroOp.getResult());

  // Replace the RepeatOp with an AddOp to perform implicit repeat.
  rewriter.replaceOpWithNewOp<ttnn::AddOp>(op, op.getResult().getType(),
                                           addInputs);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
