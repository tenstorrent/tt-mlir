// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Broadcast.h"

#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttnn {

// Convert Broadcast Ops to an addOp with zero in order to apply implicit
// broadcast for all operands.
// TODO(uazizTT): Remove this workaround once implicit broadcast for all
// operands is supported.
LogicalResult
TTNNBroadcastWorkaround::matchAndRewrite(ttnn::BroadcastOp op,
                                         PatternRewriter &rewriter) const {

  Value device = ttnn::utils::getOrInsertDevice(
      rewriter, op.getOperand(1).getDefiningOp());
  float fillValue = 0;
  ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);

  ttnn::FullOp zeroOp = rewriter.create<ttnn::FullOp>(
      op->getLoc(), op.getResult().getType(), device, fillValueAttr);

  rewriter.replaceOpWithNewOp<ttnn::AddOp>(op, op.getInput(),
                                           zeroOp.getResult(), op.getOutput());

  return success();
}

} // namespace mlir::tt::ttnn
