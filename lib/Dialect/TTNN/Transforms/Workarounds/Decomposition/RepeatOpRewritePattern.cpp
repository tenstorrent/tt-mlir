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
  Value device =
      ttnn::utils::getOrInsertDevice(rewriter, op.getOperand().getDefiningOp());
  float fillValue = 0;
  ::mlir::FloatAttr fillValueAttr = rewriter.getF32FloatAttr(fillValue);

  // Create a zero Full Op to be used with AddOp
  ttnn::FullOp zeroOp = rewriter.create<ttnn::FullOp>(
      op->getLoc(), op.getResult().getType(), device, fillValueAttr);

  SmallVector<Value> addInputs;
  addInputs.push_back(op.getOperand());
  addInputs.push_back(zeroOp.getResult());

  // Create an EmptyOp as AddOp is a DPS Op.
  // Get ttnn::TTNNLayoutAttr of the result type
  //
  ttnn::TTNNLayoutAttr layoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(op.getResult().getType().getEncoding());

  // Get the shape of the tensor, tensor layout, and data type
  //
  ttnn::ShapeAttr shapeAttr = ttnn::ShapeAttr::get(
      rewriter.getContext(),
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape());
  DataType dtype = layoutAttr.getDataType();
  ttnn::Layout ttnnLayoutEnum = ttnn::Layout::RowMajor;
  if (layoutAttr.isTiled()) {
    ttnnLayoutEnum = ttnn::Layout::Tile;
  } else {
    ttnnLayoutEnum = ttnn::Layout::RowMajor;
  }
  DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(), dtype);
  ttnn::LayoutAttr tensorLayoutAttr =
      ttnn::LayoutAttr::get(op.getContext(), ttnnLayoutEnum);

  // Create MemoryConfigAttr
  //
  ttnn::BufferTypeAttr bufferTypeAttr =
      ttnn::BufferTypeAttr::get(op.getContext(), layoutAttr.getBufferType());
  ttnn::ShardSpecAttr shardSpecAttr = ttnn::ShardSpecAttr::get(
      op.getContext(),
      ttnn::ShapeAttr::get(op.getContext(), layoutAttr.getShardShape()));
  ttnn::MemoryConfigAttr memoryConfigAttr =
      ttnn::MemoryConfigAttr::get(op.getContext(), bufferTypeAttr,
                                  shardSpecAttr, layoutAttr.getMemLayout());

  // Create EmptyOp
  //
  ttnn::EmptyOp emptyOp = rewriter.create<ttnn::EmptyOp>(
      op->getLoc(), op.getType(), shapeAttr, dTypeAttr, tensorLayoutAttr,
      device, memoryConfigAttr);

  // Replace the RepeatOp with an AddOp to perform implicit repeat.
  rewriter.replaceOpWithNewOp<ttnn::AddOp>(op, op.getResult().getType(),
                                           addInputs, emptyOp.getResult());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
