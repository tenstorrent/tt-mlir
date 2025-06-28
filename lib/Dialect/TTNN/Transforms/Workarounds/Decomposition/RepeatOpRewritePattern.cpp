// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

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

  auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
  auto layoutAttr = mlir::cast<ttnn::TTNNLayoutAttr>(resultType.getEncoding());
  auto shapeAttr =
      ttnn::ShapeAttr::get(rewriter.getContext(), resultType.getShape());
  auto dTypeAttr = ttcore::DataTypeAttr::get(rewriter.getContext(),
                                             layoutAttr.getDataType());
  auto layout = ttnn::LayoutAttr::get(op.getContext(), layoutAttr.getLayout());

  // Create a ZerosOp to be used with AddOp
  ttnn::ZerosOp zerosOp = rewriter.create<ttnn::ZerosOp>(
      ttmlir::utils::appendLocationSuffix(op->getLoc(), "_zeros"), resultType,
      shapeAttr, dTypeAttr, layout, device, ttnn::MemoryConfigAttr());

  SmallVector<Value> addInputs;
  addInputs.push_back(op.getOperand());
  addInputs.push_back(zerosOp.getResult());

  // Replace the RepeatOp with an AddOp to perform implicit repeat.
  rewriter.replaceOpWithNewOp<ttnn::AddOp>(op, op.getResult().getType(),
                                           addInputs);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
