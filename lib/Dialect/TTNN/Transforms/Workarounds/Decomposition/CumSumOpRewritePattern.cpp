// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
CumSumOpRewritePattern::matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                        PatternRewriter &rewriter) const {
  mlir::RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  llvm::SmallVector<int64_t> inputTypeShape(inputType.getShape());
  if (inputTypeShape.size() >= 4) {
    return failure();
  }

  RankedTensorType outputType = srcOp.getResult().getType();
  int64_t inputRank = inputType.getRank();
  std::vector<int64_t> reshapedInputShape(4, 1);
  for (int idx = 0; idx < inputRank; ++idx) {
    reshapedInputShape[idx] = inputTypeShape[idx];
  }

  mlir::ArrayAttr shapeAttrI = rewriter.getI32ArrayAttr(
      std::vector<int32_t>(inputTypeShape.begin(), inputTypeShape.end()));
  ArrayAttr shapeAttr = rewriter.getI32ArrayAttr(std::vector<int32_t>(
      reshapedInputShape.begin(), reshapedInputShape.end()));
  mlir::RankedTensorType newType =
      mlir::RankedTensorType::Builder(inputType).setShape(reshapedInputShape);
  ReshapeOp inputReshapeOp = rewriter.create<mlir::tt::ttnn::ReshapeOp>(
      srcOp->getLoc(), newType, srcOp.getInput(), shapeAttr);

  TTNNLayoutAttr newOutputLayoutAttr =
      mlir::cast<TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(rewriter.getContext(), newType.getShape());

  RankedTensorType newOutputType = RankedTensorType::get(
      newType.getShape(), outputType.getElementType(), newOutputLayoutAttr);

  tensor::EmptyOp emptyOp = rewriter.create<tensor::EmptyOp>(
      srcOp.getLoc(), newOutputType.getShape(), newOutputType.getElementType(),
      newOutputType.getEncoding());
  MorehCumSumOp cumsumOp = rewriter.create<mlir::tt::ttnn::MorehCumSumOp>(
      srcOp->getLoc(), newOutputType, inputReshapeOp->getResult(0),
      srcOp.getDim(), emptyOp, nullptr);
  rewriter.replaceOpWithNewOp<mlir::tt::ttnn::ReshapeOp>(
      srcOp, outputType, cumsumOp->getResult(0), shapeAttrI);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
