// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ArgMaxOpRewritePattern.h"
#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
ArgMaxOpRewritePattern::matchAndRewrite(ttnn::ArgMaxOp srcOp,
                                        PatternRewriter &rewriter) const {
  mlir::RankedTensorType inputType =
      mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  llvm::SmallVector<int64_t> inputTypeShape(inputType.getShape());
  if (inputTypeShape.size() >= 4) {
    return failure();
  }

  int64_t inputRank = inputType.getRank();
  llvm::SmallVector<int64_t, 4> reshapeOutputShape(4 - inputRank, 1);
  reshapeOutputShape.append(inputTypeShape.begin(), inputTypeShape.end());

  llvm::ArrayRef<int64_t> reshapedShapeAttr(reshapeOutputShape);

  ReshapeOp preReshapeOp = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), reshapedShapeAttr, rewriter);

  RankedTensorType outputType = srcOp.getResult().getType();
  llvm::SmallVector<int64_t> outputTypeShape(outputType.getShape());
  llvm::SmallVector<int64_t, 4> argMaxOutputShape(4 - inputRank, 1);
  argMaxOutputShape.append(outputTypeShape.begin(), outputTypeShape.end());

  ttnn::TTNNLayoutAttr newOutputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(rewriter.getContext(), argMaxOutputShape);
  RankedTensorType newOutputType = RankedTensorType::get(
      argMaxOutputShape, outputType.getElementType(), newOutputLayoutAttr);

  DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(),
                                             newOutputLayoutAttr.getDataType());
  ttnn::LayoutAttr tensorLayoutAttr =
      ttnn::LayoutAttr::get(getContext(), newOutputLayoutAttr.getLayout());

  ttnn::ShapeAttr shapeAttr =
      ttnn::ShapeAttr::get(rewriter.getContext(), newOutputType.getShape());

  ttnn::BufferTypeAttr bufferTypeAttr = ttnn::BufferTypeAttr::get(
      getContext(), newOutputLayoutAttr.getBufferType());
  ttnn::ShardSpecAttr shardSpecAttr = ttnn::ShardSpecAttr::get(
      getContext(),
      ttnn::ShapeAttr::get(getContext(), newOutputLayoutAttr.getShardShape()));
  ttnn::MemoryConfigAttr memoryConfigAttr =
      ttnn::MemoryConfigAttr::get(getContext(), bufferTypeAttr, shardSpecAttr,
                                  newOutputLayoutAttr.getMemLayout());

  EmptyOp emptyOp = rewriter.create<ttnn::EmptyOp>(
      srcOp->getLoc(), newOutputType, shapeAttr, dTypeAttr, tensorLayoutAttr,
      ttnn::utils::getOrInsertDevice(rewriter, srcOp), memoryConfigAttr);

  mlir::IntegerAttr dimAttr;
  auto dimArg = srcOp.getDim();
  if (dimArg) {
    // Update the dimension according to reshaped input.
    int32_t dim = *dimArg + (reshapeOutputShape.size() - inputTypeShape.size());
    dimAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), dim);
  }
  ArgMaxOp argMaxOp = rewriter.create<mlir::tt::ttnn::ArgMaxOp>(
      srcOp->getLoc(), newOutputType, preReshapeOp->getResult(0),
      dimArg ? dimAttr : nullptr, false, nullptr, emptyOp);

  llvm::ArrayRef<int64_t> outputShapeAttr(outputType.getShape());
  mlir::TypedValue<mlir::RankedTensorType> argMaxOutput =
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
          argMaxOp->getResults().front());

  ReshapeOp postReshapeOp = ttir_to_ttnn::utils::generateReshape(
      argMaxOutput, outputShapeAttr, rewriter);

  rewriter.replaceOp(srcOp, postReshapeOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
