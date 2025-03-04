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

// Unsqueeze the input tensor to 4D tensor (if required) and reshape it back
// afterwards.
LogicalResult
ArgMaxOpRewritePattern::matchAndRewrite(ttnn::ArgMaxOp srcOp,
                                        PatternRewriter &rewriter) const {
  mlir::RankedTensorType inputType = srcOp.getInput().getType();
  llvm::ArrayRef<int64_t> inputTypeShape = inputType.getShape();
  if (inputTypeShape.size() >= 4) {
    return failure();
  }

  // Create new shape for input tensor to make it 4D tensor. Starting dims will
  // be set to 1.
  int64_t inputRank = inputType.getRank();
  llvm::SmallVector<int64_t, 4> reshapeOutputShape(4 - inputRank, 1);
  reshapeOutputShape.append(inputTypeShape.begin(), inputTypeShape.end());

  // Create new shape for output tensor to make it 4D tensor. Starting dims will
  // be set to 1.
  RankedTensorType outputType = srcOp.getResult().getType();
  llvm::ArrayRef<int64_t> outputTypeShape(outputType.getShape());
  llvm::SmallVector<int64_t, 4> argMaxOutputShape(4 - inputRank, 1);
  argMaxOutputShape.append(outputTypeShape.begin(), outputTypeShape.end());

  // Create reshape op to unsqueeze input tensor to 4D.
  ReshapeOp preReshapeOp = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), reshapeOutputShape, rewriter);

  // Create output layout attribute with new output tensor shape and create new
  // output type.
  ttnn::TTNNLayoutAttr newOutputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(rewriter.getContext(), argMaxOutputShape);
  RankedTensorType newOutputType = RankedTensorType::get(
      argMaxOutputShape, outputType.getElementType(), newOutputLayoutAttr);

  // Create new ttnn.empty tensor using new output shape and layout attribute.
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
  DataTypeAttr dTypeAttr = DataTypeAttr::get(rewriter.getContext(),
                                             newOutputLayoutAttr.getDataType());

  EmptyOp emptyOp = rewriter.create<ttnn::EmptyOp>(
      srcOp->getLoc(), newOutputType, shapeAttr, dTypeAttr, tensorLayoutAttr,
      ttnn::utils::getOrInsertDevice(rewriter, srcOp), memoryConfigAttr);

  // Update the dimension according to reshaped input (if required). The dim
  // attribute will be incremented exactly as increase in input tensor rank due
  // to unsqueeze operation.
  mlir::IntegerAttr dimAttr = nullptr;
  auto dimArg = srcOp.getDim();
  if (dimArg) {
    int32_t dim = *dimArg + (reshapeOutputShape.size() - inputTypeShape.size());
    dimAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(getContext(), 32), dim);
  }
  // Create new ttnn.argmax op with updated input tensor, dimension, etc.
  ArgMaxOp argMaxOp = rewriter.create<mlir::tt::ttnn::ArgMaxOp>(
      srcOp->getLoc(), newOutputType, preReshapeOp, dimAttr,
      /*use_multicore=*/false, // Default tt-metal value.
      /*memoryConfig=*/nullptr, emptyOp);

  // Create ttnn.reshape op after performing ttnn.argmax op.
  ReshapeOp postReshapeOp = ttir_to_ttnn::utils::generateReshape(
      argMaxOp, outputType.getShape(), rewriter);

  rewriter.replaceOp(srcOp, postReshapeOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
