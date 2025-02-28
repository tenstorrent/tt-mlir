// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRewritePattern.h"
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
  llvm::SmallVector<int64_t, 4> reshapeOutputShape(4, 1);
  for (int idx = 0; idx < inputRank; ++idx) {
    reshapeOutputShape[idx] = inputTypeShape[idx];
  }

  llvm::ArrayRef<int64_t> reshapedShapeAttr(reshapeOutputShape);
  ReshapeOp preReshapeOp = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), reshapedShapeAttr, rewriter);

  mlir::RankedTensorType reshapeOutputType =
      mlir::RankedTensorType::Builder(inputType).setShape(reshapeOutputShape);
  ttnn::TTNNLayoutAttr newOutputLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
          .withTensorShape(rewriter.getContext(), reshapeOutputType.getShape());
  RankedTensorType newOutputType =
      RankedTensorType::get(reshapeOutputType.getShape(),
                            outputType.getElementType(), newOutputLayoutAttr);

  MorehCumSumOp cumsumOp = rewriter.create<mlir::tt::ttnn::MorehCumSumOp>(
      srcOp->getLoc(), newOutputType, preReshapeOp->getResult(0),
      srcOp.getDim(), nullptr);

  llvm::ArrayRef<int64_t> outputShapeAttr(inputTypeShape);
  mlir::TypedValue<mlir::RankedTensorType> cumsumOutput =
      mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
          cumsumOp->getResults().front());

  ReshapeOp postReshapeOp = ttir_to_ttnn::utils::generateReshape(
      cumsumOutput, outputShapeAttr, rewriter);
  rewriter.replaceOp(srcOp, postReshapeOp);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
