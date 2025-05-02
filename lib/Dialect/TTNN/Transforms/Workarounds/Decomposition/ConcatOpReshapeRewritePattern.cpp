// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpReshapeRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult ConcatOpReshapeRewritePattern::matchAndRewrite(
    ttnn::ConcatOp srcOp, PatternRewriter &rewriter) const {
  int64_t dim = srcOp.getDim();
  mlir::RankedTensorType outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(srcOp->getResultTypes().front());
  int64_t rank = outputType.getRank();
  if ((dim + 1) != rank) {
    return failure();
  }

  auto inputs = srcOp.getInputs();
  bool isWorkaroundRequired = true;
  llvm::SmallVector<int64_t, 4> lastDims;

  for (auto input : inputs) {
    auto inputShape =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType()).getShape();
    lastDims.emplace_back(inputShape.back());
    isWorkaroundRequired &=
        llvm::all_of(inputShape, [&](int64_t index) { return index == 1; });
  }

  auto countDims =
      llvm::count_if(lastDims, [&](int64_t index) { return index == 1; });

  isWorkaroundRequired |= (countDims > 0 && countDims < rank);
  if (!isWorkaroundRequired) {
    return failure();
  }

  llvm::SmallVector<mlir::Value, 2> adaptedInputs;

  for (auto input : inputs) {
    auto inputShape =
        mlir::dyn_cast<mlir::RankedTensorType>(input.getType()).getShape();
    llvm::SmallVector<int64_t, 4> adaptedShape(inputShape);
    adaptedShape.emplace_back(1);
    auto check = dyn_cast<mlir::TypedValue<mlir::RankedTensorType>>(input);
    ReshapeOp adaptedInput =
        ttir_to_ttnn::utils::generateReshape(check, adaptedShape, rewriter);
    adaptedInputs.emplace_back(adaptedInput);
  }

  auto outputShape = outputType.getShape();
  llvm::SmallVector<int64_t, 4> adaptedOutputShape(outputShape);
  adaptedOutputShape.emplace_back(1);
  mlir::RankedTensorType adaptedOutputType =
      mlir::RankedTensorType::Builder(outputType)
          .setShape(adaptedOutputShape)
          .setEncoding(
              mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding())
                  .withTensorShape(adaptedOutputShape));

  ConcatOp adaptedConcatOp = rewriter.create<mlir::tt::ttnn::ConcatOp>(
      srcOp->getLoc(), adaptedOutputType, adaptedInputs, srcOp.getDim(),
      /*memory_config=*/nullptr);

  ReshapeOp concatOutput = ttir_to_ttnn::utils::generateReshape(
      adaptedConcatOp, srcOp.getResult().getType().getShape(), rewriter);
  rewriter.replaceOp(srcOp, concatOutput);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
