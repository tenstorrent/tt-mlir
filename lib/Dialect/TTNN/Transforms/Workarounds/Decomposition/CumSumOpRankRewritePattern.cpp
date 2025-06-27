// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRankRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult
CumSumOpRankRewritePattern::matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                            PatternRewriter &rewriter) const {
  constexpr int64_t TARGET_RANK = 4;
  mlir::RankedTensorType inputType = srcOp.getInput().getType();
  auto shape = inputType.getShape();
  if (shape.size() >= TARGET_RANK) {
    return failure();
  }

  int64_t rank = inputType.getRank();
  int64_t additionalAxes = TARGET_RANK - rank;
  llvm::SmallVector<int64_t, 4> adaptedShape(shape);
  adaptedShape.append(additionalAxes, 1);

  ReshapeOp adaptedInput = ttir_to_ttnn::utils::generateReshape(
      srcOp.getInput(), adaptedShape, rewriter, "_reshapeInput");

  RankedTensorType outputType = srcOp.getResult().getType();
  RankedTensorType adaptedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, adaptedShape);
  MorehCumSumOp adaptedCumSumOp =
      rewriter.create<mlir::tt::ttnn::MorehCumSumOp>(
          srcOp->getLoc(), adaptedOutputType, adaptedInput, srcOp.getDim(),
          /*memory_config=*/nullptr);

  ReshapeOp cumsumOutput = ttir_to_ttnn::utils::generateReshape(
      adaptedCumSumOp, srcOp.getResult().getType().getShape(), rewriter,
      "_reshapeOutput");
  rewriter.replaceOp(srcOp, cumsumOutput);

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
