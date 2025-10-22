# include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ComplexReshapePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult ComplexReshapePattern::matchAndRewrite(ttnn::ReshapeOp srcOp,
                                                     PatternRewriter &rewriter) const {
  // Skip if already decomposed by this pattern to prevent infinite loops
  if (srcOp->hasAttr("complex_reshape_decomposed")) {
    return failure();
  }

  mlir::RankedTensorType inputType = ::mlir::dyn_cast<mlir::RankedTensorType>(srcOp.getInput().getType());
  mlir::RankedTensorType outputType = ::mlir::dyn_cast<mlir::RankedTensorType>(srcOp.getResult().getType());

  if (!inputType || !outputType || !inputType.hasRank() || !outputType.hasRank()) {
    return failure();
  }

  // Only 4D tensors are eligible for this pattern.
  if (inputType.getRank() != 4 || outputType.getRank() != 4) {
    return failure();
  }

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  int64_t Bi = inputShape[0];
  int64_t Ci = inputShape[1];
  int64_t Hi = inputShape[2];
  int64_t Wi = inputShape[3];

  int64_t Bo = outputShape[0];
  int64_t Co = outputShape[1];
  int64_t Ho = outputShape[2];
  int64_t Wo = outputShape[3];

      // Check the three constraints:
    // 1. Bi == Bo (batch dimension unchanged)
    // 2. Ci * Hi == Co (channel and height merged)
    // 3. Ho * Wo == Wi (width split into height and width)
    // 4. Ho > 1 and Hi > 1 (avoids infinite recursion)
  if (Bi != Bo || Ci * Hi != Co || Ho * Wo != Wi || Ho <= 1 || Hi <= 1) {
    return failure();
  }

  llvm::SmallVector<int64_t, 4> intermediateShape = {Bi, Ci * Hi, 1, Wo};
  RankedTensorType intermediateShapeType = ttnn::utils::RankedTensorTypeFactory::create(inputType, intermediateShape);

  auto firstNewReshape = rewriter.create<ttnn::ReshapeOp>(ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_intermediate_reshape"),
  intermediateShapeType, srcOp.getInput(), rewriter.getI64ArrayAttr(intermediateShape), srcOp.getMemoryConfigAttr());
  // Mark as decomposed to prevent infinite loops
  firstNewReshape->setAttr("complex_reshape_decomposed", rewriter.getUnitAttr());

  auto secondNewReshape = rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
      srcOp, outputType, firstNewReshape.getResult(),
      rewriter.getI64ArrayAttr(outputShape), srcOp.getMemoryConfigAttr());
  // Mark as decomposed to prevent infinite loops
  secondNewReshape->setAttr("complex_reshape_decomposed", rewriter.getUnitAttr());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition