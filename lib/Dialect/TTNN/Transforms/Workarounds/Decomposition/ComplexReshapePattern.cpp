# include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ComplexReshapePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult ComplexReshapePattern::matchAndRewrite(ttnn::ReshapeOp srcOp,
                                                     PatternRewriter &rewriter) const {
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
  if (Bi != Bo || Ci * Hi != Co || Ho * Wo != Wi) {
    return failure();
  }

  llvm::SmallVector<int64_t, 4> intermediateShape = {Bi, Ci * Hi, 1, Wo};
  RankedTensorType intermediateShapeType = ttnn::utils::RankedTensorTypeFactory::create(inputType, intermediateShape);

  auto firstNewReshape = rewriter.create<ttnn::ReshapeOp>(ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_intermediate_reshape1"),
  intermediateShapeType, srcOp.getInput(), rewriter.getI64ArrayAttr(intermediateShape), srcOp.getMemoryConfigAttr());

  auto secondNewReshape = rewriter.create<ttnn::ReshapeOp>(ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_intermediate_reshape2"),
  outputType, firstNewReshape, rewriter.getI64ArrayAttr(outputShape), srcOp.getMemoryConfigAttr());

  rewriter.replaceOp(srcOp, secondNewReshape.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition