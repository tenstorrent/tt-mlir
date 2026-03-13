// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpOutputShapeRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Calculate the output shape of a matmul operation following tt-metal's logic.
static SmallVector<int64_t>
computeMatmulOutputShape(llvm::ArrayRef<int64_t> shapeA, bool transposeA,
                         llvm::ArrayRef<int64_t> shapeB, bool transposeB) {
  int64_t rankA = shapeA.size();
  int64_t rankB = shapeB.size();

  SmallVector<int64_t> outputShape;

  if (rankA == 1 && rankB == 1) {
    outputShape.push_back(1);
    return outputShape;
  }

  if (rankA == 1) {
    outputShape.append(shapeB.begin(), shapeB.end() - 2);
    outputShape.push_back(transposeB ? shapeB[rankB - 2] : shapeB[rankB - 1]);
    return outputShape;
  }

  if (rankB == 1) {
    if (transposeA) {
      outputShape.append(shapeA.begin(), shapeA.end() - 2);
      outputShape.push_back(shapeA[rankA - 1]);
    } else {
      outputShape.append(shapeA.begin(), shapeA.end() - 1);
    }
    return outputShape;
  }

  // Both inputs are at least rank 2
  SmallVector<int64_t> batchShapeA(shapeA.begin(), shapeA.end() - 2);
  SmallVector<int64_t> batchShapeB(shapeB.begin(), shapeB.end() - 2);
  mlir::OpTrait::util::getBroadcastedShape(batchShapeA, batchShapeB,
                                           outputShape);

  outputShape.push_back(transposeA ? shapeA[rankA - 1] : shapeA[rankA - 2]);
  outputShape.push_back(transposeB ? shapeB[rankB - 2] : shapeB[rankB - 1]);

  return outputShape;
}

LogicalResult LinearOpOutputShapeRewritePattern::matchAndRewrite(
    ttnn::LinearOp srcOp, PatternRewriter &rewriter) const {

  if (!srcOp.getBias()) {
    return failure();
  }

  // Bail for vector-vector products — the matmul produces a scalar and
  // reshaping it to the broadcast shape is not valid.
  if (srcOp.getA().getType().getRank() == 1 &&
      srcOp.getB().getType().getRank() == 1) {
    return failure();
  }

  RankedTensorType biasType = srcOp.getBias().getType();
  ArrayRef<int64_t> biasShape = biasType.getShape();

  // Check if this LinearOp will use the fused kernel path.
  // The fused kernel is used when the padded bias second-to-last dim
  // equals TILE_HEIGHT. In that case, the hardware output shape is
  // the matmul shape, not the broadcasted shape.
  SmallVector<int64_t> paddedBiasShape =
      ttnn::utils::getTilePaddedShape(biasShape);
  bool isFusedKernelPath =
      paddedBiasShape.size() <= 1 ||
      paddedBiasShape[paddedBiasShape.size() - 2] == TILE_HEIGHT;

  if (!isFusedKernelPath) {
    return failure();
  }

  // Compute matmul output shape.
  RankedTensorType inputAType = srcOp.getA().getType();
  RankedTensorType inputBType = srcOp.getB().getType();
  SmallVector<int64_t> matmulShape =
      computeMatmulOutputShape(inputAType.getShape(), srcOp.getTransposeA(),
                               inputBType.getShape(), srcOp.getTransposeB());

  RankedTensorType currentOutputType = srcOp.getResult().getType();
  ArrayRef<int64_t> currentOutputShape = currentOutputType.getShape();

  // If the output shape already equals the matmul shape, nothing to do.
  if (llvm::equal(currentOutputShape, matmulShape)) {
    return failure();
  }

  // Replace LinearOp with matmul-shaped output + ReshapeOp.
  auto matmulOutputType =
      utils::RankedTensorTypeFactory::create(currentOutputType, matmulShape);

  auto newLinearOp = ttnn::LinearOp::create(
      rewriter, srcOp.getLoc(), matmulOutputType, srcOp.getA(), srcOp.getB(),
      srcOp.getBias(), srcOp.getTransposeA(), srcOp.getTransposeB(),
      /*matmul_program_config=*/nullptr, srcOp.getActivationAttr(),
      /*compute_config=*/srcOp.getComputeConfigAttr());

  if (auto weightDtype = srcOp->getAttr("ttcore.weight_dtype")) {
    newLinearOp->setAttr("ttcore.weight_dtype", weightDtype);
  }

  // Reshape back to the original broadcasted shape.
  ttnn::ReshapeOp reshapeOp = ttir_to_ttnn::utils::generateReshape(
      newLinearOp.getResult(), currentOutputShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(),
                                          "_output_shape_workaround"));

  rewriter.replaceOp(srcOp, reshapeOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
