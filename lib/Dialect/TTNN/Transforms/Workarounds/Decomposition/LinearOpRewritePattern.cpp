// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Rewrite Linear op into matmul + add if input B is batched.
// Follows
// third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul.cpp.
static bool isBatchedLinearOp(ttnn::LinearOp linearOp) {
  RankedTensorType inputBType = linearOp.getB().getType();
  auto inputBShape = inputBType.getShape();
  int64_t rank = inputBShape.size();

  // if rank < 2, it cannot be batched. Return false.
  // Check if batched: any dimension before the last 2 has size > 1.
  // i.e. <1x3x128x32xbf16> is batched because of 3.
  // i.e. <1x1x128x32xbf16> is not batched because all dims before last 2 are 1.
  // i.e. <128xbf16> is not batched because rank < 2.
  return rank >= 2 && llvm::any_of(inputBShape.drop_back(2),
                                   [](int64_t dim) { return dim > 1; });
}

// Calculate the output shape of a matmul operation following tt-metal's logic.
// Reference: ttnn/cpp/ttnn/operations/matmul/matmul.cpp
static SmallVector<int64_t>
computeMatmulOutputShape(llvm::ArrayRef<int64_t> shapeA,
                         llvm::ArrayRef<int64_t> shapeB) {
  int64_t rankA = shapeA.size();
  int64_t rankB = shapeB.size();

  // Input B is a vector -> We drop the last dimension of shapeA.
  // (n_1,..., n_k, p, q) x (q, (implicit 1)) -> (n_1,..., n_k, p)
  if (rankB == 1) {
    return llvm::SmallVector<int64_t>(shapeA.begin(), shapeA.end() - 1);
  }
  // Input A is a vector -> We drop the second-to-last dimension of shapeB.
  int64_t outRank = std::max(rankA, rankB);
  SmallVector<int64_t> outputShape(outRank, 1);

  int64_t aOffset = outRank - rankA;
  int64_t bOffset = outRank - rankB;

  // Broadcast batch dimensions
  for (int64_t i = 0; i < outRank - 2; ++i) {
    int64_t aDim = (i >= aOffset) ? shapeA[i - aOffset] : 1;
    int64_t bDim = (i >= bOffset) ? shapeB[i - bOffset] : 1;
    outputShape[i] = std::max(aDim, bDim);
  }

  // Matmul inner dims: (…, p, q) x (…, q, r) -> (…, p, r)
  outputShape[outRank - 2] = shapeA[rankA - 2];
  outputShape[outRank - 1] = shapeB[rankB - 1];

  return outputShape;
}

LogicalResult
LinearOpRewritePattern::matchAndRewrite(ttnn::LinearOp srcOp,
                                        PatternRewriter &rewriter) const {

  // Only decompose if input B is batched AND bias exists
  if (!isBatchedLinearOp(srcOp) || !srcOp.getBias()) {
    return failure();
  }

  RankedTensorType inputAType = srcOp.getA().getType();
  RankedTensorType inputBType = srcOp.getB().getType();
  RankedTensorType outputType = srcOp.getResult().getType();

  // Compute matmul output shape
  SmallVector<int64_t> matmulShape =
      computeMatmulOutputShape(inputAType.getShape(), inputBType.getShape());

  // Create matmul output type
  auto outputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());
  auto matmulOutputType =
      utils::RankedTensorTypeFactory::create(outputType, matmulShape);

  auto dataTypeAttr = mlir::tt::ttcore::DataTypeAttr::get(
      rewriter.getContext(), outputEncoding.getDataType());

  // Step 1: Create MatMul operation
  MatmulOp matmulOp = rewriter.create<ttnn::MatmulOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_matmul"),
      matmulOutputType, srcOp.getA(), srcOp.getB(), srcOp.getTransposeA(),
      srcOp.getTransposeB(),
      /*matmul_program_config=*/mlir::Attribute());

  // Step 2: Create Add operation with bias
  AddOp addOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_decomp_add"),
      outputType, matmulOp.getResult(), srcOp.getBias(),
      /*dtype=*/dataTypeAttr,
      /*memory_config=*/ttnn::MemoryConfigAttr());

  rewriter.replaceOp(srcOp, addOp.getResult());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
