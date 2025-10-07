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
  RankedTensorType inputBType =
      mlir::cast<RankedTensorType>(linearOp.getB().getType());
  auto inputBShape = inputBType.getShape();
  int64_t rank = inputBShape.size();

  // Check if batched: any dimension before the last 2 has size > 1
  if (rank < 2) {
    return false;
  }

  for (int64_t i = 0; i < rank - 2; ++i) {
    if (inputBShape[i] > 1) {
      return true;
    }
  }
  return false;
}

// Calculate the output shape of a matmul operation following tt-metal's logic.
// Reference: ttnn/cpp/ttnn/operations/matmul/matmul.cpp
static SmallVector<int64_t>
computeMatmulOutputShape(llvm::ArrayRef<int64_t> shapeA,
                         llvm::ArrayRef<int64_t> shapeB, bool transposeA,
                         bool transposeB) {

  int64_t rankA = shapeA.size();
  int64_t rankB = shapeB.size();

  // Rank difference will be used to align batch dimensions
  int64_t outRank = std::max(rankA, rankB) - (rankA == 1 || rankB == 1);
  int64_t rankDifference = std::max<int64_t>(0, outRank - rankA);

  // Initialize output shape based on the tensor with higher rank
  SmallVector<int64_t> outputShape;
  if (rankB > rankA) {
    outputShape.assign(shapeB.begin(), shapeB.end());
  } else {
    outputShape.assign(shapeA.begin(), shapeA.end());
  }

  // Handle batch dimensions for the case where rankB > rankA
  for (int64_t index = 0; index < rankDifference; ++index) {
    // tt-metal requires these front dimensions to be 1
    // For our purposes, we'll just copy them
    outputShape[index] = shapeB[index];
  }

  // Copy dimensions from shapeA except the last one
  for (int64_t index = 0; index < rankA - 1; ++index) {
    outputShape[rankDifference + index] = shapeA[index];
  }

  // The last dimension comes from input_tensor_b
  outputShape[outputShape.size() - 1] = shapeB[shapeB.size() - 1];

  // Handle the vector matmul case: if rankA == 1, remove the second-to-last
  // dimension
  if (rankA == 1 && outputShape.size() > 1) {
    SmallVector<int64_t> newShape;
    newShape.reserve(outputShape.size() - 1);
    // Copy all elements except the second-to-last dimension
    for (size_t srcIdx = 0; srcIdx < outputShape.size(); ++srcIdx) {
      if (srcIdx != outputShape.size() - 2) {
        newShape.push_back(outputShape[srcIdx]);
      }
    }
    outputShape = std::move(newShape);
  }

  // Handle the case where rankB == 1, remove the last dimension
  if (rankB == 1) {
    SmallVector<int64_t> newShape;
    newShape.reserve(outputShape.size() - 1);
    for (size_t index = 0; index < outputShape.size() - 1; ++index) {
      newShape.push_back(outputShape[index]);
    }
    outputShape = std::move(newShape);
  }

  return outputShape;
}

LogicalResult
LinearOpRewritePattern::matchAndRewrite(ttnn::LinearOp srcOp,
                                        PatternRewriter &rewriter) const {

  // Only decompose if input B is batched AND bias exists
  if (!isBatchedLinearOp(srcOp) || !srcOp.getBias()) {
    return failure();
  }

  RankedTensorType inputAType =
      mlir::cast<RankedTensorType>(srcOp.getA().getType());
  RankedTensorType inputBType =
      mlir::cast<RankedTensorType>(srcOp.getB().getType());
  RankedTensorType outputType =
      mlir::cast<RankedTensorType>(srcOp.getResult().getType());

  // Compute matmul output shape
  SmallVector<int64_t> matmulShape =
      computeMatmulOutputShape(inputAType.getShape(), inputBType.getShape(),
                               srcOp.getTransposeA(), srcOp.getTransposeB());

  // Create matmul output type
  auto outputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(outputType.getEncoding());
  auto matmulOutputType = RankedTensorType::get(
      matmulShape, outputType.getElementType(), outputType.getEncoding());

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
