// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MultiplyOpDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult MultiplyOpDecompositionRewritePattern::matchAndRewrite(
    ttnn::MultiplyOp originalMultiplyOp, PatternRewriter &rewriter) const {
  Location loc = originalMultiplyOp.getLoc();
  auto lhs = originalMultiplyOp.getLhs();
  auto rhs = originalMultiplyOp.getRhs();

  // If inputs are direct arguments, we don't control their deallocation.
  if (lhs.getDefiningOp() == nullptr || rhs.getDefiningOp() == nullptr) {
    return failure();
  }

  RankedTensorType lhsType = lhs.getType();
  RankedTensorType rhsType = rhs.getType();
  RankedTensorType outputType = originalMultiplyOp.getResult().getType();

  if (lhsType.getRank() != 4 || rhsType.getRank() != 4 ||
      outputType.getRank() != 4) {
    return failure();
  }

  auto outputShape = outputType.getShape();
  // Only apply workaround if output dimensions are greater than (1024, 1024, 1,
  // 1) Check that dims 2 and 3 are equal to 1 (or at least small) and first two
  // dims are large
  if (outputShape.size() != 4 || outputShape[0] < 1024 ||
      outputShape[1] < 1024 || outputShape[2] != 1 || outputShape[3] != 1) {
    return failure();
  }

  // Define the permutation (2, 3, 0, 1)
  llvm::SmallVector<int64_t> permutation = {2, 3, 0, 1};
  DenseI64ArrayAttr permutationAttr =
      rewriter.getDenseI64ArrayAttr(permutation);

  // Apply permutation to lhs input
  llvm::SmallVector<int64_t> lhsPermutedShape =
      ttmlir::utils::applyPermutation(lhsType.getShape(), permutation);
  RankedTensorType lhsPermutedType =
      utils::RankedTensorTypeFactory::create(lhsType, lhsPermutedShape);

  PermuteOp lhsPermuted = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_lhs_permute"), lhsPermutedType,
      lhs, permutationAttr, ttnn::MemoryConfigAttr(), mlir::FloatAttr());

  // Apply permutation to rhs input
  llvm::SmallVector<int64_t> rhsPermutedShape =
      ttmlir::utils::applyPermutation(rhsType.getShape(), permutation);
  RankedTensorType rhsPermutedType =
      utils::RankedTensorTypeFactory::create(rhsType, rhsPermutedShape);

  PermuteOp rhsPermuted = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rhs_permute"), rhsPermutedType,
      rhs, permutationAttr, ttnn::MemoryConfigAttr(), mlir::FloatAttr());

  // Create the multiply operation on permuted inputs
  llvm::SmallVector<int64_t> permutedOutputShape =
      ttmlir::utils::applyPermutation(outputShape, permutation);
  RankedTensorType permutedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, permutedOutputShape);

  MultiplyOp permutedMultiply = rewriter.create<ttnn::MultiplyOp>(
      loc, permutedOutputType, lhsPermuted.getResult(),
      rhsPermuted.getResult());

  // Apply reverse permutation to output (which is the same as forward: (2, 3,
  // 0, 1))
  PermuteOp finalResult = rewriter.replaceOpWithNewOp<ttnn::PermuteOp>(
      originalMultiplyOp, outputType, permutedMultiply.getResult(),
      permutationAttr, ttnn::MemoryConfigAttr(), mlir::FloatAttr());
  finalResult->setLoc(
      ttmlir::utils::appendLocationSuffix(loc, "_output_permute"));

  rewriter.moveOpAfter(lhsPermuted, lhs.getDefiningOp());
  rewriter.moveOpAfter(rhsPermuted, rhs.getDefiningOp());

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
