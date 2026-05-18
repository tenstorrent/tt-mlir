// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ArgMaxOpDimRewritePattern.h"

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

// Permute the input tensor so that the reduction dimension is last, perform
// argmax on the last dimension, then permute back.
LogicalResult
ArgMaxOpDimRewritePattern::matchAndRewrite(ttnn::ArgMaxOp srcOp,
                                           PatternRewriter &rewriter) const {
  auto dimArg = srcOp.getDim();
  if (!dimArg) {
    return failure();
  }

  int64_t rank = srcOp.getInput().getType().getRank();
  int64_t dim = *dimArg;
  if (dim < 0) {
    dim += rank;
  }

  // No permute needed if already reducing on last dim.
  if (dim == rank - 1) {
    return failure();
  }

  // Build permutation swapping dim with last dim.
  llvm::SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(rank));
  std::swap(permutation[dim], permutation[rank - 1]);

  // Compute permuted input shape and type.
  mlir::RankedTensorType inputType = srcOp.getInput().getType();
  llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
  llvm::SmallVector<int64_t> permutedShape =
      ttmlir::utils::applyPermutation(originalShape, permutation);
  RankedTensorType permutedInputType =
      utils::RankedTensorTypeFactory::create(inputType, permutedShape);

  // Forward permute: move reduction dim to last position.
  auto forwardPermute = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_permute"),
      permutedInputType, srcOp.getInput(),
      rewriter.getDenseI64ArrayAttr(permutation),
      /*pad_value=*/mlir::FloatAttr());

  // Compute permuted output shape and inverse permutation for the output.
  RankedTensorType outputType = srcOp.getResult().getType();
  llvm::SmallVector<int64_t> permutedOutputShape;
  llvm::SmallVector<int64_t> outputInversePerm;

  if (srcOp.getKeepDim()) {
    // keep_dim=true: output has same rank as input, use same permutation.
    permutedOutputShape =
        ttmlir::utils::applyPermutation(outputType.getShape(), permutation);
    outputInversePerm = ttmlir::utils::inversePermutation(permutation);
  } else {
    // keep_dim=false: output has rank-1. Permuted output is the permuted input
    // shape with last dim removed.
    permutedOutputShape.assign(permutedShape.begin(), permutedShape.end() - 1);

    // Compute reduced inverse permutation: move element at position dim to the
    // end, shifting elements after it to the left.
    outputInversePerm = llvm::to_vector(llvm::seq<int64_t>(rank - 1));
    std::rotate(outputInversePerm.begin() + dim,
                outputInversePerm.begin() + dim + 1, outputInversePerm.end());
  }

  RankedTensorType permutedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, permutedOutputShape);

  // Create argmax on last dim of permuted tensor.
  mlir::IntegerAttr lastDimAttr = mlir::IntegerAttr::get(
      mlir::IntegerType::get(getContext(), 32), rank - 1);
  auto argMaxOp = rewriter.create<ttnn::ArgMaxOp>(
      srcOp->getLoc(), permutedOutputType, forwardPermute, lastDimAttr,
      srcOp.getKeepDimAttr(), srcOp.getUseMulticoreAttr());

  // Inverse permute to restore original dimension order.
  auto inversePermute = rewriter.replaceOpWithNewOp<ttnn::PermuteOp>(
      srcOp, outputType, argMaxOp,
      rewriter.getDenseI64ArrayAttr(outputInversePerm),
      /*pad_value=*/mlir::FloatAttr());
  inversePermute->setLoc(ttmlir::utils::appendLocationSuffix(
      inversePermute.getLoc(), "_permuteInverse"));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
