// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpDimRewritePattern.h"

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

LogicalResult
CumSumOpDimRewritePattern::matchAndRewrite(ttnn::MorehCumSumOp srcOp,
                                           PatternRewriter &rewriter) const {
  uint64_t dim = srcOp.getDim();
  if (dim <= 1) {
    return failure();
  }

  int64_t rank = srcOp.getInput().getType().getRank();
  llvm::SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(rank));
  std::swap(permutation[0], permutation[dim]);

  mlir::RankedTensorType inputType = srcOp.getInput().getType();
  llvm::ArrayRef<int64_t> originalShape = inputType.getShape();
  llvm::SmallVector<int64_t> adaptedShape =
      ttmlir::utils::applyPermutation(originalShape, permutation);
  RankedTensorType adaptedInputType =
      utils::RankedTensorTypeFactory::create(inputType, adaptedShape);
  auto adaptedInput = rewriter.create<ttnn::PermuteOp>(
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_permute"),
      adaptedInputType, srcOp.getInput(),
      rewriter.getDenseI64ArrayAttr(permutation),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());

  mlir::RankedTensorType outputType = srcOp.getResult().getType();
  RankedTensorType adaptedOutputType =
      utils::RankedTensorTypeFactory::create(outputType, adaptedShape);
  auto adaptedCumSumOp = rewriter.create<mlir::tt::ttnn::MorehCumSumOp>(
      srcOp->getLoc(), adaptedOutputType, adaptedInput, /*dim=*/0,
      /*memory_config=*/nullptr);

  auto permute = rewriter.replaceOpWithNewOp<ttnn::PermuteOp>(
      srcOp, outputType, adaptedCumSumOp,
      rewriter.getDenseI64ArrayAttr(
          ttmlir::utils::inversePermutation(permutation)),
      /*memory_config=*/ttnn::MemoryConfigAttr(),
      /*pad_value=*/mlir::FloatAttr());
  permute->setLoc(
      ttmlir::utils::appendLocationSuffix(permute.getLoc(), "_permuteInverse"));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
