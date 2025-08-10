// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {
LogicalResult EmbeddingOpSqueezeWeightRewritePattern::matchAndRewrite(
    ttnn::EmbeddingOp srcOp, PatternRewriter &rewriter) const {
  mlir::RankedTensorType weightType = srcOp.getWeight().getType();
  if (weightType.getRank() <= 4) {
    return failure();
  }

  llvm::ArrayRef<int64_t> weightShape = weightType.getShape();
  assert(std::all_of(weightShape.begin(), weightShape.end() - 2,
                     [](int64_t dim) { return dim == 1; }) &&
         "Weight tensor must be effectively 2D tensor");

  // Create new shape for weight tensor to make it 4D tensor. It's going to be
  // (1, 1, weightShape[-2], weightShape[-1]).
  llvm::SmallVector<int64_t, 4> adaptedWeightShape(2, 1);
  adaptedWeightShape.append(weightShape.end() - 2, weightShape.end());

  // Create reshape op to squeeze weight tensor to 4D.
  ReshapeOp reshapeWeightOp = ttir_to_ttnn::utils::generateReshape(
      srcOp.getWeight(), adaptedWeightShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp->getLoc(), "_reshape"));

  rewriter.modifyOpInPlace(
      srcOp, [&]() { srcOp.getWeightMutable().assign(reshapeWeightOp); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
