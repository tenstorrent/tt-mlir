// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GroupNormAffineReshapeRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

LogicalResult GroupNormAffineReshapeRewritePattern::matchAndRewrite(
    ttnn::GroupNormOp srcOp, PatternRewriter &rewriter) const {
  constexpr int64_t tileWidth = 32;

  auto reshapeAffineIfNeeded = [&](Value affine, StringRef suffix) -> Value {
    if (!affine) {
      return affine;
    }
    auto affineType = mlir::cast<RankedTensorType>(affine.getType());
    if (affineType.getRank() != 1) {
      return affine;
    }

    int64_t c = affineType.getShape()[0];
    if (c <= 0 || c % tileWidth != 0) {
      return affine;
    }

    llvm::SmallVector<int64_t> reshapedShape = {1, 1, c / tileWidth, tileWidth};
    return ttir_to_ttnn::utils::generateReshape(
        mlir::cast<TypedValue<RankedTensorType>>(affine), reshapedShape,
        rewriter, ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), suffix));
  };

  Value newWeight = reshapeAffineIfNeeded(srcOp.getWeight(), "_reshape_weight");
  Value newBias = reshapeAffineIfNeeded(srcOp.getBias(), "_reshape_bias");
  if (newWeight == srcOp.getWeight() && newBias == srcOp.getBias()) {
    return failure();
  }

  auto newOp = ttnn::GroupNormOp::create(
      rewriter, srcOp.getLoc(), srcOp.getResult().getType(), srcOp.getInput(),
      srcOp.getInputMask(), newWeight, newBias, srcOp.getNumGroupsAttr(),
      srcOp.getEpsilonAttr(), srcOp.getMemoryConfigAttr(),
      srcOp.getCoreGridAttr());

  rewriter.replaceOp(srcOp, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
