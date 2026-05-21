// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/RotaryEmbeddingDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult RotaryEmbeddingDecompositionRewritePattern::matchAndRewrite(
    ttnn::RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter) const {

  auto inputType = mlir::cast<RankedTensorType>(ropeOp.getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(ropeOp.getResult().getType());

  // When validation config is provided, validate first.
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult = validator.validateOp<ttnn::RotaryEmbeddingOp>(
        ropeOp.getOperation(), ropeOp.getLoc(), {resultType}, ropeOp.getInput(),
        ropeOp.getCosCache(), ropeOp.getSinCache(), ropeOp.getTokenIndexAttr(),
        /*compute_config=*/nullptr);

    if (validationResult.isSuccess()) {
      return failure(); // Op is valid, keep it.
    }

    TTMLIR_DEBUG(
        ttmlir::LogComponent::IsolatedIRValidationWrapper,
        "RotaryEmbedding decomposition triggered (validation failed): {0}",
        validationResult.errorMessage);
  }

  // Decompose: result = x*cos + rotate_half(x)*sin
  // where rotate_half(x) = concat(neg(x[D/2:]), x[:D/2])

  auto loc = ropeOp.getLoc();
  auto inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  int64_t lastDim = rank - 1;
  int64_t fullDim = inputShape[lastDim];
  int64_t halfDim = fullDim / 2;

  // Build slice attributes for first half [0, D/2) and second half [D/2, D).
  SmallVector<Attribute> beginsLo, endsLo, beginsHi, endsHi, steps;
  for (int64_t i = 0; i < rank; ++i) {
    beginsLo.push_back(rewriter.getI32IntegerAttr(0));
    beginsHi.push_back(rewriter.getI32IntegerAttr(i == lastDim ? halfDim : 0));
    endsLo.push_back(
        rewriter.getI32IntegerAttr(i == lastDim ? halfDim : inputShape[i]));
    endsHi.push_back(rewriter.getI32IntegerAttr(inputShape[i]));
    steps.push_back(rewriter.getI32IntegerAttr(1));
  }

  // Compute half-dim shape and type.
  SmallVector<int64_t> halfShape(inputShape);
  halfShape[lastDim] = halfDim;
  auto halfType = utils::RankedTensorTypeFactory::create(resultType, halfShape);

  auto stepsAttr = rewriter.getArrayAttr(steps);

  // x_lo = x[:D/2], x_hi = x[D/2:]
  auto xLo = rewriter.create<ttnn::SliceStaticOp>(
      loc, halfType, ropeOp.getInput(), rewriter.getArrayAttr(beginsLo),
      rewriter.getArrayAttr(endsLo), stepsAttr);
  auto xHi = rewriter.create<ttnn::SliceStaticOp>(
      loc, halfType, ropeOp.getInput(), rewriter.getArrayAttr(beginsHi),
      rewriter.getArrayAttr(endsHi), stepsAttr);

  // neg(x_hi)
  auto negHi = rewriter.create<ttnn::NegOp>(loc, halfType, xHi.getResult());

  // rotated = concat(neg(x_hi), x_lo) on last dim
  auto rotated = rewriter.create<ttnn::ConcatOp>(
      loc, resultType, ValueRange{negHi.getResult(), xLo.getResult()},
      static_cast<int32_t>(lastDim));

  // x * cos
  auto xCos = rewriter.create<ttnn::MultiplyOp>(
      loc, resultType, ropeOp.getInput(), ropeOp.getCosCache());

  // rotated * sin
  auto rotSin = rewriter.create<ttnn::MultiplyOp>(
      loc, resultType, rotated.getResult(), ropeOp.getSinCache());

  // result = x*cos + rotated*sin
  auto result = rewriter.create<ttnn::AddOp>(loc, resultType, xCos.getResult(),
                                             rotSin.getResult());

  rewriter.replaceOp(ropeOp, result.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
