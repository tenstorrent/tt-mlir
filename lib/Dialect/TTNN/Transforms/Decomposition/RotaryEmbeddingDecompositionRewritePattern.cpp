// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/RotaryEmbeddingDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

// If `value` is `ttnn.concat(half, half, dim=last)` with the *same* SSA value
// on both operands (the half-D → full-D cos/sin duplication pattern emitted by
// Pattern-1 RoPE models), return the half-D input. Otherwise return nullopt.
static std::optional<Value> matchSelfConcatLastDim(Value value) {
  auto concatOp = value.getDefiningOp<ttnn::ConcatOp>();
  if (!concatOp || concatOp.getNumOperands() != 2 ||
      concatOp.getOperand(0) != concatOp.getOperand(1)) {
    return std::nullopt;
  }
  auto resultType = mlir::cast<RankedTensorType>(concatOp.getType());
  // ttnn.concat verifier accepts negative dims (e.g. -1) and normalizes them
  // (concat verifier: `if (dim < 0) dim += rank`). Match the normalized form
  // so equivalent IR with dim=-1 still triggers the pattern.
  int32_t rawDim = concatOp.getDim();
  int64_t lastDim = resultType.getRank() - 1;
  int64_t normalizedDim = rawDim < 0 ? rawDim + resultType.getRank() : rawDim;
  if (normalizedDim != lastDim) {
    return std::nullopt;
  }
  return concatOp.getOperand(0);
}

LogicalResult RotaryEmbeddingDecompositionRewritePattern::matchAndRewrite(
    ttnn::RotaryEmbeddingOp ropeOp, PatternRewriter &rewriter) const {

  // Don't decompose decode-mode RoPE (token_index set). These are
  // intentionally created by DecodeRoPELayoutOptimization and should be
  // preserved for efficient decode execution.
  if (ropeOp.getTokenIndex()) {
    return failure();
  }


  auto inputType = mlir::cast<RankedTensorType>(ropeOp.getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(ropeOp.getResult().getType());

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

  auto loc = ropeOp.getLoc();
  auto inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  int64_t lastDim = rank - 1;
  int64_t fullDim = inputShape[lastDim];
  int64_t halfDim = fullDim / 2;

  // Slice attrs for x_lo = x[..., :D/2] and x_hi = x[..., D/2:].
  SmallVector<Attribute> beginsLo, endsLo, beginsHi, endsHi, steps;
  for (int64_t i = 0; i < rank; ++i) {
    beginsLo.push_back(rewriter.getI32IntegerAttr(0));
    beginsHi.push_back(rewriter.getI32IntegerAttr(i == lastDim ? halfDim : 0));
    endsLo.push_back(
        rewriter.getI32IntegerAttr(i == lastDim ? halfDim : inputShape[i]));
    endsHi.push_back(rewriter.getI32IntegerAttr(inputShape[i]));
    steps.push_back(rewriter.getI32IntegerAttr(1));
  }
  auto stepsAttr = rewriter.getArrayAttr(steps);

  SmallVector<int64_t> halfShape(inputShape);
  halfShape[lastDim] = halfDim;
  auto halfType = utils::RankedTensorTypeFactory::create(resultType, halfShape);

  auto xLo = rewriter.create<ttnn::SliceStaticOp>(
      loc, halfType, ropeOp.getInput(), rewriter.getArrayAttr(beginsLo),
      rewriter.getArrayAttr(endsLo), stepsAttr);
  auto xHi = rewriter.create<ttnn::SliceStaticOp>(
      loc, halfType, ropeOp.getInput(), rewriter.getArrayAttr(beginsHi),
      rewriter.getArrayAttr(endsHi), stepsAttr);

  // If cos/sin are both half-D self-duplications, emit complex-rotation form:
  //   concat( x_lo*cos - x_hi*sin ,  x_hi*cos + x_lo*sin )
  // This is algebraically identical to the rotate_half form below, but uses
  // only half-D ops and a single concat — strictly fewer ops, no full-D
  // multiplies, no concat(x, x) ops left behind.
  //
  // Broadcast safety: the matcher guarantees `cosHalf.shape[-1] == halfDim`
  // (concat-of-two halves on the last dim doubles it), and `cosHalf`'s
  // non-last dims equal `cosCache`'s non-last dims. So `multiply(xLo,
  // cosHalf)` broadcasts iff `multiply(input, cosCache)` did — i.e. this
  // branch is no less safe than the rotate_half fallback below, which makes
  // the same broadcast assumption.
  if (auto cosHalf = matchSelfConcatLastDim(ropeOp.getCosCache())) {
    if (auto sinHalf = matchSelfConcatLastDim(ropeOp.getSinCache())) {
      auto loCos = rewriter.create<ttnn::MultiplyOp>(loc, halfType,
                                                     xLo.getResult(), *cosHalf);
      auto hiSin = rewriter.create<ttnn::MultiplyOp>(loc, halfType,
                                                     xHi.getResult(), *sinHalf);
      auto first = rewriter.create<ttnn::SubtractOp>(
          loc, halfType, loCos.getResult(), hiSin.getResult());

      auto hiCos = rewriter.create<ttnn::MultiplyOp>(loc, halfType,
                                                     xHi.getResult(), *cosHalf);
      auto loSin = rewriter.create<ttnn::MultiplyOp>(loc, halfType,
                                                     xLo.getResult(), *sinHalf);
      auto second = rewriter.create<ttnn::AddOp>(
          loc, halfType, hiCos.getResult(), loSin.getResult());

      auto result = rewriter.create<ttnn::ConcatOp>(
          loc, resultType, ValueRange{first.getResult(), second.getResult()},
          static_cast<int32_t>(lastDim));

      rewriter.replaceOp(ropeOp, result.getResult());
      return success();
    }
  }

  // Fallback (cos/sin not self-duplications): rotate_half form
  //   result = x*cos + concat(neg(x_hi), x_lo)*sin
  auto negHi = rewriter.create<ttnn::NegOp>(loc, halfType, xHi.getResult());
  auto rotated = rewriter.create<ttnn::ConcatOp>(
      loc, resultType, ValueRange{negHi.getResult(), xLo.getResult()},
      static_cast<int32_t>(lastDim));
  auto xCos = rewriter.create<ttnn::MultiplyOp>(
      loc, resultType, ropeOp.getInput(), ropeOp.getCosCache());
  auto rotSin = rewriter.create<ttnn::MultiplyOp>(
      loc, resultType, rotated.getResult(), ropeOp.getSinCache());
  auto result = rewriter.create<ttnn::AddOp>(loc, resultType, xCos.getResult(),
                                             rotSin.getResult());

  rewriter.replaceOp(ropeOp, result.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
