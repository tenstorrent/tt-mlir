// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

struct FoldHit {
  MultiplyOp multiplyOp; // the multiply to bypass
  Value bypassValue;     // value that should replace multiplyOp's result
  float scalar;          // the scalar to fold into SDPA's scale
};

// Walks back from `v` along the def-use chain through any one-use ttnn.typecast
// or ttnn.permute ops, looking for a one-use ttnn.multiply whose other input
// is a scalar constant. Returns the found multiply and scalar, or nullopt.
static std::optional<FoldHit> findUpstreamScaleMultiply(Value v) {
  Value cur = v;
  while (Operation *defOp = cur.getDefiningOp()) {
    if (isa<TypecastOp, PermuteOp>(defOp)) {
      if (!defOp->hasOneUse()) {
        return std::nullopt;
      }
      cur = defOp->getOperand(0);
      continue;
    }

    if (auto mulOp = dyn_cast<MultiplyOp>(defOp)) {
      if (!mulOp->hasOneUse()) {
        return std::nullopt;
      }
      auto [bypass, scalar] =
          mlir::tt::ttnn::utils::extractMultiplyWithScalarConstant(mulOp);
      if (!scalar) {
        return std::nullopt;
      }
      return FoldHit{mulOp, bypass, *scalar};
    }

    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

LogicalResult ScaledDotProductAttentionFoldScaleRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const {
  auto qHit = findUpstreamScaleMultiply(op.getQuery());
  auto kHit = findUpstreamScaleMultiply(op.getKey());

  if (!qHit && !kHit) {
    return failure();
  }

  float currentScale =
      op.getScale().has_value() ? op.getScale()->convertToFloat() : 1.0f;
  float qScalar = qHit ? qHit->scalar : 1.0f;
  float kScalar = kHit ? kHit->scalar : 1.0f;
  float newScale = currentScale * qScalar * kScalar;

  // Bypass each multiply (RAUW its result with its non-scalar input). Safe
  // because multiply with a 1x1x1x1 broadcast preserves the larger operand's
  // shape, so output type == non-scalar-input type.
  if (qHit) {
    rewriter.replaceOp(qHit->multiplyOp, qHit->bypassValue);
  }
  if (kHit) {
    rewriter.replaceOp(kHit->multiplyOp, kHit->bypassValue);
  }

  rewriter.modifyOpInPlace(
      op, [&] { op.setScaleAttr(rewriter.getF32FloatAttr(newScale)); });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
