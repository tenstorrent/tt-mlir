// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SDPAFusingPattern.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttir::fusing {

namespace {

// SDPA Query, Key, Value tensors have shape [B, H, S, D]
// (Batch, NumHeads, SeqLen, HeadDim).
constexpr int64_t kSdpaRank = 4;
constexpr int64_t kNumHeadsDim = 1;
constexpr int64_t kSeqLenDim = 2;
constexpr int64_t kHeadDim = 3;

struct SDPAComponents {
  Value query;
  Value key;
  Value value;
  Value mask;
  std::optional<float> scale;
  MatmulOp attentionMatmul;
  SoftmaxOp softmax;
  // Raw operands of the score op (matmul or linear). Q/K are extracted from
  // these (with optional pre-scale on Q and transpose-with-optional-pre-scale
  // on K).
  Value scoreQRaw;
  Value scoreKRaw;
};

// Strips at most one TypecastOp. Used only at the two softmax-precision
// boundary positions — NOT a generic look-through.
Value peelOneCast(Value v) {
  if (auto cast = v.getDefiningOp<TypecastOp>()) {
    return cast.getInput();
  }
  return v;
}

// Walk through BroadcastOp, ReshapeOp, and TypecastOp to find a scalar
// constant. Used only when extracting the scale constant — the constant
// may be produced as a scalar `ttir.full` and then broadcast/reshaped to
// match the operand shape. This is a constrained look-through (one direction,
// one target op).
std::optional<float> extractConstant(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp, BroadcastOp, ReshapeOp>(defOp)) {
      v = defOp->getOperand(0);
      continue;
    }
    break;
  }

  auto fullOp = v.getDefiningOp<FullOp>();
  if (!fullOp) {
    return std::nullopt;
  }
  if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
    return attr.getValue().convertToFloat();
  }
  return std::nullopt;
}

// If v is defined by MultiplyOp or DivOp with one operand being a scalar
// constant, return (otherOperand, scale). For DivOp scale is 1/divisor
// (divisor==0 rejects the peel). Otherwise return (v, nullopt).
std::pair<Value, std::optional<float>> peelScale(Value v) {
  if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
    if (auto s = extractConstant(mulOp.getRhs())) {
      return {mulOp.getLhs(), s};
    }
    if (auto s = extractConstant(mulOp.getLhs())) {
      return {mulOp.getRhs(), s};
    }
  }
  if (auto divOp = v.getDefiningOp<DivOp>()) {
    if (auto d = extractConstant(divOp.getRhs())) {
      if (*d != 0.0f) {
        return {divOp.getLhs(), 1.0f / *d};
      }
    }
  }
  return {v, std::nullopt};
}

// Match a ttir.transpose that swaps the last two dimensions of a 4D tensor.
// dim0/dim1 may be specified as negative indices.
bool isLastTwoDimsTranspose(TransposeOp transposeOp) {
  auto inputType =
      mlir::cast<RankedTensorType>(transposeOp.getInput().getType());
  int64_t rank = inputType.getRank();
  auto normalize = [rank](int32_t d) -> int64_t {
    return d < 0 ? rank + d : d;
  };
  int64_t d0 = normalize(transposeOp.getDim0());
  int64_t d1 = normalize(transposeOp.getDim1());
  if (d0 > d1) {
    std::swap(d0, d1);
  }
  return d0 == rank - 2 && d1 == rank - 1;
}

// Validate Q, K, V have compatible 4D SDPA shapes:
//   Q [B, Hq, Sq, D], K/V [B, Hkv, Sk, D].
bool validateShapes(Value query, Value key, Value value) {
  auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
  auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
  auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!qType || !kType || !vType) {
    return false;
  }
  if (qType.getRank() != kSdpaRank || kType.getRank() != kSdpaRank ||
      vType.getRank() != kSdpaRank) {
    return false;
  }

  auto qShape = qType.getShape();
  auto kShape = kType.getShape();
  auto vShape = vType.getShape();

  // Head dim must agree across Q/K/V.
  if (qShape[kHeadDim] != kShape[kHeadDim] ||
      kShape[kHeadDim] != vShape[kHeadDim]) {
    return false;
  }
  // K and V must share batch, num kv heads, kv seq len.
  if (kShape[0] != vShape[0] || kShape[kNumHeadsDim] != vShape[kNumHeadsDim] ||
      kShape[kSeqLenDim] != vShape[kSeqLenDim]) {
    return false;
  }
  // Q and K must share batch.
  if (qShape[0] != kShape[0]) {
    return false;
  }
  // Hq must be a multiple of Hkv (GQA constraint; MHA and MQA included).
  int64_t qNumHeads = qShape[kNumHeadsDim];
  int64_t kNumHeadsVal = kShape[kNumHeadsDim];
  if (kNumHeadsVal == 0 || qNumHeads % kNumHeadsVal != 0) {
    return false;
  }
  return true;
}

// Validate the attention mask has rank 4 with shape compatible with
// [1|B, 1|Hq, Sq, Sk]. No reshape/typecast look-through.
bool validateMask(Value mask, ArrayRef<int64_t> qShape, int64_t kSeqLen) {
  auto maskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType || maskType.getRank() != kSdpaRank) {
    return false;
  }
  auto maskShape = maskType.getShape();
  if (maskShape[0] != 1 && maskShape[0] != qShape[0]) {
    return false;
  }
  if (maskShape[kNumHeadsDim] != 1 &&
      maskShape[kNumHeadsDim] != qShape[kNumHeadsDim]) {
    return false;
  }
  if (maskShape[kSeqLenDim] != qShape[kSeqLenDim]) {
    return false;
  }
  if (maskShape[kHeadDim] != kSeqLen) {
    return false;
  }
  return true;
}

} // namespace

mlir::LogicalResult
SDPAFusingPattern::matchAndRewrite(MatmulOp srcOp,
                                   mlir::PatternRewriter &rewriter) const {
  // The TTIR matmul op must not transpose either input — SDPA's
  // post-softmax matmul is a plain matmul.
  if (srcOp.getTransposeA() || srcOp.getTransposeB()) {
    return failure();
  }

  SDPAComponents c;
  c.attentionMatmul = srcOp;
  c.value = srcOp.getB();

  // Softmax-precision-out boundary.
  Value softmaxOut = peelOneCast(srcOp.getA());
  auto softmax = softmaxOut.getDefiningOp<SoftmaxOp>();
  if (!softmax) {
    return failure();
  }
  // Softmax dimension must be the last (kv seq len) axis.
  auto smInputType =
      mlir::cast<RankedTensorType>(softmax.getInput().getType());
  int64_t smDim = softmax.getDimension();
  int64_t smRank = smInputType.getRank();
  int64_t normalizedDim = smDim < 0 ? smRank + smDim : smDim;
  if (normalizedDim != smRank - 1) {
    return failure();
  }
  c.softmax = softmax;

  // The softmax result must feed only the attention matmul (allowing for
  // an intermediate single-use typecast peeled above).
  if (!softmax.getResult().hasOneUse()) {
    return failure();
  }

  // Softmax-precision-in boundary, score+mask shape.
  // The chain root may be: LinearOp (combined matmul+mask), AddOp
  // (matmul + add(mask)), or a plain MatmulOp (no mask, optionally
  // wrapped in post-scale). LinearOp is produced by the first sub-phase
  // of TTIRFusing (MatmulWithBiasFusionPattern), which runs before us.
  Value chain = peelOneCast(softmax.getInput());

  if (auto linearOp = chain.getDefiningOp<LinearOp>()) {
    if (linearOp.getTransposeA() || linearOp.getTransposeB()) {
      return failure();
    }
    if (!linearOp.getBias()) {
      return failure();
    }
    c.scoreQRaw = linearOp.getA();
    c.scoreKRaw = linearOp.getB();
    c.mask = linearOp.getBias();
  } else if (auto addOp = chain.getDefiningOp<AddOp>()) {
    auto tryChain = [&](Value candidate, Value other) -> bool {
      auto [stripped, postScale] = peelScale(candidate);
      auto matmul = stripped.getDefiningOp<MatmulOp>();
      if (!matmul || matmul == c.attentionMatmul) {
        return false;
      }
      if (matmul.getTransposeA() || matmul.getTransposeB()) {
        return false;
      }
      c.scoreQRaw = matmul.getA();
      c.scoreKRaw = matmul.getB();
      if (postScale) {
        c.scale = postScale;
      }
      c.mask = other;
      return true;
    };
    if (!tryChain(addOp.getLhs(), addOp.getRhs()) &&
        !tryChain(addOp.getRhs(), addOp.getLhs())) {
      return failure();
    }
  } else {
    auto [stripped, postScale] = peelScale(chain);
    auto matmul = stripped.getDefiningOp<MatmulOp>();
    if (!matmul || matmul == c.attentionMatmul) {
      return failure();
    }
    if (matmul.getTransposeA() || matmul.getTransposeB()) {
      return failure();
    }
    c.scoreQRaw = matmul.getA();
    c.scoreKRaw = matmul.getB();
    if (postScale) {
      c.scale = postScale;
    }
  }

  // Q with optional pre-scale.
  auto [qStripped, qPreScale] = peelScale(c.scoreQRaw);
  c.query = qStripped;

  // K via transpose with optional pre-scale before the transpose.
  auto kTranspose = c.scoreKRaw.getDefiningOp<TransposeOp>();
  if (!kTranspose || !isLastTwoDimsTranspose(kTranspose)) {
    return failure();
  }
  auto [kStripped, kPreScale] = peelScale(kTranspose.getInput());
  c.key = kStripped;

  // Combine scales, reject double-scaling.
  bool hasPostScale = c.scale.has_value();
  bool hasPreScale = qPreScale.has_value() || kPreScale.has_value();
  if (hasPostScale && hasPreScale) {
    return failure();
  }
  if (hasPreScale) {
    c.scale = qPreScale.value_or(1.0f) * kPreScale.value_or(1.0f);
  }

  // Shape validation (strict 4D, no unsqueezing).
  if (!validateShapes(c.query, c.key, c.value)) {
    return failure();
  }
  auto qShape = mlir::cast<RankedTensorType>(c.query.getType()).getShape();
  int64_t kSeqLen =
      mlir::cast<RankedTensorType>(c.key.getType()).getShape()[kSeqLenDim];

  if (c.mask && !validateMask(c.mask, qShape, kSeqLen)) {
    return failure();
  }

  // The op verifier requires full-type equality (element type + encoding),
  // not just the matching shapes validateShapes checked: key == value and
  // query == result. Guard here so an asymmetric typecast on the K-vs-V (or
  // Q-vs-output) paths makes the matcher decline rather than emit an op that
  // fails its own verifier.
  mlir::Type resultType = c.attentionMatmul.getResult().getType();
  if (c.key.getType() != c.value.getType() ||
      c.query.getType() != resultType) {
    return failure();
  }

  // Emit ttir.scaled_dot_product_attention.
  // A matched score chain with no scale op computed unscaled softmax(QKᵀ),
  // i.e. an identity (1.0) scale. The op interprets an *omitted* scale
  // attribute as the default 1/sqrt(D), so always emit an explicit scale to
  // preserve the source semantics: the matched scale when present, else 1.0.
  FloatAttr scaleAttr = rewriter.getF32FloatAttr(c.scale.value_or(1.0f));
  auto newOp = rewriter.create<ScaledDotProductAttentionOp>(
      c.attentionMatmul.getLoc(),
      /*resultType=*/resultType,
      /*query=*/c.query,
      /*key=*/c.key,
      /*value=*/c.value,
      /*attention_mask=*/c.mask,
      /*is_causal=*/rewriter.getBoolAttr(false),
      /*scale=*/scaleAttr,
      /*sliding_window_size=*/IntegerAttr(),
      /*attention_sink=*/Value());

  rewriter.replaceOp(c.attentionMatmul, newOp.getResult());
  return success();
}

} // namespace mlir::tt::ttir::fusing
