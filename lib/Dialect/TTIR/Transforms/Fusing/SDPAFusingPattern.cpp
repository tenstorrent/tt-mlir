// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/SDPAFusingPattern.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

#include <limits>

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

// Strips at most one BroadcastOp. Used to recover the per-row predicate a
// NaN-safety select was broadcast from.
Value peelBroadcast(Value v) {
  if (auto bcast = v.getDefiningOp<BroadcastOp>()) {
    return bcast.getInput();
  }
  return v;
}

// True if v is a constant all-zeros tensor (ttir.zeros, or ttir.full with a
// zero fill), looking through a broadcast.
bool isZerosConstant(Value v) {
  v = peelBroadcast(v);
  if (v.getDefiningOp<ZerosOp>()) {
    return true;
  }
  if (auto full = v.getDefiningOp<FullOp>()) {
    if (auto f = mlir::dyn_cast<FloatAttr>(full.getFillValue())) {
      return f.getValue().isZero();
    }
    if (auto i = mlir::dyn_cast<IntegerAttr>(full.getFillValue())) {
      return i.getValue().isZero();
    }
  }
  return false;
}

// Build the "query row is fully masked" predicate from the additive mask alone,
// independent of the QK^T scores: rowDead[b, ., q, 0] = (max over the kv axis
// of mask == -inf). This is equivalent to the model's score-derived predicate
// (finite_score + (-inf) == -inf, finite_score + 0 stays finite, so
// eq(scores + mask, -inf) == eq(mask, -inf)), but it drops the dependency on
// the score matmul so it can be DCE'd, and the predicate becomes const-foldable
// (causal mask -> all-false -> the NaN-safety select folds away) or at worst
// loop-invariant (dynamic padding mask -> hoisted/CSE'd once). Returns a
// [B, m, Sq, 1] i1 predicate, or null if the mask shape/type is unexpected (the
// caller then falls back to the model's predicate).
Value buildRowFullyMaskedCond(PatternRewriter &rewriter, Location loc,
                              Value mask) {
  auto maskType = mlir::dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType || maskType.getRank() != kSdpaRank ||
      !mlir::isa<FloatType>(maskType.getElementType())) {
    return nullptr;
  }
  llvm::SmallVector<int64_t> reducedShape(maskType.getShape());
  reducedShape.back() = 1; // keep_dim over the kv (last) axis
  auto reducedType =
      RankedTensorType::get(reducedShape, maskType.getElementType());
  auto dimAttr =
      rewriter.getI32ArrayAttr({static_cast<int32_t>(maskType.getRank() - 1)});
  Value rowMax = rewriter.create<MaxOp>(
      loc, reducedType, mask, /*keep_dim=*/rewriter.getBoolAttr(true), dimAttr);
  Value negInf = rewriter.create<FullOp>(
      loc, reducedType,
      rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()));
  // Emit the predicate in the mask's (float) element type, not i1: the TTNN
  // runtime has no Bool dtype (toTTNNDataType rejects i1), and comparison ops
  // that reach it must carry a supported type. This matches how the model's own
  // comparison lowers (ttnn.eq -> bf16) and ttir-predicate handling elsewhere.
  return rewriter.create<EqualOp>(loc, reducedType, rowMax, negInf);
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

// Match a ttir.permute that swaps only the last two dimensions (identity on all
// leading dims). This is the form Kᵀ takes after dot_general decomposition,
// which real models lower to instead of ttir.transpose.
bool isLastTwoDimsPermute(PermuteOp permuteOp) {
  ArrayRef<int64_t> perm = permuteOp.getPermutation();
  int64_t rank = static_cast<int64_t>(perm.size());
  if (rank < 2) {
    return false;
  }
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (perm[i] != i) {
      return false;
    }
  }
  return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2;
}

// A GQA head-expansion on the num-heads dim, optionally wrapped in one
// element-type typecast: either a ttir.repeat_interleave, or HF repeat_kv's
// unsqueeze/broadcast/reshape form before it is canonicalized to one. Real
// models expand K/V from Hkv to Hq heads with this before the score matmul;
// SDPA does GQA natively, so it is peeled.
struct GqaExpansion {
  Value native;    // the un-expanded (Hkv-head) tensor
  TypecastOp cast; // the surrounding cast, or null
  std::optional<uint32_t> repeats;
};

// Detect (without mutating IR) whether v is a head-dim repeat_interleave,
// optionally under a single typecast. Returns repeats == nullopt if not.
GqaExpansion detectGqaExpansion(Value v) {
  TypecastOp cast = v.getDefiningOp<TypecastOp>();
  Value underCast = cast ? cast.getInput() : v;
  if (auto repeatOp = underCast.getDefiningOp<RepeatInterleaveOp>()) {
    auto inType =
        mlir::dyn_cast<RankedTensorType>(repeatOp.getInput().getType());
    if (!inType) {
      return {v, nullptr, std::nullopt};
    }
    int64_t dim = repeatOp.getDim();
    int64_t rank = inType.getRank();
    int64_t normDim = dim < 0 ? rank + dim : dim;
    if (normDim != kNumHeadsDim) {
      return {v, nullptr, std::nullopt};
    }
    return {repeatOp.getInput(), cast, repeatOp.getRepeats()};
  }

  // Also match HF repeat_kv's pre-canonicalization form (unsqueeze/broadcast/
  // reshape). Its reshape/broadcast -> repeat_interleave canonicalization is
  // co-scheduled with this fuser and may not have run yet; unless it is peeled
  // here, SDPA is handed an Hq-head copy instead of the grouped Hkv-head cache
  // (a large f32 GQA expansion per decode step).
  auto finalReshape = underCast.getDefiningOp<ReshapeOp>();
  if (!finalReshape) {
    return {v, nullptr, std::nullopt};
  }
  auto bcast = finalReshape.getInput().getDefiningOp<BroadcastOp>();
  if (!bcast || !bcast->hasOneUse()) {
    return {v, nullptr, std::nullopt};
  }
  Value dimInserted = bcast.getInput();
  Value native;
  if (auto unsq = dimInserted.getDefiningOp<UnsqueezeOp>()) {
    native = unsq.getInput();
  } else if (auto rs = dimInserted.getDefiningOp<ReshapeOp>()) {
    native = rs.getInput();
  } else {
    return {v, nullptr, std::nullopt};
  }
  auto nativeType = mlir::dyn_cast<RankedTensorType>(native.getType());
  auto outType =
      mlir::dyn_cast<RankedTensorType>(finalReshape.getResult().getType());
  if (!nativeType || !outType || nativeType.getRank() != kSdpaRank ||
      outType.getRank() != kSdpaRank) {
    return {v, nullptr, std::nullopt};
  }
  // The broadcast must expand exactly one dim, positioned right after the heads
  // dim (the GQA group axis).
  auto bdims = bcast.getBroadcastDimensions();
  int64_t insertedDim = -1;
  int64_t repeat = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(bdims.size()); ++i) {
    if (bdims[i] == 1) {
      continue;
    }
    if (insertedDim != -1) {
      return {v, nullptr, std::nullopt};
    }
    insertedDim = i;
    repeat = bdims[i];
  }
  if (insertedDim != kNumHeadsDim + 1 || repeat <= 1) {
    return {v, nullptr, std::nullopt};
  }
  // Net effect must be exactly Hkv -> Hkv*repeat on the heads dim, every other
  // dim unchanged.
  auto nShape = nativeType.getShape();
  auto oShape = outType.getShape();
  if (oShape[kNumHeadsDim] != nShape[kNumHeadsDim] * repeat ||
      oShape[0] != nShape[0] || oShape[kSeqLenDim] != nShape[kSeqLenDim] ||
      oShape[kHeadDim] != nShape[kHeadDim]) {
    return {v, nullptr, std::nullopt};
  }
  return {native, cast, static_cast<uint32_t>(repeat)};
}

// True if the slice keeps [0 : lastDim-1] of the last dim and is otherwise a
// full, unit-step slice — i.e. it drops exactly the last column. Used to peel
// the attention-sink softmax-padding slice.
bool isDropLastColumnSlice(SliceStaticOp sliceOp) {
  auto inType = mlir::dyn_cast<RankedTensorType>(sliceOp.getInput().getType());
  if (!inType) {
    return false;
  }
  ArrayAttr begins = sliceOp.getBeginsAttr();
  ArrayAttr ends = sliceOp.getEndsAttr();
  ArrayAttr steps = sliceOp.getStepAttr();
  int64_t rank = inType.getRank();
  if (static_cast<int64_t>(begins.size()) != rank ||
      static_cast<int64_t>(ends.size()) != rank ||
      static_cast<int64_t>(steps.size()) != rank) {
    return false;
  }
  ArrayRef<int64_t> shape = inType.getShape();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t b = mlir::cast<IntegerAttr>(begins[i]).getInt();
    int64_t e = mlir::cast<IntegerAttr>(ends[i]).getInt();
    int64_t s = mlir::cast<IntegerAttr>(steps[i]).getInt();
    int64_t expectedEnd = (i == rank - 1) ? shape[i] - 1 : shape[i];
    if (b != 0 || s != 1 || e != expectedEnd) {
      return false;
    }
  }
  return rank >= 1;
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

  // Optional attention-sink "softmax padding column": the sink logit is
  // concat'd as an extra score column before softmax, then the column is
  // sliced off after. Peel the trailing slice here; the matching concat at the
  // softmax input is peeled below and the sink fed to the op.
  Value attentionSink;
  bool expectSinkConcat = false;
  if (auto sliceOp = softmaxOut.getDefiningOp<SliceStaticOp>()) {
    if (isDropLastColumnSlice(sliceOp)) {
      softmaxOut = peelOneCast(sliceOp.getInput());
      expectSinkConcat = true;
    }
  }

  // Optional NaN-safety select: where(rowCond, 0, softmax). A fully-masked
  // query row makes softmax produce NaN; models scrub it with this select.
  // SDPA itself is not NaN-safe (matches PyTorch / tt-metal), so instead of
  // dropping the select we re-apply it to the fused op's output (see below).
  // Sound only when it zeros whole query rows — i.e. the predicate is broadcast
  // across the kv (last) axis, so zeroing weight rows equals zeroing the
  // matmul's output rows.
  Value nanSafeRowCond;
  if (auto whereOp = softmaxOut.getDefiningOp<WhereOp>()) {
    Value rowCond = peelBroadcast(whereOp.getFirst());
    auto condType = mlir::dyn_cast<RankedTensorType>(rowCond.getType());
    bool rowUniform = condType && condType.getRank() == kSdpaRank &&
                      condType.getShape().back() == 1;
    if (rowUniform && isZerosConstant(whereOp.getSecond())) {
      nanSafeRowCond = rowCond;
      softmaxOut = peelOneCast(whereOp.getThird());
    }
  }

  auto softmax = softmaxOut.getDefiningOp<SoftmaxOp>();
  if (!softmax) {
    return failure();
  }
  // Softmax dimension must be the last (kv seq len) axis.
  auto smInputType = mlir::cast<RankedTensorType>(softmax.getInput().getType());
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

  // For the attention-sink pattern the softmax input is concat(scores, sink),
  // appended along the kv axis. Peel the concat to recover the real score chain
  // and extract the sink (broadcast back to [1, Hq, 1, 1], which is what the op
  // expects — a head-granular scalar that the GQA repeat_interleave already
  // produced).
  if (expectSinkConcat) {
    auto concatOp = chain.getDefiningOp<ConcatOp>();
    if (!concatOp || concatOp.getInputs().size() != 2) {
      return failure();
    }
    auto concatType =
        mlir::cast<RankedTensorType>(concatOp.getResult().getType());
    int64_t concatRank = concatType.getRank();
    int64_t concatDim = concatOp.getDim() < 0 ? concatRank + concatOp.getDim()
                                              : concatOp.getDim();
    if (concatDim != concatRank - 1) {
      return failure();
    }
    Value sinkCol = concatOp.getInputs()[1];
    auto sinkColType = mlir::dyn_cast<RankedTensorType>(sinkCol.getType());
    if (!sinkColType || sinkColType.getRank() != kSdpaRank ||
        sinkColType.getShape().back() != 1) {
      return failure();
    }
    attentionSink = peelBroadcast(sinkCol);
    auto sinkType = mlir::dyn_cast<RankedTensorType>(attentionSink.getType());
    // The op expects a [1, Hq, 1, 1] sink (one scalar per query head).
    if (!sinkType || sinkType.getRank() != kSdpaRank ||
        sinkType.getShape()[0] != 1 ||
        sinkType.getShape()[kNumHeadsDim] !=
            concatType.getShape()[kNumHeadsDim] ||
        sinkType.getShape()[kSeqLenDim] != 1 ||
        sinkType.getShape()[kHeadDim] != 1) {
      return failure();
    }
    chain = peelOneCast(concatOp.getInputs()[0]);
  }

  // The Kᵀ transpose in the score matmul may be a standalone ttir.transpose /
  // ttir.permute on the K operand, OR folded into the matmul's transpose_b
  // attribute (what PermuteMatmulFusion produces when enabled). Both compute
  // Q·Kᵀ; when folded, the B operand is the un-transposed K. Track which form
  // the score op is in so K extraction below handles it correctly.
  bool scoreKTransposed = false;

  if (auto linearOp = chain.getDefiningOp<LinearOp>()) {
    // A transposed A would be Qᵀ (not SDPA); a transposed B is the folded Kᵀ.
    if (linearOp.getTransposeA()) {
      return failure();
    }
    if (!linearOp.getBias()) {
      return failure();
    }
    c.scoreQRaw = linearOp.getA();
    c.scoreKRaw = linearOp.getB();
    scoreKTransposed = linearOp.getTransposeB();
    c.mask = linearOp.getBias();
  } else if (auto addOp = chain.getDefiningOp<AddOp>()) {
    auto tryChain = [&](Value candidate, Value other) -> bool {
      auto [stripped, postScale] = peelScale(candidate);
      auto matmul = stripped.getDefiningOp<MatmulOp>();
      if (!matmul || matmul == c.attentionMatmul) {
        return false;
      }
      if (matmul.getTransposeA()) {
        return false;
      }
      c.scoreQRaw = matmul.getA();
      c.scoreKRaw = matmul.getB();
      scoreKTransposed = matmul.getTransposeB();
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
    if (matmul.getTransposeA()) {
      return failure();
    }
    c.scoreQRaw = matmul.getA();
    c.scoreKRaw = matmul.getB();
    scoreKTransposed = matmul.getTransposeB();
    if (postScale) {
      c.scale = postScale;
    }
  }

  // Q with optional pre-scale.
  auto [qStripped, qPreScale] = peelScale(c.scoreQRaw);
  c.query = qStripped;

  // K via a last-two-dims transpose, expressed either as ttir.transpose or as
  // ttir.permute (the form dot_general decomposition produces). A K scale may
  // sit on either side of the transpose: multiply(transpose(K)) (post — the
  // real-model form) or transpose(multiply(K)) (pre).
  auto [kOuterStripped, kScaleAfter] = peelScale(c.scoreKRaw);
  Value kTransposedInput;
  if (scoreKTransposed) {
    // The score matmul folded Kᵀ into transpose_b=true, so its B operand is the
    // un-transposed K itself — there is no explicit transpose/permute op.
    kTransposedInput = kOuterStripped;
  } else if (auto kTranspose = kOuterStripped.getDefiningOp<TransposeOp>()) {
    if (!isLastTwoDimsTranspose(kTranspose)) {
      return failure();
    }
    kTransposedInput = kTranspose.getInput();
  } else if (auto kPermute = kOuterStripped.getDefiningOp<PermuteOp>()) {
    if (!isLastTwoDimsPermute(kPermute)) {
      return failure();
    }
    kTransposedInput = kPermute.getInput();
  } else {
    return failure();
  }
  auto [kStripped, kScaleBefore] = peelScale(kTransposedInput);
  c.key = kStripped;

  // A K scale may appear before and/or after the transpose; both are equivalent
  // K-side pre-scales, so fold them into one.
  std::optional<float> kPreScale;
  if (kScaleAfter.has_value() || kScaleBefore.has_value()) {
    kPreScale = kScaleAfter.value_or(1.0f) * kScaleBefore.value_or(1.0f);
  }

  // Combine scales, reject double-scaling.
  bool hasPostScale = c.scale.has_value();
  bool hasPreScale = qPreScale.has_value() || kPreScale.has_value();
  if (hasPostScale && hasPreScale) {
    return failure();
  }
  if (hasPreScale) {
    c.scale = qPreScale.value_or(1.0f) * kPreScale.value_or(1.0f);
  }

  // GQA: models expand K/V from Hkv to Hq heads with a head-dim
  // repeat_interleave before the score matmul. SDPA handles Hkv < Hq natively,
  // so peel a matching expansion from both K and V (only when both match, to
  // keep their head counts equal) and feed the un-expanded tensors. Any
  // element-type cast that sat outside the expansion is re-applied on the
  // smaller native tensor so K/V keep the element type they had post-expansion.
  GqaExpansion kGqa = detectGqaExpansion(c.key);
  GqaExpansion vGqa = detectGqaExpansion(c.value);
  if (kGqa.repeats.has_value() && vGqa.repeats.has_value() &&
      *kGqa.repeats == *vGqa.repeats) {
    auto reapplyCast = [&](GqaExpansion g) -> Value {
      if (!g.cast) {
        return g.native;
      }
      auto nativeType = mlir::cast<RankedTensorType>(g.native.getType());
      auto castElemType =
          mlir::cast<RankedTensorType>(g.cast.getType()).getElementType();
      auto nativeCastType = RankedTensorType::get(
          nativeType.getShape(), castElemType, nativeType.getEncoding());
      return rewriter.create<TypecastOp>(g.cast.getLoc(), nativeCastType,
                                         g.native);
    };
    c.key = reapplyCast(kGqa);
    c.value = reapplyCast(vGqa);
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
  if (c.key.getType() != c.value.getType() || c.query.getType() != resultType) {
    return failure();
  }

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
      /*attention_sink=*/attentionSink);

  Value result = newOp.getResult();
  if (nanSafeRowCond) {
    // Re-apply the NaN-safety select on the SDPA output. Zeroing whole rows of
    // the attention weights equals zeroing the same output rows (the matmul
    // contracts over the kv axis), so
    //   matmul(where(rowCond, 0, softmax), V) == where(rowCond, 0, sdpa).
    // rowCond is [B, H, Sq, 1]; broadcast it across the output head dim.
    auto loc = c.attentionMatmul.getLoc();
    auto outType = mlir::cast<RankedTensorType>(resultType);

    // Prefer a mask-derived predicate: it is equivalent to the model's
    // score-derived one but does not depend on QK^T, so the score matmul is
    // freed to DCE and the predicate const-folds (causal masks) / hoists
    // (dynamic masks). Fall back to the model's predicate if the mask is
    // absent or an unexpected shape.
    Value cond = nanSafeRowCond;
    if (c.mask) {
      if (Value fromMask = buildRowFullyMaskedCond(rewriter, loc, c.mask)) {
        cond = fromMask;
      }
    }

    auto condType = mlir::cast<RankedTensorType>(cond.getType());
    auto condOutType =
        RankedTensorType::get(outType.getShape(), condType.getElementType());
    Value condBcast = rewriter.create<BroadcastOp>(
        loc, condOutType, cond,
        ttmlir::utils::getBroadcastDimensions<int64_t>(condType.getShape(),
                                                       outType.getShape()));
    Value zeros = rewriter.create<ZerosOp>(
        loc, outType, llvm::to_vector_of<int32_t>(outType.getShape()));
    result = rewriter.create<WhereOp>(loc, outType, condBcast, zeros, result);
  }

  rewriter.replaceOp(c.attentionMatmul, result);
  return success();
}

} // namespace mlir::tt::ttir::fusing
