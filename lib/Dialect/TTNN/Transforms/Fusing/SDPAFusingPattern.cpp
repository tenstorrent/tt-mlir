// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::fusing {

// SDPA Query, Key, Value tensors have shape [B, H, S, D]
// (Batch, NumHeads, SeqLen, HeadDim).
static constexpr int64_t kNumHeadsDim = 1;
static constexpr int64_t kSeqLenDim = 2;

// Permutation to convert query from [B, H, S, D] -> [S, B, H, D] for SDPA
// decode op.
static constexpr std::array<int64_t, 4> kToDecodePermutation = {2, 0, 1, 3};

struct SDPAFusing::SDPAComponents {
  Value query, key, value, mask;
  Value attentionSink;
  std::optional<float> scale;
  MatmulOp attentionMatmul;
  SoftmaxOp softmax;
  Operation *scoreOp = nullptr;
};

// Normalize a shape to 4D by prepending 1s. Used for validation only, does not
// mutate the IR.
static SmallVector<int64_t> normalizeTo4D(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> result(4 - shape.size(), 1);
  result.append(shape.begin(), shape.end());
  return result;
}

// Squeeze the SDPA output back to the original rank if the inputs were
// unsqueezed from < 4D.
static Value squeezeToOriginalRank(Value v, RankedTensorType originalType,
                                   PatternRewriter &rewriter, Location loc) {
  auto currentType = mlir::cast<RankedTensorType>(v.getType());
  if (currentType.getRank() == originalType.getRank()) {
    return v;
  }
  auto newType = utils::RankedTensorTypeFactory::create(
      currentType, originalType.getShape());
  SmallVector<int32_t> shapeAttr(originalType.getShape().begin(),
                                 originalType.getShape().end());
  return rewriter
      .create<ReshapeOp>(loc, newType, v, rewriter.getI32ArrayAttr(shapeAttr),
                         /*memory_config=*/MemoryConfigAttr())
      .getResult();
}

// ============================================================================
// Layout / Transpose Utilities
// ============================================================================

bool SDPAFusing::isTransposeOnLastTwoDims(ArrayRef<int64_t> perm) {
  if (perm.size() < 2) {
    return false;
  }

  size_t n = perm.size();
  for (size_t i = 0; i < n - 2; ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }

  return perm[n - 2] == static_cast<int64_t>(n - 1) &&
         perm[n - 1] == static_cast<int64_t>(n - 2);
}

bool SDPAFusing::isKeyTransposed(Value key, Value query, Value value) const {
  Operation *defOp = key.getDefiningOp();
  if (auto splitOp =
          dyn_cast_or_null<SplitQueryKeyValueAndSplitHeadsOp>(defOp)) {
    return splitOp.getTransposeKey();
  }

  auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
  auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
  auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

  if (!kType || !qType || !vType) {
    return false;
  }

  auto kRank = kType.getRank();
  auto qRank = qType.getRank();
  auto vRank = vType.getRank();

  if (kRank < 3 || kRank > 4 || qRank < 3 || qRank > 4 || vRank < 3 ||
      vRank > 4) {
    return false;
  }

  auto kShape = normalizeTo4D(kType.getShape());
  auto qShape = normalizeTo4D(qType.getShape());
  auto vShape = normalizeTo4D(vType.getShape());

  int64_t qHeadDim = qShape[3];
  int64_t vSeqLen = vShape[kSeqLenDim];

  bool kDim2MatchesHeadDim = kShape[2] == qHeadDim;
  bool kDim3MatchesSeqLen = kShape[3] == vSeqLen;

  return kDim2MatchesHeadDim && kDim3MatchesSeqLen;
}

// ============================================================================
// Constant Extraction
// ============================================================================

std::optional<float> SDPAFusing::extractConstant(Value v) const {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
      return attr.getValue().convertToFloat();
    }
  }

  if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
    auto callee = loadCached.getCallee();
    auto moduleOp = loadCached->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return std::nullopt;
    }

    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
    if (!funcOp) {
      return std::nullopt;
    }

    std::optional<float> result;
    funcOp.walk([&](FullOp fullOp) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        result = attr.getValue().convertToFloat();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  return std::nullopt;
}

std::pair<Value, std::optional<float>>
SDPAFusing::extractMultiplyWithConstant(Value v) const {
  if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
    if (auto s = extractConstant(mulOp.getRhs())) {
      return {mulOp.getLhs(), s};
    }
    if (auto s = extractConstant(mulOp.getLhs())) {
      return {mulOp.getRhs(), s};
    }
  }
  return {v, std::nullopt};
}

// ============================================================================
// Pattern Matching with Backtracking
// ============================================================================

bool SDPAFusing::matchSoftmaxPath(Value v, SDPAComponents &c) const {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  // Peel optional slice_static. Some frontends pad the softmax input with
  // extra columns via concat for numeric stability (preventing NaN when all
  // attention scores are masked to -inf), then slice off the padding after
  // softmax. The hardware SDPA op handles this internally, so we look through
  // the slice here and the corresponding concat in matchScoreComputation.
  //
  // Verify the slice trims only the last dimension (starts at 0 with step 1
  // and produces a smaller extent than the input).
  if (auto sliceOp = v.getDefiningOp<SliceStaticOp>()) {
    auto inputType = mlir::cast<RankedTensorType>(sliceOp.getInput().getType());
    int64_t lastDim = inputType.getRank() - 1;

    auto getI32 = [](ArrayAttr attr, int64_t idx) -> int32_t {
      return mlir::cast<IntegerAttr>(attr[idx]).getInt();
    };

    bool isLastDimTrim =
        getI32(sliceOp.getBegins(), lastDim) == 0 &&
        getI32(sliceOp.getStep(), lastDim) == 1 &&
        getI32(sliceOp.getEnds(), lastDim) < inputType.getShape()[lastDim];
    if (isLastDimTrim) {
      v = ttmlir::utils::lookThrough<TypecastOp>(sliceOp.getInput());
    }
  }

  // Peel optional where. Some frontends wrap softmax output with
  // where(cond, zeros, softmax) to replace NaN rows with zeros when all
  // attention scores in a row are masked to -inf. The third operand carries
  // the actual softmax result.
  if (auto whereOp = v.getDefiningOp<WhereOp>()) {
    v = ttmlir::utils::lookThrough<TypecastOp>(whereOp.getThird());
  }

  if (auto softmax = v.getDefiningOp<SoftmaxOp>()) {
    c.softmax = softmax;
    return true;
  }

  return false;
}

bool SDPAFusing::matchScoreComputation(Value v, SDPAComponents &c) const {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  // Peel optional concat — the other half of the softmax padding pattern
  // described in matchSoftmaxPath. The concat appends extra columns to the
  // score tensor before softmax; we skip it to reach the actual scores.
  // Save the padding tensor as the attention sink for the SDPA decode op.
  //
  // Verify the concat is on the last dimension with exactly two inputs.
  if (auto concatOp = v.getDefiningOp<ConcatOp>()) {
    auto resultType =
        mlir::cast<RankedTensorType>(concatOp.getResult().getType());
    int64_t lastDim = resultType.getRank() - 1;

    if (concatOp.getInputs().size() == 2 && concatOp.getDim() == lastDim) {
      v = ttmlir::utils::lookThrough<TypecastOp>(concatOp.getInputs()[0]);
      c.attentionSink =
          ttmlir::utils::lookThrough<TypecastOp>(concatOp.getInputs()[1]);
    }
  }

  if (auto linearOp = v.getDefiningOp<LinearOp>()) {
    c.scoreOp = linearOp;
    c.query = linearOp.getA();
    c.key = linearOp.getB();
    if (linearOp.getBias()) {
      c.mask = linearOp.getBias();
    }
    return true;
  }

  if (auto addOp = v.getDefiningOp<AddOp>()) {
    if (matchScoreChain(addOp.getLhs(), c)) {
      c.mask = addOp.getRhs();
      return true;
    }
    if (matchScoreChain(addOp.getRhs(), c)) {
      c.mask = addOp.getLhs();
      return true;
    }
    return false;
  }

  return matchScoreChain(v, c);
}

bool SDPAFusing::matchScoreChain(Value v, SDPAComponents &c) const {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
    if (auto scale = extractConstant(mulOp.getRhs())) {
      c.scale = scale;
      v = ttmlir::utils::lookThrough<TypecastOp>(mulOp.getLhs());
    } else if (auto scale = extractConstant(mulOp.getLhs())) {
      c.scale = scale;
      v = ttmlir::utils::lookThrough<TypecastOp>(mulOp.getRhs());
    }
  }

  else if (auto divOp = v.getDefiningOp<DivideOp>()) {
    if (auto divisor = extractConstant(divOp.getRhs())) {
      if (*divisor != 0.0f) {
        c.scale = 1.0f / *divisor;
        v = ttmlir::utils::lookThrough<TypecastOp>(divOp.getLhs());
      }
    }
  }

  if (auto matmul = v.getDefiningOp<MatmulOp>()) {
    if (matmul != c.attentionMatmul) {
      c.scoreOp = matmul;
      c.query = matmul.getA();
      c.key = matmul.getB();
      return true;
    }
  }

  return false;
}

// ============================================================================
// Input Canonicalization
// ============================================================================

std::pair<Value, std::optional<float>> SDPAFusing::analyzeQ(Value v) const {
  // Look through typecast to find multiply-with-constant (pre-scale on Q).
  Value skipped = ttmlir::utils::lookThrough<TypecastOp>(v);
  auto [inner, scale] = extractMultiplyWithConstant(skipped);
  if (scale) {
    v = inner;
  }

  if (auto typecastOp = v.getDefiningOp<TypecastOp>()) {
    v = typecastOp.getInput();
  }

  return {v, scale};
}

std::tuple<Value, bool, std::optional<float>>
SDPAFusing::analyzeK(Value v) const {
  bool skippedTranspose = false;
  std::optional<float> scale;

  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp>(defOp)) {
      v = defOp->getOperand(0);
      continue;
    }

    // Walk through multiply-with-constant (pre-scale on K).
    // The elementwise commute pass may place this before or after the permute.
    if (!scale) {
      auto [inner, s] = extractMultiplyWithConstant(v);
      if (s) {
        scale = s;
        v = inner;
        continue;
      }
    }

    if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
      if (repeatOp.getDim() == kNumHeadsDim) {
        v = repeatOp.getInput();
        continue;
      }
      break;
    }

    if (auto permuteOp = dyn_cast<PermuteOp>(defOp)) {
      if (isTransposeOnLastTwoDims(permuteOp.getPermutation())) {
        v = permuteOp.getInput();
        skippedTranspose = true;
        continue;
      }
    }

    break;
  }

  return {v, skippedTranspose, scale};
}

Value SDPAFusing::analyzeV(Value v) const {
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp>(defOp)) {
      v = defOp->getOperand(0);
      continue;
    }

    if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
      if (repeatOp.getDim() == kNumHeadsDim) {
        v = repeatOp.getInput();
        continue;
      }
      break;
    }

    break;
  }

  return v;
}

// Prepare SDPA inputs for the hardware op. This involves several steps:
//
//  1. Analyze Q, K, V to look through typecasts, repeat-interleave, multiply
//     (pre-scale), and permute ops, extracting pre-scales and detecting key
//     transposition.
//  2. Combine pre-scales (from Q and K) into c.scale.
//  3. Validate and accept the "prepared" (looked-through) versions of Q, K, V
//     if their shapes are compatible.
//  4. Un-transpose K if needed ([B, H, D, S] -> [B, H, S, D]).
//  5. Unsqueeze the attention mask to 4D if it is lower-rank.
//  6. Handle attention sink batch dimension from load_cached ops.
//  7. Unsqueeze Q, K, V to 4D by prepending 1s if they are 3D. Some frontends
//     squeeze away the head dimension when num_heads=1.
bool SDPAFusing::prepareInputsForSDPA(SDPAComponents &c,
                                      PatternRewriter &rewriter) const {
  auto [preparedQ, qPreScale] = analyzeQ(c.query);
  auto [preparedK, skippedKTranspose, kPreScale] = analyzeK(c.key);
  Value preparedV = analyzeV(c.value);

  // Reject ambiguous double-scaling (both post-matmul and pre-matmul scales).
  bool hasPostScale = c.scale.has_value();
  bool hasPreScale = qPreScale.has_value() || kPreScale.has_value();
  if (hasPostScale && hasPreScale) {
    return false;
  }

  // Combine pre-scales into c.scale.
  if (hasPreScale) {
    c.scale = qPreScale.value_or(1.0f) * kPreScale.value_or(1.0f);
  }

  if (validateShapes(preparedQ, c.key, c.value)) {
    c.query = preparedQ;
  }

  if (validateShapes(c.query, preparedK, preparedV)) {
    c.key = preparedK;
    c.value = preparedV;
  }

  if (!skippedKTranspose) {
    c.key = unTransposeKeyIfNeeded(c.query, c.key, c.value, rewriter,
                                   c.attentionMatmul.getLoc());
  }

  if (c.mask) {
    c.mask = ttmlir::utils::lookThrough<TypecastOp, RepeatOp>(c.mask);

    // tt-metal requires the attention mask to be a 4D tensor. TTIR fusing can
    // produce lower-rank masks when it folds matmul+add into linear and drops
    // the reshape to 4D. Unsqueeze leading dimensions to reach rank 4.
    auto maskType = mlir::cast<RankedTensorType>(c.mask.getType());
    if (maskType.getRank() < 4) {
      SmallVector<int64_t> newShape(4 - maskType.getRank(), 1);
      newShape.append(maskType.getShape().begin(), maskType.getShape().end());
      auto newType = utils::RankedTensorTypeFactory::create(maskType, newShape);
      SmallVector<int32_t> shapeAttr(newShape.begin(), newShape.end());
      c.mask =
          rewriter
              .create<ReshapeOp>(c.attentionMatmul.getLoc(), newType, c.mask,
                                 rewriter.getI32ArrayAttr(shapeAttr),
                                 /*memory_config=*/MemoryConfigAttr())
              .getResult();
    }
  }

  if (c.attentionSink) {
    c.attentionSink =
        ttmlir::utils::lookThrough<TypecastOp, RepeatOp>(c.attentionSink);

    // If the sink still has a batch dimension > 1, it may come from a
    // load_cached op where the repeat is baked inside the const_eval function.
    // Look inside the const_eval, strip the repeat from its return value, and
    // update the load_cached result type to match.
    auto sinkType = mlir::cast<RankedTensorType>(c.attentionSink.getType());
    if (sinkType.getRank() >= 1 && sinkType.getShape()[0] > 1) {
      if (auto loadCached =
              c.attentionSink.getDefiningOp<ttcore::LoadCachedOp>()) {
        auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            loadCached, loadCached.getCalleeAttr());
        if (funcOp) {
          unsigned resultIdx =
              mlir::cast<OpResult>(c.attentionSink).getResultNumber();
          auto &block = funcOp.getBody().front();
          auto returnOp = cast<func::ReturnOp>(block.getTerminator());
          Value innerV = returnOp.getOperand(resultIdx);

          // Walk back through repeat inside the const_eval body.
          Value stripped =
              ttmlir::utils::lookThrough<TypecastOp, RepeatOp>(innerV);
          if (stripped != innerV) {
            // Update the const_eval function: return the pre-repeat value.
            returnOp.setOperand(resultIdx, stripped);

            auto strippedType =
                mlir::cast<RankedTensorType>(stripped.getType());

            // Update function signature return type.
            auto funcType = funcOp.getFunctionType();
            SmallVector<Type> newResultTypes(funcType.getResults());
            newResultTypes[resultIdx] = strippedType;
            funcOp.setType(FunctionType::get(
                funcOp.getContext(), funcType.getInputs(), newResultTypes));

            // Update load_cached result type.
            loadCached.getResult(resultIdx).setType(strippedType);
            c.attentionSink = loadCached.getResult(resultIdx);
          }
        }
      }
    }
  }

  // Unsqueeze Q, K, V to 4D by prepending 1s if needed. The SDPA op requires
  // 4D inputs [B, H, S, D], but some frontends squeeze away the head dimension
  // when num_heads=1, producing 3D tensors.
  auto unsqueezeTo4D = [&](Value v) -> Value {
    auto type = mlir::cast<RankedTensorType>(v.getType());
    if (type.getRank() >= 4) {
      return v;
    }
    SmallVector<int64_t> newShape(4 - type.getRank(), 1);
    newShape.append(type.getShape().begin(), type.getShape().end());
    auto newType = utils::RankedTensorTypeFactory::create(type, newShape);
    SmallVector<int32_t> shapeAttr(newShape.begin(), newShape.end());
    return rewriter
        .create<ReshapeOp>(c.attentionMatmul.getLoc(), newType, v,
                           rewriter.getI32ArrayAttr(shapeAttr),
                           /*memory_config=*/MemoryConfigAttr())
        .getResult();
  };

  c.query = unsqueezeTo4D(c.query);
  c.key = unsqueezeTo4D(c.key);
  c.value = unsqueezeTo4D(c.value);

  return true;
}

// ============================================================================
// Key Un-transpose
// ============================================================================

Value SDPAFusing::unTransposeKeyIfNeeded(Value query, Value key, Value value,
                                         mlir::PatternRewriter &rewriter,
                                         Location loc) const {
  if (!isKeyTransposed(key, query, value)) {
    return key;
  }

  // Use a rank-appropriate permutation to swap the last two dims.
  auto keyRank = mlir::cast<RankedTensorType>(key.getType()).getRank();
  SmallVector<int64_t> perm;
  for (int64_t i = 0; i < keyRank; ++i) {
    perm.push_back(i);
  }
  std::swap(perm[keyRank - 2], perm[keyRank - 1]);

  return ttir_to_ttnn::utils::generatePermute(
      mlir::cast<TypedValue<RankedTensorType>>(key), perm, rewriter, loc);
}

// ============================================================================
// Validation
// ============================================================================

bool SDPAFusing::validateShapes(Value query, Value key, Value value) const {
  if (!query || !key || !value) {
    return false;
  }

  auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
  auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
  auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

  if (!qType || !kType || !vType) {
    return false;
  }

  auto qRank = qType.getRank();
  auto kRank = kType.getRank();
  auto vRank = vType.getRank();

  if (qRank < 3 || qRank > 4 || kRank < 3 || kRank > 4 || vRank < 3 ||
      vRank > 4) {
    return false;
  }

  auto qShape = normalizeTo4D(qType.getShape());
  auto kShape = normalizeTo4D(kType.getShape());
  auto vShape = normalizeTo4D(vType.getShape());

  int64_t qHeadDim = qShape[3];
  int64_t vSeqLen = vShape[kSeqLenDim];
  int64_t vHeadDim = vShape[3];

  bool keyTransposed = isKeyTransposed(key, query, value);
  int64_t kSeqLen = keyTransposed ? kShape[3] : kShape[kSeqLenDim];
  int64_t kHeadDim = keyTransposed ? kShape[kSeqLenDim] : kShape[3];

  if (kSeqLen != vSeqLen) {
    return false;
  }

  if (qHeadDim != kHeadDim || kHeadDim != vHeadDim) {
    return false;
  }

  if (qShape[0] != kShape[0] || kShape[0] != vShape[0]) {
    return false;
  }

  int64_t qNumHeads = qShape[kNumHeadsDim];
  int64_t kNumHeads = kShape[kNumHeadsDim];
  int64_t vNumHeads = vShape[kNumHeadsDim];

  if (kNumHeads != vNumHeads) {
    return false;
  }

  if (qNumHeads % kNumHeads != 0) {
    return false;
  }

  return true;
}

bool SDPAFusing::validateSemantics(const SDPAComponents &c) const {
  if (!c.query || !c.key || !c.value || !c.softmax || !c.scoreOp) {
    return false;
  }

  if (!validateShapes(c.query, c.key, c.value)) {
    return false;
  }

  if (!c.softmax->hasOneUse()) {
    return false;
  }

  if (auto *softmaxUser = *c.softmax->getUsers().begin()) {
    if (isa<TypecastOp>(softmaxUser) && !softmaxUser->hasOneUse()) {
      return false;
    }
  }

  return true;
}

// ============================================================================
// matchAndRewrite
// ============================================================================

mlir::LogicalResult
SDPAFusing::matchAndRewrite(MatmulOp srcOp,
                            mlir::PatternRewriter &rewriter) const {
  SDPAComponents c;
  c.attentionMatmul = srcOp;
  c.value = srcOp.getB();

  if (!matchSoftmaxPath(srcOp.getA(), c)) {
    return failure();
  }

  if (!matchScoreComputation(c.softmax.getInput(), c)) {
    return failure();
  }

  if (!validateSemantics(c)) {
    return failure();
  }

  if (!prepareInputsForSDPA(c, rewriter)) {
    return failure();
  }

  return createSDPAOp(rewriter, c);
}

// ============================================================================
// Op Creation
// ============================================================================

mlir::LogicalResult SDPAFusing::createSDPAOp(mlir::PatternRewriter &rewriter,
                                             SDPAComponents &c) const {
  float scale = c.scale.value_or(1.0f);
  FloatAttr scaleAttr = rewriter.getF32FloatAttr(scale);

  auto originalOutputType =
      mlir::cast<RankedTensorType>(c.attentionMatmul.getResult().getType());

  // Workaround for https://github.com/tenstorrent/tt-metal/issues/40470:
  // tt-metal's SDPA kernel computes exp((sink - max) * scale), applying scale
  // to the attention sink. But the sink is already in scaled score space, so
  // this double-scales it. Pre-multiply the sink by 1/scale to compensate.
  if (c.attentionSink && scale != 0.0f && scale != 1.0f) {
    auto sinkType = mlir::cast<RankedTensorType>(c.attentionSink.getType());
    float invScale = 1.0f / scale;
    auto deviceOp =
        utils::getOrInsertDevice(rewriter, c.attentionMatmul.getOperation());
    auto invScaleOp = rewriter.create<FullOp>(
        c.attentionMatmul.getLoc(), sinkType,
        rewriter.getF32FloatAttr(invScale), deviceOp.getResult());
    c.attentionSink =
        rewriter.create<MultiplyOp>(c.attentionMatmul.getLoc(), sinkType,
                                    c.attentionSink, invScaleOp.getResult());
  }

  auto qType = mlir::cast<RankedTensorType>(c.query.getType());
  auto qShape = qType.getShape();

  FusionValidator validator(rewriter.getContext(), validationConfig);

  auto kType = mlir::cast<RankedTensorType>(c.key.getType());
  // SDPA decode requires kvSeqLen to be a multiple of 32 (k_chunk_size
  // constraint).
  int64_t kvSeqLen =
      kType.getShape().size() >= 3 ? kType.getShape()[kSeqLenDim] : 0;
  bool isDecode =
      qShape.size() == 4 && qShape[kSeqLenDim] == 1 && kvSeqLen % 32 == 0;
  if (isDecode) {
    Value permutedQuery = ttir_to_ttnn::utils::generatePermute(
        mlir::cast<TypedValue<RankedTensorType>>(c.query),
        llvm::to_vector(kToDecodePermutation), rewriter,
        c.attentionMatmul.getLoc());

    auto validationResult =
        validator.validateFusion<ScaledDotProductAttentionDecodeOp>(
            c.attentionMatmul.getOperation(), c.attentionMatmul.getLoc(),
            {permutedQuery.getType()}, permutedQuery, c.key, c.value,
            /*is_causal=*/rewriter.getBoolAttr(false), c.mask,
            /*cur_pos_tensor=*/Value(), c.attentionSink, scaleAttr,
            /*memory_config=*/MemoryConfigAttr(),
            /*program_config=*/SDPAProgramConfigAttr());

    if (!validationResult.isSuccess()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                   "SDPA decode fusion validation failed: {0}",
                   validationResult.errorMessage);
      return failure();
    }

    auto decodeOp = rewriter.create<ScaledDotProductAttentionDecodeOp>(
        c.attentionMatmul.getLoc(), permutedQuery.getType(), permutedQuery,
        c.key, c.value,
        /*is_causal=*/rewriter.getBoolAttr(false), c.mask,
        /*cur_pos_tensor=*/Value(), c.attentionSink, scaleAttr,
        /*memory_config=*/MemoryConfigAttr(),
        /*program_config=*/SDPAProgramConfigAttr());

    Value finalResult = ttir_to_ttnn::utils::generatePermute(
        decodeOp.getResult(),
        ttmlir::utils::inversePermutation(kToDecodePermutation), rewriter,
        c.attentionMatmul.getLoc());

    finalResult = squeezeToOriginalRank(finalResult, originalOutputType,
                                        rewriter, c.attentionMatmul.getLoc());

    rewriter.replaceOp(c.attentionMatmul, finalResult);
  } else {
    auto validationResult =
        validator.validateFusion<ScaledDotProductAttentionOp>(
            c.attentionMatmul.getOperation(), c.attentionMatmul.getLoc(),
            {c.query.getType()}, c.query, c.key, c.value, c.mask,
            /*is_causal=*/rewriter.getBoolAttr(false), scaleAttr,
            /*sliding_window_size=*/IntegerAttr(), c.attentionSink,
            /*memory_config=*/MemoryConfigAttr());

    if (!validationResult.isSuccess()) {
      TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                   "SDPA fusion validation failed: {0}",
                   validationResult.errorMessage);
      return failure();
    }

    auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
        c.attentionMatmul.getLoc(), c.query.getType(), c.query, c.key, c.value,
        c.mask,
        /*is_causal=*/rewriter.getBoolAttr(false), scaleAttr,
        /*sliding_window_size=*/IntegerAttr(), c.attentionSink,
        /*memory_config=*/MemoryConfigAttr());

    Value finalResult =
        squeezeToOriginalRank(sdpaOp.getResult(), originalOutputType, rewriter,
                              c.attentionMatmul.getLoc());

    rewriter.replaceOp(c.attentionMatmul, finalResult);
  }

  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing
