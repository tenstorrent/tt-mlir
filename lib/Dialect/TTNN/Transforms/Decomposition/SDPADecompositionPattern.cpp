// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include <cmath>
#include <limits>

namespace mlir::tt::ttnn::decomposition {

// Dimension indices for 4D attention tensors [B, H, S, D].
static constexpr int64_t kBatchDim = 0; // [B, H, S, D]
static constexpr int64_t kNumHeadsDim = 1;
static constexpr int64_t kSeqLenDim = 2;
static constexpr int64_t kHeadDim = 3;

static RankedTensorType createResultType(RankedTensorType sourceType,
                                         ArrayRef<int64_t> newShape) {
  return ttnn::utils::RankedTensorTypeFactory::create(sourceType, newShape);
}

/// Create a type with `shape` in f32, reusing `referenceType`'s layout.
static RankedTensorType createF32Type(RankedTensorType referenceType,
                                      ArrayRef<int64_t> shape) {
  return ttnn::utils::RankedTensorTypeFactory::create(
      createResultType(referenceType, shape), ttcore::DataType::Float32);
}

/// Build an f32 index vector 0..length-1 placed at `axis` of a 4D shape
/// [1, 1, *, *] (axis 2 -> [1, 1, length, 1], axis 3 -> [1, 1, 1, length]).
/// f32 is required: bf16 has 8 mantissa bits and cannot exactly represent
/// indices > 256, which would corrupt the index comparison at large seq_len.
static Value makeIndexVector(PatternRewriter &rewriter, Location loc,
                             RankedTensorType referenceType, int64_t length,
                             int64_t axis, Value device) {
  // ttnn.arange verifier requires a rank-1 result; reshape to 4D afterwards.
  auto arange1dType = createF32Type(referenceType, {length});
  Value indices = rewriter
                      .create<ArangeOp>(loc, arange1dType, device,
                                        /*start=*/0, /*end=*/length, /*step=*/1)
                      .getResult();

  llvm::SmallVector<int64_t> shape4d = {1, 1, 1, 1};
  shape4d[axis] = length;
  auto reshapedType = createF32Type(referenceType, shape4d);
  llvm::SmallVector<int32_t> shape4dI32(shape4d.begin(), shape4d.end());
  return rewriter
      .create<ReshapeOp>(loc, reshapedType, indices,
                         rewriter.getI32ArrayAttr(shape4dI32))
      .getResult();
}

/// Generate a causal mask [1, 1, Sq, Skv] on-device.
/// mask[i, j] = -inf if j > i else 0. Built from arange/compare/where instead
/// of a dense constant so nothing O(seq^2) is serialized.
static Value generateCausalMask(PatternRewriter &rewriter, Location loc,
                                int64_t seqLenQ, int64_t seqLenKV,
                                RankedTensorType referenceType, Value device) {
  auto maskShape = llvm::SmallVector<int64_t>{1, 1, seqLenQ, seqLenKV};

  Value rowIdx = makeIndexVector(rewriter, loc, referenceType, seqLenQ,
                                 /*axis=*/kSeqLenDim, device); // [1,1,Sq,1]
  Value colIdx = makeIndexVector(rewriter, loc, referenceType, seqLenKV,
                                 /*axis=*/kHeadDim, device); // [1,1,1,Skv]

  // Comparison/where in f32; broadcasts to [1, 1, Sq, Skv].
  auto f32MaskType = createF32Type(referenceType, maskShape);

  // isMasked = colIdx > rowIdx  (i.e. j > i).
  Value isMasked =
      rewriter.create<GreaterThanOp>(loc, f32MaskType, colIdx, rowIdx)
          .getResult();

  // Pre-broadcast branches: ttnn.where does not reliably broadcast scalar
  // branches against a multi-dim condition.
  Value zeros = rewriter
                    .create<FullOp>(loc, f32MaskType,
                                    rewriter.getF32FloatAttr(0.0f), device)
                    .getResult();
  Value negInf =
      rewriter
          .create<FullOp>(
              loc, f32MaskType,
              rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()),
              device)
          .getResult();

  Value mask =
      rewriter.create<WhereOp>(loc, f32MaskType, isMasked, negInf, zeros)
          .getResult();

  // Typecast the {0, -inf} mask to the reference dtype for the downstream add.
  auto maskType = createResultType(referenceType, maskShape);
  return rewriter.create<TypecastOp>(loc, maskType, mask).getResult();
}

/// Generate a sliding window mask [1, 1, Sq, Skv] on-device.
/// Positions outside the window are -inf, others 0. Built from
/// arange/subtract/compare/where instead of a dense constant.
static Value generateSlidingWindowMask(PatternRewriter &rewriter, Location loc,
                                       int64_t seqLenQ, int64_t seqLenKV,
                                       uint32_t windowSize, bool isCausal,
                                       RankedTensorType referenceType,
                                       Value device) {
  // Window topology (must match the tt-metal kernel), diff = i - j:
  //   causal:     in-window iff 0 <= diff < W          (last W tokens)
  //   non-causal: in-window iff -W/2 <= diff <= W/2    (W+1 tokens centered)
  // The out-of-window positions are masked (-inf), built on-device.
  int64_t halfWindow = static_cast<int64_t>(windowSize) / 2;
  auto maskShape = llvm::SmallVector<int64_t>{1, 1, seqLenQ, seqLenKV};

  Value rowIdx = makeIndexVector(rewriter, loc, referenceType, seqLenQ,
                                 /*axis=*/kSeqLenDim, device); // [1,1,Sq,1]
  Value colIdx = makeIndexVector(rewriter, loc, referenceType, seqLenKV,
                                 /*axis=*/kHeadDim, device); // [1,1,1,Skv]

  auto f32MaskType = createF32Type(referenceType, maskShape);

  // diff = rowIdx - colIdx, broadcasts to [1, 1, Sq, Skv].
  Value diff =
      rewriter.create<SubtractOp>(loc, f32MaskType, rowIdx, colIdx).getResult();

  // Bounds and branch tensors as full [1, 1, Sq, Skv] f32 (compare ops and
  // ttnn.where do not reliably broadcast scalar operands).
  auto makeConst = [&](float v) {
    return rewriter
        .create<FullOp>(loc, f32MaskType, rewriter.getF32FloatAttr(v), device)
        .getResult();
  };

  // The zeros tensor is both the causal lower bound (diff < 0) and the
  // in-window branch of the final where; build it once and reuse.
  Value zeros = makeConst(0.0f);
  Value negInf = makeConst(-std::numeric_limits<float>::infinity());

  Value violLow;  // below the window
  Value violHigh; // above the window
  if (isCausal) {
    // out-of-window iff diff < 0  OR  diff >= W
    violLow =
        rewriter.create<LessThanOp>(loc, f32MaskType, diff, zeros).getResult();
    violHigh =
        rewriter
            .create<GreaterEqualOp>(loc, f32MaskType, diff,
                                    makeConst(static_cast<float>(windowSize)))
            .getResult();
  } else {
    // out-of-window iff diff < -halfWindow  OR  diff > halfWindow
    violLow =
        rewriter
            .create<LessThanOp>(loc, f32MaskType, diff,
                                makeConst(-static_cast<float>(halfWindow)))
            .getResult();
    violHigh =
        rewriter
            .create<GreaterThanOp>(loc, f32MaskType, diff,
                                   makeConst(static_cast<float>(halfWindow)))
            .getResult();
  }

  // mask = where(violLow, -inf, where(violHigh, -inf, 0))
  Value inner =
      rewriter.create<WhereOp>(loc, f32MaskType, violHigh, negInf, zeros)
          .getResult();
  Value mask =
      rewriter.create<WhereOp>(loc, f32MaskType, violLow, negInf, inner)
          .getResult();

  auto maskType = createResultType(referenceType, maskShape);
  return rewriter.create<TypecastOp>(loc, maskType, mask).getResult();
}

LogicalResult
SDPADecompositionPattern::matchAndRewrite(ttnn::ScaledDotProductAttentionOp op,
                                          PatternRewriter &rewriter) const {

  auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());

  // When validation config is provided, validate the SDPA op using
  // IsolatedIRValidationWrapper. If validation succeeds, keep the op as-is.
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult =
        validator.validateOp<ttnn::ScaledDotProductAttentionOp>(
            op.getOperation(), op.getLoc(), {qType}, op.getQuery(), op.getKey(),
            op.getValue(), op.getAttentionMask(), op.getIsCausalAttr(),
            op.getScaleAttr(), op.getSlidingWindowSizeAttr(),
            op.getAttentionSink());

    if (validationResult.isSuccess()) {
      // Op is valid — keep it as-is.
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "SDPA decomposition triggered (validation failed): {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();

  auto kType = mlir::cast<RankedTensorType>(op.getKey().getType());

  auto qShape = qType.getShape();
  auto kShape = kType.getShape();

  int64_t numHeads = qShape[kNumHeadsDim];
  int64_t numKVHeads = kShape[kNumHeadsDim];
  int64_t seqLenQ = qShape[kSeqLenDim];
  int64_t seqLenKV = kShape[kSeqLenDim];
  int64_t headDim = qShape[kHeadDim];
  int64_t batch = qShape[kBatchDim];

  Value query = op.getQuery();
  Value key = op.getKey();
  Value value = op.getValue();

  // ---- GQA head expansion ----
  // If num_heads != num_kv_heads, repeat K and V along the heads dimension.
  if (numHeads != numKVHeads) {
    assert(numHeads % numKVHeads == 0 &&
           "num_heads must be divisible by num_kv_heads");
    uint32_t repeats = static_cast<uint32_t>(numHeads / numKVHeads);

    llvm::SmallVector<int64_t> expandedKVShape = {batch, numHeads, seqLenKV,
                                                  headDim};
    auto expandedKType = createResultType(kType, expandedKVShape);
    auto expandedVType = createResultType(
        mlir::cast<RankedTensorType>(value.getType()), expandedKVShape);

    key = rewriter
              .create<RepeatInterleaveOp>(
                  loc, expandedKType, key, rewriter.getUI32IntegerAttr(repeats),
                  rewriter.getSI32IntegerAttr(kNumHeadsDim))
              .getResult();

    value =
        rewriter
            .create<RepeatInterleaveOp>(
                loc, expandedVType, value, rewriter.getUI32IntegerAttr(repeats),
                rewriter.getSI32IntegerAttr(kNumHeadsDim))
            .getResult();
  }

  // ---- Transpose K ----
  // K: [B, H, Skv, D] -> [B, H, D, Skv]
  llvm::SmallVector<int64_t> kTransposedShape = {batch, numHeads, headDim,
                                                 seqLenKV};
  auto kTransposedType = createResultType(
      mlir::cast<RankedTensorType>(key.getType()), kTransposedShape);

  Value kTransposed = rewriter
                          .create<TransposeOp>(loc, kTransposedType, key,
                                               rewriter.getSI32IntegerAttr(-2),
                                               rewriter.getSI32IntegerAttr(-1))
                          .getResult();

  // ---- Matmul Q @ K^T ----
  // [B, H, Sq, D] x [B, H, D, Skv] -> [B, H, Sq, Skv]
  llvm::SmallVector<int64_t> scoresShape = {batch, numHeads, seqLenQ, seqLenKV};
  auto scoresType = createResultType(qType, scoresShape);

  Value scores =
      rewriter
          .create<MatmulOp>(loc, scoresType, query, kTransposed,
                            /*transpose_a=*/false, /*transpose_b=*/false,
                            /*matmul_program_config=*/nullptr,
                            /*activation=*/nullptr)
          .getResult();

  // ---- Scale ----
  // Use provided scale or default 1/sqrt(D).
  float scaleValue;
  if (op.getScaleAttr()) {
    scaleValue = op.getScaleAttr().getValueAsDouble();
  } else {
    scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
  }

  // Create a scalar tensor filled with the scale value.
  llvm::SmallVector<int64_t> scalarShape = {1, 1, 1, 1};
  auto scalarType = createResultType(qType, scalarShape);
  Value device =
      utils::getOrInsertDevice(rewriter, op.getOperation()).getResult();
  Value scaleTensor =
      rewriter
          .create<FullOp>(loc, scalarType, rewriter.getF32FloatAttr(scaleValue),
                          device)
          .getResult();

  scores = rewriter.create<MultiplyOp>(loc, scoresType, scores, scaleTensor)
               .getResult();

  // ---- Add attention mask ----
  if (op.getAttentionMask()) {
    Value mask = op.getAttentionMask();

    // Broadcast mask along the heads dimension if needed.
    // Mask may be [B, 1, Sq, Skv] while scores are [B, H, Sq, Skv].
    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto maskShape = maskType.getShape();
    if (maskShape[kNumHeadsDim] == 1 && numHeads > 1) {
      llvm::SmallVector<int64_t> broadcastShape(maskShape);
      broadcastShape[kNumHeadsDim] = numHeads;
      auto broadcastType = createResultType(maskType, broadcastShape);
      llvm::SmallVector<int64_t> repeatDims = {1, numHeads, 1, 1};
      mask = rewriter
                 .create<RepeatOp>(
                     loc, broadcastType, mask,
                     ShapeAttr::get(rewriter.getContext(), repeatDims))
                 .getResult();
    }

    scores = rewriter.create<AddOp>(loc, scoresType, scores, mask).getResult();
  }

  // Generate positional mask (sliding window and/or causal).
  if (op.getSlidingWindowSizeAttr()) {
    uint32_t windowSize = op.getSlidingWindowSizeAttr().getUInt();
    Value windowMask =
        generateSlidingWindowMask(rewriter, loc, seqLenQ, seqLenKV, windowSize,
                                  op.getIsCausal(), qType, device);
    scores =
        rewriter.create<AddOp>(loc, scoresType, scores, windowMask).getResult();
  } else if (!op.getAttentionMask() && op.getIsCausal()) {
    // Causal-only mask (no sliding window, no explicit mask).
    Value causalMask =
        generateCausalMask(rewriter, loc, seqLenQ, seqLenKV, qType, device);
    scores =
        rewriter.create<AddOp>(loc, scoresType, scores, causalMask).getResult();
  }

  // ---- Attention sink (concat) ----
  // Kernel computes softmax([scale*QK, scale*sink]) — the user-provided sink
  // is in raw logit units (same coord frame as raw QK). Pre-scale the sink so
  // it lives in the same units as the already-scaled scores before concat.
  // The sink is broadcast across batch and Sq before concat since ttnn.concat
  // requires matching non-concat dimensions (unlike the fused kernel which
  // broadcasts implicitly).
  Value attentionSink = op.getAttentionSink();
  int64_t originalSeqLenKV = seqLenKV;
  if (attentionSink) {
    auto sinkType = mlir::cast<RankedTensorType>(attentionSink.getType());
    int64_t sinkCols = sinkType.getShape().back();

    Value scaledSink =
        rewriter.create<MultiplyOp>(loc, sinkType, attentionSink, scaleTensor)
            .getResult();

    // Broadcast sink [1, Hq, 1, sinkCols] -> [B, Hq, Sq, sinkCols].
    llvm::SmallVector<int64_t> broadcastSinkShape = {batch, numHeads, seqLenQ,
                                                     sinkCols};
    if (sinkType.getShape() != ArrayRef<int64_t>(broadcastSinkShape)) {
      auto broadcastSinkType = createResultType(qType, broadcastSinkShape);
      llvm::SmallVector<int64_t> sinkRepeatDims = {batch, 1, seqLenQ, 1};
      scaledSink = rewriter
                       .create<RepeatOp>(loc, broadcastSinkType, scaledSink,
                                         ShapeAttr::get(rewriter.getContext(),
                                                        sinkRepeatDims))
                       .getResult();
    }

    llvm::SmallVector<int64_t> concatShape = {batch, numHeads, seqLenQ,
                                              seqLenKV + sinkCols};
    auto concatType = createResultType(qType, concatShape);

    SmallVector<Value> concatInputs = {scores, scaledSink};
    scores = rewriter
                 .create<ConcatOp>(loc, concatType, concatInputs,
                                   static_cast<int32_t>(-1))
                 .getResult();
  }

  // ---- Softmax ----
  // numericStable=true subtracts rowwise max before exp, matching the fused
  // kernel's online-softmax. Required because scale·QK can exceed bf16's exp
  // range (e.g. head_dim=256 with non-standard scale produces scores ~100, and
  // exp(100) overflows bf16) — without max subtraction softmax returns NaN.
  auto softmaxInputType = mlir::cast<RankedTensorType>(scores.getType());
  Value softmaxOut =
      rewriter
          .create<SoftmaxOp>(loc, softmaxInputType, scores, /*dimension=*/-1,
                             /*numericStable=*/true)
          .getResult();

  // ---- Slice to remove sink columns ----
  if (attentionSink) {
    // Slice back to [B, H, Sq, originalSeqLenKV] removing the sink columns.
    llvm::SmallVector<int64_t> slicedShape = {batch, numHeads, seqLenQ,
                                              originalSeqLenKV};
    auto slicedType = createResultType(qType, slicedShape);

    SmallVector<int32_t> begins = {0, 0, 0, 0};
    SmallVector<int32_t> ends = {
        static_cast<int32_t>(batch), static_cast<int32_t>(numHeads),
        static_cast<int32_t>(seqLenQ), static_cast<int32_t>(originalSeqLenKV)};
    SmallVector<int32_t> steps = {1, 1, 1, 1};

    softmaxOut = rewriter
                     .create<SliceStaticOp>(loc, slicedType, softmaxOut,
                                            rewriter.getI32ArrayAttr(begins),
                                            rewriter.getI32ArrayAttr(ends),
                                            rewriter.getI32ArrayAttr(steps))
                     .getResult();
  }

  // ---- Matmul softmax_out @ V ----
  // [B, H, Sq, Skv] x [B, H, Skv, D] -> [B, H, Sq, D]
  llvm::SmallVector<int64_t> resultShape = {batch, numHeads, seqLenQ, headDim};
  auto resultType = createResultType(qType, resultShape);

  Value result =
      rewriter
          .create<MatmulOp>(loc, resultType, softmaxOut, value,
                            /*transpose_a=*/false, /*transpose_b=*/false,
                            /*matmul_program_config=*/nullptr,
                            /*activation=*/nullptr)
          .getResult();

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
