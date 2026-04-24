// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>
#include <limits>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDECOMPOSECOMPOSITES
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// RMSNorm pattern
//===----------------------------------------------------------------------===//

struct DecomposeRMSNormPattern : public OpRewritePattern<RMSNormOp> {
  using OpRewritePattern<RMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RMSNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = op.getInput();
    auto inputType = cast<RankedTensorType>(x.getType());
    int64_t rank = inputType.getRank();

    auto xSquared = rewriter.create<MultiplyOp>(loc, inputType, x, x);

    // `normalized_shape` lists the trailing k input dims over which RMS is
    // taken (see RMSNormOp::verify). Mean of x^2 must run over all of them.
    ArrayRef<int64_t> normalizedShape = op.getNormalizedShape();
    const int64_t normRank = static_cast<int64_t>(normalizedShape.size());

    SmallVector<int64_t> reducedShape(inputType.getShape());
    for (int64_t i = 0; i < normRank; ++i) {
      reducedShape[rank - normRank + i] = 1;
    }
    auto reducedType = RankedTensorType::get(
        reducedShape, inputType.getElementType(), inputType.getEncoding());

    SmallVector<int32_t> reduceDims;
    reduceDims.reserve(normRank);
    for (int64_t i = 0; i < normRank; ++i) {
      reduceDims.push_back(static_cast<int32_t>(rank - normRank + i));
    }

    auto meanOp = rewriter.create<MeanOp>(
        loc, reducedType, xSquared.getResult(), rewriter.getBoolAttr(true),
        rewriter.getI32ArrayAttr(reduceDims));

    float epsilon = op.getEpsilon().convertToFloat();
    auto epsOp = rewriter.create<FullOp>(loc, reducedType,
                                         rewriter.getF32FloatAttr(epsilon));

    auto addEps = rewriter.create<AddOp>(loc, reducedType, meanOp.getResult(),
                                         epsOp.getResult());

    auto rsqrt = rewriter.create<RsqrtOp>(loc, reducedType, addEps.getResult());

    auto normalized =
        rewriter.create<MultiplyOp>(loc, inputType, x, rsqrt.getResult());

    Value result = normalized.getResult();

    auto reshapeToInputRank = [&](Value v) -> Value {
      auto vType = cast<RankedTensorType>(v.getType());
      if (vType.getRank() == rank) {
        return v;
      }
      SmallVector<int64_t> newShape(rank - vType.getRank(), 1);
      newShape.append(vType.getShape().begin(), vType.getShape().end());
      auto reshapedType = RankedTensorType::get(
          newShape, vType.getElementType(), vType.getEncoding());
      SmallVector<int32_t> shapeI32(newShape.begin(), newShape.end());
      return rewriter.create<ReshapeOp>(loc, reshapedType, v,
                                        rewriter.getI32ArrayAttr(shapeI32));
    };

    if (op.getWeight()) {
      Value weight = reshapeToInputRank(op.getWeight());
      result = rewriter.create<MultiplyOp>(loc, inputType, result, weight)
                   .getResult();
    }

    if (op.getBias()) {
      Value bias = reshapeToInputRank(op.getBias());
      result = rewriter.create<AddOp>(loc, inputType, result, bias).getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SDPA pattern
//===----------------------------------------------------------------------===//

static Value reshapeTo(Location loc, PatternRewriter &rewriter, Value v,
                       ArrayRef<int64_t> newShape, Type elemType,
                       Attribute encoding) {
  auto newType = RankedTensorType::get(newShape, elemType, encoding);
  SmallVector<int32_t> shapeI32(newShape.begin(), newShape.end());
  return rewriter
      .create<ReshapeOp>(loc, newType, v, rewriter.getI32ArrayAttr(shapeI32))
      .getResult();
}

struct DecomposeSDPAPattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
  using OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSlidingWindowSize().has_value() || op.getAttentionSink()) {
      return rewriter.notifyMatchFailure(
          op, "sliding_window_size / attention_sink not supported");
    }

    Location loc = op.getLoc();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();

    auto queryType = cast<RankedTensorType>(query.getType());
    auto keyType = cast<RankedTensorType>(key.getType());
    ArrayRef<int64_t> qShape = queryType.getShape();
    ArrayRef<int64_t> kShape = keyType.getShape();

    int64_t batch = qShape[0];
    int64_t numHeads = qShape[1];
    int64_t querySeqLen = qShape[2];
    int64_t headSize = qShape[3];
    int64_t numKVHeads = kShape[1];
    int64_t kvSeqLen = kShape[2];

    Type elemType = queryType.getElementType();
    auto encoding = queryType.getEncoding();

    bool isGQA = (numHeads != numKVHeads);

    // For GQA, reshape Q so its head dimension matches K's kv-head dimension:
    //   Q: [B, NH, Sq, D] -> [B, NKV, groups*Sq, D]
    // This lets us batch-matmul against K without expanding K.
    int64_t groups = isGQA ? numHeads / numKVHeads : 1;
    Value q = query;
    if (isGQA) {
      q = reshapeTo(loc, rewriter, q,
                    {batch, numKVHeads, groups * querySeqLen, headSize},
                    elemType, encoding);
    }

    // Transpose K: [B, NKV, Sk, D] -> [B, NKV, D, Sk]
    // Explicit permute because D2M matmul doesn't support transpose_b (TODO
    // #2591).
    auto keyTransposedType =
        RankedTensorType::get({kShape[0], kShape[1], kShape[3], kShape[2]},
                              elemType, keyType.getEncoding());
    SmallVector<int64_t> keyPerm = {0, 1, 3, 2};
    auto keyT = rewriter.create<PermuteOp>(
        loc, keyTransposedType, key, rewriter.getDenseI64ArrayAttr(keyPerm));

    // scores = matmul(Q, K_T)  ->  [B, NH, Sq, Sk] (or grouped if GQA)
    auto fullScoresType = RankedTensorType::get(
        {batch, numHeads, querySeqLen, kvSeqLen}, elemType, encoding);
    auto matmulScoresType =
        isGQA ? RankedTensorType::get(
                    {batch, numKVHeads, groups * querySeqLen, kvSeqLen},
                    elemType, encoding)
              : fullScoresType;
    auto scores =
        rewriter.create<MatmulOp>(loc, matmulScoresType, q, keyT.getResult());

    // Reshape scores back to [B, NH, Sq, Sk] for scaling/masking/softmax.
    Value scoresVal = scores.getResult();
    if (isGQA) {
      scoresVal = reshapeTo(loc, rewriter, scoresVal,
                            {batch, numHeads, querySeqLen, kvSeqLen}, elemType,
                            encoding);
    }

    // Add attention mask before scaling to match tt-metal FlashAttention
    // semantics: exp((QK + mask - max) * scale).
    Value attnInput = scoresVal;
    if (op.getAttentionMask()) {
      attnInput = rewriter
                      .create<AddOp>(loc, fullScoresType, attnInput,
                                     op.getAttentionMask())
                      .getResult();
    }

    // Apply causal mask: lower-triangular additive mask where future positions
    // (j > i) are set to -inf so softmax drives them to zero probability.
    if (op.getIsCausal()) {
      auto maskType = RankedTensorType::get({1, 1, querySeqLen, kvSeqLen},
                                            elemType, encoding);

      auto rowIdx = rewriter.create<ArangeOp>(loc, maskType, /*start=*/0,
                                              /*end=*/querySeqLen,
                                              /*step=*/1,
                                              /*arange_dimension=*/2);
      auto colIdx = rewriter.create<ArangeOp>(loc, maskType, /*start=*/0,
                                              /*end=*/kvSeqLen,
                                              /*step=*/1,
                                              /*arange_dimension=*/3);

      // ge(i, j) -> 1.0 where i >= j (causal), 0.0 otherwise.
      // Uses elemType (not i1) since PredicateTypeAlignment already ran.
      auto causalBool = rewriter.create<GreaterEqualOp>(
          loc, maskType, rowIdx.getResult(), colIdx.getResult());

      auto zeros = rewriter.create<FullOp>(loc, maskType,
                                           rewriter.getF32FloatAttr(0.0f));
      float negInfVal = -std::numeric_limits<float>::infinity();
      auto negInf = rewriter.create<FullOp>(
          loc, maskType, rewriter.getF32FloatAttr(negInfVal));

      // where(mask, 0, -inf): 0 for allowed positions, -inf for blocked.
      auto causalMask =
          rewriter.create<WhereOp>(loc, maskType, causalBool.getResult(),
                                   zeros.getResult(), negInf.getResult());

      attnInput = rewriter
                      .create<AddOp>(loc, fullScoresType, attnInput,
                                     causalMask.getResult())
                      .getResult();
    }

    // scale = 1 / sqrt(head_size) unless user-provided.
    float scaleVal = 1.0f / std::sqrt(static_cast<float>(headSize));
    if (auto scaleAttr = op.getScaleAttr()) {
      scaleVal = static_cast<float>(scaleAttr.getValueAsDouble());
    }

    auto scaleConst = rewriter.create<FullOp>(
        loc, fullScoresType, rewriter.getF32FloatAttr(scaleVal));
    attnInput = rewriter
                    .create<MultiplyOp>(loc, fullScoresType, attnInput,
                                        scaleConst.getResult())
                    .getResult();

    // Softmax along last dimension.
    int32_t softmaxDim = static_cast<int32_t>(fullScoresType.getRank() - 1);
    auto probs = rewriter.create<SoftmaxOp>(
        loc, fullScoresType, attnInput, rewriter.getSI32IntegerAttr(softmaxDim),
        rewriter.getBoolAttr(true));

    // For GQA, reshape probs back to grouped form for V matmul:
    //   [B, NH, Sq, Sk] -> [B, NKV, groups*Sq, Sk]
    Value probsVal = probs.getResult();
    if (isGQA) {
      probsVal = reshapeTo(loc, rewriter, probsVal,
                           {batch, numKVHeads, groups * querySeqLen, kvSeqLen},
                           elemType, encoding);
    }

    // output = matmul(probs, V)
    auto outputType = RankedTensorType::get(
        {batch, numHeads, querySeqLen, headSize}, elemType, encoding);
    auto matmulOutType =
        isGQA ? RankedTensorType::get(
                    {batch, numKVHeads, groups * querySeqLen, headSize},
                    elemType, encoding)
              : outputType;
    auto output =
        rewriter.create<MatmulOp>(loc, matmulOutType, probsVal, value);

    // Reshape back to [B, NH, Sq, D] if GQA.
    Value result = output.getResult();
    if (isGQA) {
      result = reshapeTo(loc, rewriter, result,
                         {batch, numHeads, querySeqLen, headSize}, elemType,
                         encoding);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Softmax pattern
//===----------------------------------------------------------------------===//

// softmax(x, dim) = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)
struct DecomposeSoftmaxPattern : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    int32_t dim = op.getDimension();
    if (dim < 0) {
      dim += inputType.getRank();
    }

    // Reduced shape: same as input but with dim collapsed to 1.
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[dim] = 1;
    auto reducedType = RankedTensorType::get(
        reducedShape, inputType.getElementType(), inputType.getEncoding());

    auto dimAttr = rewriter.getI32ArrayAttr({dim});

    Value expInput = input;
    if (op.getNumericStable()) {
      // m = max(x, dim, keep_dim=true)
      auto maxVal = rewriter.create<MaxOp>(loc, reducedType, input,
                                           rewriter.getBoolAttr(true), dimAttr);
      // x_shifted = x - m (broadcasts along reduced dim)
      expInput =
          rewriter.create<SubtractOp>(loc, inputType, input, maxVal.getResult())
              .getResult();
    }

    // e = exp(x) or exp(x - max)
    auto expVal = rewriter.create<ExpOp>(loc, inputType, expInput);

    // s = sum(e, dim, keep_dim=true)
    auto sumVal = rewriter.create<SumOp>(loc, reducedType, expVal.getResult(),
                                         rewriter.getBoolAttr(true), dimAttr);

    // result = e / s
    // Use divide instead of reciprocal + multiply to avoid a tile-level
    // column-broadcast multiply bug (FPU mul_tiles with bcast produces
    // incorrect results for certain grid configurations).
    auto result = rewriter.create<DivOp>(loc, inputType, expVal.getResult(),
                                         sumVal.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class TTIRDecomposeComposites
    : public impl::TTIRDecomposeCompositesBase<TTIRDecomposeComposites> {
public:
  using impl::TTIRDecomposeCompositesBase<
      TTIRDecomposeComposites>::TTIRDecomposeCompositesBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    // SDPA gets highest benefit so the greedy rewriter applies it first;
    // the softmax ops it produces are then caught by the softmax pattern
    // on subsequent iterations.
    patterns.add<DecomposeSDPAPattern>(&getContext(), /*benefit=*/2);
    patterns.add<DecomposeRMSNormPattern>(&getContext(), /*benefit=*/1);
    patterns.add<DecomposeSoftmaxPattern>(&getContext(), /*benefit=*/0);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
