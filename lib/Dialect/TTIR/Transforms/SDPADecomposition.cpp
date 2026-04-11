// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"

#include <cmath>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSDPADECOMPOSITION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static Value reshapeTo(Location loc, IRRewriter &rewriter, Value v,
                       ArrayRef<int64_t> newShape, Type elemType,
                       Attribute encoding) {
  auto newType = RankedTensorType::get(newShape, elemType, encoding);
  SmallVector<int32_t> shapeI32(newShape.begin(), newShape.end());
  return rewriter
      .create<ReshapeOp>(loc, newType, v, rewriter.getI32ArrayAttr(shapeI32))
      .getResult();
}

static void decomposeSDPA(ScaledDotProductAttentionOp op,
                          IRRewriter &rewriter) {
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
  int64_t groups = numHeads / numKVHeads;

  // For GQA, reshape Q so its head dimension matches K's kv-head dimension:
  //   Q: [B, NH, Sq, D] -> [B, NKV, groups*Sq, D]
  // This lets us batch-matmul against K without expanding K.
  Value q = query;
  if (isGQA) {
    q = reshapeTo(loc, rewriter, q,
                  {batch, numKVHeads, groups * querySeqLen, headSize}, elemType,
                  encoding);
  }

  // Transpose K: [B, NKV, Sk, D] -> [B, NKV, D, Sk]
  // Explicit permute because D2M matmul doesn't support transpose_b (TODO #2591).
  int64_t keyRank = keyType.getRank();
  SmallVector<int64_t> keyPerm;
  for (int64_t i = 0; i < keyRank - 2; ++i) {
    keyPerm.push_back(i);
  }
  keyPerm.push_back(keyRank - 1);
  keyPerm.push_back(keyRank - 2);
  auto keyTransposedType = RankedTensorType::get(
      {kShape[0], kShape[1], kShape[3], kShape[2]}, elemType,
      keyType.getEncoding());
  auto keyT = rewriter.create<PermuteOp>(
      loc, keyTransposedType, key, rewriter.getDenseI64ArrayAttr(keyPerm));

  // scores = matmul(Q, K_T)
  // If GQA:  [B, NKV, groups*Sq, D] x [B, NKV, D, Sk] = [B, NKV, groups*Sq, Sk]
  // Else:    [B, NH, Sq, D] x [B, NH, D, Sk] = [B, NH, Sq, Sk]
  int64_t scoresHead = isGQA ? numKVHeads : numHeads;
  int64_t scoresSeq = isGQA ? groups * querySeqLen : querySeqLen;
  auto scoresType = RankedTensorType::get(
      {batch, scoresHead, scoresSeq, kvSeqLen}, elemType, encoding);
  auto scores = rewriter.create<MatmulOp>(loc, scoresType, q,
                                          keyT.getResult());

  // Reshape scores back to [B, NH, Sq, Sk] for scaling/masking/softmax.
  auto fullScoresType = RankedTensorType::get(
      {batch, numHeads, querySeqLen, kvSeqLen}, elemType, encoding);
  Value scoresVal = scores.getResult();
  if (isGQA) {
    scoresVal =
        reshapeTo(loc, rewriter, scoresVal,
                  {batch, numHeads, querySeqLen, kvSeqLen}, elemType, encoding);
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

  // scale = 1 / sqrt(head_size) unless user-provided.
  float scaleVal;
  if (op.getScale().has_value()) {
    scaleVal = op.getScale()->convertToFloat();
  } else {
    scaleVal = 1.0f / std::sqrt(static_cast<float>(headSize));
  }

  auto scaleConst = rewriter.create<FullOp>(loc, fullScoresType,
                                            rewriter.getF32FloatAttr(scaleVal));
  attnInput = rewriter
                  .create<MultiplyOp>(loc, fullScoresType, attnInput,
                                      scaleConst.getResult())
                  .getResult();

  // softmax along last dimension.
  int32_t softmaxDim = static_cast<int32_t>(fullScoresType.getRank() - 1);
  auto probs = rewriter.create<SoftmaxOp>(
      loc, fullScoresType, attnInput, rewriter.getSI32IntegerAttr(softmaxDim),
      rewriter.getBoolAttr(false));

  // For GQA, reshape probs back to grouped form for V matmul:
  //   [B, NH, Sq, Sk] -> [B, NKV, groups*Sq, Sk]
  Value probsVal = probs.getResult();
  if (isGQA) {
    probsVal = reshapeTo(loc, rewriter, probsVal,
                         {batch, numKVHeads, groups * querySeqLen, kvSeqLen},
                         elemType, encoding);
  }

  // output = matmul(probs, V)
  // If GQA:  [B, NKV, groups*Sq, Sk] x [B, NKV, Sk, D] = [B, NKV, groups*Sq, D]
  // Else:    [B, NH, Sq, Sk] x [B, NH, Sk, D] = [B, NH, Sq, D]
  int64_t outHead = isGQA ? numKVHeads : numHeads;
  int64_t outSeq = isGQA ? groups * querySeqLen : querySeqLen;
  auto outMatmulType = RankedTensorType::get(
      {batch, outHead, outSeq, headSize}, elemType, encoding);
  auto output =
      rewriter.create<MatmulOp>(loc, outMatmulType, probsVal, value);

  // Reshape back to [B, NH, Sq, D] if GQA.
  Value result = output.getResult();
  if (isGQA) {
    result = reshapeTo(loc, rewriter, result,
                       {batch, numHeads, querySeqLen, headSize}, elemType,
                       encoding);
  }

  rewriter.replaceOp(op, result);
}

class TTIRSDPADecomposition
    : public impl::TTIRSDPADecompositionBase<TTIRSDPADecomposition> {
public:
  using impl::TTIRSDPADecompositionBase<
      TTIRSDPADecomposition>::TTIRSDPADecompositionBase;

  void runOnOperation() final {
    llvm::SmallVector<ScaledDotProductAttentionOp> opsToDecompose;
    getOperation()->walk([&](ScaledDotProductAttentionOp op) {
      opsToDecompose.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (ScaledDotProductAttentionOp op : opsToDecompose) {
      rewriter.setInsertionPoint(op);
      decomposeSDPA(op, rewriter);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
