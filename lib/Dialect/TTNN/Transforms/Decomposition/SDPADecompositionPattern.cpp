// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"

#include <cmath>

namespace mlir::tt::ttnn::decomposition {

// Dimension indices for 4D attention tensors [B, H, S, D].
static constexpr int64_t kBatchDim = 0;
static constexpr int64_t kNumHeadsDim = 1;
static constexpr int64_t kSeqLenDim = 2;
static constexpr int64_t kHeadDim = 3;

// Create a RankedTensorType preserving TTNN layout encoding when present.
static RankedTensorType createResultType(RankedTensorType sourceType,
                                         ArrayRef<int64_t> newShape) {
  if (sourceType.getEncoding()) {
    return ttnn::utils::RankedTensorTypeFactory::create(sourceType, newShape);
  }
  return RankedTensorType::get(newShape, sourceType.getElementType());
}

LogicalResult
SDPADecompositionPattern::matchAndRewrite(ScaledDotProductAttentionOp op,
                                          PatternRewriter &rewriter) const {

  // ---- Validation gate ----
  if (!forceDecompose) {
    FusionValidator validator(rewriter.getContext(), *validationConfig);

    auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());
    auto validationResult =
        validator.validateFusion<ScaledDotProductAttentionOp>(
            op.getOperation(), op.getLoc(), {qType}, op.getQuery(), op.getKey(),
            op.getValue(), op.getAttentionMask(), op.getIsCausal(),
            op.getScaleAttr(), op.getSlidingWindowSizeAttr(),
            op.getAttentionSink(), op.getMemoryConfigAttr());

    if (validationResult.isSuccess()) {
      return failure(); // Op is valid on device, no decomposition needed.
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::FusionValidator,
                 "SDPA validation failed, decomposing: {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();

  auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());
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

  // ---- Step 1: GQA head expansion ----
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
                  rewriter.getSI32IntegerAttr(kNumHeadsDim),
                  /*memory_config=*/MemoryConfigAttr())
              .getResult();

    value =
        rewriter
            .create<RepeatInterleaveOp>(
                loc, expandedVType, value, rewriter.getUI32IntegerAttr(repeats),
                rewriter.getSI32IntegerAttr(kNumHeadsDim),
                /*memory_config=*/MemoryConfigAttr())
            .getResult();
  }

  // ---- Step 2: Transpose K ----
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

  // ---- Step 3: Matmul Q @ K^T ----
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

  // ---- Step 4: Scale ----
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

  // ---- Step 5: Add attention mask ----
  if (op.getAttentionMask()) {
    scores =
        rewriter.create<AddOp>(loc, scoresType, scores, op.getAttentionMask())
            .getResult();
  }

  // ---- Step 6: Attention sink (concat) ----
  // If attention_sink is present, concatenate it along the last dimension.
  Value attentionSink = op.getAttentionSink();
  int64_t originalSeqLenKV = seqLenKV;
  if (attentionSink) {
    auto sinkType = mlir::cast<RankedTensorType>(attentionSink.getType());
    int64_t sinkCols = sinkType.getShape().back();

    llvm::SmallVector<int64_t> concatShape = {batch, numHeads, seqLenQ,
                                              seqLenKV + sinkCols};
    auto concatType = createResultType(qType, concatShape);

    SmallVector<Value> concatInputs = {scores, attentionSink};
    scores = rewriter
                 .create<ConcatOp>(loc, concatType, concatInputs,
                                   static_cast<int32_t>(-1),
                                   /*memory_config=*/MemoryConfigAttr())
                 .getResult();

    // Update seqLenKV to reflect concatenated dimension.
    seqLenKV += sinkCols;
  }

  // ---- Step 7: Softmax ----
  auto softmaxInputType = mlir::cast<RankedTensorType>(scores.getType());
  Value softmaxOut =
      rewriter
          .create<SoftmaxOp>(loc, softmaxInputType, scores, /*dimension=*/-1)
          .getResult();

  // ---- Step 8: Slice to remove sink columns ----
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

  // ---- Step 9: Matmul softmax_out @ V ----
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
