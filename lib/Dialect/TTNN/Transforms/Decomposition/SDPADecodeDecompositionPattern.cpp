// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include <cmath>
#include <limits>

namespace mlir::tt::ttnn::decomposition {

// SDPADecode Q shape: [1, B, H, D]
// SDPA Q shape:       [B, H, Sq, D]
// Permutation: [1, 2, 0, 3] maps [1, B, H, D] -> [B, H, 1, D]
static constexpr std::array<int64_t, 4> kToSDPAPermutation = {1, 2, 0, 3};
// Inverse: [2, 0, 1, 3] maps [B, H, 1, D] -> [1, B, H, D]
static constexpr std::array<int64_t, 4> kFromSDPAPermutation = {2, 0, 1, 3};
// SDPADecode mask shape: [B, 1, Hq, Sk]   (heads at dim 2)
// SDPA      mask shape: [B, Hq, Sq, Sk]   (heads at dim 1, Sq=1 after dispatch)
// Permutation: [0, 2, 1, 3] maps [B, 1, Hq, Sk] -> [B, Hq, 1, Sk]
static constexpr std::array<int64_t, 4> kMaskToSDPAPermutation = {0, 2, 1, 3};

static Value createPermute(Value input, ArrayRef<int64_t> permutation,
                           PatternRewriter &rewriter, Location loc) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outputShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);

  return rewriter.create<PermuteOp>(loc, outputType, input,
                                    rewriter.getDenseI64ArrayAttr(permutation),
                                    /*pad_value=*/mlir::FloatAttr());
}

LogicalResult SDPADecodeDecompositionPattern::matchAndRewrite(
    ttnn::ScaledDotProductAttentionDecodeOp op,
    PatternRewriter &rewriter) const {

  auto qType = mlir::cast<RankedTensorType>(op.getQuery().getType());

  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult =
        validator.validateOp<ttnn::ScaledDotProductAttentionDecodeOp>(
            op.getOperation(), op.getLoc(), {qType}, op.getQuery(), op.getKey(),
            op.getValue(), op.getIsCausalAttr(), op.getAttentionMask(),
            op.getCurPosTensor(), op.getAttentionSink(), op.getScaleAttr(),
            op.getProgramConfigAttr());

    if (validationResult.isSuccess()) {
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "SDPA decode decomposition triggered (validation failed): {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();

  // Permute Q from [1, B, H, D] to [B, H, 1, D].
  Value permutedQuery =
      createPermute(op.getQuery(), kToSDPAPermutation, rewriter, loc);

  // Common scalars: scale value, device handle, Q's data type.
  float scaleValue;
  if (op.getScaleAttr()) {
    scaleValue = op.getScaleAttr().getValueAsDouble();
  } else {
    int64_t headDim = qType.getShape().back();
    scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
  }
  Value device =
      utils::getOrInsertDevice(rewriter, op.getOperation()).getResult();

  ttcore::DataType qDataType = ttcore::DataType::BFloat16;
  if (auto qLayoutAttr =
          mlir::dyn_cast_if_present<TTNNLayoutAttr>(qType.getEncoding())) {
    qDataType = qLayoutAttr.getDataType();
  }

  // Q is [1, B, Hq, D]; K is [B, Hkv, Sk, D].
  int64_t batch = qType.getShape()[1];
  auto kType = mlir::cast<RankedTensorType>(op.getKey().getType());
  int64_t seqLenKV = kType.getShape()[2];

  Value scaledMask = op.getAttentionMask();
  if (scaledMask) {
    // The two SDPA ops have different mask conventions at the API level:
    //   decode:  softmax(scale * (QK + mask)) = softmax(scale*QK + scale*mask)
    //   prefill: softmax(scale * QK + mask)                  (PyTorch-style)
    // Both kernels internally compute softmax(scale*(QK + mask)); prefill's
    // host wrapper compensates by pre-dividing the user mask by scale, decode
    // does not. To make a prefill op produce the decode result on the same
    // mask m, feed prefill m' = scale * m.
    auto maskType = mlir::cast<RankedTensorType>(scaledMask.getType());
    llvm::SmallVector<int64_t> scalarShape(maskType.getRank(), 1);
    RankedTensorType scalarType =
        ttnn::utils::RankedTensorTypeFactory::create(maskType, scalarShape);
    Value scaleTensor =
        rewriter
            .create<FullOp>(loc, scalarType,
                            rewriter.getF32FloatAttr(scaleValue), device)
            .getResult();
    scaledMask =
        rewriter.create<MultiplyOp>(loc, maskType, scaledMask, scaleTensor)
            .getResult();

    // Decode mask layout is [B, 1, Hq, Sk] (heads at dim 2). The downstream
    // regular SDPA op expects [B, Hq, Sq, Sk] (heads at dim 1, Sq=1 here).
    // Permute dims 1 and 2 to bridge the convention difference.
    scaledMask =
        createPermute(scaledMask, kMaskToSDPAPermutation, rewriter, loc);
  } else if (op.getIsCausal() && op.getCurPosTensor()) {
    // Synthesize a per-batch causal mask from cur_pos_tensor: positions
    // j > cur_pos[b] are -inf, otherwise 0. Shape [B, 1, 1, Sk] aligns with
    // the prefill mask convention (heads dim broadcast).
    Value curPos = op.getCurPosTensor();
    auto curPosType = mlir::cast<RankedTensorType>(curPos.getType());

    // 1. Reshape cur_pos [B] -> [B, 1, 1, 1] so it broadcasts against arange.
    llvm::SmallVector<int64_t> reshapedCurPosShape = {batch, 1, 1, 1};
    auto reshapedCurPosType = ttnn::utils::RankedTensorTypeFactory::create(
        curPosType, reshapedCurPosShape);
    llvm::SmallVector<int32_t> reshapeShapeI32 = {static_cast<int32_t>(batch),
                                                  1, 1, 1};
    curPos =
        rewriter
            .create<ttnn::ReshapeOp>(loc, reshapedCurPosType, curPos,
                                     rewriter.getI32ArrayAttr(reshapeShapeI32))
            .getResult();

    // 2. Cast cur_pos from int32 to Q's dtype so comparison with arange works.
    auto castedCurPosType = ttnn::utils::RankedTensorTypeFactory::create(
        reshapedCurPosType, qDataType);
    curPos =
        rewriter.create<TypecastOp>(loc, castedCurPosType, curPos).getResult();

    // 3. Arange [0..Sk) -> rank-1 [Sk] in Q's dtype (ttnn.arange verifier
    // requires rank-1 output), then reshape to [1, 1, 1, Sk] for broadcast.
    auto arange1dType = ttnn::utils::RankedTensorTypeFactory::create(
        qType, llvm::SmallVector<int64_t>{seqLenKV});
    Value indices =
        rewriter
            .create<ttnn::ArangeOp>(loc, arange1dType, device, /*start=*/0,
                                    /*end=*/seqLenKV, /*step=*/1)
            .getResult();

    auto arange4dType = ttnn::utils::RankedTensorTypeFactory::create(
        qType, llvm::SmallVector<int64_t>{1, 1, 1, seqLenKV});
    llvm::SmallVector<int32_t> arange4dShapeI32 = {
        1, 1, 1, static_cast<int32_t>(seqLenKV)};
    indices =
        rewriter
            .create<ttnn::ReshapeOp>(loc, arange4dType, indices,
                                     rewriter.getI32ArrayAttr(arange4dShapeI32))
            .getResult();

    // 4. gt(arange, cur_pos): broadcasts to [B, 1, 1, Sk]. The result is in
    // Q's dtype carrying 0.0/1.0 (TTNN comparison ops follow input dtype).
    auto maskShape = llvm::SmallVector<int64_t>{batch, 1, 1, seqLenKV};
    auto maskType =
        ttnn::utils::RankedTensorTypeFactory::create(qType, maskShape);
    Value isMasked =
        rewriter.create<GreaterThanOp>(loc, maskType, indices, curPos)
            .getResult();

    // 5. Zeros and -inf at full mask shape. ttnn.where doesn't reliably
    // broadcast scalar branches against a multi-dim condition, so the t/f
    // branches must be pre-broadcast to match the condition's shape.
    Value zeros = rewriter
                      .create<FullOp>(loc, maskType,
                                      rewriter.getF32FloatAttr(0.0f), device)
                      .getResult();
    Value negInf =
        rewriter
            .create<FullOp>(loc, maskType,
                            rewriter.getF32FloatAttr(
                                -std::numeric_limits<float>::infinity()),
                            device)
            .getResult();

    // 6. where(isMasked, -inf, 0) -> [B, 1, 1, Sk] mask in prefill convention.
    // The mask is in {0, -inf} so pre-scaling by `scale` is a no-op
    // (scale*0=0, scale*-inf=-inf); no MultiplyOp needed.
    scaledMask =
        rewriter.create<WhereOp>(loc, maskType, isMasked, negInf, zeros)
            .getResult();
  }

  // Attention sink: decode layout is [Hq, 32] (tile-padded), prefill expects
  // [1, Hq, 1, 1]. Take column 0 of the tile (the meaningful per-head value)
  // and add the broadcast dims.
  Value attentionSink = op.getAttentionSink();
  if (attentionSink) {
    auto sinkType = mlir::cast<RankedTensorType>(attentionSink.getType());
    auto sinkShape = sinkType.getShape();
    if (sinkShape.size() == 2) {
      int64_t numHeadsQ = sinkShape[0];
      int64_t tileCols = sinkShape[1];

      // Reshape [Hq, tileCols] -> [1, Hq, 1, tileCols].
      llvm::SmallVector<int64_t> sink4dShape = {1, numHeadsQ, 1, tileCols};
      auto sink4dType =
          ttnn::utils::RankedTensorTypeFactory::create(sinkType, sink4dShape);
      llvm::SmallVector<int32_t> sink4dShapeI32 = {
          1, static_cast<int32_t>(numHeadsQ), 1,
          static_cast<int32_t>(tileCols)};
      attentionSink =
          rewriter
              .create<ttnn::ReshapeOp>(loc, sink4dType, attentionSink,
                                       rewriter.getI32ArrayAttr(sink4dShapeI32))
              .getResult();

      // Slice column 0: [1, Hq, 1, tileCols] -> [1, Hq, 1, 1].
      llvm::SmallVector<int64_t> slicedSinkShape = {1, numHeadsQ, 1, 1};
      auto slicedSinkType = ttnn::utils::RankedTensorTypeFactory::create(
          sinkType, slicedSinkShape);
      llvm::SmallVector<int32_t> begins = {0, 0, 0, 0};
      llvm::SmallVector<int32_t> ends = {1, static_cast<int32_t>(numHeadsQ), 1,
                                         1};
      llvm::SmallVector<int32_t> steps = {1, 1, 1, 1};
      attentionSink =
          rewriter
              .create<SliceStaticOp>(loc, slicedSinkType, attentionSink,
                                     rewriter.getI32ArrayAttr(begins),
                                     rewriter.getI32ArrayAttr(ends),
                                     rewriter.getI32ArrayAttr(steps))
              .getResult();
    }
  }

  // Create ScaledDotProductAttentionOp.
  // Result type matches permuted Q shape: [B, H, 1, D].
  // is_causal is always false because causality, when present, has been
  // baked into the synthesized mask via cur_pos_tensor.
  auto sdpaOp = rewriter.create<ttnn::ScaledDotProductAttentionOp>(
      loc, permutedQuery.getType(), permutedQuery, op.getKey(), op.getValue(),
      scaledMask,
      /*is_causal=*/rewriter.getBoolAttr(false), op.getScaleAttr(),
      /*sliding_window_size=*/IntegerAttr(), attentionSink);

  // Permute result back from [B, H, 1, D] to [1, B, H, D].
  Value finalResult =
      createPermute(sdpaOp.getResult(), kFromSDPAPermutation, rewriter, loc);

  rewriter.replaceOp(op, finalResult);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
