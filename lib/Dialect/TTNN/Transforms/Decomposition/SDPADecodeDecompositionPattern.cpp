// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include <cmath>

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

  // Decode applies the mask as softmax(scale*(QK + mask)), but regular SDPA
  // applies it as softmax(scale*QK + mask). Pre-scale the mask so the
  // downstream regular SDPA produces decode-equivalent values.
  Value scaledMask = op.getAttentionMask();
  if (scaledMask) {
    float scaleValue;
    if (op.getScaleAttr()) {
      scaleValue = op.getScaleAttr().getValueAsDouble();
    } else {
      int64_t headDim = qType.getShape().back();
      scaleValue = 1.0f / std::sqrt(static_cast<float>(headDim));
    }
    auto maskType = mlir::cast<RankedTensorType>(scaledMask.getType());
    llvm::SmallVector<int64_t> scalarShape(maskType.getRank(), 1);
    RankedTensorType scalarType =
        ttnn::utils::RankedTensorTypeFactory::create(maskType, scalarShape);
    Value device =
        utils::getOrInsertDevice(rewriter, op.getOperation()).getResult();
    Value scaleTensor =
        rewriter
            .create<FullOp>(loc, scalarType,
                            rewriter.getF32FloatAttr(scaleValue), device)
            .getResult();
    scaledMask = rewriter
                     .create<MultiplyOp>(loc, maskType, scaledMask, scaleTensor)
                     .getResult();

    // Decode mask layout is [B, 1, Hq, Sk] (heads at dim 2). The downstream
    // regular SDPA op expects [B, Hq, Sq, Sk] (heads at dim 1, Sq=1 here).
    // Permute dims 1 and 2 to bridge the convention difference.
    scaledMask =
        createPermute(scaledMask, kMaskToSDPAPermutation, rewriter, loc);
  }

  // Create ScaledDotProductAttentionOp.
  // Result type matches permuted Q shape: [B, H, 1, D].
  auto sdpaOp = rewriter.create<ttnn::ScaledDotProductAttentionOp>(
      loc, permutedQuery.getType(), permutedQuery, op.getKey(), op.getValue(),
      scaledMask,
      /*is_causal=*/rewriter.getBoolAttr(false), op.getScaleAttr(),
      /*sliding_window_size=*/IntegerAttr(), op.getAttentionSink());

  // Permute result back from [B, H, 1, D] to [1, B, H, D].
  Value finalResult =
      createPermute(sdpaOp.getResult(), kFromSDPAPermutation, rewriter, loc);

  rewriter.replaceOp(op, finalResult);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
