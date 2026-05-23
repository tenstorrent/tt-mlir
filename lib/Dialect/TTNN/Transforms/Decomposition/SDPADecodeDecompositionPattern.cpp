// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/SDPADecodeDecompositionPattern.h"

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {

// SDPADecode Q shape: [1, B, H, D]
// SDPA Q shape:       [B, H, Sq, D]
// Permutation: [1, 2, 0, 3] maps [1, B, H, D] -> [B, H, 1, D]
static constexpr std::array<int64_t, 4> kToSDPAPermutation = {1, 2, 0, 3};
// Inverse: [2, 0, 1, 3] maps [B, H, 1, D] -> [1, B, H, D]
static constexpr std::array<int64_t, 4> kFromSDPAPermutation = {2, 0, 1, 3};

// Create a permute op. Uses RankedTensorTypeFactory when TTNN layout encoding
// is present, falls back to plain RankedTensorType otherwise.
static Value createPermute(Value input, ArrayRef<int64_t> permutation,
                           PatternRewriter &rewriter, Location loc) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  llvm::SmallVector<int64_t> outputShape =
      ttmlir::utils::applyPermutation(inputType.getShape(), permutation);

  RankedTensorType outputType;
  if (inputType.getEncoding()) {
    outputType =
        ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);
  } else {
    outputType = RankedTensorType::get(outputShape, inputType.getElementType());
  }

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

  // Step 1: Permute Q from [1, B, H, D] to [B, H, 1, D].
  Value permutedQuery =
      createPermute(op.getQuery(), kToSDPAPermutation, rewriter, loc);

  // Step 2: Create ScaledDotProductAttentionOp.
  // Result type matches permuted Q shape: [B, H, 1, D].
  auto sdpaOp = rewriter.create<ttnn::ScaledDotProductAttentionOp>(
      loc, permutedQuery.getType(), permutedQuery, op.getKey(), op.getValue(),
      op.getAttentionMask(),
      /*is_causal=*/rewriter.getBoolAttr(false), op.getScaleAttr(),
      /*sliding_window_size=*/IntegerAttr(), op.getAttentionSink());

  // Step 3: Permute result back from [B, H, 1, D] to [1, B, H, D].
  Value finalResult =
      createPermute(sdpaOp.getResult(), kFromSDPAPermutation, rewriter, loc);

  rewriter.replaceOp(op, finalResult);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
