// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DitRMSNormUnaryFusedDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

// Create the TTNN unary activation op matching `activation`. Returns nullptr
// (and reports a match failure) if the activation string is unsupported.
static mlir::Value createActivationOp(PatternRewriter &rewriter, Location loc,
                                      Type resultType, mlir::Value input,
                                      llvm::StringRef activation) {
  if (activation == "silu") {
    return rewriter.create<ttnn::SiluOp>(loc, resultType, input).getResult();
  }
  if (activation == "gelu") {
    return rewriter.create<ttnn::GeluOp>(loc, resultType, input).getResult();
  }
  if (activation == "relu") {
    return rewriter.create<ttnn::ReluOp>(loc, resultType, input).getResult();
  }
  return nullptr;
}

LogicalResult DitRMSNormUnaryFusedDecompositionRewritePattern::matchAndRewrite(
    ttnn::DitRMSNormUnaryFusedOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  // If an activation is present it must be one we can express with a TTNN
  // unary op. Bail out early (before any validation) if not, so we never
  // produce an incorrect decomposition.
  mlir::StringAttr activationAttr = op.getActivationAttr();
  if (activationAttr && activationAttr.getValue() != "silu" &&
      activationAttr.getValue() != "gelu" &&
      activationAttr.getValue() != "relu") {
    return rewriter.notifyMatchFailure(
        op, "unsupported activation for decomposition");
  }

  // When validation config is provided, validate the fused op. If validation
  // succeeds, keep the op as-is (only decompose on failure).
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult = validator.validateOp<ttnn::DitRMSNormUnaryFusedOp>(
        op.getOperation(), op.getLoc(), {resultType}, op.getInput(),
        op.getWeight(), op.getBias(), op.getResidualInput(),
        op.getEpsilonAttr(), op.getActivationAttr(), op.getMemoryConfigAttr(),
        op.getComputeConfigAttr());

    if (validationResult.isSuccess()) {
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "dit_rms_norm_unary_fused decomposition triggered (validation "
                 "failed): {0}",
                 validationResult.errorMessage);
  }

  Location loc = op.getLoc();

  // Fused residual pre-add: x = input + residual_input.
  mlir::Value normInput = op.getInput();
  if (op.getResidualInput()) {
    normInput = rewriter
                    .create<ttnn::AddOp>(
                        ttmlir::utils::appendLocationSuffix(loc, "_residual"),
                        inputType, op.getInput(), op.getResidualInput())
                    .getResult();
  }

  // n = rms_norm(x, weight, bias, epsilon, compute_config).
  auto rmsNorm = rewriter.create<ttnn::RMSNormOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rms_norm"), resultType,
      normInput, op.getWeight(), op.getBias(), op.getEpsilonAttr(),
      op.getComputeConfigAttr());

  // out = activation(n) if present, otherwise the rms_norm result.
  mlir::Value result = rmsNorm.getResult();
  if (activationAttr) {
    result = createActivationOp(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_activation"),
        resultType, result, activationAttr.getValue());
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
