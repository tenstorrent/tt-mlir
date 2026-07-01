// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/GroupNormDecompositionRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult GroupNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::GroupNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  // When validation config is provided, validate the GroupNorm operation using
  // IsolatedIRValidationWrapper. If validation succeeds, keep the op as-is.
  if (validationConfig.has_value()) {
    IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                          *validationConfig);

    auto validationResult = validator.validateOp<ttnn::GroupNormOp>(
        op.getOperation(), op.getLoc(), {resultType}, op.getInput(),
        op.getInputMask(), op.getWeight(), op.getBias(), op.getNumGroupsAttr(),
        op.getEpsilonAttr());

    if (validationResult.isSuccess()) {
      return failure();
    }

    TTMLIR_DEBUG(ttmlir::LogComponent::IsolatedIRValidationWrapper,
                 "GroupNorm decomposition triggered (validation failed): {0}",
                 validationResult.errorMessage);
  }

  // The TTIRToTTNN conversion places GroupNorm inputs in [N, 1, H*W, C] form
  // (channels in the last dim). If we ever see something different, bail out
  // and leave the op as-is rather than emit an incorrect decomposition.
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (inputShape.size() != 4) {
    return rewriter.notifyMatchFailure(
        op, "expected rank-4 input in canonical [N, 1, H*W, C] form");
  }

  const int64_t C = inputShape[3];
  const int64_t G = op.getNumGroups();
  if (G <= 0 || C % G != 0) {
    return rewriter.notifyMatchFailure(
        op, "num_groups must evenly divide the channel dimension");
  }
  // Emit the primitive-op decomposition (shared with the TTIRToTTNN conversion
  // so there is a single source of truth for the group-norm math).
  mlir::Value result = ttir_to_ttnn::utils::decomposeGroupNorm(
      mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput()),
      op.getWeight(), op.getBias(), op.getNumGroups(), op.getEpsilon(),
      resultType, op.getOperation(), rewriter, op.getLoc());

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
