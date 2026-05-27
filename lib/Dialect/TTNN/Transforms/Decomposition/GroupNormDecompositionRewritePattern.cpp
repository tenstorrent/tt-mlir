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

  const int64_t N = inputShape[0];
  const int64_t S = inputShape[2];
  const int64_t C = inputShape[3];
  const int64_t G = op.getNumGroups();
  if (G <= 0 || C % G != 0) {
    return rewriter.notifyMatchFailure(
        op, "num_groups must evenly divide the channel dimension");
  }
  const int64_t Cpg = C / G;

  Location loc = op.getLoc();

  // Reshape [N, 1, S, C] -> [N, S, G, Cpg] so per-group reductions can happen
  // over a contiguous (S, Cpg) sub-tensor.
  SmallVector<int64_t> groupedShape = {N, S, G, Cpg};
  mlir::Value grouped =
      ttir_to_ttnn::utils::generateReshape(
          mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput()),
          groupedShape, rewriter,
          ttmlir::utils::appendLocationSuffix(loc, "_group_reshape"))
          .getResult();
  RankedTensorType groupedType =
      mlir::cast<RankedTensorType>(grouped.getType());

  SmallVector<int64_t> statsShape = {N, 1, G, 1};
  RankedTensorType statsType =
      utils::RankedTensorTypeFactory::create(inputType, statsShape);
  ArrayAttr reduceDims = rewriter.getI32ArrayAttr({1, 3});

  // mean = mean(grouped, dims=[1,3], keep_dim=true)
  auto meanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_mean"), statsType, grouped,
      /*keep_dim=*/true, reduceDims);

  // centered = grouped - mean  (broadcasts dims 1 and 3 of mean)
  auto centeredOp = rewriter.create<ttnn::SubtractOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_center"), groupedType, grouped,
      meanOp.getResult());

  // variance = mean(centered * centered, dims=[1,3], keep_dim=true)
  auto squaredOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_square"), groupedType,
      centeredOp.getResult(), centeredOp.getResult());
  auto varianceOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_variance"), statsType,
      squaredOp.getResult(), /*keep_dim=*/true, reduceDims);

  // eps tensor: ttnn.full of stats shape filled with epsilon.
  auto deviceOp = utils::getOrInsertDevice(rewriter, op);
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_eps"), statsType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()),
      deviceOp.getResult());

  // inv_std = rsqrt(variance + eps)
  auto stabilizedOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_add_eps"), statsType,
      varianceOp.getResult(), epsTensor.getResult());
  auto invStdOp = rewriter.create<ttnn::RsqrtOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rsqrt"), statsType,
      stabilizedOp.getResult());

  // normalized = centered * inv_std (broadcasts)
  auto normalizedOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_normalize"), groupedType,
      centeredOp.getResult(), invStdOp.getResult());

  // Restore the original [N, 1, S, C] shape.
  mlir::Value result =
      ttir_to_ttnn::utils::generateReshape(
          normalizedOp.getResult(), inputShape, rewriter,
          ttmlir::utils::appendLocationSuffix(loc, "_unreshape"))
          .getResult();

  // Per-channel affine parameters. After TTIRToTTNN they are materialized as
  // 1-D tensors of shape [C]; we reshape to [1, 1, 1, C] so that the broadcast
  // is explicit and unambiguous to downstream layout selection.
  SmallVector<int64_t> affineShape = {1, 1, 1, C};

  // Optional weight: result = result * reshape(weight, [1,1,1,C]).
  if (op.getWeight()) {
    mlir::Value reshapedWeight =
        ttir_to_ttnn::utils::generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getWeight()),
            affineShape, rewriter,
            ttmlir::utils::appendLocationSuffix(loc, "_weight_reshape"))
            .getResult();
    result = rewriter
                 .create<ttnn::MultiplyOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_weight_mul"),
                     resultType, result, reshapedWeight)
                 .getResult();
  }

  // Optional bias: result = result + reshape(bias, [1,1,1,C]).
  if (op.getBias()) {
    mlir::Value reshapedBias =
        ttir_to_ttnn::utils::generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getBias()),
            affineShape, rewriter,
            ttmlir::utils::appendLocationSuffix(loc, "_bias_reshape"))
            .getResult();
    result = rewriter
                 .create<ttnn::AddOp>(
                     ttmlir::utils::appendLocationSuffix(loc, "_bias_add"),
                     resultType, result, reshapedBias)
                 .getResult();
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
