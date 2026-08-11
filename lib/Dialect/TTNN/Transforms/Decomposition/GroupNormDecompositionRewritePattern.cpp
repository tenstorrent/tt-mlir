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

  // The fused ttnn.group_norm kernel used to reduce over the tile-padding rows
  // as if they were data, so it was silently wrong for a non-tile-aligned
  // per-sample H*W and this pattern decomposed every such shape. tt-metal
  // #52685 makes the kernel EXCLUDE those rows -- it switches to a row-masked
  // copy of the input mask on the final row-tile of each batch -- so the
  // alignment precondition is gone and the op model alone decides: anything it
  // accepts keeps the fused kernel. Without a validationConfig (optimizer off)
  // we still decompose.
  //
  // Requires a tt-metal carrying #52685. Against an older one this gate hands
  // non-tile-aligned shapes to a kernel that reads its own tile padding as data.
  //
  // The old K-dependent error term is GONE. #50682's original correction
  // back-corrected the padding's contribution as Var - K*E[x]^2 (K =
  // padded/logical - 1), a bfloat16 cancellation whose error grew with K; that
  // subtraction no longer exists, because the padding never enters either sum.
  // Re-measured on Wormhole B0, uniform inputs, 0.08 op tolerance, each against
  // a tile-aligned control:
  //
  //   H*W (K)      mean 0      aligned control
  //   16 (1.00)    0.0447      0.0436
  //   17 (0.88)    0.0473      0.0436
  //   24 (0.33)    0.0415      0.0436
  //   33 (0.94)    0.0456      0.0488
  //   50 (0.28)    0.0507      0.0488
  //   100 (0.28)   0.0406      0.0549
  //
  // Error no longer tracks K at all -- H*W=16 at K=1.00 now beats H*W=50 at
  // K=0.28 -- and every case sits at the aligned control. The previous "only
  // H*W <= 16 exceeds tolerance" caveat is obsolete.
  //
  // What DOES still hurt is a large input mean, and it hurts tile-ALIGNED
  // shapes just as much or more: at mean 100 the non-aligned H*W=16 measures
  // 0.313 while its aligned control measures 1.975. That is the fused kernel's
  // own bfloat16 behaviour when (x - E[x]) cancels, not anything to do with
  // alignment, so it is not a reason to keep decomposing non-aligned shapes
  // specifically.
  //
  // Blocker 1 (padding contents) is CLOSED by #52685. Verified: the kernel's
  // output is now bit-identical for tile padding of 0.0 / 7.0 / 1.0 / -3.5 /
  // 0.5 / 100.0 on both interleaved and block-sharded inputs, and the XTTS-v2
  // conditioning encoder recovered from PCC 0.822335 to 0.999100.
  //
  // *** Blocker 2 REMAINS, and is a judgement call, not a bug: ***
  //
  //    On the XTTS-v2 conditioning encoder (real activations) fused reaches PCC
  //    0.999100 against the decomposed path's 0.999552 -- better than the
  //    0.998208 recorded before #52685, but still short. The gap is NOT the
  //    padding handling: on the same real tensor, tile-ALIGNED shapes measure
  //    the same or worse (H*W=256 -> 0.5683 max err vs H*W=259 -> 0.4888), so
  //    it is the fused kernel's own bfloat16 accuracy on inputs with a wide
  //    per-group variance spread (173x here). Relaxing the gate therefore
  //    trades a little accuracy for the fused kernel and has to be accepted
  //    deliberately.
  //
  // Perf note: this lowering passes no input_mask, so ttnn::group_norm builds
  // one per call. For non-tile-aligned H*W that mask is twice the size (it
  // carries the row-masked set), measured 158 -> 255 us/call interleaved. If
  // group_norm lands in a hot loop, hoist the mask via
  // create_group_norm_input_mask(..., rows_in_last_tile = H*W % 32).
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
