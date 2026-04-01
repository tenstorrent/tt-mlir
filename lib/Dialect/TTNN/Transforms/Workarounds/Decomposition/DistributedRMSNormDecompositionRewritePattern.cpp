// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Returns true if the input shape is (1,1,32,M) which is the only shape
// supported by the fused_rms_minimal kernel.
static bool isSupportedByFusedKernel(ArrayRef<int64_t> shape) {
  return shape.size() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == 32;
}

LogicalResult DistributedRMSNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Only decompose when the fused kernel cannot handle the shape.
  if (isSupportedByFusedKernel(inputShape)) {
    return failure();
  }

  Location loc = op.getLoc();
  int64_t rank = inputType.getRank();
  uint32_t clusterAxis = op.getClusterAxis();
  mlir::Value input = op.getInput();

  // Look up the mesh shape to determine how many devices are along the
  // cluster axis. This is needed to compute the all_gather output shape.
  auto deviceDesc = ttcore::lookupDevice(op);
  ArrayRef<int64_t> meshShape = deviceDesc.getMeshShape();
  int64_t numDevices = meshShape[clusterAxis];

  // If residual is present, add it to the input first.
  // rms_norm_pre_all_gather in tt-metal accepts residual and computes
  // stats on (input + residual).
  mlir::Value x = input;
  if (op.getResidual()) {
    auto addOp = rewriter.create<ttnn::AddOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_residual_add"), inputType, x,
        op.getResidual());
    x = addOp.getResult();
  }

  // --- Pre all-gather: compute local E(x^2) ---

  // x_sq = multiply(x, x)
  auto xSqOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_square"), inputType, x, x);

  // local_stats = mean(x_sq, dim=-1, keep_dim=true)
  // Output shape: input shape with last dim replaced by 1. The mean op
  // handles the shape reduction internally.
  SmallVector<int64_t> statsShape(inputShape.begin(), inputShape.end());
  statsShape.back() = 1;
  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  RankedTensorType statsType =
      RankedTensorType::get(statsShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(statsShape));

  ArrayAttr dimArg = rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1)});
  auto localMeanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_local_mean"), statsType,
      xSqOp.getResult(), /*keep_dim=*/true, dimArg);

  // --- All-gather: gather local stats across devices ---

  // After all_gather on the last dim, shape becomes (..., numDevices).
  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = numDevices;
  RankedTensorType gatheredType =
      RankedTensorType::get(gatheredShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(gatheredShape));

  auto allGatherOp = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_all_gather"), gatheredType,
      localMeanOp.getResult(),
      /*all_gather_dim=*/static_cast<int32_t>(rank - 1),
      /*cluster_axis=*/clusterAxis,
      /*sub_device_id=*/nullptr,
      /*memory_config=*/nullptr,
      /*num_links=*/op.getNumLinksAttr(),
      /*topology=*/op.getTopologyAttr());

  // --- Post all-gather: normalize using global stats ---

  // global_stats = mean(gathered_stats, dim=-1, keep_dim=true)
  auto globalMeanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_global_mean"), statsType,
      allGatherOp.getResult(), /*keep_dim=*/true, dimArg);

  // eps_tensor = full(epsilon)
  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_epsilon"), statsType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()),
      device.getResult());

  // stabilized = add(global_stats, eps_tensor)
  auto addEpsOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_add_eps"), statsType,
      globalMeanOp.getResult(), epsTensor.getResult());

  // inv_rms = rsqrt(stabilized)
  auto rsqrtOp = rewriter.create<ttnn::RsqrtOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rsqrt"), statsType,
      addEpsOp.getResult());

  // normalized = multiply(x, inv_rms) — broadcasts inv_rms across last dim
  auto normalizedOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_normalize"), inputType, x,
      rsqrtOp.getResult());

  // Apply optional weight (gamma).
  mlir::Value result = normalizedOp.getResult();
  if (op.getWeight()) {
    auto weightOp = rewriter.create<ttnn::MultiplyOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_weight"), inputType, result,
        op.getWeight());
    result = weightOp.getResult();
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
