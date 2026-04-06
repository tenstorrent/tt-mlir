// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {

// Returns true if the op can be lowered to the fused_rms_minimal kernel.
// The kernel requires:
//   - input shape (1,1,32,M) where M is a multiple of 32
//     (rms_allgather_device_operation.cpp: shape[0]==1, shape[1]==1,
//      shape[2]==32, shape[3]%32==0)
//   - a weight (gamma) tensor must be present; the kernel asserts
//     gamma.has_value() (https://github.com/tenstorrent/tt-metal/issues/38211)
static bool isSupportedByFusedKernel(ttnn::DistributedRMSNormOp op) {
  ArrayRef<int64_t> shape =
      mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
  return shape.size() == 4 && shape[0] == 1 && shape[1] == 1 &&
         shape[2] == 32 && shape[3] % 32 == 0 && op.getWeight();
}

LogicalResult DistributedRMSNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  // Only decompose when the fused kernel cannot handle the op.
  if (isSupportedByFusedKernel(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  int64_t rank = inputType.getRank();
  uint32_t clusterAxis = op.getClusterAxis();
  mlir::Value input = op.getInput();

  // Determine how many devices are along the cluster axis by inspecting the
  // GetDeviceOp's mesh_shape attribute. This avoids requiring a system
  // descriptor and is consistent with the IR representation.
  auto getDeviceOp = mlir::dyn_cast_if_present<ttnn::GetDeviceOp>(
      op.getDevice().getDefiningOp());
  assert(getDeviceOp && "expected device to be defined by a GetDeviceOp");
  ttnn::MeshShapeAttr meshShapeAttr = getDeviceOp.getMeshShapeAttr();
  assert(meshShapeAttr &&
         "expected GetDeviceOp to have a mesh_shape attribute");
  // MeshShapeAttr stores (y, x); clusterAxis 0 = y-axis, 1 = x-axis.
  int64_t numDevices =
      (clusterAxis == 0) ? meshShapeAttr.getY() : meshShapeAttr.getX();

  // If residual is present, add it to the input first.
  // rms_norm_pre_all_gather in tt-metal accepts residual and computes
  // stats on (input + residual).
  mlir::Value x = input;
  if (op.getResidual()) {
    auto addOp = ttnn::AddOp::create(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_residual_add"),
        inputType, x, op.getResidual());
    x = addOp.getResult();
  }

  // --- Pre all-gather: compute local E(x^2) ---

  // x_sq = multiply(x, x)
  auto xSqOp = ttnn::MultiplyOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_square"), inputType,
      x, x);

  // local_stats = mean(x_sq, dim=-1, keep_dim=true)
  SmallVector<int64_t> statsShape(inputShape.begin(), inputShape.end());
  statsShape.back() = 1;
  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  RankedTensorType statsType =
      RankedTensorType::get(statsShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(statsShape));

  ArrayAttr dimArg = rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1)});
  auto localMeanOp = ttnn::MeanOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_local_mean"),
      statsType, xSqOp.getResult(), /*keep_dim=*/true, dimArg);

  // --- All-gather: gather local stats across devices ---

  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = numDevices;
  RankedTensorType gatheredType =
      RankedTensorType::get(gatheredShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(gatheredShape));

  auto allGatherOp = ttnn::AllGatherOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_all_gather"),
      gatheredType, localMeanOp.getResult(),
      /*all_gather_dim=*/static_cast<int32_t>(rank - 1),
      /*cluster_axis=*/clusterAxis,
      /*sub_device_id=*/nullptr,
      /*memory_config=*/nullptr,
      /*num_links=*/op.getNumLinksAttr(),
      /*topology=*/op.getTopologyAttr());

  // --- Post all-gather: normalize using global stats ---

  // global_stats = mean(gathered_stats, dim=-1, keep_dim=true)
  auto globalMeanOp = ttnn::MeanOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_global_mean"),
      statsType, allGatherOp.getResult(), /*keep_dim=*/true, dimArg);

  // eps_tensor = full(epsilon)
  auto epsTensor = ttnn::FullOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_epsilon"), statsType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()),
      op.getDevice());

  // stabilized = add(global_stats, eps_tensor)
  auto addEpsOp = ttnn::AddOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_add_eps"), statsType,
      globalMeanOp.getResult(), epsTensor.getResult());

  // inv_rms = rsqrt(stabilized)
  auto rsqrtOp = ttnn::RsqrtOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_rsqrt"), statsType,
      addEpsOp.getResult());

  // normalized = multiply(x, inv_rms) — broadcasts inv_rms across last dim
  auto normalizedOp = ttnn::MultiplyOp::create(
      rewriter, ttmlir::utils::appendLocationSuffix(loc, "_normalize"),
      resultType, x, rsqrtOp.getResult());

  // Apply optional weight (gamma).
  mlir::Value result = normalizedOp.getResult();
  if (op.getWeight()) {
    auto weightOp = ttnn::MultiplyOp::create(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_weight"),
        resultType, result, op.getWeight());
    result = weightOp.getResult();
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
