// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedLayerNormDecompositionRewritePattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {

LogicalResult DistributedLayerNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedLayerNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  Location loc = op.getLoc();
  int64_t rank = inputType.getRank();
  uint32_t clusterAxis = op.getClusterAxis();

  // Determine how many devices are along the cluster axis by inspecting the
  // GetDeviceOp's mesh_shape attribute.
  auto getDeviceOp = mlir::dyn_cast_if_present<ttnn::GetDeviceOp>(
      op.getDevice().getDefiningOp());
  if (!getDeviceOp) {
    return op->emitOpError("expected device to be defined by a GetDeviceOp");
  }
  ttnn::MeshShapeAttr meshShapeAttr = getDeviceOp.getMeshShapeAttr();
  if (!meshShapeAttr) {
    return op->emitOpError(
        "expected GetDeviceOp to have a mesh_shape attribute");
  }
  // MeshShapeAttr stores (y, x); clusterAxis 0 = y-axis, 1 = x-axis.
  int64_t numDevices =
      (clusterAxis == 0) ? meshShapeAttr.getY() : meshShapeAttr.getX();

  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());

  // --- Step 1: Optional residual add ---
  // norm_input = input + residual (if residual is present)
  // norm_input is passed to both pre_all_gather (for stats computation) and
  // post_all_gather (for normalization), so the add is computed only once.
  mlir::Value normInput = op.getInput();
  if (op.getResidual()) {
    auto addOp = rewriter.create<ttnn::AddOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_residual_add"), inputType,
        normInput, op.getResidual());
    normInput = addOp.getResult();
  }

  // --- Step 2: layer_norm_pre_all_gather ---
  // Computes local partial statistics (sum(x) and sum(x^2)) on the
  // local shard of norm_input. Output shape has last dim =
  // ttnn::LAYER_NORM_STATS_WIDTH
  // (= 64 = 2 * TILE_WIDTH).
  SmallVector<int64_t> statsShape(inputShape.begin(), inputShape.end());
  statsShape.back() = ttnn::LAYER_NORM_STATS_WIDTH;
  RankedTensorType statsType =
      RankedTensorType::get(statsShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(statsShape));

  auto dtypeAttr = ttcore::DataTypeAttr::get(
      rewriter.getContext(),
      ttcore::elementTypeToDataType(inputType.getElementType()));

  auto preAllGatherOp = rewriter.create<ttnn::LayerNormPreAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_pre_all_gather"), statsType,
      normInput,
      /*residual_input=*/mlir::Value{},
      /*recip=*/mlir::Value{},
      /*dtype=*/dtypeAttr,
      /*memory_config=*/nullptr,
      /*compute_config=*/nullptr,
      /*program_config=*/nullptr);

  // --- Step 3: all_gather ---
  // Gather partial statistics from all devices along cluster_axis.
  // The gathered stats tensor has last dim = ttnn::LAYER_NORM_STATS_WIDTH *
  // numDevices.

  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = ttnn::LAYER_NORM_STATS_WIDTH * numDevices;
  RankedTensorType gatheredType =
      RankedTensorType::get(gatheredShape, inputType.getElementType(),
                            inputEncoding.withTensorShape(gatheredShape));

  auto allGatherOp = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_all_gather"), gatheredType,
      preAllGatherOp.getResult(),
      /*all_gather_dim=*/static_cast<int32_t>(rank - 1),
      /*cluster_axis=*/clusterAxis,
      /*sub_device_id=*/nullptr,
      /*memory_config=*/nullptr,
      /*num_links=*/nullptr,
      /*topology=*/nullptr);

  // --- Step 4: layer_norm_post_all_gather ---
  // Normalize norm_input using the globally gathered statistics.
  // Optionally apply weight (gamma) and bias (beta).

  auto postAllGatherOp = rewriter.create<ttnn::LayerNormPostAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_post_all_gather"), resultType,
      normInput, allGatherOp.getResult(), op.getWeight(), op.getBias(),
      op.getEpsilonAttr(),
      /*dtype=*/dtypeAttr,
      /*memory_config=*/nullptr,
      /*compute_config=*/nullptr,
      /*program_config=*/nullptr);

  rewriter.replaceOp(op, postAllGatherOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
