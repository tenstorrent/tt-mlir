// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedLayerNormDecompositionRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {
namespace {

// Metal's layer_norm_pre/post_all_gather wrappers default fp32_dest_acc_en to
// false. Welford fatals on Float32 input unless that flag is true (precision
// is otherwise silently lost in the unpacker). Match RMSNorm's high-precision
// default so bf16 is also legal.
DeviceComputeKernelConfigAttr
createDistributedLayerNormComputeConfig(MLIRContext *context) {
  return DeviceComputeKernelConfigAttr::get(
      context,
      /*mathFidelity=*/MathFidelity::HiFi4,
      /*mathApproxMode=*/BoolAttr::get(context, false),
      /*fp32DestAccEn=*/BoolAttr::get(context, true),
      /*packerL1Acc=*/BoolAttr::get(context, true),
      /*dstFullSyncEn=*/nullptr);
}

// layer_norm_pre_all_gather's runtime indexes output_shape[3], so it expects
// a rank-4 tensor. Left-pad lower-rank inputs with ones, run the existing
// rank-4 decomposition on the wrapped op, then reshape the result back.
LogicalResult rewriteDistributedLayerNormWithReshape(
    ttnn::DistributedLayerNormOp op, PatternRewriter &rewriter,
    ArrayRef<int64_t> targetShape) {
  Location loc = op.getLoc();
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  mlir::Value reshapedInput =
      ttir_to_ttnn::utils::generateReshape(
          mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput()),
          targetShape, rewriter, loc)
          .getResult();

  mlir::Value reshapedResidual = op.getResidual();
  if (reshapedResidual) {
    reshapedResidual =
        ttir_to_ttnn::utils::generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(reshapedResidual),
            targetShape, rewriter, loc)
            .getResult();
  }

  RankedTensorType canonicalResultType =
      utils::RankedTensorTypeFactory::create(resultType, targetShape);

  auto newOp = rewriter.create<ttnn::DistributedLayerNormOp>(
      loc, canonicalResultType, reshapedInput, op.getWeight(), op.getBias(),
      reshapedResidual, op.getDevice(), op.getClusterAxis(), op.getEpsilon());

  mlir::Value reshapedResult =
      ttir_to_ttnn::utils::generateReshape(newOp.getResult(),
                                           resultType.getShape(), rewriter, loc)
          .getResult();

  rewriter.replaceOp(op, reshapedResult);
  return success();
}

} // namespace

LogicalResult DistributedLayerNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedLayerNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  Location loc = op.getLoc();
  int64_t rank = inputType.getRank();
  // The decomposition lowers through layer_norm_pre_all_gather, whose runtime
  // expects a rank-4 tensor. Left-pad lower-rank shapes with ones so HxW and
  // 1xHxW become 1x1xHxW before decomposition. The greedy rewriter then
  // matches the wrapped rank-4 op and emits pre/post all-gather with
  // all_gather_dim on the last axis.
  if (rank < 4) {
    SmallVector<int64_t> canonicalShapeForPreAllGather;
    canonicalShapeForPreAllGather.append(4 - rank, 1);
    canonicalShapeForPreAllGather.append(inputShape.begin(), inputShape.end());
    return rewriteDistributedLayerNormWithReshape(
        op, rewriter, canonicalShapeForPreAllGather);
  }

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
  ttnn::TTNNLayoutAttr statsEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, statsShape);
  RankedTensorType statsType = RankedTensorType::get(
      statsShape, inputType.getElementType(), statsEncoding);

  DeviceComputeKernelConfigAttr computeConfig =
      createDistributedLayerNormComputeConfig(rewriter.getContext());

  auto preAllGatherOp = rewriter.create<ttnn::LayerNormPreAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_pre_all_gather"), statsType,
      normInput,
      /*residual_input=*/mlir::Value{},
      /*recip=*/mlir::Value{},
      computeConfig,
      /*program_config=*/nullptr);

  // --- Step 3: all_gather ---
  // Gather partial statistics from all devices along cluster_axis.
  // The gathered stats tensor has last dim = ttnn::LAYER_NORM_STATS_WIDTH *
  // numDevices.

  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = ttnn::LAYER_NORM_STATS_WIDTH * numDevices;
  ttnn::TTNNLayoutAttr gatheredEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, gatheredShape);
  RankedTensorType gatheredType = RankedTensorType::get(
      gatheredShape, inputType.getElementType(), gatheredEncoding);

  auto allGatherOp = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_all_gather"), gatheredType,
      preAllGatherOp.getResult(),
      /*all_gather_dim=*/static_cast<int32_t>(rank - 1),
      /*cluster_axis=*/clusterAxis,
      /*sub_device_id=*/nullptr,
      /*num_links=*/nullptr,
      /*topology=*/nullptr);

  // --- Step 4: layer_norm_post_all_gather ---
  // Normalize norm_input using the globally gathered statistics.
  // Optionally apply weight (gamma) and bias (beta).

  auto postAllGatherOp = rewriter.create<ttnn::LayerNormPostAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_post_all_gather"), resultType,
      normInput, allGatherOp.getResult(), op.getWeight(), op.getBias(),
      op.getEpsilonAttr(), computeConfig,
      /*program_config=*/nullptr);

  rewriter.replaceOp(op, postAllGatherOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
