// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::decomposition {

namespace {

// Returns true if the op can be lowered to the fused_rms_minimal kernel, i.e.
// the input is already in canonical (1,1,32,M) shape or can be reshaped there.
// The kernel requires:
//   - a weight (gamma) tensor must be present; the kernel asserts
//     gamma.has_value() (https://github.com/tenstorrent/tt-metal/issues/38211)
//   - input shape with second-to-last dim == 32, last dim a multiple of 32,
//     and all leading dims == 1 (canonical (1,...,1,32,M)).
bool isEligibleForFusedKernel(ttnn::DistributedRMSNormOp op) {
  if (!op.getWeight()) {
    return false;
  }
  ArrayRef<int64_t> shape =
      mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
  if (shape.size() < 2) {
    return false;
  }
  if (shape[shape.size() - 2] != 32 || shape.back() % 32 != 0) {
    return false;
  }
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    if (shape[i] != 1) {
      return false;
    }
  }
  return true;
}

// Returns true when the shape is already the canonical rank-4 (1,1,32,M) form
// that the fused kernel consumes directly.
bool isAlreadyCanonicalShape(ArrayRef<int64_t> shape) {
  return shape.size() == 4 && shape[0] == 1 && shape[1] == 1;
}

mlir::Value reshapeTo(PatternRewriter &rewriter, Location loc, mlir::Value v,
                      ArrayRef<int64_t> targetShape) {
  auto srcType = mlir::cast<RankedTensorType>(v.getType());
  RankedTensorType targetType =
      utils::RankedTensorTypeFactory::create(srcType, targetShape);
  SmallVector<int32_t> targetShapeI32(targetShape.begin(), targetShape.end());
  return rewriter
      .create<ttnn::ReshapeOp>(loc, targetType, v,
                               rewriter.getI32ArrayAttr(targetShapeI32))
      .getResult();
}

// The fused_rms_minimal kernel requires the weight (gamma) tensor to be
// 2D with width equal to tile_width (32). If the weight is still 1D (N,),
// reshape it to (N/32, 32). The Tile -> RowMajor layout conversion is
// handled separately by the layout/sharding workaround pattern.
//
// Returns the (possibly new) weight value, or the original value if no
// reshape is needed.
mlir::Value maybeReshapeWeightToTileWidth(PatternRewriter &rewriter,
                                          Location loc, mlir::Value weight) {
  if (!weight) {
    return weight;
  }
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  if (weightType.getRank() != 1) {
    return weight;
  }
  int64_t totalElements = weightType.getShape()[0];
  if (totalElements % TILE_WIDTH != 0) {
    // Cannot reshape to (N/32, 32) cleanly; leave as-is and let downstream
    // verification surface the error.
    return weight;
  }
  SmallVector<int64_t> reshapedShape = {totalElements / TILE_WIDTH, TILE_WIDTH};
  return reshapeTo(rewriter, loc, weight, reshapedShape);
}

LogicalResult rewriteDistributedRMSNormWithReshape(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter,
    ArrayRef<int64_t> targetShape, mlir::Value weight) {
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

  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      loc, canonicalResultType, reshapedInput, weight, reshapedResidual,
      op.getStats(), op.getSemaphore(), op.getDevice(), op.getClusterAxis(),
      op.getEpsilon(), op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
      op.getTopologyAttr(), op.getComputeConfigAttr(),
      op.getProgramConfigAttr());

  mlir::Value reshapedResult =
      ttir_to_ttnn::utils::generateReshape(newOp.getResult(),
                                           resultType.getShape(), rewriter, loc)
          .getResult();

  rewriter.replaceOp(op, reshapedResult);
  return success();
}

} // namespace

LogicalResult DistributedRMSNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  if (isEligibleForFusedKernel(op)) {
    Location loc = op.getLoc();

    // The fused kernel requires the weight in (N/32, 32) form. Reshape
    // here, alongside the input reshape, so that all kernel-shape prep
    // happens in one place. The Tile -> RowMajor layout conversion stays
    // in the workaround pattern (it is a layout change, not a shape one).
    mlir::Value reshapedWeight =
        maybeReshapeWeightToTileWidth(rewriter, loc, op.getWeight());

    if (isAlreadyCanonicalShape(inputShape)) {
      if (reshapedWeight == op.getWeight()) {
        // Already (1,1,32,M) and weight already in tile-width form — the
        // fused kernel handles this directly.
        return failure();
      }
      // Input is canonical but weight needed reshaping: rebuild the op
      // with the new weight and forward all other operands/attrs through.
      auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
          loc, resultType, op.getInput(), reshapedWeight, op.getResidual(),
          op.getStats(), op.getSemaphore(), op.getDevice(), op.getClusterAxis(),
          op.getEpsilon(), op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
          op.getTopologyAttr(), op.getComputeConfigAttr(),
          op.getProgramConfigAttr());
      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }

    // Reshape to canonical (1,1,32,M), forward to the fused kernel, then
    // reshape the result back to the original shape.
    SmallVector<int64_t> canonicalShapeForFusedKernel = {1, 1, 32,
                                                         inputShape.back()};
    return rewriteDistributedRMSNormWithReshape(
        op, rewriter, canonicalShapeForFusedKernel, reshapedWeight);
  }

  int64_t rank = inputType.getRank();
  // The fallback decomposition lowers through rms_norm_pre_all_gather, whose
  // runtime expects a rank-4 tensor. Left-pad lower-rank shapes with ones so
  // HxW and 1xHxW become 1x1xHxW before decomposition.
  if (rank < 4) {
    SmallVector<int64_t> canonicalShapeForPreAllGather;
    canonicalShapeForPreAllGather.append(4 - rank, 1);
    canonicalShapeForPreAllGather.append(inputShape.begin(), inputShape.end());
    return rewriteDistributedRMSNormWithReshape(
        op, rewriter, canonicalShapeForPreAllGather, op.getWeight());
  }

  Location loc = op.getLoc();
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
    auto addOp = rewriter.create<ttnn::AddOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_residual_add"), inputType, x,
        op.getResidual());
    x = addOp.getResult();
  }

  // --- Pre all-gather: compute local E(x^2) ---
  // Output shape has last dim = TTNN::TILE_WIDTH (32)
  SmallVector<int64_t> statsShape(inputShape.begin(), inputShape.end());
  statsShape.back() = ttnn::TILE_WIDTH;
  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  ttnn::TTNNLayoutAttr statsEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, statsShape);
  RankedTensorType statsType = RankedTensorType::get(
      statsShape, inputType.getElementType(), statsEncoding);

  auto preAllGatherOp = rewriter.create<ttnn::RMSNormPreAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_pre_all_gather"), statsType,
      /*input*/ x,
      /*residual_input=*/mlir::Value{},
      /*compute_config=*/nullptr,
      /*program_config=*/nullptr,
      /*use_2d_core_grid*/ nullptr);

  // --- All-gather: gather local stats across devices ---

  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = ttnn::TILE_WIDTH * numDevices;
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
      /*num_links=*/op.getNumLinksAttr(),
      /*topology=*/op.getTopologyAttr());

  // --- Post all-gather: normalize using global stats ---

  // rms_norm_pre_all_gather produces per-device statistics with shape
  // [..., TILE_WIDTH], where only column 0 stores E(x^2) and the remaining
  // columns are padding/unused.
  //
  // After all_gather, the stats tensor becomes:
  // [..., numDevices * TILE_WIDTH]
  //
  // We cannot directly apply MeanOp over the last dimension because that would
  // incorrectly average across both valid statistics and padded columns,
  // resulting in normalization by (numDevices * TILE_WIDTH).
  //
  // To recover the per-device statistics, reshape:
  // [..., numDevices * TILE_WIDTH] -> [..., numDevices, TILE_WIDTH]
  //
  // Then slice column 0 to extract the valid E(x^2) values from each device:
  // [..., numDevices, TILE_WIDTH] -> [..., numDevices, 1]
  //
  // Flatten [..., numDevices, 1] back to [..., numDevices]
  //
  // Finally, compute the mean across the last dimension to obtain the global
  // E(x^2) used for RMS normalization.

  // [..., numDevices * TILE_WIDTH] -> [..., numDevices, TILE_WIDTH]
  SmallVector<int64_t> reshapedStatsShape(statsType.getShape().begin(),
                                          statsType.getShape().end());
  reshapedStatsShape.back() = numDevices;
  reshapedStatsShape.push_back(ttnn::TILE_WIDTH);
  int64_t reshapedRank = static_cast<int64_t>(reshapedStatsShape.size());

  auto reshapedStatsEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, reshapedStatsShape);

  RankedTensorType reshapedStatsType = RankedTensorType::get(
      reshapedStatsShape, inputType.getElementType(), reshapedStatsEncoding);

  SmallVector<int32_t> reshapedStatsShape32(reshapedStatsShape.begin(),
                                            reshapedStatsShape.end());
  auto reshapedStats = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_reshape_stats"),
      reshapedStatsType, allGatherOp.getResult(),
      rewriter.getI32ArrayAttr(reshapedStatsShape32));

  // Slice column 0: [..., numDevices, TILE_WIDTH] -> [..., numDevices, 1]
  SmallVector<int64_t> slicedShape(reshapedStatsShape);
  slicedShape.back() = 1;

  auto slicedEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, slicedShape);

  RankedTensorType slicedType = RankedTensorType::get(
      slicedShape, inputType.getElementType(), slicedEncoding);

  SmallVector<int32_t> sliceBegins(reshapedRank, 0);
  SmallVector<int32_t> sliceEnds;
  for (int64_t dim : slicedShape) {
    sliceEnds.push_back(static_cast<int32_t>(dim));
  }
  SmallVector<int32_t> sliceSteps(reshapedRank, 1);

  auto sliceEx2 = rewriter.create<ttnn::SliceStaticOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_slice_ex2"), slicedType,
      reshapedStats.getResult(), rewriter.getI32ArrayAttr(sliceBegins),
      rewriter.getI32ArrayAttr(sliceEnds),
      rewriter.getI32ArrayAttr(sliceSteps));

  // Now we have the [..., numDevices, 1], we could either sum across numDevices
  // (reshapedRankDim - 2) then reshape, or we could just reshape [...,
  // numDevices, 1] to [..., numDevices * 1] and pass it to existing
  // globalMeanOp.
  // Flatten [..., numDevices, 1] back to [..., numDevices]
  SmallVector<int64_t> flattenedStatsShape(slicedShape.begin(),
                                           slicedShape.end() - 1);

  auto flattenedEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, flattenedStatsShape);

  RankedTensorType flattenedStatsType = RankedTensorType::get(
      flattenedStatsShape, inputType.getElementType(), flattenedEncoding);

  SmallVector<int32_t> flattenedStatsShapeI32(flattenedStatsShape.begin(),
                                              flattenedStatsShape.end());

  auto flattenedStats = rewriter.create<ttnn::ReshapeOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_flatten_stats"),
      flattenedStatsType, sliceEx2.getResult(),
      rewriter.getI32ArrayAttr(flattenedStatsShapeI32));

  // Mean across device dimension (now last dim).
  // [..., numDevices] to [..., 1]
  SmallVector<int64_t> scalarRowShape(inputShape.begin(), inputShape.end());
  scalarRowShape.back() = 1;

  auto scalarEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, scalarRowShape);

  RankedTensorType scalarRowType = RankedTensorType::get(
      scalarRowShape, inputType.getElementType(), scalarEncoding);

  ArrayAttr dimArg = rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1)});
  // mean of the per-device partial stats across devices.
  auto globalMeanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_global_mean"), scalarRowType,
      flattenedStats.getResult(), /*keep_dim=*/true, dimArg);

  // Stats are per-device sum(x^2); divide by hidden size N to get E(x^2).
  double invHiddenPerDevice = 1.0 / static_cast<double>(inputShape.back());
  auto invHiddenTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_inv_hidden"), scalarRowType,
      rewriter.getF32FloatAttr(static_cast<float>(invHiddenPerDevice)),
      op.getDevice());
  // global_stats = E(x^2) = mean_over_devices(sum(x^2)) / N
  auto globalStatsOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_scale_ex2"), scalarRowType,
      globalMeanOp.getResult(), invHiddenTensor.getResult());

  // eps_tensor = full(epsilon)
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_epsilon"), scalarRowType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()),
      op.getDevice());

  // stabilized = add(global_stats, eps_tensor)
  auto addEpsOp = rewriter.create<ttnn::AddOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_add_eps"), scalarRowType,
      globalStatsOp.getResult(), epsTensor.getResult());

  // inv_rms = rsqrt(stabilized)
  auto rsqrtOp = rewriter.create<ttnn::RsqrtOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_rsqrt"), scalarRowType,
      addEpsOp.getResult());

  // normalized = multiply(x, inv_rms) — broadcasts inv_rms across last dim
  auto normalizedOp = rewriter.create<ttnn::MultiplyOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_normalize"), resultType, x,
      rsqrtOp.getResult());

  // Apply optional weight (gamma).
  mlir::Value result = normalizedOp.getResult();
  if (op.getWeight()) {
    auto weightOp = rewriter.create<ttnn::MultiplyOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_weight"), resultType, result,
        op.getWeight());
    result = weightOp.getResult();
  }

  rewriter.replaceOp(op, result);
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
