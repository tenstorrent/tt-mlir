// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Decomposition/DistributedRMSNormDecompositionRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
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
constexpr int64_t kTileWidth = 32;

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
  if (totalElements % kTileWidth != 0) {
    // Cannot reshape to (N/32, 32) cleanly; leave as-is and let downstream
    // verification surface the error.
    return weight;
  }
  SmallVector<int64_t> reshapedShape = {totalElements / kTileWidth, kTileWidth};
  return reshapeTo(rewriter, loc, weight, reshapedShape);
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
          op.getStats(), op.getSemaphore(), op.getDevice(),
          op.getClusterAxis(), op.getEpsilon(), op.getSubDeviceIdAttr(),
          op.getMemoryConfigAttr(), op.getNumLinksAttr(), op.getTopologyAttr(),
          op.getComputeConfigAttr(), op.getProgramConfigAttr());
      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }

    // Reshape to canonical (1,1,32,M), forward to the fused kernel, then
    // reshape the result back to the original shape.
    SmallVector<int64_t> canonicalShape = {1, 1, 32, inputShape.back()};

    mlir::Value reshapedInput =
        ttir_to_ttnn::utils::generateReshape(
            mlir::cast<mlir::TypedValue<RankedTensorType>>(op.getInput()),
            canonicalShape, rewriter, loc)
            .getResult();

    mlir::Value reshapedResidual = op.getResidual();
    if (reshapedResidual) {
      reshapedResidual =
          ttir_to_ttnn::utils::generateReshape(
              mlir::cast<mlir::TypedValue<RankedTensorType>>(reshapedResidual),
              canonicalShape, rewriter, loc)
              .getResult();
    }

    RankedTensorType canonicalResultType =
        utils::RankedTensorTypeFactory::create(resultType, canonicalShape);

    auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
        loc, canonicalResultType, reshapedInput, reshapedWeight,
        reshapedResidual, op.getStats(), op.getSemaphore(), op.getDevice(),
        op.getClusterAxis(), op.getEpsilon(), op.getSubDeviceIdAttr(),
        op.getNumLinksAttr(), op.getTopologyAttr(),
        op.getComputeConfigAttr(), op.getProgramConfigAttr());

    mlir::Value reshapedResult =
        ttir_to_ttnn::utils::generateReshape(
            newOp.getResult(), resultType.getShape(), rewriter, loc)
            .getResult();

    rewriter.replaceOp(op, reshapedResult);
    return success();
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
  SmallVector<int64_t> statsShape(inputShape.begin(), inputShape.end());
  statsShape.back() = 1;
  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  ttnn::TTNNLayoutAttr statsEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, statsShape);
  RankedTensorType statsType = RankedTensorType::get(
      statsShape, inputType.getElementType(), statsEncoding);

  ArrayAttr dimArg = rewriter.getI32ArrayAttr({static_cast<int32_t>(rank - 1)});
  auto localMeanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_local_mean"), statsType,
      xSqOp.getResult(), /*keep_dim=*/true, dimArg);

  // --- All-gather: gather local stats across devices ---

  SmallVector<int64_t> gatheredShape(statsShape.begin(), statsShape.end());
  gatheredShape.back() = numDevices;
  ttnn::TTNNLayoutAttr gatheredEncoding =
      ttnn::TTNNLayoutAttr::Builder(inputEncoding, gatheredShape);
  RankedTensorType gatheredType = RankedTensorType::get(
      gatheredShape, inputType.getElementType(), gatheredEncoding);

  auto allGatherOp = rewriter.create<ttnn::AllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_all_gather"), gatheredType,
      localMeanOp.getResult(),
      /*all_gather_dim=*/static_cast<int32_t>(rank - 1),
      /*cluster_axis=*/clusterAxis,
      /*sub_device_id=*/nullptr,
      /*num_links=*/op.getNumLinksAttr(),
      /*topology=*/op.getTopologyAttr());

  // --- Post all-gather: normalize using global stats ---

  // global_stats = mean(gathered_stats, dim=-1, keep_dim=true)
  auto globalMeanOp = rewriter.create<ttnn::MeanOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_global_mean"), statsType,
      allGatherOp.getResult(), /*keep_dim=*/true, dimArg);

  // eps_tensor = full(epsilon)
  auto epsTensor = rewriter.create<ttnn::FullOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_epsilon"), statsType,
      rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()),
      op.getDevice());

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
