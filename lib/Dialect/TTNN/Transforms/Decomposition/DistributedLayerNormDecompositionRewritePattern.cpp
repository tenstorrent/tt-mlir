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

mlir::Value reshapeTo(PatternRewriter &rewriter, Location loc, mlir::Value v,
                      ArrayRef<int64_t> targetShape) {
  return ttir_to_ttnn::utils::generateReshape(
             mlir::cast<mlir::TypedValue<RankedTensorType>>(v), targetShape,
             rewriter, loc)
      .getResult();
}

mlir::Value typecastTo(PatternRewriter &rewriter, Location loc, mlir::Value v,
                       ttcore::DataType dataType) {
  RankedTensorType srcType = mlir::cast<RankedTensorType>(v.getType());
  RankedTensorType dstType =
      utils::RankedTensorTypeFactory::create(srcType, dataType);
  if (srcType == dstType) {
    return v;
  }
  return rewriter.create<ttnn::TypecastOp>(loc, dstType, v).getResult();
}

Type getScalarElementType(RankedTensorType type) {
  Type elem = type.getElementType();
  if (auto tile = mlir::dyn_cast<ttcore::TileType>(elem)) {
    elem = tile.getElementType();
  }
  return elem;
}

bool isBF16ElementType(RankedTensorType type) {
  return mlir::isa<BFloat16Type>(getScalarElementType(type));
}

bool isFusedKernelElementType(RankedTensorType type) {
  Type elem = getScalarElementType(type);
  return mlir::isa<BFloat16Type>(elem) || mlir::isa<Float32Type>(elem);
}

bool affineParamIsFusedForm(mlir::Value value, int64_t hidden) {
  if (!value) {
    return false;
  }
  RankedTensorType type = mlir::cast<RankedTensorType>(value.getType());
  ArrayRef<int64_t> shape = type.getShape();
  return type.getRank() == 2 && shape.back() == hidden &&
         (shape[0] == 1 || shape[0] > 1);
}

// DiT fused LN: no residual, weight and bias present, last dim tile-aligned.
// Element type is bf16 or f32 (XLA LayerNorm upcasts to f32). Rank 3
// `[1,N,H]` (Wan AdaLN) or rank 4 `[1,B,N,H]`. Rank-4 with 1D affine params
// stays on the sandwich so existing goldens/lit keep pre/post. f32 is
// typecast to TILE bf16 around the fused op. After the rank-3 rewrite, an
// already-canonical bf16 rank-4 op is left intact (`return failure()`).
bool isEligibleForDitFusedKernel(ttnn::DistributedLayerNormOp op) {
  if (op.getResidual() || !op.getWeight() || !op.getBias()) {
    return false;
  }
  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  if (!isFusedKernelElementType(inputType)) {
    return false;
  }
  ArrayRef<int64_t> shape = inputType.getShape();
  if (shape.size() < 2 || shape.back() <= 0 ||
      shape.back() % TILE_WIDTH != 0) {
    return false;
  }
  if (shape.size() == 3) {
    return shape[0] == 1;
  }
  if (shape.size() == 4) {
    if (shape[0] != 1) {
      return false;
    }
    return affineParamIsFusedForm(op.getWeight(), shape.back()) &&
           affineParamIsFusedForm(op.getBias(), shape.back());
  }
  return false;
}

ttnn::DistributedLayerNormOp
createDistributedLayerNorm(PatternRewriter &rewriter, Location loc,
                            RankedTensorType resultType, mlir::Value input,
                            mlir::Value weight, mlir::Value bias,
                            mlir::Value residual, mlir::Value device,
                            ttnn::DistributedLayerNormOp srcOp) {
  return rewriter.create<ttnn::DistributedLayerNormOp>(
      loc, resultType, input, weight, bias, residual,
      srcOp.getStats(), srcOp.getSemaphore(), device,
      srcOp.getClusterAxis(), srcOp.getEpsilon(), srcOp.getNumLinksAttr(),
      srcOp.getTopologyAttr(), srcOp.getComputeConfigAttr());
}

LogicalResult rewriteToDitFusedForm(ttnn::DistributedLayerNormOp op,
                                     PatternRewriter &rewriter) {
  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  Location loc = op.getLoc();

  SmallVector<int64_t> fusedInputShape(inputShape.begin(), inputShape.end());
  if (inputType.getRank() == 3) {
    fusedInputShape = {1, 1, inputShape[1], inputShape[2]};
  }

  mlir::Value fusedInput = op.getInput();
  bool changed = false;
  if (inputShape != ArrayRef<int64_t>(fusedInputShape)) {
    fusedInput = reshapeTo(rewriter, loc, fusedInput, fusedInputShape);
    changed = true;
  }

  SmallVector<int64_t> affineShape = {1, inputShape.back()};
  mlir::Value fusedWeight = op.getWeight();
  mlir::Value fusedBias = op.getBias();
  if (mlir::cast<RankedTensorType>(fusedWeight.getType()).getRank() == 1) {
    fusedWeight = reshapeTo(rewriter, loc, fusedWeight, affineShape);
    changed = true;
  }
  if (mlir::cast<RankedTensorType>(fusedBias.getType()).getRank() == 1) {
    fusedBias = reshapeTo(rewriter, loc, fusedBias, affineShape);
    changed = true;
  }

  // Metal dit_fused_distributed_layernorm is TILE bf16. XLA LayerNorm
  // upcasts to f32; insert casts around the fused op and restore the
  // original element type on the result.
  bool needsBf16Cast = !isBF16ElementType(inputType);
  if (needsBf16Cast) {
    fusedInput =
        typecastTo(rewriter, loc, fusedInput, ttcore::DataType::BFloat16);
    fusedWeight =
        typecastTo(rewriter, loc, fusedWeight, ttcore::DataType::BFloat16);
    fusedBias =
        typecastTo(rewriter, loc, fusedBias, ttcore::DataType::BFloat16);
    changed = true;
  }

  if (!changed) {
    return failure();
  }

  RankedTensorType fusedResultType =
      utils::RankedTensorTypeFactory::create(resultType, fusedInputShape);
  if (needsBf16Cast) {
    fusedResultType = utils::RankedTensorTypeFactory::create(
        fusedResultType, ttcore::DataType::BFloat16);
  }
  auto fusedOp = createDistributedLayerNorm(
      rewriter, loc, fusedResultType, fusedInput, fusedWeight, fusedBias,
      /*residual=*/mlir::Value{}, op.getDevice(), op);

  mlir::Value result = fusedOp.getResult();
  if (needsBf16Cast) {
    RankedTensorType origElemFusedType =
        utils::RankedTensorTypeFactory::create(resultType, fusedInputShape);
    result = rewriter.create<ttnn::TypecastOp>(loc, origElemFusedType, result)
                 .getResult();
  }
  if (resultType.getShape() != ArrayRef<int64_t>(fusedInputShape)) {
    result = reshapeTo(rewriter, loc, result, resultType.getShape());
  }
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult rewriteDistributedLayerNormWithReshape(
    ttnn::DistributedLayerNormOp op, PatternRewriter &rewriter,
    ArrayRef<int64_t> targetShape) {
  Location loc = op.getLoc();
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  mlir::Value reshapedInput = reshapeTo(rewriter, loc, op.getInput(), targetShape);

  mlir::Value reshapedResidual = op.getResidual();
  if (reshapedResidual) {
    reshapedResidual = reshapeTo(rewriter, loc, reshapedResidual, targetShape);
  }

  RankedTensorType canonicalResultType =
      utils::RankedTensorTypeFactory::create(resultType, targetShape);

  auto newOp = createDistributedLayerNorm(
      rewriter, loc, canonicalResultType, reshapedInput, op.getWeight(),
      op.getBias(), reshapedResidual, op.getDevice(), op);

  mlir::Value reshapedResult =
      reshapeTo(rewriter, loc, newOp.getResult(), resultType.getShape());
  rewriter.replaceOp(op, reshapedResult);
  return success();
}

} // namespace

LogicalResult DistributedLayerNormDecompositionRewritePattern::matchAndRewrite(
    ttnn::DistributedLayerNormOp op, PatternRewriter &rewriter) const {

  if (isEligibleForDitFusedKernel(op)) {
    return rewriteToDitFusedForm(op, rewriter);
  }

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());
  RankedTensorType resultType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();

  Location loc = op.getLoc();
  int64_t rank = inputType.getRank();
  if (rank < 4) {
    SmallVector<int64_t> canonicalShapeForPreAllGather;
    canonicalShapeForPreAllGather.append(4 - rank, 1);
    canonicalShapeForPreAllGather.append(inputShape.begin(), inputShape.end());
    return rewriteDistributedLayerNormWithReshape(
        op, rewriter, canonicalShapeForPreAllGather);
  }

  uint32_t clusterAxis = op.getClusterAxis();

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
  int64_t numDevices =
      (clusterAxis == 0) ? meshShapeAttr.getY() : meshShapeAttr.getX();

  auto inputEncoding =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputType.getEncoding());

  mlir::Value normInput = op.getInput();
  if (op.getResidual()) {
    auto addOp = rewriter.create<ttnn::AddOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_residual_add"), inputType,
        normInput, op.getResidual());
    normInput = addOp.getResult();
  }

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

  auto postAllGatherOp = rewriter.create<ttnn::LayerNormPostAllGatherOp>(
      ttmlir::utils::appendLocationSuffix(loc, "_post_all_gather"), resultType,
      normInput, allGatherOp.getResult(), op.getWeight(), op.getBias(),
      op.getEpsilonAttr(), computeConfig,
      /*program_config=*/nullptr);

  rewriter.replaceOp(op, postAllGatherOp.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::decomposition
