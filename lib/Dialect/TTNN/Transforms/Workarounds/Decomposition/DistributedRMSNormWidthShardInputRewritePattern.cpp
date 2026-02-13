// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Prepares DistributedRMSNormOp for the tt-metal fused_rms_minimal kernel.
//
// The workaround addresses the following requirements:
//
//  1. Input and residual tensors must be width-sharded in L1 with a ROW_MAJOR
//     shard spec (so the program factory can read the shard grid/shape).
//  2. The weight tensor must be in ROW_MAJOR layout with width equal to
//     tile_width (32). We reshape from 1D (N,) to 2D (N/32, 32) and convert.
//  3. The output memory config must match the input shard spec
//     (skip_write_back = true). The fused kernel only all-gathers stats,
//     not data, so output shape == input shape.
//  4. A compute_config must be set. If absent, a default HiFi4 config with
//     fp32_dest_acc_en=true is created.
//  5. A stats scratch tensor (1x1x32x32, width-sharded on core (0,0), L1) is
//     created as an EmptyOp. Its dtype (Float32 or BFloat16) is derived from
//     fp32_dest_acc_en in the compute_config.
//  6. A LayerNormShardedMultiCoreProgramConfig is computed from the input's
//     shard spec (grid size, block_h, block_w) and attached as an attribute.
//
// If the original output layout differs from the sharded config (e.g.
// interleaved DRAM), a to_memory_config op is inserted after the norm.
//
LogicalResult DistributedRMSNormWidthShardInputRewritePattern::matchAndRewrite(
    ttnn::DistributedRMSNormOp op, PatternRewriter &rewriter) const {

  RankedTensorType inputType =
      mlir::cast<RankedTensorType>(op.getInput().getType());

  // Check if the input is already width-sharded in L1.
  ttnn::TTNNLayoutAttr currentInputLayout =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(inputType.getEncoding());
  if (currentInputLayout) {
    auto memLayout = currentInputLayout.getMemLayout();
    if (memLayout &&
        memLayout.getValue() == ttnn::TensorMemoryLayout::WidthSharded) {
      return failure();
    }
  }

  auto inputElementType = inputType.getElementType();
  if (ttcore::TileType inputTileType =
          mlir::dyn_cast_or_null<ttcore::TileType>(inputElementType)) {
    inputElementType = inputTileType.getElementType();
  }

  // Compute the number of tiles along the width (last) dimension.
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t tileWidth = 32;
  int64_t numWidthTiles = (inputShape.back() + tileWidth - 1) / tileWidth;

  // Retrieve the physical grid shape for the device.
  auto physicalGrid =
      ttcore::getCurrentScopeSystemDesc(op).getChipDescs()[0].getGrid();
  int64_t maxCores =
      physicalGrid[0] * physicalGrid[1]; // total cores in physical grid

  // Pick the largest core count that evenly divides numWidthTiles,
  // capped by the physical grid size.
  int64_t numCores = 1;
  for (int64_t c = std::min(maxCores, numWidthTiles); c >= 1; --c) {
    if (numWidthTiles % c == 0) {
      numCores = c;
      break;
    }
  }
  SmallVector<int64_t> virtualGridSize = {1, numCores};

  // Create an affine map that translates the virtual grid layout to the
  // physical grid layout.
  auto affineMap = mlir::tt::ttnn::optimizer_utils::
      createSingleDeviceVirtualToPhysicalAffineMap(
          rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded,
          physicalGrid);
  auto grid = mlir::tt::ttcore::GridAttr::get(rewriter.getContext(),
                                              virtualGridSize, affineMap);
  auto memLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded);

  // Create layout attribute for the input tensor with width-sharded L1 config.
  ttnn::TTNNLayoutAttr desiredInputLayout =
      ttnn::TTNNLayoutAttr::get(rewriter.getContext(), inputType.getShape(),
                                ttcore::TileType::get(inputElementType),
                                ttnn::BufferType::L1, grid, memLayoutAttr);

  if (currentInputLayout == desiredInputLayout) {
    return failure();
  }

  // Apply ToLayoutOp to convert the input tensor to width-sharded L1.
  ttnn::MemoryConfigAttr inputMemoryConfig =
      ttnn::MemoryConfigAttr::get(desiredInputLayout, grid);
  RankedTensorType memoryConfigedInputType =
      inputType.cloneWithEncoding(desiredInputLayout);
  auto inputToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
      op.getLoc(), memoryConfigedInputType, op.getInput(),
      tt::ttnn::Layout::Tile,
      ttcore::DataTypeAttr::get(
          rewriter.getContext(),
          ttcore::elementTypeToDataType(inputElementType)),
      inputMemoryConfig);

  // The runtime requires the weight tensor in ROW_MAJOR layout with
  // width = tile_width (32). Reshape from 1D (N,) to 2D (N/32, 32) first.
  mlir::Value weight = op.getWeight();
  if (weight) {
    RankedTensorType weightType =
        mlir::cast<RankedTensorType>(weight.getType());
    ttnn::TTNNLayoutAttr weightLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(weightType.getEncoding());

    auto weightElementType = weightType.getElementType();
    if (ttcore::TileType weightTileType =
            mlir::dyn_cast_or_null<ttcore::TileType>(weightElementType)) {
      weightElementType = weightTileType.getElementType();
    }

    // Reshape weight to 2D: (N,) -> (N/32, 32) so width matches tile width.
    int64_t totalElements = 1;
    for (int64_t dim : weightType.getShape()) {
      totalElements *= dim;
    }
    SmallVector<int64_t> reshapedShape = {totalElements / tileWidth, tileWidth};
    SmallVector<int32_t> reshapedShapeI32(reshapedShape.begin(),
                                          reshapedShape.end());
    RankedTensorType reshapedWeightType =
        ttnn::utils::RankedTensorTypeFactory::create(weightType, reshapedShape);
    auto reshapeOp = rewriter.create<ttnn::ReshapeOp>(
        op.getLoc(), reshapedWeightType, weight,
        rewriter.getI32ArrayAttr(reshapedShapeI32), ttnn::MemoryConfigAttr());
    weight = reshapeOp.getResult();

    // Convert to ROW_MAJOR layout.
    weightType = mlir::cast<RankedTensorType>(weight.getType());
    weightLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(weightType.getEncoding());
    if (weightLayout && weightLayout.isTiled()) {
      ttnn::TTNNLayoutAttr rowMajorLayout = weightLayout.withLayout(
          ttnn::Layout::RowMajor, weightType.getShape());
      RankedTensorType rowMajorWeightType =
          weightType.cloneWithEncoding(rowMajorLayout);
      auto weightMemConfig = ttnn::MemoryConfigAttr::get(
          rewriter.getContext(),
          ttnn::TensorMemoryLayoutAttr::get(
              rewriter.getContext(), ttnn::TensorMemoryLayout::Interleaved),
          ttnn::BufferTypeAttr::get(rewriter.getContext(),
                                    weightLayout.getBufferType()),
          /*shardSpec=*/std::nullopt);
      auto weightToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), rowMajorWeightType, weight, tt::ttnn::Layout::RowMajor,
          ttcore::DataTypeAttr::get(
              rewriter.getContext(),
              ttcore::elementTypeToDataType(weightElementType)),
          weightMemConfig);
      weight = weightToLayoutOp.getResult();
    }
  }

  // The runtime requires the residual to have the same shard spec as input.
  mlir::Value residual = op.getResidual();
  if (residual) {
    RankedTensorType residualType =
        mlir::cast<RankedTensorType>(residual.getType());
    ttnn::TTNNLayoutAttr residualLayout =
        mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
            residualType.getEncoding());
    if (residualLayout != desiredInputLayout) {
      RankedTensorType shardedResidualType =
          residualType.cloneWithEncoding(desiredInputLayout);
      auto residualToLayoutOp = rewriter.create<ttnn::ToLayoutOp>(
          op.getLoc(), shardedResidualType, residual, tt::ttnn::Layout::Tile,
          ttcore::DataTypeAttr::get(
              rewriter.getContext(),
              ttcore::elementTypeToDataType(inputElementType)),
          inputMemoryConfig);
      residual = residualToLayoutOp.getResult();
    }
  }

  // Ensure compute_config is set (needed for stats dtype and by the kernel).
  auto computeConfigAttr = op.getComputeConfigAttr();
  if (!computeConfigAttr) {
    computeConfigAttr = DeviceComputeKernelConfigAttr::get(
        rewriter.getContext(),
        /*mathFidelity=*/MathFidelity::HiFi4,
        /*mathApproxMode=*/BoolAttr::get(rewriter.getContext(), false),
        /*fp32DestAccEn=*/BoolAttr::get(rewriter.getContext(), true),
        /*packerL1Acc=*/BoolAttr::get(rewriter.getContext(), true),
        /*dstFullSyncEn=*/nullptr);
  }

  // Derive stats tensor dtype from fp32_dest_acc_en in compute_config.
  // The CB data format must match: Float32 when fp32_dest_acc_en, else
  // BFloat16.
  bool fp32DestAccEn = true;
  if (auto fp32Attr = computeConfigAttr.getFp32DestAccEn()) {
    fp32DestAccEn = fp32Attr.getValue();
  }
  auto statsElementType =
      fp32DestAccEn
          ? static_cast<Type>(Float32Type::get(rewriter.getContext()))
          : static_cast<Type>(BFloat16Type::get(rewriter.getContext()));
  auto statsDataType =
      fp32DestAccEn ? ttcore::DataType::Float32 : ttcore::DataType::BFloat16;

  // Create stats scratch tensor: one tile (32x32) per device, width-sharded
  // on core (0,0) in L1. The fused kernel writes partial RMS statistics here
  // and exchanges them across devices via the allgather.
  SmallVector<int64_t> statsGridShape = {1, 1};
  auto statsGrid = mlir::tt::ttcore::GridAttr::get(rewriter.getContext(),
                                                   statsGridShape, affineMap);
  auto statsMemLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      rewriter.getContext(), ttnn::TensorMemoryLayout::WidthSharded);

  SmallVector<int64_t> statsShape = {1, 1, 32, 32};
  ttnn::TTNNLayoutAttr statsLayout = ttnn::TTNNLayoutAttr::get(
      rewriter.getContext(), statsShape,
      ttcore::TileType::get(statsElementType), ttnn::BufferType::L1, statsGrid,
      statsMemLayoutAttr);

  auto statsShapeAttr = ttnn::ShapeAttr::get(rewriter.getContext(), statsShape);
  auto statsDtypeAttr =
      ttcore::DataTypeAttr::get(rewriter.getContext(), statsDataType);
  auto statsLayoutAttr =
      ttnn::LayoutAttr::get(rewriter.getContext(), ttnn::Layout::Tile);
  auto statsMemConfig = ttnn::MemoryConfigAttr::get(statsLayout, statsGrid);

  RankedTensorType statsResultType =
      RankedTensorType::get(statsShape, statsElementType, statsLayout);

  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
  auto statsEmptyOp = rewriter.create<ttnn::EmptyOp>(
      op.getLoc(), statsResultType, device, statsShapeAttr, statsDtypeAttr,
      statsLayoutAttr, statsMemConfig);

  // The fused kernel output shape == input shape (only stats are all-gathered,
  // not data). Use the input's width-sharded memory config for the output too
  // (skip_write_back = true in the program factory).
  RankedTensorType shardedOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType())
          .cloneWithEncoding(desiredInputLayout);

  // Compute LayerNormShardedMultiCoreProgramConfig from the shard spec.
  // This mirrors ttnn::prim::create_program_config(shardSpec).
  auto scalarShardShape = desiredInputLayout.getScalarShardShape();
  int64_t blockH = scalarShardShape[0] / tileWidth;
  int64_t blockW = scalarShardShape[1] / tileWidth;
  int64_t gridW = std::min(numCores, physicalGrid[0]);
  int64_t gridH = (numCores + physicalGrid[0] - 1) / physicalGrid[0];
  auto programConfigAttr =
      ttnn::LayerNormShardedMultiCoreProgramConfigAttr::get(
          rewriter.getContext(),
          ttnn::CoreCoordAttr::get(rewriter.getContext(), gridW, gridH),
          /*subblock_w=*/1, blockH, blockW, /*inplace=*/false);

  auto newOp = rewriter.create<ttnn::DistributedRMSNormOp>(
      op.getLoc(), shardedOutputType, inputToLayoutOp.getResult(), weight,
      residual, statsEmptyOp.getResult(),
      static_cast<uint32_t>(op.getClusterAxis()), op.getEpsilon(),
      op.getSubDeviceIdAttr(), inputMemoryConfig, op.getNumLinksAttr(),
      op.getTopologyAttr(), computeConfigAttr, programConfigAttr);

  // If the original output had a different layout (e.g. interleaved DRAM),
  // insert a to_memory_config to convert back.
  RankedTensorType originalOutputType =
      mlir::cast<RankedTensorType>(op.getResult().getType());
  auto originalOutputLayout = mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(
      originalOutputType.getEncoding());
  if (originalOutputLayout && originalOutputLayout != desiredInputLayout) {
    auto originalMemConfig = ttnn::MemoryConfigAttr::get(
        originalOutputLayout, originalOutputLayout.getGrid());
    auto toMemConfigOp = rewriter.create<ttnn::ToMemoryConfigOp>(
        op.getLoc(), originalOutputType, newOp.getResult(), originalMemConfig);
    rewriter.replaceOp(op, toMemConfigOp.getResult());
  } else {
    rewriter.replaceOp(op, newOp.getResult());
  }
  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
