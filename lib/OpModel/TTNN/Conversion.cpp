// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <stdexcept>
#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::tt::op_model::ttnn {

namespace conversion {

::tt::tt_metal::DataType getDataType(const DataType dataType) {
  switch (dataType) {
  case tt::DataType::Float32:
    return ::tt::tt_metal::DataType::FLOAT32;
  case tt::DataType::BFloat16:
    return ::tt::tt_metal::DataType::BFLOAT16;
  case tt::DataType::BFP_BFloat8:
    return ::tt::tt_metal::DataType::BFLOAT8_B;
  case tt::DataType::BFP_BFloat4:
    return ::tt::tt_metal::DataType::BFLOAT4_B;
  case tt::DataType::UInt32:
    return ::tt::tt_metal::DataType::UINT32;
  case tt::DataType::UInt16:
    return ::tt::tt_metal::DataType::UINT16;
  case tt::DataType::UInt8:
    return ::tt::tt_metal::DataType::UINT8;
  case tt::DataType::Int32:
    return ::tt::tt_metal::DataType::INT32;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

tt::DataType getDataType(const ::tt::tt_metal::DataType dataType) {
  switch (dataType) {
  case ::tt::tt_metal::DataType::FLOAT32:
    return tt::DataType::Float32;
  case ::tt::tt_metal::DataType::BFLOAT16:
    return tt::DataType::BFloat16;
  case ::tt::tt_metal::DataType::BFLOAT8_B:
    return tt::DataType::BFP_BFloat8;
  case ::tt::tt_metal::DataType::BFLOAT4_B:
    return tt::DataType::BFP_BFloat4;
  case ::tt::tt_metal::DataType::UINT32:
    return tt::DataType::UInt32;
  case ::tt::tt_metal::DataType::UINT16:
    return tt::DataType::UInt16;
  case ::tt::tt_metal::DataType::UINT8:
    return tt::DataType::UInt8;
  case ::tt::tt_metal::DataType::INT32:
    return tt::DataType::Int32;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

::ttnn::Shape getShape(const ::llvm::ArrayRef<int64_t> shape) {
  ::ttsl::SmallVector<uint32_t> small_vector_shape;
  for (const auto &dim : shape) {
    small_vector_shape.push_back(static_cast<uint32_t>(dim));
  }

  return ::ttnn::Shape(small_vector_shape);
}

llvm::SmallVector<int64_t> getShape(const ::ttnn::Shape &shape) {
  return llvm::SmallVector<int64_t>(shape.cbegin(), shape.cend());
}

const std::array<uint32_t, 2>
getShardShape(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  const auto layoutShardTile = layout.getScalarShardShape();

  if (layoutShardTile.size() != 2) {
    llvm::errs() << "ERROR: layout_shard_tile.size() != 2\n";
    return {0, 0};
  }

  std::array<uint32_t, 2> shardShape;
  shardShape[0] = layoutShardTile[0];
  shardShape[1] = layoutShardTile[1];
  return shardShape;
}

::tt::tt_metal::Layout
getPageLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return layout.isTiled() ? ::tt::tt_metal::Layout::TILE
                          : ::tt::tt_metal::Layout::ROW_MAJOR;
}

::tt::tt_metal::Layout getPageLayout(mlir::tt::ttnn::Layout layout) {
  switch (layout) {
  case ::mlir::tt::ttnn::Layout::RowMajor:
    return ::tt::tt_metal::Layout::ROW_MAJOR;
  case ::mlir::tt::ttnn::Layout::Tile:
    return ::tt::tt_metal::Layout::TILE;
  case ::mlir::tt::ttnn::Layout::Invalid:
    return ::tt::tt_metal::Layout::INVALID;
  }
}

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  std::set<::tt::tt_metal::CoreRange> coreRangeSet;
  assert(layout.getGrid().getMapping().isEmpty() == false);
  for (const auto &[loc, size] : utils::toCoreRangeSet(
           layout.getGrid().getShape(), layout.getGrid().getMapping())) {
    coreRangeSet.insert(::tt::tt_metal::CoreRange(
        CoreCoord(loc[0], loc[1]),
        CoreCoord(loc[0] + size[0] - 1, loc[1] + size[1] - 1)));
  }
  return ::tt::tt_metal::CoreRangeSet(coreRangeSet);
}

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  if (layout.getIgnorePhysicalLayout()) {
    return std::nullopt;
  }

  if (!isShardedMemoryLayout(layout.getMemLayout().getValue())) {
    return std::nullopt;
  }

  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO(jserbedzija): with issue #620
  return ::tt::tt_metal::ShardSpec(getCoreRangeSet(layout),
                                   getShardShape(layout),
                                   ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
}

::tt::tt_metal::BufferType
getBufferType(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto bufferType = layout.getBufferType();

  switch (bufferType) {
  case mlir::tt::ttnn::BufferType::DRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case mlir::tt::ttnn::BufferType::L1:
    return ::tt::tt_metal::BufferType::L1;
  case mlir::tt::ttnn::BufferType::SystemMemory:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case mlir::tt::ttnn::BufferType::L1Small:
    return ::tt::tt_metal::BufferType::L1_SMALL;
  case mlir::tt::ttnn::BufferType::Trace:
    return ::tt::tt_metal::BufferType::TRACE;
  }
}

mlir::tt::ttnn::BufferType
getBufferType(const ::tt::tt_metal::BufferType bufferType) {
  switch (bufferType) {
  case ::tt::tt_metal::BufferType::DRAM:
    return mlir::tt::ttnn::BufferType::DRAM;
  case ::tt::tt_metal::BufferType::L1:
    return mlir::tt::ttnn::BufferType::L1;
  case ::tt::tt_metal::BufferType::SYSTEM_MEMORY:
    return mlir::tt::ttnn::BufferType::SystemMemory;
  case ::tt::tt_metal::BufferType::L1_SMALL:
    return mlir::tt::ttnn::BufferType::L1Small;
  case ::tt::tt_metal::BufferType::TRACE:
    return mlir::tt::ttnn::BufferType::Trace;
  }
}

::tt::tt_metal::TensorMemoryLayout getTensorMemoryLayout(
    const mlir::tt::ttnn::TensorMemoryLayout tensorMemoryLayout) {
  switch (tensorMemoryLayout) {
  case mlir::tt::ttnn::TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case mlir::tt::ttnn::TensorMemoryLayout::SingleBank:
    return ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK;
  case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
  }
}
mlir::tt::ttnn::TensorMemoryLayout
getTensorMemoryLayout(const ::tt::tt_metal::TensorMemoryLayout memLayout) {
  switch (memLayout) {
  case ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED:
    return mlir::tt::ttnn::TensorMemoryLayout::Interleaved;
  case ::tt::tt_metal::TensorMemoryLayout::SINGLE_BANK:
    return mlir::tt::ttnn::TensorMemoryLayout::SingleBank;
  case ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
    return mlir::tt::ttnn::TensorMemoryLayout::HeightSharded;
  case ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
    return mlir::tt::ttnn::TensorMemoryLayout::WidthSharded;
  case ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED:
    return mlir::tt::ttnn::TensorMemoryLayout::BlockSharded;
  }
}

::tt::tt_metal::TensorMemoryLayout getTensorMemoryLayout(
    const mlir::tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr) {
  auto tensorMemoryLayout = memLayoutAttr.getValue();
  return getTensorMemoryLayout(tensorMemoryLayout);
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = getTensorMemoryLayout(layout.getMemLayout());
  auto bufferType = getBufferType(layout);

  auto shardSpec = getShardSpec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::TensorLayout
getTensorLayout(const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  return ::tt::tt_metal::TensorLayout(getDataType(layout.getDataType()),
                                      getPageLayout(layout),
                                      getMemoryConfig(layout));
}

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const mlir::tt::ttnn::TTNNLayoutAttr &layout) {
  assert(!layout.getIgnorePhysicalLayout() &&
         "TensorSpecs cannot be created without physical layouts");
  return ::ttnn::TensorSpec(getShape(shape), getTensorLayout(layout));
}

bool validateTensorSpec(const ::ttnn::TensorSpec &tensorSpec,
                        const ::tt::tt_metal::CoreCoord &computeGridSize) {
  // Check the shard bounding box
  auto memoryConfig = tensorSpec.memory_config();
  if (memoryConfig.is_sharded() && memoryConfig.shard_spec().has_value()) {
    ::tt::tt_metal::CoreRange shardBoundingBox =
        memoryConfig.shard_spec().value().grid.bounding_box();
    ::tt::tt_metal::CoreRangeSet deviceWorkerCores{::tt::tt_metal::CoreRange{
        ::tt::tt_metal::CoreCoord{0, 0},
        ::tt::tt_metal::CoreCoord{computeGridSize.x - 1,
                                  computeGridSize.y - 1}}};
    if (!deviceWorkerCores.contains(shardBoundingBox)) {
      return false;
    }
  }

  // Check attributes required for allocation
  // May call TT_THROW or TT_FATAL for malformed TensorSpecs
  try {
    tensorSpec.compute_packed_buffer_size_bytes();
    tensorSpec.compute_page_size_bytes();
    tensorSpec.compute_distribution_spec();
  } catch (const std::exception &e) {
    return false;
  }
  return true;
}

::ttsl::SmallVector<int>
convertLLVMSmallVecToTTNNSmallVec(const ::llvm::ArrayRef<int64_t> vec) {
  return ::ttsl::SmallVector<int>(vec.begin(), vec.end());
}

std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig> getConv2dConfig(
    const std::optional<mlir::tt::ttnn::Conv2dConfigAttr> &conv2dConfig) {
  if (!conv2dConfig) {
    return std::nullopt;
  }

  // TODO(#2130): config.core_grid is hardcoded to nullopt until we add
  // CoreRangeSet as an IR attribute.
  assert(!conv2dConfig->getCoreGrid() && "CoreGrid is not supported yet");

  ::ttnn::operations::conv::conv2d::Conv2dConfig config;

  if (conv2dConfig->getDtype()) {
    config.dtype = getDataType(*conv2dConfig->getDtype());
  }

  if (conv2dConfig->getWeightsDtype()) {
    config.weights_dtype = getDataType(*conv2dConfig->getWeightsDtype());
  }

  if (conv2dConfig->getActivation()) {
    config.activation = conv2dConfig->getActivation().getValue().str();
  }

  if (conv2dConfig->getDeallocateActivation()) {
    config.deallocate_activation =
        conv2dConfig->getDeallocateActivation().getValue();
  }

  if (conv2dConfig->getReallocateHaloOutput()) {
    config.reallocate_halo_output =
        conv2dConfig->getReallocateHaloOutput().getValue();
  }

  if (conv2dConfig->getActBlockHOverride()) {
    config.act_block_h_override = *conv2dConfig->getActBlockHOverride();
  }

  if (conv2dConfig->getActBlockWDiv()) {
    config.act_block_w_div = *conv2dConfig->getActBlockWDiv();
  }

  if (conv2dConfig->getReshardIfNotOptimal()) {
    config.reshard_if_not_optimal =
        conv2dConfig->getReshardIfNotOptimal().getValue();
  }

  if (conv2dConfig->getOverrideShardingConfig()) {
    config.override_sharding_config =
        conv2dConfig->getOverrideShardingConfig().getValue();
  }

  if (conv2dConfig->getShardLayout()) {
    config.shard_layout =
        getTensorMemoryLayout(*conv2dConfig->getShardLayout());
  } else {
    config.shard_layout = std::nullopt;
  }

  config.core_grid = std::nullopt;

  if (conv2dConfig->getTransposeShards()) {
    config.transpose_shards = conv2dConfig->getTransposeShards().getValue();
  }

  if (conv2dConfig->getOutputLayout()) {
    config.output_layout = getPageLayout(*conv2dConfig->getOutputLayout());
  }

  if (conv2dConfig->getPreprocessWeightsOnDevice()) {
    config.preprocess_weights_on_device =
        conv2dConfig->getPreprocessWeightsOnDevice().getValue();
  }

  if (conv2dConfig->getAlwaysPreprocessWeights()) {
    config.always_preprocess_weights =
        conv2dConfig->getAlwaysPreprocessWeights().getValue();
  }

  if (conv2dConfig->getEnableActDoubleBuffer()) {
    config.enable_act_double_buffer =
        conv2dConfig->getEnableActDoubleBuffer().getValue();
  }

  if (conv2dConfig->getEnableWeightsDoubleBuffer()) {
    config.enable_weights_double_buffer =
        conv2dConfig->getEnableWeightsDoubleBuffer().getValue();
  }

  if (conv2dConfig->getEnableSplitReader()) {
    config.enable_split_reader =
        conv2dConfig->getEnableSplitReader().getValue();
  }

  if (conv2dConfig->getEnableSubblockPadding()) {
    config.enable_subblock_padding =
        conv2dConfig->getEnableSubblockPadding().getValue();
  }

  return config;
}

// sgholamiTT: I was on the fence for publicly exposing this API. Right now
// there's no clear usecase for it other than conversion from
// mlir::tt::ttnn::MathFidelity to ::ttnn::MathFidelity. Therefore, I decided to
// not expose it for now. Subject to change in the future.
MathFidelity getMathFidelity(mlir::tt::ttnn::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case mlir::tt::ttnn::MathFidelity::LoFi:
    return MathFidelity::LoFi;
  case mlir::tt::ttnn::MathFidelity::HiFi2:
    return MathFidelity::HiFi2;
  case mlir::tt::ttnn::MathFidelity::HiFi3:
    return MathFidelity::HiFi3;
  case mlir::tt::ttnn::MathFidelity::HiFi4:
    return MathFidelity::HiFi4;
  }
}

std::optional<::ttnn::DeviceComputeKernelConfig> getDeviceComputeKernelConfig(
    const std::optional<mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        &deviceComputeKernelConfig) {
  if (!deviceComputeKernelConfig || !deviceComputeKernelConfig.has_value()) {
    return std::nullopt;
  }
  mlir::tt::ttnn::DeviceComputeKernelConfigAttr devConfig =
      deviceComputeKernelConfig.value();

  // Note: Currently, we only support creating WormholeComputeKernelConfig.
  // If we need to support GrayskullComputeKernelConfig in the future, we
  // need to pass in the device information to this function or include it in
  // DeviceComputeKernelConfigAttr.
  ::ttnn::WormholeComputeKernelConfig config;
  if (devConfig.getFp32DestAccEn()) {
    config.fp32_dest_acc_en = devConfig.getFp32DestAccEn().getValue();
  }
  if (devConfig.getPackerL1Acc()) {
    config.packer_l1_acc = devConfig.getPackerL1Acc().getValue();
  }
  if (devConfig.getMathApproxMode()) {
    config.math_approx_mode = devConfig.getMathApproxMode().getValue();
  }
  if (devConfig.getDstFullSyncEn()) {
    config.dst_full_sync_en = devConfig.getDstFullSyncEn().getValue();
  }
  if (devConfig.getMathFidelity().has_value()) {
    config.math_fidelity = getMathFidelity(devConfig.getMathFidelity().value());
  }
  return config;
}

llvm::SmallVector<int64_t>
getLogicalGridShape(const ::tt::tt_metal::MemoryConfig &memoryConfig,
                    const llvm::ArrayRef<int64_t> &gridPhyCores) {

  if (memoryConfig.memory_layout() ==
      ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED) {
    assert(memoryConfig.shard_spec().has_value());
    return {memoryConfig.shard_spec()->num_cores(), 1};
  }

  if (memoryConfig.memory_layout() ==
      ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
    assert(memoryConfig.shard_spec().has_value());
    return {1, memoryConfig.shard_spec()->num_cores()};
  }

  if (memoryConfig.memory_layout() ==
      ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED) {
    assert(memoryConfig.shard_spec().has_value());
    CoreRange boundingGrid = memoryConfig.shard_spec()->grid.bounding_box();
    assert(memoryConfig.shard_spec()->num_cores() == boundingGrid.size());
    return {static_cast<int64_t>(boundingGrid.grid_size().y),
            static_cast<int64_t>(boundingGrid.grid_size().x)};
  }

  // interleaved
  return {gridPhyCores[0], gridPhyCores[1]};
}

mlir::tt::ttnn::TTNNLayoutAttr
getLayoutAttrFromTensorSpec(MLIRContext *context,
                            const ::ttnn::TensorSpec &tensorSpec,
                            llvm::ArrayRef<int64_t> deviceGrid) {
  llvm::SmallVector<int64_t> shape;
  if (tensorSpec.logical_shape().size() > 0) {
    shape = getShape(tensorSpec.logical_shape());
  } else {
    // Scalar. This can result from reduction operations. Convert it to (1,1)
    // for compatibility
    shape = {1, 1};
  }

  Type elementType;
  if (tensorSpec.layout() == ::tt::tt_metal::Layout::TILE) {
    elementType = mlir::tt::TileType::get(
        context,
        {tensorSpec.page_config().get_tile().get_height(),
         tensorSpec.page_config().get_tile().get_width()},
        getDataType(tensorSpec.data_type()));
  } else {
    elementType =
        dataTypeToElementType(context, getDataType(tensorSpec.data_type()));
  }

  mlir::tt::ttnn::BufferType bufferType =
      getBufferType(tensorSpec.memory_config().buffer_type());
  auto memoryLayoutAttr = mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
      context,
      getTensorMemoryLayout(tensorSpec.memory_config().memory_layout()));

  GridAttr grid = mlir::tt::GridAttr::get(
      context, getLogicalGridShape(tensorSpec.memory_config(), deviceGrid),
      ::mlir::tt::ttnn::optimizer_utils::
          createSingleDeviceVirtualToPhysicalAffineMap(
              context, memoryLayoutAttr.getValue(), deviceGrid));

  return mlir::tt::ttnn::TTNNLayoutAttr::get(
      context, shape, elementType, bufferType, grid, memoryLayoutAttr);
}

} // namespace conversion
} // namespace mlir::tt::op_model::ttnn
#endif // TTMLIR_ENABLE_OPMODEL
