// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "tt/runtime/detail/common/logger.h"

namespace ttnn_op_invoke::operations::utils {

bool inSystemMemory(const ::tt::target::ttnn::TensorRefT &tensorRef) {
  const ::tt::target::ttnn::StorageType storageType =
      tensorRef.desc->layout->memory_desc->storage_type;
  return storageType == ::tt::target::ttnn::StorageType::Host;
}

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout) {
  return tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::HeightSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::WidthSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
}

::ttnn::DataType toTTNNDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return ::ttnn::DataType::FLOAT32;
  case ::tt::target::DataType::BFloat16:
    return ::ttnn::DataType::BFLOAT16;
  case ::tt::target::DataType::BFP_BFloat8:
    return ::ttnn::DataType::BFLOAT8_B;
  case ::tt::target::DataType::BFP_BFloat4:
    return ::ttnn::DataType::BFLOAT4_B;
  case ::tt::target::DataType::UInt32:
    return ::ttnn::DataType::UINT32;
  case ::tt::target::DataType::UInt16:
    return ::ttnn::DataType::UINT16;
  case ::tt::target::DataType::UInt8:
    return ::ttnn::DataType::UINT8;
  case ::tt::target::DataType::Int32:
    return ::ttnn::DataType::INT32;

  default:
    LOG_FATAL("Unsupported data type");
  }
}
::tt::tt_metal::MathFidelity
toTTNNMathFidelity(::tt::target::MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case ::tt::target::MathFidelity::LoFi:
    return ::tt::tt_metal::MathFidelity::LoFi;
  case ::tt::target::MathFidelity::HiFi2:
    return ::tt::tt_metal::MathFidelity::HiFi2;
  case ::tt::target::MathFidelity::HiFi3:
    return ::tt::tt_metal::MathFidelity::HiFi3;
  case ::tt::target::MathFidelity::HiFi4:
    return ::tt::tt_metal::MathFidelity::HiFi4;
  }
}

::ttnn::Layout toTTNNLayout(::tt::target::TensorLayout layout) {
  switch (layout) {
  case ::tt::target::TensorLayout::Tile:
    return ::ttnn::Layout::TILE;
  case ::tt::target::TensorLayout::RowMajor:
    return ::ttnn::Layout::ROW_MAJOR;
  default:
    LOG_FATAL("Unsupported layout");
  }
}

::ttnn::TensorMemoryLayout toTTNNTensorMemoryLayout(
    ::tt::target::ttnn::TensorMemoryLayout tensorMemoryLayout) {

  switch (tensorMemoryLayout) {
  case ::tt::target::ttnn::TensorMemoryLayout::Interleaved:
    return ::ttnn::TensorMemoryLayout::INTERLEAVED;
  case ::tt::target::ttnn::TensorMemoryLayout::HeightSharded:
    return ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::WidthSharded:
    return ::ttnn::TensorMemoryLayout::WIDTH_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::BlockSharded:
    return ::ttnn::TensorMemoryLayout::BLOCK_SHARDED;
  case ::tt::target::ttnn::TensorMemoryLayout::NDSharded:
    return ::ttnn::TensorMemoryLayout::ND_SHARDED;
  }
}

::ttnn::BufferType toTTNNBufferType(::tt::target::BufferType bufferType) {

  switch (bufferType) {
  case ::tt::target::BufferType::DRAM:
    return ::ttnn::BufferType::DRAM;
  case ::tt::target::BufferType::L1:
    return ::ttnn::BufferType::L1;
  case ::tt::target::BufferType::SystemMemory:
    return ::ttnn::BufferType::SYSTEM_MEMORY;
  case ::tt::target::BufferType::L1Small:
    return ::ttnn::BufferType::L1_SMALL;
  case ::tt::target::BufferType::Trace:
    return ::ttnn::BufferType::TRACE;
  }
}

tt::tt_metal::CoreCoord
toTTNNCoreCoord(const ::tt::target::ttnn::CoreCoord &coreCoord) {
  return tt::tt_metal::CoreCoord(coreCoord.x(), coreCoord.y());
}

tt::tt_metal::CoreRange
toTTNNCoreRange(const tt::target::ttnn::CoreRange &coreRange) {
  tt::tt_metal::CoreCoord start = toTTNNCoreCoord(coreRange.start_coord());
  tt::tt_metal::CoreCoord end = toTTNNCoreCoord(coreRange.end_coord());
  return tt::tt_metal::CoreRange(start, end);
}

tt::tt_metal::CoreRangeSet
toTTNNCoreRangeSet(const tt::target::ttnn::CoreRangeSetT &coreRangeSet) {
  // Even though `CoreRangeSet`'s naming implies usage of std::set,
  // it's important that the order of core ranges is preserved,
  // so we use std::vector here to maintain the order as originally defined.
  std::vector<tt::tt_metal::CoreRange> coreRanges;
  coreRanges.reserve(coreRangeSet.core_ranges.size());
  for (const auto &coreRange : coreRangeSet.core_ranges) {
    coreRanges.emplace_back(toTTNNCoreRange(coreRange));
  }
  return tt::tt_metal::CoreRangeSet(std::move(coreRanges));
}

::ttnn::ShardOrientation
toTTNNShardOrientation(tt::target::ttnn::ShardOrientation orientation) {
  switch (orientation) {
  case tt::target::ttnn::ShardOrientation::RowMajor:
    return ::ttnn::ShardOrientation::ROW_MAJOR;
  case tt::target::ttnn::ShardOrientation::ColMajor:
    return ::ttnn::ShardOrientation::COL_MAJOR;
  }
}

tt::tt_metal::ShardDistributionStrategy toTTNNShardDistributionStrategy(
    tt::target::ttnn::ShardDistributionStrategy distributionStrategy) {
  switch (distributionStrategy) {
  case tt::target::ttnn::ShardDistributionStrategy::RoundRobin1D:
    return tt::tt_metal::ShardDistributionStrategy::ROUND_ROBIN_1D;
  case tt::target::ttnn::ShardDistributionStrategy::Grid2D:
    return tt::tt_metal::ShardDistributionStrategy::GRID_2D;
  }
}

const ::tt::target::ttnn::MemoryConfigT
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRefT &tensorRef) {
  return *tensorRef.desc->layout->memory_desc->memory_config;
}

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfigT &memcfg,
                           CallType callType) {
  const auto targetBufferType = memcfg.buffer_type;
  LOG_ASSERT(targetBufferType == ::tt::target::BufferType::DRAM ||
                 targetBufferType == ::tt::target::BufferType::L1,
             "Memory config buffer type should be DRAM or L1");
  const auto ttnnBufferType = toTTNNBufferType(targetBufferType);

  const auto targetMemLayout = memcfg.tensor_memory_layout;
  const auto memLayout = toTTNNTensorMemoryLayout(targetMemLayout);

  // Verify that shard spec is present only for sharded memory layouts
  if (callType == CallType::EXECUTE) {
    const bool hasShardSpec =
        (memcfg.shard_spec != nullptr) || (memcfg.nd_shard_spec != nullptr);
    LOG_ASSERT(
        hasShardSpec == isSharded(targetMemLayout),
        "A shard spec must be present if and only if the tensor is sharded");
  }

  // Handle (legacy) shard spec
  if (const auto &shardSpec = memcfg.shard_spec) {
    const auto &shardShape = shardSpec->shape;
    LOG_ASSERT(shardShape.size() == 2,
               "Only 2D shard shape is supported in TTNN backend");
    std::array<uint32_t, 2> shape;
    std::copy(shardShape.begin(), shardShape.end(), shape.begin());

    const tt::tt_metal::CoreRangeSet coreRangeSet =
        toTTNNCoreRangeSet(*shardSpec->core_range_set);
    const ::ttnn::ShardOrientation orientation =
        toTTNNShardOrientation(shardSpec->orientation);
    auto metalShardSpec =
        ::tt::tt_metal::ShardSpec(coreRangeSet, shape, orientation);

    return ::ttnn::MemoryConfig{memLayout, ttnnBufferType, metalShardSpec};
  }

  // Handle ND shard spec
  if (const auto &ndShardSpec = memcfg.nd_shard_spec) {
    const auto &shardShape = ndShardSpec->shape;
    std::vector<uint32_t> shape(shardShape.begin(), shardShape.end());

    const tt::tt_metal::CoreRangeSet coreRangeSet =
        toTTNNCoreRangeSet(*ndShardSpec->core_range_set);
    const ::ttnn::ShardOrientation orientation =
        toTTNNShardOrientation(ndShardSpec->orientation);
    const tt::tt_metal::ShardDistributionStrategy strategy =
        toTTNNShardDistributionStrategy(ndShardSpec->distribution_strategy);
    auto metalNdShardSpec = tt::tt_metal::NdShardSpec(
        tt::tt_metal::Shape(ttsl::Span<const uint32_t>(shape)), coreRangeSet,
        orientation, strategy);

    return ::ttnn::MemoryConfig{ttnnBufferType, metalNdShardSpec};
  }

  // Non-sharded memory config
  return ::ttnn::MemoryConfig{memLayout, ttnnBufferType};
}

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRefT &tensorRef) {
  return toTTNNDataType(tensorRef.desc->layout->memory_desc->data_type);
}

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfigT &config) {
  ::ttnn::WormholeComputeKernelConfig computeKernelConfig;

  if (config.math_fidelity) {
    computeKernelConfig.math_fidelity =
        operations::utils::toTTNNMathFidelity(*config.math_fidelity);
  }

  if (config.math_approx_mode) {
    computeKernelConfig.math_approx_mode = *config.math_approx_mode;
  }

  if (config.fp32_dest_acc_en) {
    computeKernelConfig.fp32_dest_acc_en = *config.fp32_dest_acc_en;
  }

  if (config.packer_l1_acc) {
    computeKernelConfig.packer_l1_acc = *config.packer_l1_acc;
  }

  if (config.dst_full_sync_en) {
    computeKernelConfig.dst_full_sync_en = *config.dst_full_sync_en;
  }

  return computeKernelConfig;
}

} // namespace ttnn_op_invoke::operations::utils
