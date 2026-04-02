#include "utils/utils.h"
#include "tt/runtime/detail/common/logger.h"

namespace unifiedOpLib::operations::utils {

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
  std::set<tt::tt_metal::CoreRange> coreRanges;
  for (const tt::target::ttnn::CoreRange coreRange : coreRangeSet.core_ranges) {
    coreRanges.emplace(toTTNNCoreRange(coreRange));
  }
  return tt::tt_metal::CoreRangeSet(coreRanges);
}

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::UnaryOpType unaryOpType) {
  using FbUnaryOpType = ::tt::target::ttnn::UnaryOpType;
  using TTNNUnaryOpType = ::ttnn::operations::unary::UnaryOpType;

  static const std::unordered_map<FbUnaryOpType, TTNNUnaryOpType> opTypeMap = {
      {FbUnaryOpType::Exp, TTNNUnaryOpType::EXP},
      {FbUnaryOpType::Recip, TTNNUnaryOpType::RECIP},
      {FbUnaryOpType::Gelu, TTNNUnaryOpType::GELU},
      {FbUnaryOpType::Relu, TTNNUnaryOpType::RELU},
      {FbUnaryOpType::Sqrt, TTNNUnaryOpType::SQRT},
      {FbUnaryOpType::Sigmoid, TTNNUnaryOpType::SIGMOID},
      {FbUnaryOpType::Log, TTNNUnaryOpType::LOG},
      {FbUnaryOpType::Tanh, TTNNUnaryOpType::TANH},
      {FbUnaryOpType::Log2, TTNNUnaryOpType::LOG2},
      {FbUnaryOpType::Log10, TTNNUnaryOpType::LOG10},
      {FbUnaryOpType::Sin, TTNNUnaryOpType::SIN},
      {FbUnaryOpType::Cos, TTNNUnaryOpType::COS},
      {FbUnaryOpType::Abs, TTNNUnaryOpType::ABS},
      {FbUnaryOpType::AbsInt32, TTNNUnaryOpType::ABS_INT32},
      {FbUnaryOpType::Sign, TTNNUnaryOpType::SIGN},
      {FbUnaryOpType::Square, TTNNUnaryOpType::SQUARE},
      {FbUnaryOpType::Eqz, TTNNUnaryOpType::EQZ},
      {FbUnaryOpType::Nez, TTNNUnaryOpType::NEZ},
      {FbUnaryOpType::Gtz, TTNNUnaryOpType::GTZ},
      {FbUnaryOpType::Ltz, TTNNUnaryOpType::LTZ},
      {FbUnaryOpType::Gez, TTNNUnaryOpType::GEZ},
      {FbUnaryOpType::Lez, TTNNUnaryOpType::LEZ},
      {FbUnaryOpType::ReluMax, TTNNUnaryOpType::RELU_MAX},
      {FbUnaryOpType::ReluMin, TTNNUnaryOpType::RELU_MIN},
      {FbUnaryOpType::Power, TTNNUnaryOpType::POWER},
      {FbUnaryOpType::LeakyRelu, TTNNUnaryOpType::LEAKY_RELU},
      {FbUnaryOpType::Elu, TTNNUnaryOpType::ELU},
      {FbUnaryOpType::Exp2, TTNNUnaryOpType::EXP2},
      {FbUnaryOpType::Heaviside, TTNNUnaryOpType::HEAVISIDE},
      {FbUnaryOpType::Expm1, TTNNUnaryOpType::EXPM1},
      {FbUnaryOpType::Signbit, TTNNUnaryOpType::SIGNBIT},
      {FbUnaryOpType::Asin, TTNNUnaryOpType::ASIN},
      {FbUnaryOpType::Acos, TTNNUnaryOpType::ACOS},
      {FbUnaryOpType::Rsqrt, TTNNUnaryOpType::RSQRT},
      {FbUnaryOpType::Relu6, TTNNUnaryOpType::RELU6},
      {FbUnaryOpType::Hardsigmoid, TTNNUnaryOpType::HARDSIGMOID},
      {FbUnaryOpType::Atan, TTNNUnaryOpType::ATAN},
      {FbUnaryOpType::Erf, TTNNUnaryOpType::ERF},
      {FbUnaryOpType::Erfc, TTNNUnaryOpType::ERFC},
      {FbUnaryOpType::Isinf, TTNNUnaryOpType::ISINF},
      {FbUnaryOpType::Isposinf, TTNNUnaryOpType::ISPOSINF},
      {FbUnaryOpType::Isneginf, TTNNUnaryOpType::ISNEGINF},
      {FbUnaryOpType::Isnan, TTNNUnaryOpType::ISNAN},
      {FbUnaryOpType::LogicalNotUnary, TTNNUnaryOpType::LOGICAL_NOT_UNARY},
      {FbUnaryOpType::Isfinite, TTNNUnaryOpType::ISFINITE},
      {FbUnaryOpType::Erfinv, TTNNUnaryOpType::ERFINV},
      {FbUnaryOpType::I0, TTNNUnaryOpType::I0},
      {FbUnaryOpType::I1, TTNNUnaryOpType::I1},
      {FbUnaryOpType::Tan, TTNNUnaryOpType::TAN},
      {FbUnaryOpType::Rsub, TTNNUnaryOpType::RSUB},
      {FbUnaryOpType::Rdiv, TTNNUnaryOpType::RDIV},
      {FbUnaryOpType::Silu, TTNNUnaryOpType::SILU},
      {FbUnaryOpType::Softplus, TTNNUnaryOpType::SOFTPLUS},
      {FbUnaryOpType::Identity, TTNNUnaryOpType::IDENTITY},
      {FbUnaryOpType::Neg, TTNNUnaryOpType::NEG},
      {FbUnaryOpType::AddUnarySfpu, TTNNUnaryOpType::ADD_UNARY_SFPU},
      {FbUnaryOpType::SubUnarySfpu, TTNNUnaryOpType::SUB_UNARY_SFPU},
      {FbUnaryOpType::MulUnarySfpu, TTNNUnaryOpType::MUL_UNARY_SFPU},
      {FbUnaryOpType::DivUnarySfpu, TTNNUnaryOpType::DIV_UNARY_SFPU},
      {FbUnaryOpType::IdentityUint32, TTNNUnaryOpType::IDENTITY},
      {FbUnaryOpType::UnaryNe, TTNNUnaryOpType::UNARY_NE},
      {FbUnaryOpType::UnaryGt, TTNNUnaryOpType::UNARY_GT},
      {FbUnaryOpType::UnaryLt, TTNNUnaryOpType::UNARY_LT},
      {FbUnaryOpType::TiledProd, TTNNUnaryOpType::TILED_PROD},
      {FbUnaryOpType::Typecast, TTNNUnaryOpType::TYPECAST},
      {FbUnaryOpType::BitwiseXor, TTNNUnaryOpType::BITWISE_XOR},
      {FbUnaryOpType::BitwiseNot, TTNNUnaryOpType::BITWISE_NOT},
      {FbUnaryOpType::BitwiseAnd, TTNNUnaryOpType::BITWISE_AND},
      {FbUnaryOpType::BitwiseOr, TTNNUnaryOpType::BITWISE_OR},
      {FbUnaryOpType::RightShift, TTNNUnaryOpType::RIGHT_SHIFT},
      {FbUnaryOpType::Floor, TTNNUnaryOpType::FLOOR},
      {FbUnaryOpType::Ceil, TTNNUnaryOpType::CEIL},
      {FbUnaryOpType::Round, TTNNUnaryOpType::ROUND},
      {FbUnaryOpType::LeftShift, TTNNUnaryOpType::LEFT_SHIFT},
      {FbUnaryOpType::Remainder, TTNNUnaryOpType::REMAINDER},
      {FbUnaryOpType::Fmod, TTNNUnaryOpType::FMOD},
      {FbUnaryOpType::Dropout, TTNNUnaryOpType::DROPOUT},
      {FbUnaryOpType::Fill, TTNNUnaryOpType::FILL},
      {FbUnaryOpType::PreluSfpu, TTNNUnaryOpType::PRELU_SFPU},
      {FbUnaryOpType::ZeroPoint, TTNNUnaryOpType::ZERO_POINT},
  };

  auto it = opTypeMap.find(unaryOpType);
  if (it != opTypeMap.end()) {
    return it->second;
  }

  LOG_FATAL("Unsupported UnaryOpType");
}

::ttnn::operations::unary::UnaryWithParam toTTNNUnaryWithParam(
    const ::tt::target::ttnn::UnaryWithParamT &unaryWithParam) {
  return ::ttnn::operations::unary::UnaryWithParam(
      toTTNNUnaryOpType(unaryWithParam.op_type),
      std::vector<float>(unaryWithParam.params.begin(),
                         unaryWithParam.params.end()));
}

/////////////////

const ::tt::target::ttnn::MemoryConfigT
getTensorRefMemoryConfig(const ::tt::target::ttnn::TensorRefT &tensorRef) {
  return *tensorRef.desc->layout->memory_desc->memory_config;
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
};

bool isSharded(
    const ::tt::target::ttnn::TensorMemoryLayout &tensorMemoryLayout) {
  return tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::HeightSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::WidthSharded ||
         tensorMemoryLayout ==
             ::tt::target::ttnn::TensorMemoryLayout::BlockSharded;
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

bool inSystemMemory(const ::tt::target::ttnn::TensorRefT &tensorRef) {
  const ::tt::target::ttnn::StorageType storageType =
      tensorRef.desc->layout->memory_desc->storage_type;
  return storageType == ::tt::target::ttnn::StorageType::Host;
}

std::optional<::ttnn::MemoryConfig>
createMemoryConfigIfNeeded(const ::tt::target::ttnn::MemoryConfigT &memcfg) {
  const auto targetBufferType = memcfg.buffer_type;
  LOG_ASSERT(targetBufferType == ::tt::target::BufferType::DRAM ||
                 targetBufferType == ::tt::target::BufferType::L1,
             "Memory config buffer type should be DRAM or L1");
  const auto ttnnBufferType = toTTNNBufferType(targetBufferType);

  const auto targetMemLayout = memcfg.tensor_memory_layout;
  const auto memLayout = toTTNNTensorMemoryLayout(targetMemLayout);

  // Verify that shard spec is present only for sharded memory layouts
  const bool hasShardSpec =
      (memcfg.shard_spec != nullptr) || (memcfg.nd_shard_spec != nullptr);
  LOG_ASSERT(
      hasShardSpec == isSharded(targetMemLayout),
      "A shard spec must be present if and only if the tensor is sharded");

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

::ttnn::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfigT &config) {
  ::ttnn::Conv2dConfig conv2dConfig;

  if (config.weights_dtype) {
    conv2dConfig.weights_dtype = toTTNNDataType(*config.weights_dtype);
  }

  if (config.activation) {
    conv2dConfig.activation =
        std::optional<::ttnn::operations::unary::UnaryWithParam>(
            toTTNNUnaryWithParam(*config.activation));
  }

  if (config.deallocate_activation) {
    conv2dConfig.deallocate_activation = *config.deallocate_activation;
  }

  if (config.reallocate_halo_output) {
    conv2dConfig.reallocate_halo_output = *config.reallocate_halo_output;
  }

  if (config.act_block_h_override) {
    conv2dConfig.act_block_h_override = *config.act_block_h_override;
  }

  if (config.act_block_w_div) {
    conv2dConfig.act_block_w_div = *config.act_block_w_div;
  }

  if (config.reshard_if_not_optimal) {
    conv2dConfig.reshard_if_not_optimal = *config.reshard_if_not_optimal;
  }

  if (config.override_sharding_config) {
    conv2dConfig.override_sharding_config = *config.override_sharding_config;
  }

  if (config.shard_layout) {
    conv2dConfig.shard_layout = toTTNNTensorMemoryLayout(*config.shard_layout);
  }

  if (config.core_grid) {
    conv2dConfig.core_grid =
        std::make_optional(toTTNNCoreRangeSet(*config.core_grid));
  }

  if (config.transpose_shards) {
    conv2dConfig.transpose_shards = *config.transpose_shards;
  }

  if (config.output_layout) {
    conv2dConfig.output_layout = toTTNNLayout(*config.output_layout);
  }

  if (config.enable_act_double_buffer) {
    conv2dConfig.enable_act_double_buffer = *config.enable_act_double_buffer;
  }

  if (config.enable_weights_double_buffer) {
    conv2dConfig.enable_weights_double_buffer =
        *config.enable_weights_double_buffer;
  }

  if (config.enable_kernel_stride_folding) {
    conv2dConfig.enable_kernel_stride_folding =
        *config.enable_kernel_stride_folding;
  }

  if (config.config_tensors_in_dram) {
    conv2dConfig.config_tensors_in_dram = *config.config_tensors_in_dram;
  }

  return conv2dConfig;
}

::ttnn::Conv2dSliceConfig::SliceType
createConv2dSliceType(::tt::target::ttnn::Conv2dSliceType sliceType) {
  switch (sliceType) {
  case ::tt::target::ttnn::Conv2dSliceType::DramHeight:
    return ::ttnn::Conv2dSliceConfig::SliceType::DRAM_HEIGHT;
  case ::tt::target::ttnn::Conv2dSliceType::DramWidth:
    return ::ttnn::Conv2dSliceConfig::SliceType::DRAM_WIDTH;
  case ::tt::target::ttnn::Conv2dSliceType::L1Full:
    return ::ttnn::Conv2dSliceConfig::SliceType::L1_FULL;
  }
}

::ttnn::Conv2dSliceConfig
createConv2dSliceConfig(const ::tt::target::ttnn::Conv2dSliceConfigT &config) {
  ::ttnn::Conv2dSliceConfig sliceConfig;

  sliceConfig.slice_type = createConv2dSliceType(config.slice_type);
  sliceConfig.num_slices = config.num_slices;

  return sliceConfig;
}

} // namespace unifiedOpLib::operations::utils