// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::utils {

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef) {
  const ::tt::target::Dim2d *tileShape =
      tensorRef->desc()->layout()->memory_desc()->tile_shape();
  return tileShape->x() == 32 && tileShape->y() == 32;
}

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef) {
  return ::tt::runtime::ttnn::utils::toTTNNDataType(
      tensorRef->desc()->layout()->memory_desc()->data_type());
}

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::ttnn::DistributionStrategy *strategy) {
  switch (strategy->strategy_type()) {
  case ::tt::target::ttnn::DistributedTensorConfig::ReplicateTensor: {
    return ::tt::tt_metal::ReplicateTensor(
        strategy->strategy_as_ReplicateTensor()->replication_factor());
  }
  case ::tt::target::ttnn::DistributedTensorConfig::ShardTensor: {
    return ::tt::tt_metal::ShardTensor(
        strategy->strategy_as_ShardTensor()->shard_dim());
  }
  case ::tt::target::ttnn::DistributedTensorConfig::ShardTensor2D: {
    uint32_t y = strategy->strategy_as_ShardTensor2D()->shard_mesh()->y();
    uint32_t x = strategy->strategy_as_ShardTensor2D()->shard_mesh()->x();
    ::tt::tt_metal::ShardMesh mesh(y, x);
    return ::tt::tt_metal::ShardTensor2D(mesh);
  }
  case ::tt::target::ttnn::DistributedTensorConfig::AllGatherTensor: {
    return ::tt::tt_metal::AllGatherTensor();
  }
  case ::tt::target::ttnn::DistributedTensorConfig::NONE: {
    LOG_FATAL("Unsupported distributed tensor config");
  }
  }
}

bool shouldSwapBinaryOperands(const ::ttnn::Tensor &lhs,
                              const ::ttnn::Tensor &rhs) {
  return (workaround::Env::get().swapBinaryOperands) &&
         (lhs.volume() < rhs.volume());
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
      {FbUnaryOpType::IdentityUint32, TTNNUnaryOpType::IDENTITY_UINT32},
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
      {FbUnaryOpType::FloorFloat32, TTNNUnaryOpType::FLOOR_FLOAT32},
      {FbUnaryOpType::Ceil, TTNNUnaryOpType::CEIL},
      {FbUnaryOpType::CeilFloat32, TTNNUnaryOpType::CEIL_FLOAT32},
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

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParam &unaryWithParam) {
  return ::ttnn::operations::unary::UnaryWithParam(
      toTTNNUnaryOpType(unaryWithParam.op_type()),
      std::vector<float>(unaryWithParam.params()->begin(),
                         unaryWithParam.params()->end()));
}

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::MatmulOp *op) {
  if (!op->matmul_program_config()) {
    return std::nullopt;
  }

  ::ttnn::operations::matmul::MatmulProgramConfig matmulProgramConfig;
  switch (op->matmul_program_config_type()) {
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseProgramConfig();
    return ::ttnn::operations::matmul::MatmulMultiCoreReuseProgramConfig{
        .compute_with_storage_grid_size =
            ::tt::runtime::ttnn::utils::toTTNNCoreCoord(
                *config->compute_with_storage_grid_size()),
        .in0_block_w = config->in0_block_w(),
        .out_subblock_h = config->out_subblock_h(),
        .out_subblock_w = config->out_subblock_w(),
        .per_core_M = config->per_core_m(),
        .per_core_N = config->per_core_n()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size =
                ::tt::runtime::ttnn::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .transpose_mcast = config->transpose_mcast(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .fuse_batch = config->fuse_batch()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCast1DProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCast1DProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size =
                ::tt::runtime::ttnn::utils::toTTNNCoreCoord(
                    *config->compute_with_storage_grid_size()),
            .in0_block_w = config->in0_block_w(),
            .out_subblock_h = config->out_subblock_h(),
            .out_subblock_w = config->out_subblock_w(),
            .out_block_h = config->out_block_h(),
            .out_block_w = config->out_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fuse_batch = config->fuse_batch(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
            .mcast_in0 = config->mcast_in0(),
            .gather_in0 = config->gather_in0(),
            .hop_cores = ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(
                *config->hop_cores()),
            .num_global_cb_receivers = config->num_global_cb_receivers()};
  }
  case ::tt::target::ttnn::MatmulProgramConfig::
      MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: {
    auto *config =
        op->matmul_program_config_as_MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig();
    return ::ttnn::operations::matmul::
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig{
            .in0_block_w = config->in0_block_w(),
            .per_core_M = config->per_core_m(),
            .per_core_N = config->per_core_n(),
            .fused_activation =
                config->fused_activation()
                    ? std::optional<::ttnn::operations::unary::UnaryWithParam>(
                          toTTNNUnaryWithParam(*config->fused_activation()))
                    : std::nullopt,
        };
  }
  default:
    LOG_FATAL("Unsupported MatmulProgramConfig type");
  }
}

::ttnn::operations::conv::conv2d::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *config) {
  ::ttnn::operations::conv::Conv2dConfig conv2dConfig;

  if (config->dtype()) {
    conv2dConfig.dtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*config->dtype());
  }

  if (config->weights_dtype()) {
    conv2dConfig.weights_dtype =
        ::tt::runtime::ttnn::utils::toTTNNDataType(*config->weights_dtype());
  }

  if (config->activation()) {
    conv2dConfig.activation = config->activation()->str();
  }

  if (config->input_channels_alignment()) {
    conv2dConfig.input_channels_alignment = *config->input_channels_alignment();
  }

  if (config->deallocate_activation()) {
    conv2dConfig.deallocate_activation = *config->deallocate_activation();
  }

  if (config->reallocate_halo_output()) {
    conv2dConfig.reallocate_halo_output = *config->reallocate_halo_output();
  }

  if (config->act_block_h_override()) {
    conv2dConfig.act_block_h_override = *config->act_block_h_override();
  }

  if (config->act_block_w_div()) {
    conv2dConfig.act_block_w_div = *config->act_block_w_div();
  }

  if (config->reshard_if_not_optimal()) {
    conv2dConfig.reshard_if_not_optimal = *config->reshard_if_not_optimal();
  }

  if (config->override_sharding_config()) {
    conv2dConfig.override_sharding_config = *config->override_sharding_config();
  }

  if (config->shard_layout()) {
    conv2dConfig.shard_layout =
        ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
            *config->shard_layout());
  }

  if (config->core_grid()) {
    conv2dConfig.core_grid = std::make_optional(
        ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*config->core_grid()));
  }

  if (config->transpose_shards()) {
    conv2dConfig.transpose_shards = *config->transpose_shards();
  }

  if (config->output_layout()) {
    conv2dConfig.output_layout =
        ::tt::runtime::ttnn::utils::toTTNNLayout(*config->output_layout());
  }

  if (config->preprocess_weights_on_device()) {
    conv2dConfig.preprocess_weights_on_device =
        *config->preprocess_weights_on_device();
  }

  if (config->always_preprocess_weights()) {
    conv2dConfig.always_preprocess_weights =
        *config->always_preprocess_weights();
  }

  if (config->enable_act_double_buffer()) {
    conv2dConfig.enable_act_double_buffer = *config->enable_act_double_buffer();
  }

  if (config->enable_weights_double_buffer()) {
    conv2dConfig.enable_weights_double_buffer =
        *config->enable_weights_double_buffer();
  }

  if (config->enable_split_reader()) {
    conv2dConfig.enable_split_reader = *config->enable_split_reader();
  }

  if (config->enable_subblock_padding()) {
    conv2dConfig.enable_subblock_padding = *config->enable_subblock_padding();
  }

  return conv2dConfig;
}

template <typename T>
static ::ttnn::Tensor
toTTNNTensorImpl(const ::flatbuffers::Vector<uint8_t> *data,
                 const ::ttnn::Shape &shape, const ::ttnn::DataType &dataType) {
  std::uint64_t numElements = shape.volume();
  size_t elementSize = sizeof(T);
  LOG_ASSERT(numElements * elementSize == data->size(), "Invalid data size");
  std::vector<T> dataVec(numElements);
  for (size_t i = 0; i < numElements; i++) {
    if constexpr (std::is_same_v<T, bfloat16>) {
      dataVec[i] = bfloat16(
          ::flatbuffers::IndirectHelper<uint16_t>::Read(data->data(), i));
    } else {
      dataVec[i] = ::flatbuffers::IndirectHelper<T>::Read(data->data(), i);
    }
  }
  return ::tt::runtime::ttnn::utils::createTTNNTensor<T>(dataVec.data(), shape,
                                                         dataType);
}

::ttnn::Tensor toTTNNTensor(const ::flatbuffers::Vector<uint8_t> *data,
                            const ::ttnn::Shape &shape,
                            const ::ttnn::DataType &dataType) {
  switch (dataType) {
  case ::ttnn::DataType::FLOAT32: {
    return toTTNNTensorImpl<float>(data, shape, dataType);
  }
  case ::ttnn::DataType::BFLOAT16: {
    return toTTNNTensorImpl<bfloat16>(data, shape, dataType);
  }
  case ::ttnn::DataType::UINT32: {
    return toTTNNTensorImpl<uint32_t>(data, shape, dataType);
  }
  case ::ttnn::DataType::UINT16: {
    return toTTNNTensorImpl<uint16_t>(data, shape, dataType);
  }
  case ::ttnn::DataType::UINT8: {
    return toTTNNTensorImpl<uint8_t>(data, shape, dataType);
  }
  case ::ttnn::DataType::INT32: {
    return toTTNNTensorImpl<int32_t>(data, shape, dataType);
  }
  default:
    LOG_FATAL("Unsupported data type");
  }
}

} // namespace tt::runtime::ttnn::operations::utils
