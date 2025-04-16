// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"
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
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg) {
  std::optional<::ttnn::TensorMemoryLayout> shardLayout;
  if (memcfg->shard_layout()) {
    shardLayout = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        memcfg->shard_layout().value());
  }
  std::optional<::ttnn::CoreRangeSet> coreGrid;
  if (memcfg->core_grid()) {
    coreGrid =
        ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*memcfg->core_grid());
  }

  ::ttnn::operations::conv::conv2d::Conv2dConfig conv2dConfig = {
      .dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(memcfg->dtype()),
      .weights_dtype =
          ::tt::runtime::ttnn::utils::toTTNNDataType(memcfg->weights_dtype()),
      .activation = memcfg->activation()->str(),
      .input_channels_alignment = memcfg->input_channels_alignment(),
      .deallocate_activation = memcfg->deallocate_activation(),
      .reallocate_halo_output = memcfg->reallocate_halo_output(),
      .act_block_h_override = memcfg->act_block_h_override(),
      .act_block_w_div = memcfg->act_block_w_div(),
      .reshard_if_not_optimal = memcfg->reshard_if_not_optimal(),
      .override_sharding_config = memcfg->override_sharding_config(),
      .shard_layout = shardLayout,
      .core_grid = coreGrid,
      .transpose_shards = memcfg->transpose_shards(),
      .output_layout =
          ::tt::runtime::ttnn::utils::toTTNNLayout(memcfg->output_layout()),
      .preprocess_weights_on_device = memcfg->preprocess_weights_on_device(),
      .always_preprocess_weights = memcfg->always_preprocess_weights(),
      .enable_act_double_buffer = memcfg->enable_act_double_buffer(),
      .enable_weights_double_buffer = memcfg->enable_weights_double_buffer(),
      .enable_split_reader = memcfg->enable_split_reader(),
      .enable_subblock_padding = memcfg->enable_subblock_padding(),
      .in_place = memcfg->in_place(),
  };

  return conv2dConfig;
}

} // namespace tt::runtime::ttnn::operations::utils
