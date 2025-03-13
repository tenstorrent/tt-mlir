// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/utils.h"

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

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::UnaryOpType unaryOpType) {
  switch (unaryOpType) {
  case ::tt::target::ttnn::UnaryOpType::Exp:
    return ::ttnn::operations::unary::UnaryOpType::EXP;
  case ::tt::target::ttnn::UnaryOpType::Recip:
    return ::ttnn::operations::unary::UnaryOpType::RECIP;
  case ::tt::target::ttnn::UnaryOpType::Gelu:
    return ::ttnn::operations::unary::UnaryOpType::GELU;
  case ::tt::target::ttnn::UnaryOpType::Relu:
    return ::ttnn::operations::unary::UnaryOpType::RELU;
  case ::tt::target::ttnn::UnaryOpType::Sqrt:
    return ::ttnn::operations::unary::UnaryOpType::SQRT;
  case ::tt::target::ttnn::UnaryOpType::Sigmoid:
    return ::ttnn::operations::unary::UnaryOpType::SIGMOID;
  case ::tt::target::ttnn::UnaryOpType::Log:
    return ::ttnn::operations::unary::UnaryOpType::LOG;
  case ::tt::target::ttnn::UnaryOpType::Tanh:
    return ::ttnn::operations::unary::UnaryOpType::TANH;
  case ::tt::target::ttnn::UnaryOpType::Log2:
    return ::ttnn::operations::unary::UnaryOpType::LOG2;
  case ::tt::target::ttnn::UnaryOpType::Log10:
    return ::ttnn::operations::unary::UnaryOpType::LOG10;
  case ::tt::target::ttnn::UnaryOpType::Sin:
    return ::ttnn::operations::unary::UnaryOpType::SIN;
  case ::tt::target::ttnn::UnaryOpType::Cos:
    return ::ttnn::operations::unary::UnaryOpType::COS;
  case ::tt::target::ttnn::UnaryOpType::Abs:
    return ::ttnn::operations::unary::UnaryOpType::ABS;
  case ::tt::target::ttnn::UnaryOpType::AbsInt32:
    return ::ttnn::operations::unary::UnaryOpType::ABS_INT32;
  case ::tt::target::ttnn::UnaryOpType::Sign:
    return ::ttnn::operations::unary::UnaryOpType::SIGN;
  case ::tt::target::ttnn::UnaryOpType::Square:
    return ::ttnn::operations::unary::UnaryOpType::SQUARE;
  case ::tt::target::ttnn::UnaryOpType::Eqz:
    return ::ttnn::operations::unary::UnaryOpType::EQZ;
  case ::tt::target::ttnn::UnaryOpType::Nez:
    return ::ttnn::operations::unary::UnaryOpType::NEZ;
  case ::tt::target::ttnn::UnaryOpType::Gtz:
    return ::ttnn::operations::unary::UnaryOpType::GTZ;
  case ::tt::target::ttnn::UnaryOpType::Ltz:
    return ::ttnn::operations::unary::UnaryOpType::LTZ;
  case ::tt::target::ttnn::UnaryOpType::Gez:
    return ::ttnn::operations::unary::UnaryOpType::GEZ;
  case ::tt::target::ttnn::UnaryOpType::Lez:
    return ::ttnn::operations::unary::UnaryOpType::LEZ;
  case ::tt::target::ttnn::UnaryOpType::ReluMax:
    return ::ttnn::operations::unary::UnaryOpType::RELU_MAX;
  case ::tt::target::ttnn::UnaryOpType::ReluMin:
    return ::ttnn::operations::unary::UnaryOpType::RELU_MIN;
  case ::tt::target::ttnn::UnaryOpType::Power:
    return ::ttnn::operations::unary::UnaryOpType::POWER;
  case ::tt::target::ttnn::UnaryOpType::LeakyRelu:
    return ::ttnn::operations::unary::UnaryOpType::LEAKY_RELU;
  case ::tt::target::ttnn::UnaryOpType::Elu:
    return ::ttnn::operations::unary::UnaryOpType::ELU;
  case ::tt::target::ttnn::UnaryOpType::Exp2:
    return ::ttnn::operations::unary::UnaryOpType::EXP2;
  case ::tt::target::ttnn::UnaryOpType::Heaviside:
    return ::ttnn::operations::unary::UnaryOpType::HEAVISIDE;
  case ::tt::target::ttnn::UnaryOpType::Expm1:
    return ::ttnn::operations::unary::UnaryOpType::EXPM1;
  case ::tt::target::ttnn::UnaryOpType::Signbit:
    return ::ttnn::operations::unary::UnaryOpType::SIGNBIT;
  case ::tt::target::ttnn::UnaryOpType::Asin:
    return ::ttnn::operations::unary::UnaryOpType::ASIN;
  case ::tt::target::ttnn::UnaryOpType::Acos:
    return ::ttnn::operations::unary::UnaryOpType::ACOS;
  case ::tt::target::ttnn::UnaryOpType::Rsqrt:
    return ::ttnn::operations::unary::UnaryOpType::RSQRT;
  case ::tt::target::ttnn::UnaryOpType::Relu6:
    return ::ttnn::operations::unary::UnaryOpType::RELU6;
  case ::tt::target::ttnn::UnaryOpType::Atan:
    return ::ttnn::operations::unary::UnaryOpType::ATAN;
  case ::tt::target::ttnn::UnaryOpType::Erf:
    return ::ttnn::operations::unary::UnaryOpType::ERF;
  case ::tt::target::ttnn::UnaryOpType::Erfc:
    return ::ttnn::operations::unary::UnaryOpType::ERFC;
  case ::tt::target::ttnn::UnaryOpType::Isinf:
    return ::ttnn::operations::unary::UnaryOpType::ISINF;
  case ::tt::target::ttnn::UnaryOpType::Isposinf:
    return ::ttnn::operations::unary::UnaryOpType::ISPOSINF;
  case ::tt::target::ttnn::UnaryOpType::Isneginf:
    return ::ttnn::operations::unary::UnaryOpType::ISNEGINF;
  case ::tt::target::ttnn::UnaryOpType::Isnan:
    return ::ttnn::operations::unary::UnaryOpType::ISNAN;
  case ::tt::target::ttnn::UnaryOpType::LogicalNotUnary:
    return ::ttnn::operations::unary::UnaryOpType::LOGICAL_NOT_UNARY;
  case ::tt::target::ttnn::UnaryOpType::Isfinite:
    return ::ttnn::operations::unary::UnaryOpType::ISFINITE;
  case ::tt::target::ttnn::UnaryOpType::Erfinv:
    return ::ttnn::operations::unary::UnaryOpType::ERFINV;
  case ::tt::target::ttnn::UnaryOpType::I0:
    return ::ttnn::operations::unary::UnaryOpType::I0;
  case ::tt::target::ttnn::UnaryOpType::I1:
    return ::ttnn::operations::unary::UnaryOpType::I1;
  case ::tt::target::ttnn::UnaryOpType::Tan:
    return ::ttnn::operations::unary::UnaryOpType::TAN;
  case ::tt::target::ttnn::UnaryOpType::Rsub:
    return ::ttnn::operations::unary::UnaryOpType::RSUB;
  case ::tt::target::ttnn::UnaryOpType::Rdiv:
    return ::ttnn::operations::unary::UnaryOpType::RDIV;
  case ::tt::target::ttnn::UnaryOpType::Silu:
    return ::ttnn::operations::unary::UnaryOpType::SILU;
  case ::tt::target::ttnn::UnaryOpType::Softplus:
    return ::ttnn::operations::unary::UnaryOpType::SOFTPLUS;
  case ::tt::target::ttnn::UnaryOpType::Identity:
    return ::ttnn::operations::unary::UnaryOpType::IDENTITY;
  case ::tt::target::ttnn::UnaryOpType::Neg:
    return ::ttnn::operations::unary::UnaryOpType::NEG;
  case ::tt::target::ttnn::UnaryOpType::AddUnarySfpu:
    return ::ttnn::operations::unary::UnaryOpType::ADD_UNARY_SFPU;
  case ::tt::target::ttnn::UnaryOpType::SubUnarySfpu:
    return ::ttnn::operations::unary::UnaryOpType::SUB_UNARY_SFPU;
  case ::tt::target::ttnn::UnaryOpType::MulUnarySfpu:
    return ::ttnn::operations::unary::UnaryOpType::MUL_UNARY_SFPU;
  case ::tt::target::ttnn::UnaryOpType::DivUnarySfpu:
    return ::ttnn::operations::unary::UnaryOpType::DIV_UNARY_SFPU;
  case ::tt::target::ttnn::UnaryOpType::IdentityUint32:
    return ::ttnn::operations::unary::UnaryOpType::IDENTITY_UINT32;
  case ::tt::target::ttnn::UnaryOpType::UnaryNe:
    return ::ttnn::operations::unary::UnaryOpType::UNARY_NE;
  case ::tt::target::ttnn::UnaryOpType::UnaryGt:
    return ::ttnn::operations::unary::UnaryOpType::UNARY_GT;
  case ::tt::target::ttnn::UnaryOpType::UnaryLt:
    return ::ttnn::operations::unary::UnaryOpType::UNARY_LT;
  case ::tt::target::ttnn::UnaryOpType::TiledProd:
    return ::ttnn::operations::unary::UnaryOpType::TILED_PROD;
  case ::tt::target::ttnn::UnaryOpType::Typecast:
    return ::ttnn::operations::unary::UnaryOpType::TYPECAST;
  case ::tt::target::ttnn::UnaryOpType::BitwiseXor:
    return ::ttnn::operations::unary::UnaryOpType::BITWISE_XOR;
  case ::tt::target::ttnn::UnaryOpType::BitwiseNot:
    return ::ttnn::operations::unary::UnaryOpType::BITWISE_NOT;
  case ::tt::target::ttnn::UnaryOpType::BitwiseAnd:
    return ::ttnn::operations::unary::UnaryOpType::BITWISE_AND;
  case ::tt::target::ttnn::UnaryOpType::BitwiseOr:
    return ::ttnn::operations::unary::UnaryOpType::BITWISE_OR;
  case ::tt::target::ttnn::UnaryOpType::RightShift:
    return ::ttnn::operations::unary::UnaryOpType::RIGHT_SHIFT;
  case ::tt::target::ttnn::UnaryOpType::Floor:
    return ::ttnn::operations::unary::UnaryOpType::FLOOR;
  case ::tt::target::ttnn::UnaryOpType::FloorFloat32:
    return ::ttnn::operations::unary::UnaryOpType::FLOOR_FLOAT32;
  case ::tt::target::ttnn::UnaryOpType::Ceil:
    return ::ttnn::operations::unary::UnaryOpType::CEIL;
  case ::tt::target::ttnn::UnaryOpType::CeilFloat32:
    return ::ttnn::operations::unary::UnaryOpType::CEIL_FLOAT32;
  case ::tt::target::ttnn::UnaryOpType::LeftShift:
    return ::ttnn::operations::unary::UnaryOpType::LEFT_SHIFT;
  case ::tt::target::ttnn::UnaryOpType::Remainder:
    return ::ttnn::operations::unary::UnaryOpType::REMAINDER;
  case ::tt::target::ttnn::UnaryOpType::Fmod:
    return ::ttnn::operations::unary::UnaryOpType::FMOD;
  case ::tt::target::ttnn::UnaryOpType::Dropout:
    return ::ttnn::operations::unary::UnaryOpType::DROPOUT;
  case ::tt::target::ttnn::UnaryOpType::Fill:
    return ::ttnn::operations::unary::UnaryOpType::FILL;
  case ::tt::target::ttnn::UnaryOpType::PreluSfpu:
    return ::ttnn::operations::unary::UnaryOpType::PRELU_SFPU;
  case ::tt::target::ttnn::UnaryOpType::ZeroPoint:
    return ::ttnn::operations::unary::UnaryOpType::ZERO_POINT;
  }
}

::ttnn::operations::conv::conv2d::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg) {
  std::optional<::ttnn::TensorMemoryLayout> shardLayout = std::nullopt;
  if (memcfg->shard_layout()) {
    shardLayout = ::tt::runtime::ttnn::utils::toTTNNTensorMemoryLayout(
        memcfg->shard_layout().value());
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
      .core_grid = std::nullopt,
      .transpose_shards = memcfg->transpose_shards(),
      .output_layout =
          ::tt::runtime::ttnn::utils::toTTNNLayout(memcfg->output_layout()),
  };

  return conv2dConfig;
}

::ttnn::operations::matmul::MatmulProgramConfig createMatmulProgramConfig(
    const ::tt::target::ttnn::MatmulMultiCoreReuseProgramConfig
        *matmulProgramConfig) {}
} // namespace tt::runtime::ttnn::operations::utils
