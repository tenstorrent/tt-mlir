// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/Utils/CoreRangeSet.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace mlir::tt::ttnn::op_model {

namespace conversion {

::tt::tt_metal::DataType getDataType(const ttcore::DataType dataType) {
  switch (dataType) {
  case ttcore::DataType::Float32:
    return ::tt::tt_metal::DataType::FLOAT32;
  case ttcore::DataType::BFloat16:
    return ::tt::tt_metal::DataType::BFLOAT16;
  case ttcore::DataType::BFP_BFloat8:
    return ::tt::tt_metal::DataType::BFLOAT8_B;
  case ttcore::DataType::BFP_BFloat4:
    return ::tt::tt_metal::DataType::BFLOAT4_B;
  case ttcore::DataType::UInt32:
    return ::tt::tt_metal::DataType::UINT32;
  case ttcore::DataType::UInt16:
    return ::tt::tt_metal::DataType::UINT16;
  case ttcore::DataType::UInt8:
    return ::tt::tt_metal::DataType::UINT8;
  case ttcore::DataType::Int32:
    return ::tt::tt_metal::DataType::INT32;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

ttcore::DataType getDataType(const ::tt::tt_metal::DataType dataType) {
  switch (dataType) {
  case ::tt::tt_metal::DataType::FLOAT32:
    return ttcore::DataType::Float32;
  case ::tt::tt_metal::DataType::BFLOAT16:
    return ttcore::DataType::BFloat16;
  case ::tt::tt_metal::DataType::BFLOAT8_B:
    return ttcore::DataType::BFP_BFloat8;
  case ::tt::tt_metal::DataType::BFLOAT4_B:
    return ttcore::DataType::BFP_BFloat4;
  case ::tt::tt_metal::DataType::UINT32:
    return ttcore::DataType::UInt32;
  case ::tt::tt_metal::DataType::UINT16:
    return ttcore::DataType::UInt16;
  case ::tt::tt_metal::DataType::UINT8:
    return ttcore::DataType::UInt8;
  case ::tt::tt_metal::DataType::INT32:
    return ttcore::DataType::Int32;
  default:
    throw std::runtime_error("Invalid element type");
  }
}

::ttnn::operations::unary::UnaryWithParam
getUnaryWithParams(UnaryWithParamAttr attr) {
  using TTMLIRUnaryOpType = mlir::tt::ttnn::UnaryOpType;
  using TTNNUnaryOpType = ::ttnn::operations::unary::UnaryOpType;

  static const std::unordered_map<TTMLIRUnaryOpType, TTNNUnaryOpType>
      opTypeMap = {
          {TTMLIRUnaryOpType::Exp, TTNNUnaryOpType::EXP},
          {TTMLIRUnaryOpType::Recip, TTNNUnaryOpType::RECIP},
          {TTMLIRUnaryOpType::Gelu, TTNNUnaryOpType::GELU},
          {TTMLIRUnaryOpType::Relu, TTNNUnaryOpType::RELU},
          {TTMLIRUnaryOpType::Sqrt, TTNNUnaryOpType::SQRT},
          {TTMLIRUnaryOpType::Sigmoid, TTNNUnaryOpType::SIGMOID},
          {TTMLIRUnaryOpType::Log, TTNNUnaryOpType::LOG},
          {TTMLIRUnaryOpType::Tanh, TTNNUnaryOpType::TANH},
          {TTMLIRUnaryOpType::Log2, TTNNUnaryOpType::LOG2},
          {TTMLIRUnaryOpType::Log10, TTNNUnaryOpType::LOG10},
          {TTMLIRUnaryOpType::Sin, TTNNUnaryOpType::SIN},
          {TTMLIRUnaryOpType::Cos, TTNNUnaryOpType::COS},
          {TTMLIRUnaryOpType::Abs, TTNNUnaryOpType::ABS},
          {TTMLIRUnaryOpType::AbsInt32, TTNNUnaryOpType::ABS_INT32},
          {TTMLIRUnaryOpType::Sign, TTNNUnaryOpType::SIGN},
          {TTMLIRUnaryOpType::Square, TTNNUnaryOpType::SQUARE},
          {TTMLIRUnaryOpType::Eqz, TTNNUnaryOpType::EQZ},
          {TTMLIRUnaryOpType::Nez, TTNNUnaryOpType::NEZ},
          {TTMLIRUnaryOpType::Gtz, TTNNUnaryOpType::GTZ},
          {TTMLIRUnaryOpType::Ltz, TTNNUnaryOpType::LTZ},
          {TTMLIRUnaryOpType::Gez, TTNNUnaryOpType::GEZ},
          {TTMLIRUnaryOpType::Lez, TTNNUnaryOpType::LEZ},
          {TTMLIRUnaryOpType::ReluMax, TTNNUnaryOpType::RELU_MAX},
          {TTMLIRUnaryOpType::ReluMin, TTNNUnaryOpType::RELU_MIN},
          {TTMLIRUnaryOpType::Power, TTNNUnaryOpType::POWER},
          {TTMLIRUnaryOpType::LeakyRelu, TTNNUnaryOpType::LEAKY_RELU},
          {TTMLIRUnaryOpType::Elu, TTNNUnaryOpType::ELU},
          {TTMLIRUnaryOpType::Exp2, TTNNUnaryOpType::EXP2},
          {TTMLIRUnaryOpType::Heaviside, TTNNUnaryOpType::HEAVISIDE},
          {TTMLIRUnaryOpType::Expm1, TTNNUnaryOpType::EXPM1},
          {TTMLIRUnaryOpType::Signbit, TTNNUnaryOpType::SIGNBIT},
          {TTMLIRUnaryOpType::Asin, TTNNUnaryOpType::ASIN},
          {TTMLIRUnaryOpType::Acos, TTNNUnaryOpType::ACOS},
          {TTMLIRUnaryOpType::Rsqrt, TTNNUnaryOpType::RSQRT},
          {TTMLIRUnaryOpType::Relu6, TTNNUnaryOpType::RELU6},
          {TTMLIRUnaryOpType::Atan, TTNNUnaryOpType::ATAN},
          {TTMLIRUnaryOpType::Erf, TTNNUnaryOpType::ERF},
          {TTMLIRUnaryOpType::Erfc, TTNNUnaryOpType::ERFC},
          {TTMLIRUnaryOpType::IsInf, TTNNUnaryOpType::ISINF},
          {TTMLIRUnaryOpType::IsPosInf, TTNNUnaryOpType::ISPOSINF},
          {TTMLIRUnaryOpType::IsNegInf, TTNNUnaryOpType::ISNEGINF},
          {TTMLIRUnaryOpType::IsNan, TTNNUnaryOpType::ISNAN},
          {TTMLIRUnaryOpType::LogicalNotUnary,
           TTNNUnaryOpType::LOGICAL_NOT_UNARY},
          {TTMLIRUnaryOpType::IsFinite, TTNNUnaryOpType::ISFINITE},
          {TTMLIRUnaryOpType::Erfinv, TTNNUnaryOpType::ERFINV},
          {TTMLIRUnaryOpType::I0, TTNNUnaryOpType::I0},
          {TTMLIRUnaryOpType::I1, TTNNUnaryOpType::I1},
          {TTMLIRUnaryOpType::Tan, TTNNUnaryOpType::TAN},
          {TTMLIRUnaryOpType::Rsub, TTNNUnaryOpType::RSUB},
          {TTMLIRUnaryOpType::Rdiv, TTNNUnaryOpType::RDIV},
          {TTMLIRUnaryOpType::Silu, TTNNUnaryOpType::SILU},
          {TTMLIRUnaryOpType::SoftPlus, TTNNUnaryOpType::SOFTPLUS},
          {TTMLIRUnaryOpType::Identity, TTNNUnaryOpType::IDENTITY},
          {TTMLIRUnaryOpType::Neg, TTNNUnaryOpType::NEG},
          {TTMLIRUnaryOpType::AddUnarySfpu, TTNNUnaryOpType::ADD_UNARY_SFPU},
          {TTMLIRUnaryOpType::SubUnarySfpu, TTNNUnaryOpType::SUB_UNARY_SFPU},
          {TTMLIRUnaryOpType::MulUnarySfpu, TTNNUnaryOpType::MUL_UNARY_SFPU},
          {TTMLIRUnaryOpType::DivUnarySfpu, TTNNUnaryOpType::DIV_UNARY_SFPU},
          {TTMLIRUnaryOpType::IdentityUint32, TTNNUnaryOpType::IDENTITY},
          {TTMLIRUnaryOpType::UnaryNe, TTNNUnaryOpType::UNARY_NE},
          {TTMLIRUnaryOpType::UnaryGt, TTNNUnaryOpType::UNARY_GT},
          {TTMLIRUnaryOpType::UnaryLt, TTNNUnaryOpType::UNARY_LT},
          {TTMLIRUnaryOpType::TiledProd, TTNNUnaryOpType::TILED_PROD},
          {TTMLIRUnaryOpType::Typecast, TTNNUnaryOpType::TYPECAST},
          {TTMLIRUnaryOpType::BitwiseXor, TTNNUnaryOpType::BITWISE_XOR},
          {TTMLIRUnaryOpType::BitwiseNot, TTNNUnaryOpType::BITWISE_NOT},
          {TTMLIRUnaryOpType::BitwiseAnd, TTNNUnaryOpType::BITWISE_AND},
          {TTMLIRUnaryOpType::BitwiseOr, TTNNUnaryOpType::BITWISE_OR},
          {TTMLIRUnaryOpType::RightShift, TTNNUnaryOpType::RIGHT_SHIFT},
          {TTMLIRUnaryOpType::Floor, TTNNUnaryOpType::FLOOR},
          {TTMLIRUnaryOpType::Ceil, TTNNUnaryOpType::CEIL},
          {TTMLIRUnaryOpType::Round, TTNNUnaryOpType::ROUND},
          {TTMLIRUnaryOpType::LeftShift, TTNNUnaryOpType::LEFT_SHIFT},
          {TTMLIRUnaryOpType::Remainder, TTNNUnaryOpType::REMAINDER},
          {TTMLIRUnaryOpType::Fmod, TTNNUnaryOpType::FMOD},
          {TTMLIRUnaryOpType::Dropout, TTNNUnaryOpType::DROPOUT},
          {TTMLIRUnaryOpType::Fill, TTNNUnaryOpType::FILL},
          {TTMLIRUnaryOpType::PreluSfpu, TTNNUnaryOpType::PRELU_SFPU},
          {TTMLIRUnaryOpType::ZeroPoint, TTNNUnaryOpType::ZERO_POINT},
      };

  auto it = opTypeMap.find(attr.getOpType());
  if (it != opTypeMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unsupported element type.");

  std::vector<float> params(attr.getParams().size());
  for (std::size_t i = 0; i < attr.getParams().size(); ++i) {
    params[i] = attr.getParams()[i].getValue().convertToFloat();
  }
  return ::ttnn::operations::unary::UnaryWithParam(it->second, params);
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

const std::array<uint32_t, 2> getShardShape(const TTNNLayoutAttr &layout) {
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

const std::array<uint32_t, 2>
getShardShape(const llvm::ArrayRef<int64_t> &shapeAttr) {
  assert(shapeAttr.size() == 2 && "Shard shape must be 2D");
  std::array<uint32_t, 2> shape;
  shape[0] = shapeAttr[0];
  shape[1] = shapeAttr[1];
  return shape;
}

::tt::tt_metal::Layout getPageLayout(const TTNNLayoutAttr &layout) {
  return layout.isTiled() ? ::tt::tt_metal::Layout::TILE
                          : ::tt::tt_metal::Layout::ROW_MAJOR;
}

::tt::tt_metal::Layout getPageLayout(Layout layout) {
  switch (layout) {
  case Layout::RowMajor:
    return ::tt::tt_metal::Layout::ROW_MAJOR;
  case Layout::Tile:
    return ::tt::tt_metal::Layout::TILE;
  case Layout::Invalid:
    return ::tt::tt_metal::Layout::INVALID;
  }
}

::tt::tt_metal::CoreRangeSet
getCoreRangeSet(const CoreRangeSetAttr &coreRangeSetAttr) {
  std::set<::tt::tt_metal::CoreRange> coreRangeSet;
  for (const CoreRangeAttr &coreRange : coreRangeSetAttr.getCoreRanges()) {
    coreRangeSet.insert(
        ::tt::tt_metal::CoreRange(CoreCoord(coreRange.getStartCoord().getX(),
                                            coreRange.getStartCoord().getY()),
                                  CoreCoord(coreRange.getEndCoord().getX(),
                                            coreRange.getEndCoord().getY())));
  }
  return ::tt::tt_metal::CoreRangeSet(coreRangeSet);
}

::tt::tt_metal::CoreRangeSet getCoreRangeSet(const TTNNLayoutAttr &layout) {
  std::set<::tt::tt_metal::CoreRange> coreRangeSet;
  assert(layout.getGrid().getMapping().isEmpty() == false);
  for (const auto &[loc, size] : ttcore::utils::toCoreRangeSet(
           layout.getGrid().getShape(), layout.getGrid().getMapping())) {
    coreRangeSet.insert(::tt::tt_metal::CoreRange(
        CoreCoord(loc[0], loc[1]),
        CoreCoord(loc[0] + size[0] - 1, loc[1] + size[1] - 1)));
  }
  return ::tt::tt_metal::CoreRangeSet(coreRangeSet);
}

std::optional<::tt::tt_metal::ShardSpec>
getShardSpec(const TTNNLayoutAttr &layout) {
  if (layout.getIgnorePhysicalLayout()) {
    return std::nullopt;
  }

  if (!isShardedMemoryLayout(
          layout.getMemLayoutOpt().value_or(TensorMemoryLayout::Interleaved))) {
    return std::nullopt;
  }

  // tt_ShardOrientation is not part of ttnn::TTNNLayoutAttr;
  // defaulting to ROW_MAJOR. TODO(jserbedzija): with issue #620
  return ::tt::tt_metal::ShardSpec(getCoreRangeSet(layout),
                                   getShardShape(layout),
                                   ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
}

::tt::tt_metal::ShardOrientation
getShardOrientation(const ShardOrientationAttr &shardOrientationAttr) {
  switch (shardOrientationAttr.getValue()) {
  case ShardOrientation::RowMajor:
    return ::tt::tt_metal::ShardOrientation::ROW_MAJOR;
  case ShardOrientation::ColMajor:
    return ::tt::tt_metal::ShardOrientation::COL_MAJOR;
  }
}

::tt::tt_metal::ShardSpec getShardSpec(const ShardSpecAttr &shardSpecAttr) {
  ::tt::tt_metal::CoreRangeSet coreRangeSet =
      getCoreRangeSet(shardSpecAttr.getCoreRangeSet());
  std::array<uint32_t, 2> shape =
      getShardShape(shardSpecAttr.getShape().getShape());
  ::tt::tt_metal::ShardOrientation orientation =
      getShardOrientation(shardSpecAttr.getShardOrientation());
  return ::tt::tt_metal::ShardSpec(coreRangeSet, shape, orientation);
}

::tt::tt_metal::BufferType getBufferType(const BufferType &bufferType) {
  switch (bufferType) {
  case BufferType::DRAM:
    return ::tt::tt_metal::BufferType::DRAM;
  case BufferType::L1:
    return ::tt::tt_metal::BufferType::L1;
  case BufferType::SystemMemory:
    return ::tt::tt_metal::BufferType::SYSTEM_MEMORY;
  case BufferType::L1Small:
    return ::tt::tt_metal::BufferType::L1_SMALL;
  case BufferType::Trace:
    return ::tt::tt_metal::BufferType::TRACE;
  }
}

::tt::tt_metal::BufferType getBufferType(const TTNNLayoutAttr &layout) {
  auto bufferType = layout.getBufferType();
  return getBufferType(bufferType);
}

BufferType getBufferType(const ::tt::tt_metal::BufferType bufferType) {
  switch (bufferType) {
  case ::tt::tt_metal::BufferType::DRAM:
    return BufferType::DRAM;
  case ::tt::tt_metal::BufferType::L1:
    return BufferType::L1;
  case ::tt::tt_metal::BufferType::SYSTEM_MEMORY:
    return BufferType::SystemMemory;
  case ::tt::tt_metal::BufferType::L1_SMALL:
    return BufferType::L1Small;
  case ::tt::tt_metal::BufferType::TRACE:
    return BufferType::Trace;
  }
}

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const TensorMemoryLayout tensorMemoryLayout) {
  switch (tensorMemoryLayout) {
  case TensorMemoryLayout::Interleaved:
    return ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  case TensorMemoryLayout::HeightSharded:
    return ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED;
  case TensorMemoryLayout::WidthSharded:
    return ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
  case TensorMemoryLayout::BlockSharded:
    return ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
  }
}
TensorMemoryLayout
getTensorMemoryLayout(const ::tt::tt_metal::TensorMemoryLayout memLayout) {
  switch (memLayout) {
  case ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED:
    return TensorMemoryLayout::Interleaved;
  case ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED:
    return TensorMemoryLayout::HeightSharded;
  case ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED:
    return TensorMemoryLayout::WidthSharded;
  case ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED:
    return TensorMemoryLayout::BlockSharded;
  }
}

::tt::tt_metal::TensorMemoryLayout
getTensorMemoryLayout(const TensorMemoryLayoutAttr memLayoutAttr) {
  auto tensorMemoryLayout = memLayoutAttr.getValue();
  return getTensorMemoryLayout(tensorMemoryLayout);
}

::tt::tt_metal::MemoryConfig getMemoryConfig(const TTNNLayoutAttr &layout) {
  auto tensorMemoryLayout = getTensorMemoryLayout(
      layout.getMemLayoutOpt().value_or(TensorMemoryLayout::Interleaved));
  auto bufferType = getBufferType(layout);

  auto shardSpec = getShardSpec(layout);
  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::MemoryConfig
getMemoryConfig(const MemoryConfigAttr &memConfigAttr) {
  // Get tensor memory layout if available, otherwise use INTERLEAVED as default
  ::tt::tt_metal::TensorMemoryLayout tensorMemoryLayout =
      ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
  if (memConfigAttr.getTensorMemoryLayout()) {
    tensorMemoryLayout =
        getTensorMemoryLayout(memConfigAttr.getTensorMemoryLayout());
  }

  // Convert buffer type enum
  ::tt::tt_metal::BufferType bufferType =
      ::tt::tt_metal::BufferType::DRAM; // Default to DRAM
  if (memConfigAttr.getBufferType()) {
    bufferType = getBufferType(memConfigAttr.getBufferType().getValue());
  }

  // Shard spec is not implemented for this version
  std::optional<::tt::tt_metal::ShardSpec> shardSpec = std::nullopt;
  if (memConfigAttr.getShardSpec()) {
    shardSpec = getShardSpec(memConfigAttr.getShardSpec().value());
  }

  return ::tt::tt_metal::MemoryConfig(tensorMemoryLayout, bufferType,
                                      shardSpec);
}

::tt::tt_metal::TensorLayout getTensorLayout(const TTNNLayoutAttr &layout) {
  return ::tt::tt_metal::TensorLayout(getDataType(layout.getDataType()),
                                      getPageLayout(layout),
                                      getMemoryConfig(layout));
}

::ttnn::TensorSpec getTensorSpec(const ::llvm::ArrayRef<int64_t> shape,
                                 const TTNNLayoutAttr &layout) {
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
    tensorSpec.compute_buffer_sharding_args();
  } catch (const std::exception &e) {
    return false;
  }
  return true;
}

::ttsl::SmallVector<int>
convertLLVMSmallVecToTTNNSmallVec(const ::llvm::ArrayRef<int64_t> vec) {
  return ::ttsl::SmallVector<int>(vec.begin(), vec.end());
}

std::optional<::ttnn::operations::conv::conv2d::Conv2dConfig>
getConv2dConfig(const std::optional<Conv2dConfigAttr> &conv2dConfig) {
  if (!conv2dConfig || !conv2dConfig.has_value() || !conv2dConfig.value()) {
    // TODO (azecevic): Has to be set explicitly to false, otherwise it will
    // assert for flattened Conv2dOp.
    // https://github.com/tenstorrent/tt-metal/issues/30985
    auto config = ::ttnn::operations::conv::conv2d::Conv2dConfig();
    config.enable_kernel_stride_folding = false;
    return config;
  }

  // TODO(#2130): config.core_grid is hardcoded to nullopt until we add
  // CoreRangeSet as an IR attribute.
  assert(!conv2dConfig->getCoreGrid() && "CoreGrid is not supported yet");

  ::ttnn::operations::conv::conv2d::Conv2dConfig config;

  if (conv2dConfig->getWeightsDtype()) {
    config.weights_dtype = getDataType(*conv2dConfig->getWeightsDtype());
  }

  if (conv2dConfig->getActivation()) {
    config.activation = getUnaryWithParams(conv2dConfig->getActivation());
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

  if (conv2dConfig->getEnableActDoubleBuffer()) {
    config.enable_act_double_buffer =
        conv2dConfig->getEnableActDoubleBuffer().getValue();
  }

  if (conv2dConfig->getEnableWeightsDoubleBuffer()) {
    config.enable_weights_double_buffer =
        conv2dConfig->getEnableWeightsDoubleBuffer().getValue();
  }

  if (conv2dConfig->getInPlace()) {
    config.in_place = conv2dConfig->getInPlace().getValue();
  }

  if (conv2dConfig->getEnableKernelStrideFolding()) {
    config.enable_kernel_stride_folding =
        conv2dConfig->getEnableKernelStrideFolding().getValue();
  }

  return config;
}

// sgholamiTT: I was on the fence for publicly exposing this API. Right now
// there's no clear usecase for it other than conversion from
// MathFidelity to ::ttnn::MathFidelity. Therefore, I decided to
// not expose it for now. Subject to change in the future.
::MathFidelity getMathFidelity(MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case MathFidelity::LoFi:
    return ::MathFidelity::LoFi;
  case MathFidelity::HiFi2:
    return ::MathFidelity::HiFi2;
  case MathFidelity::HiFi3:
    return ::MathFidelity::HiFi3;
  case MathFidelity::HiFi4:
    return ::MathFidelity::HiFi4;
  }
}

std::optional<::ttnn::DeviceComputeKernelConfig>
getDeviceComputeKernelConfig(const std::optional<DeviceComputeKernelConfigAttr>
                                 &deviceComputeKernelConfig) {
  if (!deviceComputeKernelConfig || !deviceComputeKernelConfig.has_value() ||
      !deviceComputeKernelConfig.value()) {
    return std::nullopt;
  }
  const DeviceComputeKernelConfigAttr &devConfig =
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

TTNNLayoutAttr getLayoutAttrFromTensorSpec(MLIRContext *context,
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
    elementType =
        ttcore::TileType::get(context,
                              {tensorSpec.page_config().get_tile().get_height(),
                               tensorSpec.page_config().get_tile().get_width()},
                              getDataType(tensorSpec.data_type()));
  } else {
    elementType =
        dataTypeToElementType(context, getDataType(tensorSpec.data_type()));
  }

  BufferType bufferType =
      getBufferType(tensorSpec.memory_config().buffer_type());
  auto memoryLayoutAttr = TensorMemoryLayoutAttr::get(
      context,
      getTensorMemoryLayout(tensorSpec.memory_config().memory_layout()));

  ttcore::GridAttr gridAttr = ttcore::GridAttr::get(context);
  if (isL1BufferType(bufferType)) {
    gridAttr = ttcore::GridAttr::get(
        context, getLogicalGridShape(tensorSpec.memory_config(), deviceGrid),
        optimizer_utils::createSingleDeviceVirtualToPhysicalAffineMap(
            context, memoryLayoutAttr.getValue(), deviceGrid));
  }

  return TTNNLayoutAttr::get(context, shape, elementType, bufferType, gridAttr,
                             memoryLayoutAttr);
}

} // namespace conversion
} // namespace mlir::tt::ttnn::op_model
#endif // TTMLIR_ENABLE_OPMODEL
