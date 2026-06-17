// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/SmallVector.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpInvoke/TTNN/Conv/Conv2dOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/Conv3dOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/ConvTranspose2dOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConv2dBiasOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConv2dWeightsOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConvTranspose2dBiasOp.h"
#include "ttmlir/OpInvoke/TTNN/Conv/PrepareConvTranspose2dWeightsOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/AssignOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/ConcatOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/GatherOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/PadOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/PermuteOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/RepeatInterleaveOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/RepeatOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/ReshapeOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/ScatterOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/SortOp.h"
#include "ttmlir/OpInvoke/TTNN/DataMovement/TransposeOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryCompositeOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Binary/EltwiseBinaryOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Quantization/EltwiseQuantizationOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Ternary/EltwiseTernaryOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryCompositeOp.h"
#include "ttmlir/OpInvoke/TTNN/Eltwise/Unary/EltwiseUnaryOp.h"
#include "ttmlir/OpInvoke/TTNN/Embedding/EmbeddingBackwardOp.h"
#include "ttmlir/OpInvoke/TTNN/Embedding/EmbeddingOp.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/FillCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/PagedFillCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/PagedUpdateCacheOp.h"
#include "ttmlir/OpInvoke/TTNN/Matmul/MatmulOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/BatchNormOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/GroupNormOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPostAllGatherOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/LayerNormPreAllGatherOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/RMSNormPreAllGatherOp.h"
#include "ttmlir/OpInvoke/TTNN/Normalization/SoftmaxOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/ArgMaxOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/CumSumOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/ProdOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/ReductionOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/TopKOp.h"
#include "ttmlir/OpInvoke/TTNN/Reduction/TopKRouterGptOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/ConcatenateHeadsOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPCreateQKVHeadsDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/PagedFlashMultiLatentAttentionDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/PagedScaledDotProductAttentionDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingLlamaOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/RotaryEmbeddingOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/ScaledDotProductAttentionDecodeOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/ScaledDotProductAttentionOp.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/SplitQueryKeyValueAndSplitHeadsOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#endif // TTMLIR_ENABLE_OPMODEL

namespace mlir::tt::ttnn::op_model {

#ifdef TTMLIR_ENABLE_OPMODEL

namespace operation {

/// RAII helper to preserve and restore the program cache state.
struct ProgramCacheState {
  ::tt::tt_metal::distributed::MeshDevice *device_ = nullptr;
  bool was_enabled_ = false;

  ProgramCacheState(::tt::tt_metal::distributed::MeshDevice *device)
      : device_(device) {
    was_enabled_ = device_->get_program_cache().is_enabled();
  }

  ~ProgramCacheState() {
    if (was_enabled_) {
      device_->enable_program_cache();
    }
  }
};

/**
 * @brief Executes a constraint query and validates the response.
 *
 * This helper function attempts to execute the provided callable to obtain
 * constraint query information. It handles exceptions and validates that the
 * response contains the required data.
 *
 * @param callable A callable object that performs the query.
 * @return A ConstraintQueryResponse if successful, or an error.
 */
template <class Callable>
llvm::Expected<::ttnn::graph::ConstraintQueryResponse>
executeConstraintQuery(Callable &callable) {
  ::ttnn::graph::ConstraintQueryResponse query;
  try {
    auto *device = SingletonDeviceContext::getInstance().getDevice();
    ::ttnn::graph::detail::LogLevelGuard log_guard(
        spdlog::level::level_enum::off);
    ProgramCacheState pcState(device);
    device->disable_and_clear_program_cache();
    query = callable();
  } catch (const std::exception &e) {
    // We expect that query will handle exceptions and set error message. If
    // not, we should not continue.
    // TODO(rpavlovicTT): This should be a TT_FATAL.
    llvm::errs() << "Exception thrown during op constraints query: " << e.what()
                 << "\n";
    assert(false && "Exception thrown during op constraints query");
  }

  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Op constraint query failed with error: " +
            query.error_message.value_or("<error message not set>"));
  }

  if (!query.output_tensor_specs.has_value() ||
      query.output_tensor_specs->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Op constraint query missing output tensor");
  }

  return query;
}

/**
 * @brief Retrieves operation constraints based on the provided operation name
 * and callable.
 *
 * This function attempts to query operation constraints using the provided
 * callable and arguments. If successful, it returns a tuple with resource usage
 * details and the actual layout of the output tensor of the op. Otherwise, an
 * error message.
 *
 * @param name The name of the operation to query constraints for.
 * @param context The MLIRContext to use for creating the TTNNLayoutAttr for the
 * output tensor
 * @param callable A callable object that performs the query.
 * @return A tuple containing query results or a string error.
 */
template <class Callable>
llvm::Expected<OpConstraints> getOpConstraints(MLIRContext *context,
                                               Callable &callable) {

  llvm::Expected<::ttnn::graph::ConstraintQueryResponse> query =
      executeConstraintQuery<Callable>(callable);
  if (auto error = query.takeError()) {
    return error;
  }

  ::ttnn::graph::ConstraintQueryResponse response = query.get();

  // The worker grid used to build interleaved output layouts is sourced from
  // the open device rather than threaded in from the IR: the two are equivalent
  // (the system desc that produced the IR's DeviceAttr is itself derived from
  // this grid), and this is the only place the value is consumed. The context
  // caches it across device open/reset, so this is a cheap lookup.
  const llvm::ArrayRef<int64_t> deviceGrid =
      SingletonDeviceContext::getInstance().getComputeGridShape();

  llvm::SmallVector<TTNNLayoutAttr> layoutAttrs;
  for (const auto &outputTensorSpec : response.output_tensor_specs.value()) {
    layoutAttrs.push_back(conversion::getLayoutAttrFromTensorSpec(
        context, outputTensorSpec, deviceGrid));
  }

  return OpConstraints(response.resource_usage.cb_peak_size_per_core,
                       response.resource_usage.l1_buffers_peak_per_core,
                       response.resource_usage.peak_memory_usage_per_core,
                       response.resource_usage.l1_output_buffer_per_core,
                       layoutAttrs);
}

template <class Callable>
llvm::Expected<size_t> getOpRuntime(Callable &callable) {
  if (SingletonDeviceContext::getInstance().isMockDevice()) {
    return llvm::createStringError(
        "getOpRuntime is not supported in mock device mode");
  }

  ::ttnn::graph::RuntimeQueryResponse query;
  try {
    query = callable();
  } catch (const std::exception &e) {
    query.status = ::ttnn::graph::ExecutionStatus::Error;
    query.error_message = e.what();
  }

  // Check if query was successful
  if (query.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        query.error_message.value_or("<error message not set>"));
  }

  return query.runtime;
}

} // namespace operation

namespace detail {
/**
 * @brief Convenience wrapper to create and validate a tensor spec
 *
 * @param device Pointer to an open device to obtain the compute grid size
 */
llvm::Expected<::ttnn::TensorSpec>
convertToTensorSpec(::tt::tt_metal::distributed::MeshDevice *device,
                    llvm::ArrayRef<int64_t> shape, TTNNLayoutAttr layout) {
  const ::ttnn::TensorSpec spec = conversion::getTensorSpec(shape, layout);
  if (conversion::validateTensorSpec(
          spec, device->compute_with_storage_grid_size())) {
    return spec;
  }

  return llvm::createStringError(
      "Unable to create TensorSpec out of given shape and layout");
}

std::optional<::ttnn::TensorSpec>
convertToOptionalTensorSpec(::tt::tt_metal::distributed::MeshDevice *device,
                            std::optional<llvm::ArrayRef<int64_t>> shape,
                            std::optional<TTNNLayoutAttr> layout) {
  std::optional<::ttnn::TensorSpec> ret = std::nullopt;
  if (shape.has_value() && layout.has_value()) {
    auto retExp =
        detail::convertToTensorSpec(device, shape.value(), layout.value());
    if (!retExp) {
      assert(false && "Failed to convert to TensorSpec");
      return std::nullopt;
    }
    ret = retExp.get();
  }
  return ret;
}

/**
 * @brief Convenience wrapper to get a memory config from a TTNNLayout attr that
 * may be a nullptr. Returns std::nullopt if layout is nullptr
 */
std::optional<::tt::tt_metal::MemoryConfig>
getNullableMemoryConfig(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getMemoryConfig(layout);
}

std::optional<::tt::target::ttnn::MemoryConfigT>
getNullableMemoryConfigT(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getMemoryConfigT(layout);
}

std::unique_ptr<::tt::target::ttnn::TensorRefT>
getOutputTensorRefT(TTNNLayoutAttr layout) {
  auto memoryConfigNative = getNullableMemoryConfigT(layout);
  if (!memoryConfigNative.has_value()) {
    return nullptr;
  }

  auto tensorRefNative = std::make_unique<::tt::target::ttnn::TensorRefT>();
  tensorRefNative->desc = std::make_unique<::tt::target::ttnn::TensorDescT>();
  tensorRefNative->desc->layout =
      std::make_unique<::tt::target::ttnn::LayoutDescT>();
  tensorRefNative->desc->layout->memory_desc =
      std::make_unique<::tt::target::ttnn::MemoryDescT>();
  tensorRefNative->desc->layout->memory_desc->memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          memoryConfigNative.value());

  tensorRefNative->desc->layout->memory_desc->data_type =
      toNative(layout.getDataType());

  if (auto tileType =
          mlir::dyn_cast<ttcore::TileType>(layout.getElementType())) {
    tensorRefNative->desc->layout->memory_desc->tile_shape =
        std::make_unique<::tt::target::Dim2d>(tileType.getHeight(),
                                              tileType.getWidth());
  } else {
    tensorRefNative->desc->layout->memory_desc->tile_shape =
        std::make_unique<::tt::target::Dim2d>(1, 1);
  }

  return tensorRefNative;
}

/**
 * @brief Reorder pool2d padding from IR convention to tt-metal convention.
 *
 * IR stores padding as [H_low, W_low, H_high, W_high] (top, left, bottom,
 * right) but tt-metal expects [top, bottom, left, right] (H_low, H_high,
 * W_low, W_high). The runtime does this reordering when executing from
 * flatbuffers, but the op_model constraint query path must do it too.
 */
std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>>
reorderPool2dPadding(llvm::ArrayRef<int32_t> padding) {
  if (padding.size() == 2) {
    return conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding);
  }
  return std::array<uint32_t, 4>{
      static_cast<uint32_t>(padding[0]), // top
      static_cast<uint32_t>(padding[2]), // bottom
      static_cast<uint32_t>(padding[1]), // left
      static_cast<uint32_t>(padding[3]), // right
  };
}

/**
 * @brief Convenience wrapper to get a DataType from a TTNNLayout attr that
 * may be a nullptr. Returns std::nullopt if layout is nullptr
 */
std::optional<::tt::tt_metal::DataType>
getNullableDataType(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return conversion::getDataType(layout.getDataType());
}

std::optional<::tt::target::DataType>
getNullableDataTypeT(TTNNLayoutAttr layout) {
  if (!layout) {
    return std::nullopt;
  }
  return toNative(layout.getDataType());
}

/**
 * @brief Checks if a C++ type T is compatible with a given MLIR type.
 *
 * @param elType The type to check.
 * @return True if the type is compatible, false otherwise.
 */
template <typename T>
bool isCompatibleType(mlir::Type elType) {
  if constexpr (std::is_same_v<T, float>) {
    return elType.isF32();
  } else if constexpr (std::is_same_v<T, double>) {
    return elType.isF64();
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return elType.isInteger(8);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return elType.isInteger(16);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return elType.isInteger(32);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return elType.isInteger(64);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return elType.isUnsignedInteger(8);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return elType.isUnsignedInteger(16);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return elType.isUnsignedInteger(32);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return elType.isUnsignedInteger(64);
  }
  return false;
}

/**
 * @brief This function populates a vector with the data that the ElementsAttr
 * contains. It checks for type compatibility and handles DenseElementsAttr and
 * SplatElementsAttr.
 */
template <typename T>
llvm::Expected<std::vector<T>>
getRawDataFromElementsAttr(mlir::ElementsAttr attr) {
  std::vector<T> result;
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!isCompatibleType<T>(denseAttr.getType().getElementType())) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch");
    }
    // Iterate over the elements
    for (auto value : denseAttr.getValues<T>()) {
      result.push_back(value);
    }

  } else if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
    // Handle splat attributes, Although this is not expected to be triggered
    // (since we have other ops to cover splat attributes, such as FullOp,
    // EmptyOp, etc), we can handle it here to avoid unnecessary failures.
    if (!isCompatibleType<T>(splatAttr.getType().getElementType())) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch");
    }
    auto splatValue = splatAttr.getSplatValue<T>();
    auto numElements = splatAttr.getType().getNumElements();
    result.resize(numElements, splatValue);
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unsupported ElementsAttr type");
  }
  return result;
}

// Template specialization for bfloat16 - MLIR doesn't have built-in support
// for bfloat16 in DenseElementsAttr::getValues<T>(), so we extract as uint16_t
// and convert to bfloat16
template <>
llvm::Expected<std::vector<bfloat16>>
getRawDataFromElementsAttr<bfloat16>(mlir::ElementsAttr attr) {
  std::vector<bfloat16> result;
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(attr)) {
    if (!denseAttr.getType().getElementType().isBF16()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch - expected BF16");
    }
    // Extract raw bytes as uint16_t and convert to bfloat16
    for (auto value : denseAttr.getValues<llvm::APFloat>()) {
      uint16_t rawBits =
          static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
      result.emplace_back(rawBits);
    }
  } else if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
    if (!splatAttr.getType().getElementType().isBF16()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Element type mismatch - expected BF16");
    }
    auto splatValue = splatAttr.getSplatValue<llvm::APFloat>();
    uint16_t rawBits =
        static_cast<uint16_t>(splatValue.bitcastToAPInt().getZExtValue());
    auto numElements = splatAttr.getType().getNumElements();
    result.resize(numElements, bfloat16(rawBits));
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unsupported ElementsAttr type");
  }
  return result;
}

template <typename OpTy>
auto getOpSymbol() {
  if constexpr (std::is_same_v<OpTy, ReluOp>) {
    return WRAP_OP(::ttnn::relu);
  } else if constexpr (std::is_same_v<OpTy, Relu6Op>) {
    return WRAP_OP(::ttnn::relu6);
  } else if constexpr (std::is_same_v<OpTy, HardsigmoidOp>) {
    return WRAP_OP(::ttnn::hardsigmoid);
  } else if constexpr (std::is_same_v<OpTy, SqrtOp>) {
    return WRAP_OP(::ttnn::sqrt);
  } else if constexpr (std::is_same_v<OpTy, SinOp>) {
    return WRAP_OP(::ttnn::sin);
  } else if constexpr (std::is_same_v<OpTy, AsinOp>) {
    return WRAP_OP(::ttnn::asin);
  } else if constexpr (std::is_same_v<OpTy, AsinhOp>) {
    return WRAP_OP(::ttnn::asinh);
  } else if constexpr (std::is_same_v<OpTy, AbsOp>) {
    return WRAP_OP(::ttnn::abs);
  } else if constexpr (std::is_same_v<OpTy, CeilOp>) {
    return WRAP_OP(::ttnn::ceil);
  } else if constexpr (std::is_same_v<OpTy, SignOp>) {
    return WRAP_OP(::ttnn::sign);
  } else if constexpr (std::is_same_v<OpTy, FloorOp>) {
    return WRAP_OP(::ttnn::floor);
  } else if constexpr (std::is_same_v<OpTy, IsFiniteOp>) {
    return WRAP_OP(::ttnn::isfinite);
  } else if constexpr (std::is_same_v<OpTy, ExpOp>) {
    return WRAP_OP(::ttnn::exp);
  } else if constexpr (std::is_same_v<OpTy, ErfOp>) {
    return WRAP_OP(::ttnn::erf);
  } else if constexpr (std::is_same_v<OpTy, ErfcOp>) {
    return WRAP_OP(::ttnn::erfc);
  } else if constexpr (std::is_same_v<OpTy, GeluOp>) {
    return WRAP_OP(::ttnn::gelu);
  } else if constexpr (std::is_same_v<OpTy, RsqrtOp>) {
    return WRAP_OP(::ttnn::rsqrt);
  } else if constexpr (std::is_same_v<OpTy, LogicalNotOp>) {
    return WRAP_OP(::ttnn::logical_not);
  } else if constexpr (std::is_same_v<OpTy, NegOp>) {
    return WRAP_OP(::ttnn::neg);
  } else if constexpr (std::is_same_v<OpTy, TanOp>) {
    return WRAP_OP(::ttnn::tan);
  } else if constexpr (std::is_same_v<OpTy, AtanOp>) {
    return WRAP_OP(::ttnn::atan);
  } else if constexpr (std::is_same_v<OpTy, Log1pOp>) {
    return WRAP_OP(::ttnn::log1p);
  } else if constexpr (std::is_same_v<OpTy, Expm1Op>) {
    return WRAP_OP(::ttnn::expm1);
  } else if constexpr (std::is_same_v<OpTy, CosOp>) {
    return WRAP_OP(::ttnn::cos);
  } else if constexpr (std::is_same_v<OpTy, AcosOp>) {
    return WRAP_OP(::ttnn::acos);
  } else if constexpr (std::is_same_v<OpTy, TanhOp>) {
    return WRAP_OP(::ttnn::tanh);
  } else if constexpr (std::is_same_v<OpTy, LogOp>) {
    return WRAP_OP(::ttnn::log);
  } else if constexpr (std::is_same_v<OpTy, ReciprocalOp>) {
    return WRAP_OP(::ttnn::reciprocal);
  } else if constexpr (std::is_same_v<OpTy, CbrtOp>) {
    return WRAP_OP(::ttnn::cbrt);
  } else if constexpr (std::is_same_v<OpTy, BitwiseNotOp>) {
    return WRAP_OP(::ttnn::bitwise_not);
  } else if constexpr (std::is_same_v<OpTy, AddOp>) {
    return WRAP_OP(::ttnn::add);
  } else if constexpr (std::is_same_v<OpTy, MultiplyOp>) {
    return WRAP_OP(::ttnn::multiply);
  } else if constexpr (std::is_same_v<OpTy, SubtractOp>) {
    return WRAP_OP(::ttnn::subtract);
  } else if constexpr (std::is_same_v<OpTy, LogicalRightShiftOp>) {
    return WRAP_OP(::ttnn::logical_right_shift);
  } else if constexpr (std::is_same_v<OpTy, LogicalLeftShiftOp>) {
    return WRAP_OP(::ttnn::logical_left_shift);
  } else if constexpr (std::is_same_v<OpTy, DivideOp>) {
    return WRAP_OP(::ttnn::divide);
  } else if constexpr (std::is_same_v<OpTy, EqualOp>) {
    return WRAP_OP(::ttnn::eq);
  } else if constexpr (std::is_same_v<OpTy, NotEqualOp>) {
    return WRAP_OP(::ttnn::ne);
  } else if constexpr (std::is_same_v<OpTy, GreaterEqualOp>) {
    return WRAP_OP(::ttnn::ge);
  } else if constexpr (std::is_same_v<OpTy, GreaterThanOp>) {
    return WRAP_OP(::ttnn::gt);
  } else if constexpr (std::is_same_v<OpTy, LessEqualOp>) {
    return WRAP_OP(::ttnn::le);
  } else if constexpr (std::is_same_v<OpTy, LessThanOp>) {
    return WRAP_OP(::ttnn::lt);
  } else if constexpr (std::is_same_v<OpTy, LogicalAndOp>) {
    return WRAP_OP(::ttnn::logical_and);
  } else if constexpr (std::is_same_v<OpTy, LogicalOrOp>) {
    return WRAP_OP(::ttnn::logical_or);
  } else if constexpr (std::is_same_v<OpTy, LogicalXorOp>) {
    return WRAP_OP(::ttnn::logical_xor);
  } else if constexpr (std::is_same_v<OpTy, MaximumOp>) {
    return WRAP_OP(::ttnn::maximum);
  } else if constexpr (std::is_same_v<OpTy, MinimumOp>) {
    return WRAP_OP(::ttnn::minimum);
  } else if constexpr (std::is_same_v<OpTy, BitwiseAndOp>) {
    return WRAP_OP(::ttnn::bitwise_and);
  } else if constexpr (std::is_same_v<OpTy, BitwiseOrOp>) {
    return WRAP_OP(::ttnn::bitwise_or);
  } else if constexpr (std::is_same_v<OpTy, BitwiseXorOp>) {
    return WRAP_OP(::ttnn::bitwise_xor);
  } else if constexpr (std::is_same_v<OpTy, RemainderOp>) {
    return WRAP_OP(::ttnn::remainder);
  } else if constexpr (std::is_same_v<OpTy, Atan2Op>) {
    return WRAP_OP(::ttnn::atan2);
  } else if constexpr (std::is_same_v<OpTy, PowTensorOp>) {
    return WRAP_OP(::ttnn::pow);
  } else if constexpr (std::is_same_v<OpTy, WhereOp>) {
    return WRAP_OP(::ttnn::where);
  } else if constexpr (std::is_same_v<OpTy, MeanOp>) {
    return WRAP_OP(::ttnn::mean);
  } else if constexpr (std::is_same_v<OpTy, MaxOp>) {
    return WRAP_OP(::ttnn::max);
  } else if constexpr (std::is_same_v<OpTy, MinOp>) {
    return WRAP_OP(::ttnn::min);
  } else if constexpr (std::is_same_v<OpTy, SumOp>) {
    return WRAP_OP(::ttnn::sum);
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::ZerosOp>) {
    return WRAP_OP(::ttnn::zeros);
  } else if constexpr (std::is_same_v<OpTy, mlir::tt::ttnn::OnesOp>) {
    return WRAP_OP(::ttnn::ones);
  } else if constexpr (std::is_same_v<OpTy, QuantizeOp>) {
    return WRAP_OP(::ttnn::quantize);
  } else if constexpr (std::is_same_v<OpTy, DequantizeOp>) {
    return WRAP_OP(::ttnn::dequantize);
  } else if constexpr (std::is_same_v<OpTy, GlobalAvgPool2dOp>) {
    return WRAP_OP(::ttnn::avg_pool2d);
  } else if constexpr (std::is_same_v<OpTy, SiluOp>) {
    return WRAP_OP(::ttnn::silu);
  } else if constexpr (std::is_same_v<OpTy, MishOp>) {
    return WRAP_OP(::ttnn::mish);
  } else {
    static_assert(ttmlir::utils::always_false(),
                  "add mapping from TTNN dialect to TTNN lib op");
  }
}

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

bool isLayoutLegalForTensorShape(llvm::ArrayRef<int64_t> tensorShape,
                                 TTNNLayoutAttr layout,
                                 ttcore::GridAttr maxGrid) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Conversion to TensorSpec may throw if the layout is invalid, in which case
  // we return false.
  try {
    auto tensorSpec = conversion::getTensorSpec(tensorShape, layout);
    // GridAttr.getShape() returns [Y, X] (rows, cols) per createWorkerGrid
    // convention; CoreCoord(x, y) takes X first.  Pass shape[1]=X to .x and
    // shape[0]=Y to .y so the validate's worker rectangle has the right
    // extent on non-square chips (e.g., Blackhole 10x11).
    auto computeGridSize = ::tt::tt_metal::CoreCoord{
        static_cast<std::size_t>(maxGrid.getShape()[1]),
        static_cast<std::size_t>(maxGrid.getShape()[0])};
    return conversion::validateTensorSpec(tensorSpec, computeGridSize);
  } catch (const std::exception &e) {
    return false;
  }
  return true;
#else
  return true;
#endif
}

#ifdef TTMLIR_ENABLE_OPMODEL

static ::tt::tt_metal::HostBuffer
createHostBuffer(uint32_t numElements, ::tt::tt_metal::DataType dataType) {
  switch (dataType) {
  case ::tt::tt_metal::DataType::FLOAT32: {
    std::vector<float> data(numElements);
    return ::tt::tt_metal::HostBuffer(std::move(data));
  }
  case ::tt::tt_metal::DataType::BFLOAT16: {
    std::vector<bfloat16> data(numElements);
    return ::tt::tt_metal::HostBuffer(std::move(data));
  }
  default:
    llvm::report_fatal_error("Unsupported data type");
  }
}

// Allocate a ttnn tensor with the given shape and data type.
static ::tt::tt_metal::Tensor
createMetalHostTensor(llvm::ArrayRef<int64_t> shape,
                      ttcore::DataType dataType) {
  // Calculate total volume of the tensor
  uint32_t volume = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    volume *= shape[i];
  }

  auto metalDataType = conversion::getDataType(dataType);
  auto hostBuffer = createHostBuffer(volume, metalDataType);
  auto metalShape = conversion::getShape(shape);
  ::tt::tt_metal::PageConfig pageconfig(::tt::tt_metal::Layout::ROW_MAJOR);
  ::tt::tt_metal::TensorLayout layout(metalDataType, pageconfig,
                                      ::tt::tt_metal::MemoryConfig{});
  ::tt::tt_metal::TensorSpec tensorSpec(metalShape, layout);

  return ::tt::tt_metal::Tensor(std::move(hostBuffer), tensorSpec);
}

// Returns the output tensor spec of the prepared weights for a conv2d op.
// Transform the standard OIHW weights layout to the ttnn convolution internal
// layout that is desired. The output shape is dependent on the conv2d config
// and input memory config.
llvm::Expected<::ttnn::TensorSpec> getPrepareConv2dWeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool hasBias,
    bool transpose) {
  if (weightLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be on host.");
  }
  if (weightLayout.getDataType() != ttcore::DataType::Float32 &&
      weightLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d weight tensor assumed to be float32 or bfloat16.");
  }

  // TODO(rpavlovicTT):: Move this to tt-metal side #4043
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> inputDtype =
      detail::getNullableDataType(inputLayout);

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(weightLayout);

  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);

  // Create query closure
  auto prepareConv2dWeightsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv2d::prepare_conv_weights, device,
        weightTensor, inputSpec.memory_config(), inputSpec.layout(), "OIHW",
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, *inputDtype, outputDtype,
        conv2dConfigConverted,
        /* compute_config_ */ std::nullopt, sliceConfigConverted);
  };

  auto prepareConvTranspose2dWeightsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_weights,
        device, weightTensor, inputSpec.memory_config(), inputSpec.layout(),
        "IOHW", in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, *inputDtype, outputDtype,
        conv2dConfigConverted,
        /* compute_config_ */ std::nullopt,
        /* dram_slice_config_ */
        std::optional<::ttnn::Conv2dSliceConfig>{},
        /* mirror_kernel */ true);
  };

  auto output =
      transpose
          ? operation::executeConstraintQuery(
                prepareConvTranspose2dWeightsOpQuery)
          : operation::executeConstraintQuery(prepareConv2dWeightsOpQuery);

  if (!output) {
    return output.takeError();
  }

  assert(output.get().output_tensor_specs.has_value() &&
         !output.get().output_tensor_specs->empty());
  return output.get().output_tensor_specs.value()[0];
}

// Returns the output tensor spec of the prepared bias for a conv2d op.
static llvm::Expected<::ttnn::TensorSpec>
getPrepareConv2dBiasOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout,
    ::tt::tt_metal::DataType weightsDtype, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig, bool transpose) {
  if (biasLayout.getBufferType() != BufferType::SystemMemory) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Conv2d bias tensor assumed to be on host.");
  }

  // TODO(rpavlovicTT):: Move this to tt-metal side #4043
  if (biasLayout.getDataType() != ttcore::DataType::Float32 &&
      biasLayout.getDataType() != ttcore::DataType::BFloat16) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Conv2d bias tensor assumed to be float32 or bfloat16.");
  }
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> inputDtype =
      detail::getNullableDataType(inputLayout);

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(biasLayout);

  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);

  auto prepareConv2dBiasOpQuery = [=]() {
    ::ttnn::Conv2dConfig localConfig;
    if (!conv2dConfigConverted.has_value()) {
      localConfig = ::ttnn::Conv2dConfig();
    } else {
      localConfig = *conv2dConfigConverted;
    }
    // Weights dtype must always be set for prepare_conv_bias.
    // tt-metal's prepare_conv_bias accesses weights_dtype.value() without
    // checking has_value(), causing std::bad_optional_access when unset.
    localConfig.weights_dtype = weightsDtype;

    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv2d::prepare_conv_bias, device,
        biasTensor, inputSpec.memory_config(), inputSpec.layout(), in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, *inputDtype, outputDtype, localConfig,
        /*compute_config_=*/std::nullopt,
        /* conv2d_slice_config_=*/std::nullopt);
  };

  auto prepareConvTranspose2dBiasOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_bias,
        device, biasTensor, inputSpec.memory_config(), inputSpec.layout(),
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, *inputDtype, outputDtype, conv2dConfigConverted,
        /*compute_config_=*/std::nullopt,
        /* dram_slice_config_ */
        std::optional<::ttnn::Conv2dSliceConfig>{});
  };

  auto output =
      transpose
          ? operation::executeConstraintQuery(prepareConvTranspose2dBiasOpQuery)
          : operation::executeConstraintQuery(prepareConv2dBiasOpQuery);

  if (!output) {
    return output.takeError();
  }

  assert(output.get().output_tensor_specs.has_value() &&
         !output.get().output_tensor_specs->empty());
  return output.get().output_tensor_specs.value().at(0);
}

#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// Unary Eltwise Ops
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryOpT buildEltwiseUnaryOpTFromMLIR(
    TTNNLayoutAttr outputLayout,
    std::optional<llvm::APFloat> slope = std::nullopt) {
  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOp;

  if (std::is_same_v<OpTy, TanhOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::Tanh;
  } else if (std::is_same_v<OpTy, SigmoidOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::Sigmoid;
  } else if (std::is_same_v<OpTy, LeakyReluOp>) {
    eltwiseUnaryOp.type = ::tt::target::ttnn::EltwiseUnaryOpType::LeakyRelu;
    assert(slope.has_value() && "LeakyReluOp requires a slope value");
    ::tt::target::ttnn::EltwiseOpWithFloatParamsT
        eltwiseOpWithFloatParamsNative;
    eltwiseOpWithFloatParamsNative.parameter = slope.value().convertToFloat();
    eltwiseUnaryOp.params.Set(eltwiseOpWithFloatParamsNative);
  }

  eltwiseUnaryOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseOpModel<OpTy>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnary(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryOpNative, detail::getOpSymbol<OpTy>(), inputSpec,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EltwiseUnaryOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t>
UnaryEltwiseOpModel<OpTy>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                        TTNNLayoutAttr inputLayout,
                                        TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnary(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseUnaryOpNative,
            detail::getOpSymbol<OpTy>(), inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnaryOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseWithFastApproxModeOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryWithFastAndApproximateMode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryOpNative, detail::getOpSymbol<OpTy>(), inputSpec,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseUnaryWithFastAndApproximateModeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t>
UnaryEltwiseWithFastApproxModeOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryWithFastAndApproximateMode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseUnaryOpNative,
            detail::getOpSymbol<OpTy>(), inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from "
        "EltwiseUnaryWithFastAndApproximateModeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Unary Composite Eltwise Ops
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryCompositeEltwiseOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryComposite(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
            result) &&
        "Expected ConstraintQueryResponse from EltwiseUnaryCompositeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> UnaryCompositeEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryComposite(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseUnaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnaryCompositeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryCompositeEltwiseWithFastApproxModeOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeWithFastAndApproximateMode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseUnaryCompositeWithFastAndApproximateModeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t>
UnaryCompositeEltwiseWithFastApproxModeOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeWithFastAndApproximateMode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseUnaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from "
        "EltwiseUnaryCompositeWithFastAndApproximateModeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

#define INSTANTIATE_BUILD_ELTWISE_UNARY(OP)                                    \
  template ::tt::target::ttnn::EltwiseUnaryOpT                                 \
      buildEltwiseUnaryOpTFromMLIR<OP>(TTNNLayoutAttr,                         \
                                       std::optional<llvm::APFloat>);

#define INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(OP)                          \
  template ::tt::target::ttnn::EltwiseUnaryCompositeOpT                        \
      buildEltwiseUnaryCompositeOpTFromMLIR<OP>(TTNNLayoutAttr);

// Explicit template instantiation for UnaryEltwiseOpModel.
template struct UnaryEltwiseOpModel<ReluOp>;
template struct UnaryEltwiseOpModel<Relu6Op>;
template struct UnaryEltwiseOpModel<HardsigmoidOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<SqrtOp>;
template struct UnaryEltwiseOpModel<SinOp>;
template struct UnaryEltwiseOpModel<AbsOp>;
template struct UnaryEltwiseOpModel<CosOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<LogOp>;
template struct UnaryEltwiseOpModel<CeilOp>;
template struct UnaryEltwiseOpModel<SignOp>;
template struct UnaryEltwiseOpModel<FloorOp>;
template struct UnaryEltwiseOpModel<IsFiniteOp>;
template struct UnaryEltwiseOpModel<LogicalNotOp>;
template struct UnaryEltwiseOpModel<NegOp>;
template struct UnaryEltwiseOpModel<TanOp>;
template struct UnaryEltwiseOpModel<AtanOp>;
template struct UnaryEltwiseOpModel<AsinOp>;
template struct UnaryEltwiseOpModel<AsinhOp>;
template struct UnaryEltwiseOpModel<AcosOp>;
template struct UnaryEltwiseOpModel<ReciprocalOp>;
template struct UnaryCompositeEltwiseOpModel<CbrtOp>;
template struct UnaryEltwiseOpModel<BitwiseNotOp>;
template struct UnaryEltwiseOpModel<SiluOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<MishOp>;
template struct UnaryCompositeEltwiseWithFastApproxModeOpModel<Log1pOp>;
template struct UnaryEltwiseOpModel<Expm1Op>;
template struct UnaryEltwiseWithFastApproxModeOpModel<RsqrtOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ErfOp>;
template struct UnaryEltwiseOpModel<ErfcOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ExpOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<GeluOp>;

#ifdef TTMLIR_ENABLE_OPMODEL
INSTANTIATE_BUILD_ELTWISE_UNARY(ReluOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(Relu6Op);
INSTANTIATE_BUILD_ELTWISE_UNARY(HardsigmoidOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SqrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SinOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AbsOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(CosOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LogOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(CeilOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SignOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(FloorOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(IsFiniteOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LogicalNotOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(NegOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(TanOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AtanOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AsinOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AsinhOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(AcosOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ReciprocalOp);
INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(CbrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(BitwiseNotOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SiluOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(MishOp);
INSTANTIATE_BUILD_ELTWISE_UNARY_COMPOSITE(Log1pOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(Expm1Op);
INSTANTIATE_BUILD_ELTWISE_UNARY(RsqrtOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ErfOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ErfcOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(ExpOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(GeluOp);

INSTANTIATE_BUILD_ELTWISE_UNARY(TanhOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(SigmoidOp);
INSTANTIATE_BUILD_ELTWISE_UNARY(LeakyReluOp);
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// TanhOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<TanhOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                  TTNNLayoutAttr inputLayout,
                                  TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<TanhOp>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryTanh(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryOpNative, ::ttnn::tanh, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EltwiseUnaryTanhOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<TanhOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                              TTNNLayoutAttr inputLayout,
                              TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto inputSpecExp =
      detail::convertToTensorSpec(device, inputShape, inputLayout);
  if (!inputSpecExp) {
    return inputSpecExp.takeError();
  }
  ::ttnn::TensorSpec inputSpec = inputSpecExp.get();

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<TanhOp>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryTanh(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseUnaryOpNative,
            ::ttnn::tanh, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnaryTanhOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<SigmoidOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                     TTNNLayoutAttr inputLayout,
                                     TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<SigmoidOp>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnarySigmoid(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryOpNative, ::ttnn::sigmoid, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseUnarySigmoidOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<SigmoidOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 TTNNLayoutAttr inputLayout,
                                 TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<SigmoidOp>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnarySigmoid(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseUnaryOpNative,
            ::ttnn::sigmoid, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnarySigmoidOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LeakyReluOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<LeakyReluOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat slope, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<LeakyReluOp>(outputLayout, slope);

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryWithFloatParameter(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryOpNative, ::ttnn::leaky_relu, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "runEltwiseUnaryWithFloatParameterOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     leakyReluOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LeakyReluOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat slope, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryOpT eltwiseUnaryOpNative =
      buildEltwiseUnaryOpTFromMLIR<LeakyReluOp>(outputLayout, slope);

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    ttnn_op_invoke::EltwiseUnaryOpResult result =
        ttnn_op_invoke::callEltwiseUnaryWithFloatParameter(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseUnaryOpNative,
            ::ttnn::leaky_relu, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from "
        "runEltwiseUnaryWithFloatParameterOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(leakyReluOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryOpT
buildEltwiseBinaryOpTFromMLIR(TTNNLayoutAttr outputLayout,
                              ttcore::DataTypeAttr opDtypeAttr = nullptr) {
  ::tt::target::ttnn::EltwiseBinaryOpT eltwiseBinaryOp;

  eltwiseBinaryOp.out = detail::getOutputTensorRefT(outputLayout);
  if (eltwiseBinaryOp.out) {
    eltwiseBinaryOp.output_dtype =
        eltwiseBinaryOp.out->desc->layout->memory_desc->data_type;
  }
  if (opDtypeAttr) {
    eltwiseBinaryOp.output_dtype = toNative(opDtypeAttr.getValue());
    if (eltwiseBinaryOp.out && eltwiseBinaryOp.output_dtype.has_value()) {
      eltwiseBinaryOp.out->desc->layout->memory_desc->data_type =
          eltwiseBinaryOp.output_dtype.value();
    }
  }

  return eltwiseBinaryOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryEltwiseOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr opDtypeAttr) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::EltwiseBinaryOpT eltwiseBinaryOpNative =
      buildEltwiseBinaryOpTFromMLIR<OpTy>(outputLayout, opDtypeAttr);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryOpResult result =
        ttnn_op_invoke::callEltwiseBinary(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseBinaryOpNative, detail::getOpSymbol<OpTy>(), inputSpecA,
            inputSpecB, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EltwiseBinaryOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> BinaryEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::EltwiseBinaryOpT eltwiseBinaryOpNative =
      buildEltwiseBinaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryOpResult result =
        ttnn_op_invoke::callEltwiseBinary(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, eltwiseBinaryOpNative,
            detail::getOpSymbol<OpTy>(), inputSpecA, inputSpecB, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseBinaryOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseBinaryCompositeOpT
buildEltwiseBinaryCompositeOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseBinaryCompositeOpT eltwiseBinaryCompositeOp;

  eltwiseBinaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseBinaryCompositeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryCompositeOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr /*opDtypeAttr*/) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::EltwiseBinaryCompositeOpT eltwiseBinaryCompositeOpNative =
      buildEltwiseBinaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryOpResult result =
        ttnn_op_invoke::callEltwiseBinaryComposite(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseBinaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpecA, inputSpecB, device);

    assert(
        std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
            result) &&
        "Expected ConstraintQueryResponse from EltwiseBinaryCompositeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> BinaryCompositeOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::EltwiseBinaryCompositeOpT eltwiseBinaryCompositeOpNative =
      buildEltwiseBinaryCompositeOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryOpResult result =
        ttnn_op_invoke::callEltwiseBinaryComposite(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseBinaryCompositeOpNative, detail::getOpSymbol<OpTy>(),
            inputSpecA, inputSpecB, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseBinaryCompositeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

#define INSTANTIATE_BUILD_ELTWISE_BINARY(OP)                                   \
  template ::tt::target::ttnn::EltwiseBinaryOpT                                \
      buildEltwiseBinaryOpTFromMLIR<OP>(TTNNLayoutAttr, ttcore::DataTypeAttr);

#define INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(OP)                         \
  template ::tt::target::ttnn::EltwiseBinaryCompositeOpT                       \
      buildEltwiseBinaryCompositeOpTFromMLIR<OP>(TTNNLayoutAttr);

// Explicit template instantiation for BinaryEltwiseOpModel.
template struct BinaryEltwiseOpModel<AddOp>;
template struct BinaryEltwiseOpModel<MultiplyOp>;
template struct BinaryEltwiseOpModel<LogicalRightShiftOp>;
template struct BinaryEltwiseOpModel<SubtractOp>;
template struct BinaryEltwiseOpModel<MaximumOp>;
template struct BinaryEltwiseOpModel<MinimumOp>;
template struct BinaryEltwiseOpModel<DivideOp>;
template struct BinaryEltwiseOpModel<EqualOp>;
template struct BinaryEltwiseOpModel<NotEqualOp>;
template struct BinaryEltwiseOpModel<GreaterEqualOp>;
template struct BinaryEltwiseOpModel<GreaterThanOp>;
template struct BinaryEltwiseOpModel<LessEqualOp>;
template struct BinaryEltwiseOpModel<LessThanOp>;
template struct BinaryEltwiseOpModel<LogicalAndOp>;
template struct BinaryEltwiseOpModel<LogicalOrOp>;
template struct BinaryEltwiseOpModel<LogicalXorOp>;
template struct BinaryEltwiseOpModel<PowTensorOp>;
template struct BinaryEltwiseOpModel<RemainderOp>;
// BinaryCompositeOpModel
template struct BinaryCompositeOpModel<BitwiseAndOp>;
template struct BinaryCompositeOpModel<BitwiseOrOp>;
template struct BinaryCompositeOpModel<BitwiseXorOp>;
template struct BinaryCompositeOpModel<LogicalLeftShiftOp>;
template struct BinaryCompositeOpModel<Atan2Op>;

#ifdef TTMLIR_ENABLE_OPMODEL
INSTANTIATE_BUILD_ELTWISE_BINARY(AddOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MultiplyOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalRightShiftOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(SubtractOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MaximumOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(MinimumOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(DivideOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(EqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(NotEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(GreaterEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(GreaterThanOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LessEqualOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LessThanOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalAndOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalOrOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(LogicalXorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(PowTensorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY(RemainderOp);

INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseAndOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseOrOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(BitwiseXorOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(LogicalLeftShiftOp);
INSTANTIATE_BUILD_ELTWISE_BINARY_COMPOSITE(Atan2Op);
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// GeluBackwardOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<GeluBackwardOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::string approximate, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::experimental::gelu_bw, device,
                                inputSpecA, inputSpecB, approximate,
                                outputMemoryConfig);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<GeluBackwardOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::string approximate, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::gelu_bw, device, inputSpecA,
                            inputSpecB, approximate, outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "OpRuntime not yet implemented for gelu_bw");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PowScalar
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
buildEltwiseBinaryCompositeScalarOpTFromMLIR(mlir::Attribute exponent,
                                             TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
      eltwiseBinaryCompositeScalarOp;
  eltwiseBinaryCompositeScalarOp.type =
      ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpType::PowScalar;

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(exponent)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    eltwiseBinaryCompositeScalarOp.rhs.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(exponent)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    eltwiseBinaryCompositeScalarOp.rhs.Set(i32);
  } else {
    llvm::report_fatal_error("Invalid exponent");
  }

  eltwiseBinaryCompositeScalarOp.out =
      detail::getOutputTensorRefT(outputLayout);

  return eltwiseBinaryCompositeScalarOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PowScalarOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute exponent, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
      eltwiseBinaryCompositeScalarOpNative =
          buildEltwiseBinaryCompositeScalarOpTFromMLIR(exponent, outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryCompositeScalarOpResult result =
        ttnn_op_invoke::callEltwiseBinaryCompositeScalar(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseBinaryCompositeScalarOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseBinaryCompositeScalarOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PowScalarOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute exponent, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseBinaryCompositeScalarOpT
      eltwiseBinaryCompositeScalarOpNative =
          buildEltwiseBinaryCompositeScalarOpTFromMLIR(exponent, outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseBinaryCompositeScalarOpResult result =
        ttnn_op_invoke::callEltwiseBinaryCompositeScalar(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseBinaryCompositeScalarOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from "
        "EltwiseBinaryCompositeScalarOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseTernaryWhereOpT
buildEltwiseTernaryOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseTernaryWhereOpT eltwiseTernaryWhereOp;

  eltwiseTernaryWhereOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseTernaryWhereOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints> TernaryEltwiseOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecC,
      detail::convertToTensorSpec(device, inputShapeC, inputLayoutC));

  ::tt::target::ttnn::EltwiseTernaryWhereOpT eltwiseTernaryWhereOpNative =
      buildEltwiseTernaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseTernaryOpResult result =
        ttnn_op_invoke::callEltwiseTernary(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseTernaryWhereOpNative, detail::getOpSymbol<OpTy>(),
            inputSpecA, inputSpecB, inputSpecC, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EltwiseTernaryOp query");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> TernaryEltwiseOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecC,
      detail::convertToTensorSpec(device, inputShapeC, inputLayoutC));

  ::tt::target::ttnn::EltwiseTernaryWhereOpT eltwiseTernaryWhereOpNative =
      buildEltwiseTernaryOpTFromMLIR<OpTy>(outputLayout);

  // Create query closure
  auto query = [=]() {
    ttnn_op_invoke::EltwiseTernaryOpResult result =
        ttnn_op_invoke::callEltwiseTernary(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseTernaryWhereOpNative, detail::getOpSymbol<OpTy>(),
            inputSpecA, inputSpecB, inputSpecC, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseTernaryOp query");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

#define INSTANTIATE_BUILD_ELTWISE_TERNARY(OP)                                  \
  template ::tt::target::ttnn::EltwiseTernaryWhereOpT                          \
      buildEltwiseTernaryOpTFromMLIR<OP>(TTNNLayoutAttr);

// Explicit template instantiation for TernaryEltwiseOpModel.
template struct TernaryEltwiseOpModel<WhereOp>;

#ifdef TTMLIR_ENABLE_OPMODEL
INSTANTIATE_BUILD_ELTWISE_TERNARY(WhereOp);
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
static ::tt::target::ttnn::ReductionOpType getReductionOpType() {
  if constexpr (std::is_same_v<OpTy, SumOp>) {
    return ::tt::target::ttnn::ReductionOpType::Sum;
  } else if constexpr (std::is_same_v<OpTy, MeanOp>) {
    return ::tt::target::ttnn::ReductionOpType::Mean;
  } else if constexpr (std::is_same_v<OpTy, MaxOp>) {
    return ::tt::target::ttnn::ReductionOpType::Max;
  } else if constexpr (std::is_same_v<OpTy, MinOp>) {
    return ::tt::target::ttnn::ReductionOpType::Min;
  }
  llvm_unreachable("Unsupported reduction op type");
}

::tt::target::ttnn::ReductionOpT buildReductionOpTFromMLIR(
    ::tt::target::ttnn::ReductionOpType type,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ReductionOpT reductionOp;
  reductionOp.type = type;
  if (dimArg) {
    for (int64_t v : *dimArg) {
      reductionOp.dim_arg.push_back(static_cast<int32_t>(v));
    }
  }
  reductionOp.keep_dim = keepDim;
  reductionOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  reductionOp.out = detail::getOutputTensorRefT(outputLayout);
  return reductionOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints> ReductionOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReductionOpT reductionOpNative =
      buildReductionOpTFromMLIR(getReductionOpType<OpTy>(), dimArg, keepDim,
                                computeKernelConfig, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::ReductionOpResult result = ttnn_op_invoke::callReduction(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, reductionOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ReductionOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> ReductionOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReductionOpT reductionOpNative =
      buildReductionOpTFromMLIR(getReductionOpType<OpTy>(), dimArg, keepDim,
                                computeKernelConfig, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::ReductionOpResult result = ttnn_op_invoke::callReduction(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, reductionOpNative,
        inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from ReductionOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for ReductionOpModel.
template struct ReductionOpModel<MeanOp>;
template struct ReductionOpModel<SumOp>;
template struct ReductionOpModel<MaxOp>;
template struct ReductionOpModel<MinOp>;

//===----------------------------------------------------------------------===//
// Named Full Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> NamedFullOpModel<OpTy>::getOpConstraints(
    mlir::tt::ttnn::ShapeAttr shape,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    std::optional<mlir::tt::ttnn::Layout> layout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  std::optional<::tt::tt_metal::DataType> metalDtype = std::nullopt;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }
  std::optional<::ttnn::Layout> metalLayout = std::nullopt;
  if (layout.has_value()) {
    metalLayout = conversion::getPageLayout(layout.value());
  }
  std::optional<::ttnn::MemoryConfig> metalMemoryConfig = std::nullopt;
  if (outputLayout) {
    metalMemoryConfig =
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout));
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  auto namedFullOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device,
        conversion::getShape(shape.getShape()), metalDtype, metalLayout,
        deviceRef, metalMemoryConfig);
  };
  return operation::getOpConstraints(shape.getContext(), namedFullOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for NamedFullOpModel.
template struct NamedFullOpModel<ZerosOp>;
template struct NamedFullOpModel<OnesOp>;

//===----------------------------------------------------------------------===//
// SoftmaxOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::SoftmaxOpT buildSoftmaxOpTFromMLIR(
    int32_t dimension, bool numericStable,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::SoftmaxOpT softmaxOp;
  softmaxOp.dimension = dimension;
  softmaxOp.numeric_stable = numericStable;
  softmaxOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  softmaxOp.out = detail::getOutputTensorRefT(outputLayout);
  return softmaxOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<SoftmaxOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::SoftmaxOpT softmaxOpNative = buildSoftmaxOpTFromMLIR(
      dimArg, numericStable, computeKernelConfig, outputLayout);

  auto softmaxOpQuery = [=]() {
    ttnn_op_invoke::SoftmaxOpResult result = ttnn_op_invoke::callSoftmax(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, softmaxOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected SoftmaxOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), softmaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SoftmaxOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::SoftmaxOpT softmaxOpNative = buildSoftmaxOpTFromMLIR(
      dimArg, numericStable, computeKernelConfig, outputLayout);

  auto softmaxOpQuery = [=]() {
    ttnn_op_invoke::SoftmaxOpResult result =
        ttnn_op_invoke::callSoftmax(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                    softmaxOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected SoftmaxOp runtime query to return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(softmaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ScatterOpT
buildScatterOpTFromMLIR(int32_t dim, mlir::tt::ttcore::ReduceType reduceType,
                        TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ScatterOpT scatterOp;
  scatterOp.dim = dim;
  scatterOp.scatter_reduce_type = toNative(reduceType);
  scatterOp.out = detail::getOutputTensorRefT(outputLayout);
  return scatterOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ScatterOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout,
    llvm::ArrayRef<int64_t> sourceShape, TTNNLayoutAttr sourceLayout,
    int32_t dim, mlir::tt::ttcore::ReduceType reduceType,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indexSpec,
      detail::convertToTensorSpec(device, indexShape, indexLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec sourceSpec,
      detail::convertToTensorSpec(device, sourceShape, sourceLayout));

  ::tt::target::ttnn::ScatterOpT scatterOpNative =
      buildScatterOpTFromMLIR(dim, reduceType, outputLayout);

  auto scatterOpQuery = [=]() {
    ttnn_op_invoke::ScatterOpResult result = ttnn_op_invoke::callScatter(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, scatterOpNative,
        inputSpec, indexSpec, sourceSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ScatterOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), scatterOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScatterOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout,
    llvm::ArrayRef<int64_t> sourceShape, TTNNLayoutAttr sourceLayout,
    int32_t dim, mlir::tt::ttcore::ReduceType reduceType,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indexSpec,
      detail::convertToTensorSpec(device, indexShape, indexLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec sourceSpec,
      detail::convertToTensorSpec(device, sourceShape, sourceLayout));

  ::tt::target::ttnn::ScatterOpT scatterOpNative =
      buildScatterOpTFromMLIR(dim, reduceType, outputLayout);

  auto scatterOpQuery = [=]() {
    ttnn_op_invoke::ScatterOpResult result = ttnn_op_invoke::callScatter(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, scatterOpNative, inputSpec,
        indexSpec, sourceSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from ScatterOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(scatterOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ReshapeOpT
buildReshapeOpTFromMLIR(llvm::ArrayRef<int64_t> outputShape,
                        TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ReshapeOpT reshapeOp;
  reshapeOp.shape = {outputShape.begin(), outputShape.end()};
  reshapeOp.out = detail::getOutputTensorRefT(outputLayout);
  return reshapeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ReshapeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReshapeOpT reshapeOpNative =
      buildReshapeOpTFromMLIR(outputShape, outputLayout);

  auto reshapeOpQuery = [=]() {
    ttnn_op_invoke::ReshapeOpResult result = ttnn_op_invoke::callReshape(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, reshapeOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ReshapeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), reshapeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ReshapeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReshapeOpT reshapeOpNative =
      buildReshapeOpTFromMLIR(outputShape, outputLayout);

  auto reshapeOpQuery = [=]() {
    ttnn_op_invoke::ReshapeOpResult result =
        ttnn_op_invoke::callReshape(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                    reshapeOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from ReshapeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(reshapeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SliceStaticOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SliceStaticOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
    llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // convert arrays
  ::ttsl::SmallVector<int> beginsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(begins);
  ::ttsl::SmallVector<int> endsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(ends);
  ::ttsl::SmallVector<int> stepVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(step);

  ttsl::Span<const int> beginsSpan = ::ttsl::make_const_span(beginsVec);
  ttsl::Span<const int> endsSpan = ::ttsl::make_const_span(endsVec);
  ttsl::Span<const int> stepSpan = ::ttsl::make_const_span(stepVec);

  // Create query closure
  auto sliceOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::slice, device, inputSpec, beginsSpan,
                                endsSpan, stepSpan,
                                detail::getNullableMemoryConfig(outputLayout),
                                std::nullopt, std::nullopt);
  };

  return operation::getOpConstraints(inputLayout.getContext(), sliceOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SliceStaticOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
    llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Convert arrays
  ::ttsl::SmallVector<int> beginsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(begins);
  ::ttsl::SmallVector<int> endsVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(ends);
  ::ttsl::SmallVector<int> stepVec =
      conversion::convertLLVMSmallVecToTTNNSmallVec(step);

  ttsl::Span<const int> beginsSpan = ::ttsl::make_const_span(beginsVec);
  ttsl::Span<const int> endsSpan = ::ttsl::make_const_span(endsVec);
  ttsl::Span<const int> stepSpan = ::ttsl::make_const_span(stepVec);

  // Create query closure
  auto sliceOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::slice, device, inputSpec, beginsSpan,
                            endsSpan, stepSpan,
                            detail::getNullableMemoryConfig(outputLayout),
                            std::nullopt, std::nullopt);
  };

  return operation::getOpRuntime(sliceOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SliceDynamicOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<SliceDynamicOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> beginsShape, TTNNLayoutAttr beginsLayout,
    llvm::ArrayRef<int64_t> endsShape, TTNNLayoutAttr endsLayout,
    std::optional<llvm::SmallVector<int64_t>> step,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // It is not possible to use the dynamic version of slice in tt-metal since
  // the validity of the op depends on the actual data that is stored in the
  // begins/ends tensors (which is not available at compile time). Therefore,
  // here we approximate the op by using the static version and calling
  // (possibly) the worst case scenario for the static version which is slicing
  // from the beginning to one index before the end (Capturing the entire tensor
  // except for one row results in the highest memory usage). Note that this is
  // a fairly accurate approximation since the dynamic version in metal also
  // converts the three tensors (begins, ends, step) to vectors and then calls
  // the static version.
  ::ttsl::SmallVector<int> stepVec(inputShape.size(), 1);
  ::ttsl::SmallVector<int> beginsVec(inputShape.size(), 0);
  ::ttsl::SmallVector<int> endsVec(inputShape.begin(), inputShape.end());
  std::ranges::for_each(endsVec, [](int &end) { end = end - 1; });

  // Default values in tt-metal:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<float> padValue = std::nullopt;
  // Create query closure to make a call to the static version of the op:
  auto sliceOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::slice, device, inputSpec, beginsVec, endsVec, stepVec,
        detail::getNullableMemoryConfig(outputLayout), outputSpec, padValue);
  };
  return operation::getOpConstraints(inputLayout.getContext(), sliceOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SliceDynamicOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> beginsShape, TTNNLayoutAttr beginsLayout,
    llvm::ArrayRef<int64_t> endsShape, TTNNLayoutAttr endsLayout,
    std::optional<llvm::SmallVector<int64_t>> step,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // It is not possible to use the dynamic version of slice in tt-metal since
  // the validity of the op depends on the actual data that is stored in the
  // begins/ends tensors (which is not available at compile time). Therefore,
  // here we approximate the op by using the static version and calling
  // (possibly) the worst case scenario for the static version which is slicing
  // from the beginning to the end with a step of 2 (Capturing all possible
  // stripes of data from the input tensor is the most run time intensive
  // pattern).
  // Note that this is a fairly accurate approximation since the dynamic version
  // in metal also converts the three tensors (begins, ends, step) to vectors
  // and then calls the static version.
  ::ttsl::SmallVector<int> stepVec(inputShape.size(), 2);
  ::ttsl::SmallVector<int> beginsVec(inputShape.size(), 0);
  ::ttsl::SmallVector<int> endsVec(inputShape.begin(), inputShape.end());
  // Default values in tt-metal:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<float> padValue = std::nullopt;

  // Create query closure to make a call to the static version of the op:
  auto sliceOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::slice, device, inputSpec, beginsVec, endsVec, stepVec,
        detail::getNullableMemoryConfig(outputLayout), outputSpec, padValue);
  };

  return operation::getOpRuntime(sliceOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<BitcastConvertOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto bitcastOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::bitcast, device, inputSpec,
                                conversion::getDataType(dtype.getValue()),
                                detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), bitcastOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<BitcastConvertOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto bitcastOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::bitcast, device, inputSpec,
                            conversion::getDataType(dtype.getValue()),
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(bitcastOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TypecastOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<TypecastOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto typecastOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::typecast, device, inputSpec,
                                conversion::getDataType(dtype.getValue()),
                                detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(), typecastOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TypecastOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto typecastOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::typecast, device, inputSpec,
                            conversion::getDataType(dtype.getValue()),
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(typecastOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ToLayoutOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::to_layout, device, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        detail::getNullableMemoryConfig(outputLayout));
  };
  return operation::getOpConstraints(inputLayout.getContext(), toLayoutOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ToLayoutOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype) {
    dtype = conversion::getDataType(outputDtype.value());
  } else {
    dtype = std::nullopt;
  }

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::to_layout, device, inputSpec,
                            conversion::getPageLayout(outputLayout.getLayout()),
                            dtype,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(toLayoutOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ToMemoryConfigOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<ToMemoryConfigOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto toMemoryConfigOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::to_memory_config, device, inputSpec,
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout)));
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     toMemoryConfigOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ToMemoryConfigOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                        TTNNLayoutAttr inputLayout,
                                        TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto toMemoryConfigOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::to_memory_config, device, inputSpec,
                            conversion::getMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(toMemoryConfigOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ConcatOpT
buildConcatOpTFromMLIR(int32_t dim, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ConcatOpT concatOp;
  concatOp.dim = dim;
  concatOp.out = detail::getOutputTensorRefT(outputLayout);
  return concatOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ConcatOp>::getOpConstraints(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  assert(inputShapes.size() == inputLayouts.size());
  size_t numInputs = inputShapes.size();

  std::vector<::ttnn::TensorSpec> inputSpecs;
  for (size_t i = 0; i < numInputs; ++i) {
    ASSIGN_OR_RETURN(
        auto _push_tmp,
        detail::convertToTensorSpec(device, inputShapes[i], inputLayouts[i]));
    inputSpecs.push_back(std::move(_push_tmp));
  }

  ::tt::target::ttnn::ConcatOpT concatOpNative =
      buildConcatOpTFromMLIR(dim, outputLayout);

  std::vector<ttnn_op_invoke::TensorArg> inputArgs(inputSpecs.begin(),
                                                   inputSpecs.end());

  auto concatOpQuery = [=]() {
    ttnn_op_invoke::ConcatOpResult result = ttnn_op_invoke::callConcat(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, concatOpNative,
        inputArgs, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ConcatOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayouts[0].getContext(),
                                     concatOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ConcatOp>::getOpRuntime(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  assert(inputShapes.size() == inputLayouts.size());
  size_t numInputs = inputShapes.size();

  std::vector<::ttnn::TensorSpec> inputSpecs;
  for (size_t i = 0; i < numInputs; ++i) {
    ASSIGN_OR_RETURN(
        auto _push_tmp,
        detail::convertToTensorSpec(device, inputShapes[i], inputLayouts[i]));
    inputSpecs.push_back(std::move(_push_tmp));
  }

  ::tt::target::ttnn::ConcatOpT concatOpNative =
      buildConcatOpTFromMLIR(dim, outputLayout);

  std::vector<ttnn_op_invoke::TensorArg> inputArgs(inputSpecs.begin(),
                                                   inputSpecs.end());

  auto concatOpQuery = [=]() {
    ttnn_op_invoke::ConcatOpResult result =
        ttnn_op_invoke::callConcat(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                   concatOpNative, inputArgs, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from ConcatOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(concatOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::TransposeOpT
buildTransposeOpTFromMLIR(int32_t dim0, int32_t dim1,
                          TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::TransposeOpT transposeOp;
  transposeOp.dim0 = dim0;
  transposeOp.dim1 = dim1;
  transposeOp.out = detail::getOutputTensorRefT(outputLayout);
  return transposeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<TransposeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dim0, const int dim1, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::TransposeOpT transposeOpNative =
      buildTransposeOpTFromMLIR(dim0, dim1, outputLayout);

  auto transposeOpQuery = [=]() {
    ttnn_op_invoke::TransposeOpResult result = ttnn_op_invoke::callTranspose(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, transposeOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from TransposeOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     transposeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TransposeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dim0, const int dim1, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::TransposeOpT transposeOpNative =
      buildTransposeOpTFromMLIR(dim0, dim1, outputLayout);

  auto transposeOpQuery = [=]() {
    ttnn_op_invoke::TransposeOpResult result = ttnn_op_invoke::callTranspose(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, transposeOpNative,
        inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from TransposeOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(transposeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::CumSumOpT
buildCumSumOpTFromMLIR(int32_t dim, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::CumSumOpT cumSumOp;
  cumSumOp.dim = dim;
  cumSumOp.out = detail::getOutputTensorRefT(outputLayout);
  return cumSumOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<CumSumOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int32_t dim, std::optional<ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::CumSumOpT cumSumOpNative =
      buildCumSumOpTFromMLIR(dim, outputLayout);

  auto cumSumOpQuery = [=]() {
    ttnn_op_invoke::CumSumOpResult result = ttnn_op_invoke::callCumSum(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, cumSumOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from CumSumOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), cumSumOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<CumSumOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                TTNNLayoutAttr inputLayout, const int32_t dim,
                                std::optional<ttcore::DataType> dtype,
                                TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::CumSumOpT cumSumOpNative =
      buildCumSumOpTFromMLIR(dim, outputLayout);

  auto cumSumOpQuery = [=]() {
    ttnn_op_invoke::CumSumOpResult result =
        ttnn_op_invoke::callCumSum(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                   cumSumOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from CumSumOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(cumSumOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ConcatenateHeadsOpT
buildConcatenateHeadsOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ConcatenateHeadsOpT concatenateHeadsOp;
  concatenateHeadsOp.out = detail::getOutputTensorRefT(outputLayout);
  return concatenateHeadsOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ConcatenateHeadsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ConcatenateHeadsOpT concatenateHeadsOpNative =
      buildConcatenateHeadsOpTFromMLIR(outputLayout);

  auto concatenateHeadsOpQuery = [=]() {
    ttnn_op_invoke::ConcatenateHeadsOpResult result =
        ttnn_op_invoke::callConcatenateHeads(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            concatenateHeadsOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConcatenateHeadsOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     concatenateHeadsOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ConcatenateHeadsOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                          TTNNLayoutAttr inputLayout,
                                          TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ConcatenateHeadsOpT concatenateHeadsOpNative =
      buildConcatenateHeadsOpTFromMLIR(outputLayout);

  auto concatenateHeadsOpQuery = [=]() {
    ttnn_op_invoke::ConcatenateHeadsOpResult result =
        ttnn_op_invoke::callConcatenateHeads(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            concatenateHeadsOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected ConcatenateHeadsOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(concatenateHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT
buildScaledDotProductAttentionDecodeOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT
      scaledDotProductAttentionDecodeOp;
  scaledDotProductAttentionDecodeOp.is_causal = isCausal;
  if (scale.has_value()) {
    scaledDotProductAttentionDecodeOp.scale = scale->convertToFloat();
  }
  if (programConfig.has_value() && *programConfig) {
    scaledDotProductAttentionDecodeOp.program_config =
        std::make_unique<::tt::target::ttnn::SDPAConfigT>(
            toNative(*programConfig));
  }
  scaledDotProductAttentionDecodeOp.out =
      detail::getOutputTensorRefT(outputLayout);
  return scaledDotProductAttentionDecodeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT
      scaledDotProductAttentionDecodeOpNative =
          buildScaledDotProductAttentionDecodeOpTFromMLIR(
              isCausal, scale, programConfig, outputLayout);

  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    ttnn_op_invoke::ScaledDotProductAttentionDecodeOpResult result =
        ttnn_op_invoke::callScaledDotProductAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            scaledDotProductAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec, attentionMaskSpec, curPosTensorSpec, attentionSinkSpec,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ScaledDotProductAttentionDecodeOp constraints query to "
           "return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(queryLayout.getContext(),
                                     scaledDotProductAttentionDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScaledDotProductAttentionDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::ScaledDotProductAttentionDecodeOpT
      scaledDotProductAttentionDecodeOpNative =
          buildScaledDotProductAttentionDecodeOpTFromMLIR(
              isCausal, scale, programConfig, outputLayout);

  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    ttnn_op_invoke::ScaledDotProductAttentionDecodeOpResult result =
        ttnn_op_invoke::callScaledDotProductAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            scaledDotProductAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec, attentionMaskSpec, curPosTensorSpec, attentionSinkSpec,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected ScaledDotProductAttentionDecodeOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(scaledDotProductAttentionDecodeOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
buildPagedScaledDotProductAttentionDecodeOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
      pagedScaledDotProductAttentionDecodeOp;
  pagedScaledDotProductAttentionDecodeOp.is_causal = isCausal;
  if (scale.has_value()) {
    pagedScaledDotProductAttentionDecodeOp.scale =
        scale.value().convertToFloat();
  }
  if (slidingWindowSize.has_value()) {
    pagedScaledDotProductAttentionDecodeOp.sliding_window_size =
        *slidingWindowSize;
  }
  if (programConfig.has_value() && *programConfig) {
    pagedScaledDotProductAttentionDecodeOp.program_config =
        std::make_unique<::tt::target::ttnn::SDPAConfigT>(
            toNative(*programConfig));
  }
  pagedScaledDotProductAttentionDecodeOp.out =
      detail::getOutputTensorRefT(outputLayout);
  return pagedScaledDotProductAttentionDecodeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<PagedScaledDotProductAttentionDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
      pagedScaledDotProductAttentionDecodeOpNative =
          buildPagedScaledDotProductAttentionDecodeOpTFromMLIR(
              isCausal, scale, slidingWindowSize, programConfig, outputLayout);

  auto pagedScaledDotProductAttentionDecodeOpQuery = [=]() {
    ttnn_op_invoke::PagedScaledDotProductAttentionDecodeOpResult result =
        ttnn_op_invoke::callPagedScaledDotProductAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            pagedScaledDotProductAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec, pageTableSpec,
            attentionMaskSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionMaskSpec)
                : std::nullopt,
            curPosTensorSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*curPosTensorSpec)
                : std::nullopt,
            attentionSinkSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionSinkSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PagedScaledDotProductAttentionDecodeOp constraints query "
           "to return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(
      queryLayout.getContext(), pagedScaledDotProductAttentionDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<PagedScaledDotProductAttentionDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize,
    std::optional<SDPAProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);

  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
      pagedScaledDotProductAttentionDecodeOpNative =
          buildPagedScaledDotProductAttentionDecodeOpTFromMLIR(
              isCausal, scale, slidingWindowSize, programConfig, outputLayout);

  auto pagedScaledDotProductAttentionDecodeOpQuery = [=]() {
    ttnn_op_invoke::PagedScaledDotProductAttentionDecodeOpResult result =
        ttnn_op_invoke::callPagedScaledDotProductAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            pagedScaledDotProductAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec, pageTableSpec,
            attentionMaskSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionMaskSpec)
                : std::nullopt,
            curPosTensorSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*curPosTensorSpec)
                : std::nullopt,
            attentionSinkSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionSinkSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected PagedScaledDotProductAttentionDecodeOp runtime query to "
        "return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(pagedScaledDotProductAttentionDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedFlashMultiLatentAttentionDecodeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
buildPagedFlashMultiLatentAttentionDecodeOpTFromMLIR(
    uint32_t headDimV, bool isCausal, std::optional<llvm::APFloat> scale,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
      pagedFlashMultiLatentAttentionDecodeOp;
  pagedFlashMultiLatentAttentionDecodeOp.head_dim_v = headDimV;
  pagedFlashMultiLatentAttentionDecodeOp.is_causal = isCausal;
  if (scale.has_value()) {
    pagedFlashMultiLatentAttentionDecodeOp.scale =
        scale.value().convertToFloat();
  }
  pagedFlashMultiLatentAttentionDecodeOp.out =
      detail::getOutputTensorRefT(outputLayout);
  return pagedFlashMultiLatentAttentionDecodeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<PagedFlashMultiLatentAttentionDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    std::optional<llvm::ArrayRef<int64_t>> valueShape,
    std::optional<TTNNLayoutAttr> valueLayout, uint32_t headDimV,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> valueSpec =
      detail::convertToOptionalTensorSpec(device, valueShape, valueLayout);
  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
      pagedFlashMultiLatentAttentionDecodeOpNative =
          buildPagedFlashMultiLatentAttentionDecodeOpTFromMLIR(
              headDimV, isCausal, scale, outputLayout);

  auto pagedFlashMlaDecodeOpQuery = [=]() {
    ttnn_op_invoke::PagedFlashMultiLatentAttentionDecodeOpResult result =
        ttnn_op_invoke::callPagedFlashMultiLatentAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            pagedFlashMultiLatentAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*valueSpec)
                : std::nullopt,
            pageTableSpec,
            attentionMaskSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionMaskSpec)
                : std::nullopt,
            curPosTensorSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*curPosTensorSpec)
                : std::nullopt,
            attentionSinkSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionSinkSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PagedFlashMultiLatentAttentionDecodeOp constraints query "
           "to return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(queryLayout.getContext(),
                                     pagedFlashMlaDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<PagedFlashMultiLatentAttentionDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    std::optional<llvm::ArrayRef<int64_t>> valueShape,
    std::optional<TTNNLayoutAttr> valueLayout, uint32_t headDimV,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    bool isCausal, std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> curPosTensorShape,
    std::optional<TTNNLayoutAttr> curPosTensorLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout,
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> valueSpec =
      detail::convertToOptionalTensorSpec(device, valueShape, valueLayout);
  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> curPosTensorSpec =
      detail::convertToOptionalTensorSpec(device, curPosTensorShape,
                                          curPosTensorLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::PagedFlashMultiLatentAttentionDecodeOpT
      pagedFlashMultiLatentAttentionDecodeOpNative =
          buildPagedFlashMultiLatentAttentionDecodeOpTFromMLIR(
              headDimV, isCausal, scale, outputLayout);

  auto pagedFlashMlaDecodeOpQuery = [=]() {
    ttnn_op_invoke::PagedFlashMultiLatentAttentionDecodeOpResult result =
        ttnn_op_invoke::callPagedFlashMultiLatentAttentionDecode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            pagedFlashMultiLatentAttentionDecodeOpNative, querySpec, keySpec,
            valueSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*valueSpec)
                : std::nullopt,
            pageTableSpec,
            attentionMaskSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionMaskSpec)
                : std::nullopt,
            curPosTensorSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*curPosTensorSpec)
                : std::nullopt,
            attentionSinkSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*attentionSinkSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected PagedFlashMultiLatentAttentionDecodeOp runtime query to "
        "return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(pagedFlashMlaDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ScaledDotProductAttentionOpT
buildScaledDotProductAttentionOpTFromMLIR(
    bool isCausal, std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ScaledDotProductAttentionOpT scaledDotProductAttentionOp;
  scaledDotProductAttentionOp.is_causal = isCausal;
  if (scale.has_value()) {
    scaledDotProductAttentionOp.scale = scale->convertToFloat();
  }
  if (slidingWindowSize.has_value()) {
    scaledDotProductAttentionOp.sliding_window_size = *slidingWindowSize;
  }
  scaledDotProductAttentionOp.out = detail::getOutputTensorRefT(outputLayout);
  return scaledDotProductAttentionOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout, bool isCausal,
    std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::ScaledDotProductAttentionOpT
      scaledDotProductAttentionOpNative =
          buildScaledDotProductAttentionOpTFromMLIR(
              isCausal, scale, slidingWindowSize, outputLayout);

  auto scaledDotProductAttentionOpQuery = [=]() {
    ttnn_op_invoke::ScaledDotProductAttentionOpResult result =
        ttnn_op_invoke::callScaledDotProductAttention(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            scaledDotProductAttentionOpNative, querySpec, keySpec, valueSpec,
            attentionMaskSpec, attentionSinkSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ScaledDotProductAttentionOp constraints query to "
           "return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(queryLayout.getContext(),
                                     scaledDotProductAttentionOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScaledDotProductAttentionOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout, bool isCausal,
    std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valueSpec,
      detail::convertToTensorSpec(device, valueShape, valueLayout));

  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);
  std::optional<::ttnn::TensorSpec> attentionSinkSpec =
      detail::convertToOptionalTensorSpec(device, attentionSinkShape,
                                          attentionSinkLayout);

  ::tt::target::ttnn::ScaledDotProductAttentionOpT
      scaledDotProductAttentionOpNative =
          buildScaledDotProductAttentionOpTFromMLIR(
              isCausal, scale, slidingWindowSize, outputLayout);

  auto scaledDotProductAttentionOpQuery = [=]() {
    ttnn_op_invoke::ScaledDotProductAttentionOpResult result =
        ttnn_op_invoke::callScaledDotProductAttention(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            scaledDotProductAttentionOpNative, querySpec, keySpec, valueSpec,
            attentionMaskSpec, attentionSinkSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected ScaledDotProductAttentionOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(scaledDotProductAttentionOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===-----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp
// ===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RotaryEmbeddingLlamaOpT
buildRotaryEmbeddingLlamaOpTFromMLIR(
    bool isDecodeMode,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT rotaryEmbeddingLlamaOp;
  rotaryEmbeddingLlamaOp.is_decode_mode = isDecodeMode;
  rotaryEmbeddingLlamaOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  rotaryEmbeddingLlamaOp.out = detail::getOutputTensorRefT(outputLayout);
  return rotaryEmbeddingLlamaOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RotaryEmbeddingLlamaOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    llvm::ArrayRef<int64_t> transMatShape, TTNNLayoutAttr transMatLayout,
    bool isDecodeMode,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec cosSpec,
                   detail::convertToTensorSpec(device, cosShape, cosLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec sinSpec,
                   detail::convertToTensorSpec(device, sinShape, sinLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec transMatSpec,
      detail::convertToTensorSpec(device, transMatShape, transMatLayout));

  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT rotaryEmbeddingLlamaOpNative =
      buildRotaryEmbeddingLlamaOpTFromMLIR(
          isDecodeMode, deviceComputeKernelConfig, outputLayout);

  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    ttnn_op_invoke::RotaryEmbeddingLlamaOpResult result =
        ttnn_op_invoke::callRotaryEmbeddingLlama(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            rotaryEmbeddingLlamaOpNative, inputSpec, cosSpec, sinSpec,
            transMatSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected RotaryEmbeddingLlamaOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     rotaryEmbeddingLlamaOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RotaryEmbeddingLlamaOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    llvm::ArrayRef<int64_t> transMatShape, TTNNLayoutAttr transMatLayout,
    bool isDecodeMode,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec cosSpec,
                   detail::convertToTensorSpec(device, cosShape, cosLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec sinSpec,
                   detail::convertToTensorSpec(device, sinShape, sinLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec transMatSpec,
      detail::convertToTensorSpec(device, transMatShape, transMatLayout));

  ::tt::target::ttnn::RotaryEmbeddingLlamaOpT rotaryEmbeddingLlamaOpNative =
      buildRotaryEmbeddingLlamaOpTFromMLIR(
          isDecodeMode, deviceComputeKernelConfig, outputLayout);

  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    ttnn_op_invoke::RotaryEmbeddingLlamaOpResult result =
        ttnn_op_invoke::callRotaryEmbeddingLlama(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            rotaryEmbeddingLlamaOpNative, inputSpec, cosSpec, sinSpec,
            transMatSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RotaryEmbeddingLlamaOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(rotaryEmbeddingLlamaOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RotaryEmbeddingOpT buildRotaryEmbeddingOpTFromMLIR(
    std::optional<uint32_t> tokenIndex,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RotaryEmbeddingOpT rotaryEmbeddingOp;
  if (tokenIndex.has_value()) {
    rotaryEmbeddingOp.token_index = *tokenIndex;
  }
  rotaryEmbeddingOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  rotaryEmbeddingOp.out = detail::getOutputTensorRefT(outputLayout);
  return rotaryEmbeddingOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RotaryEmbeddingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    std::optional<uint32_t> tokenIndex,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec cosSpec,
                   detail::convertToTensorSpec(device, cosShape, cosLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec sinSpec,
                   detail::convertToTensorSpec(device, sinShape, sinLayout));

  ::tt::target::ttnn::RotaryEmbeddingOpT rotaryEmbeddingOpNative =
      buildRotaryEmbeddingOpTFromMLIR(tokenIndex, deviceComputeKernelConfig,
                                      outputLayout);

  auto rotaryEmbeddingOpQuery = [=]() {
    ttnn_op_invoke::RotaryEmbeddingOpResult result =
        ttnn_op_invoke::callRotaryEmbedding(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            rotaryEmbeddingOpNative, inputSpec, cosSpec, sinSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected RotaryEmbeddingOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     rotaryEmbeddingOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RotaryEmbeddingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    std::optional<uint32_t> tokenIndex,
    std::optional<::mlir::tt::ttnn::DeviceComputeKernelConfigAttr>
        deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec cosSpec,
                   detail::convertToTensorSpec(device, cosShape, cosLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec sinSpec,
                   detail::convertToTensorSpec(device, sinShape, sinLayout));

  ::tt::target::ttnn::RotaryEmbeddingOpT rotaryEmbeddingOpNative =
      buildRotaryEmbeddingOpTFromMLIR(tokenIndex, deviceComputeKernelConfig,
                                      outputLayout);

  auto rotaryEmbeddingOpQuery = [=]() {
    ttnn_op_invoke::RotaryEmbeddingOpResult result =
        ttnn_op_invoke::callRotaryEmbedding(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, rotaryEmbeddingOpNative,
            inputSpec, cosSpec, sinSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RotaryEmbeddingOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(rotaryEmbeddingOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPCreateQKVHeadsDecodeOp
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT
buildNLPCreateQKVHeadsDecodeOpTFromMLIR(uint32_t numHeads,
                                        std::optional<uint32_t> numKVHeads,
                                        std::optional<bool> overlapQKCoregrid,
                                        std::optional<uint32_t> sliceSize,
                                        TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT nlpCreateQkvHeadsDecodeOp;
  nlpCreateQkvHeadsDecodeOp.num_heads = numHeads;
  if (numKVHeads.has_value()) {
    nlpCreateQkvHeadsDecodeOp.num_kv_heads = *numKVHeads;
  }
  if (overlapQKCoregrid.has_value()) {
    nlpCreateQkvHeadsDecodeOp.overlap_qk_coregrid = *overlapQKCoregrid;
  }
  if (sliceSize.has_value()) {
    nlpCreateQkvHeadsDecodeOp.slice_size = *sliceSize;
  }
  auto memory_config = detail::getNullableMemoryConfigT(outputLayout);
  if (memory_config.has_value()) {
    nlpCreateQkvHeadsDecodeOp.memcfg =
        std::make_unique<::tt::target::ttnn::MemoryConfigT>(
            memory_config.value());
    if (nlpCreateQkvHeadsDecodeOp.memcfg) {
      llvm::WithColor::warning()
          << "Memory config should be set to nullptr to match runtime";
    }
  }
  return nlpCreateQkvHeadsDecodeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<op_model::OpConstraints>
OpModel<NLPCreateQKVHeadsDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
    std::optional<TTNNLayoutAttr> batchOffsetLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, std::optional<bool> overlapQKCoregrid,
    std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::ttnn::TensorSpec> batchOffsetSpec = std::nullopt;
  if (batchOffsetShape && batchOffsetLayout) {
    ASSIGN_OR_RETURN(batchOffsetSpec, detail::convertToTensorSpec(
                                          device, batchOffsetShape.value(),
                                          batchOffsetLayout.value()));
  }
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT
      nlpCreateQkvHeadsDecodeOpNative = buildNLPCreateQKVHeadsDecodeOpTFromMLIR(
          numHeads, numKVHeads, overlapQKCoregrid, sliceSize, outputLayout);

  auto nlpCreateQKVHeadsDecode = [=]() {
    ttnn_op_invoke::NLPCreateQKVHeadsDecodeOpResult result =
        ttnn_op_invoke::callNLPCreateQKVHeadsDecode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            nlpCreateQkvHeadsDecodeOpNative, inputSpec,
            batchOffsetSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*batchOffsetSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected NLPCreateQKVHeadsDecodeOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     nlpCreateQKVHeadsDecode);

#else
  return OpConstraints{};
#endif
}

llvm::Expected<size_t> OpModel<NLPCreateQKVHeadsDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
    std::optional<TTNNLayoutAttr> batchOffsetLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, std::optional<bool> overlapQKCoregrid,
    std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::ttnn::TensorSpec> batchOffsetSpec = std::nullopt;
  if (batchOffsetShape && batchOffsetLayout) {
    ASSIGN_OR_RETURN(batchOffsetSpec, detail::convertToTensorSpec(
                                          device, batchOffsetShape.value(),
                                          batchOffsetLayout.value()));
  }
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT
      nlpCreateQkvHeadsDecodeOpNative = buildNLPCreateQKVHeadsDecodeOpTFromMLIR(
          numHeads, numKVHeads, overlapQKCoregrid, sliceSize, outputLayout);

  auto nlpCreateQKVHeadsDecode = [=]() {
    ttnn_op_invoke::NLPCreateQKVHeadsDecodeOpResult result =
        ttnn_op_invoke::callNLPCreateQKVHeadsDecode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            nlpCreateQkvHeadsDecodeOpNative, inputSpec,
            batchOffsetSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*batchOffsetSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected NLPCreateQKVHeadsDecodeOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(nlpCreateQKVHeadsDecode);
#else
  return llvm::createStringError("Not implemented");
#endif
}

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
buildSplitQueryKeyValueAndSplitHeadsOpTFromMLIR(
    uint32_t numHeads, std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
      splitQueryKeyValueAndSplitHeadsOp;
  splitQueryKeyValueAndSplitHeadsOp.num_heads = numHeads;
  if (numKVHeads.has_value()) {
    splitQueryKeyValueAndSplitHeadsOp.num_kv_heads = *numKVHeads;
  }
  splitQueryKeyValueAndSplitHeadsOp.transpose_key = transposeKey;
  splitQueryKeyValueAndSplitHeadsOp.q_out =
      detail::getOutputTensorRefT(outputLayout);
  return splitQueryKeyValueAndSplitHeadsOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<SplitQueryKeyValueAndSplitHeadsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
    std::optional<TTNNLayoutAttr> inputKVLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> inputKVSpec = std::nullopt;
  if (inputKVShape && inputKVLayout) {
    ASSIGN_OR_RETURN(inputKVSpec,
                     detail::convertToTensorSpec(device, inputKVShape.value(),
                                                 inputKVLayout.value()));
  }

  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
      splitQueryKeyValueAndSplitHeadsOpNative =
          buildSplitQueryKeyValueAndSplitHeadsOpTFromMLIR(
              numHeads, numKVHeads, transposeKey, outputLayout);

  auto splitQueryKeyValueAndSplitHeadsOpQuery = [=]() {
    ttnn_op_invoke::SplitQueryKeyValueAndSplitHeadsOpResult result =
        ttnn_op_invoke::callSplitQueryKeyValueAndSplitHeads(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            splitQueryKeyValueAndSplitHeadsOpNative, inputSpec, inputKVSpec,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected SplitQueryKeyValueAndSplitHeadsOp constraints query "
           "to return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     splitQueryKeyValueAndSplitHeadsOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SplitQueryKeyValueAndSplitHeadsOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
    std::optional<TTNNLayoutAttr> inputKVLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> inputKVSpec = std::nullopt;
  if (inputKVShape && inputKVLayout) {
    ASSIGN_OR_RETURN(inputKVSpec,
                     detail::convertToTensorSpec(device, inputKVShape.value(),
                                                 inputKVLayout.value()));
  }

  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
      splitQueryKeyValueAndSplitHeadsOpNative =
          buildSplitQueryKeyValueAndSplitHeadsOpTFromMLIR(
              numHeads, numKVHeads, transposeKey, outputLayout);

  auto splitQueryKeyValueAndSplitHeadsOpQuery = [=]() {
    ttnn_op_invoke::SplitQueryKeyValueAndSplitHeadsOpResult result =
        ttnn_op_invoke::callSplitQueryKeyValueAndSplitHeads(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            splitQueryKeyValueAndSplitHeadsOpNative, inputSpec, inputKVSpec,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected SplitQueryKeyValueAndSplitHeadsOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(splitQueryKeyValueAndSplitHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::NLPConcatHeadsOpT
buildNLPConcatHeadsOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::NLPConcatHeadsOpT nlpConcatHeadsOp;
  nlpConcatHeadsOp.out = detail::getOutputTensorRefT(outputLayout);
  return nlpConcatHeadsOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<NLPConcatHeadsOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPConcatHeadsOpT nlpConcatHeadsOpNative =
      buildNLPConcatHeadsOpTFromMLIR(outputLayout);

  auto nlpConcatHeadsOpQuery = [=]() {
    ttnn_op_invoke::NLPConcatHeadsOpResult result =
        ttnn_op_invoke::callNLPConcatHeads(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            nlpConcatHeadsOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected NLPConcatHeadsOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     nlpConcatHeadsOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<NLPConcatHeadsOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                        TTNNLayoutAttr inputLayout,
                                        TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPConcatHeadsOpT nlpConcatHeadsOpNative =
      buildNLPConcatHeadsOpTFromMLIR(outputLayout);

  auto nlpConcatHeadsOpQuery = [=]() {
    ttnn_op_invoke::NLPConcatHeadsOpResult result =
        ttnn_op_invoke::callNLPConcatHeads(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, nlpConcatHeadsOpNative,
            inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected NLPConcatHeadsOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(nlpConcatHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::NLPConcatHeadsDecodeOpT
buildNLPConcatHeadsDecodeOpTFromMLIR(uint32_t numHeads,
                                     TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT nlpConcatHeadsDecodeOp;
  nlpConcatHeadsDecodeOp.num_heads = numHeads;
  nlpConcatHeadsDecodeOp.out = detail::getOutputTensorRefT(outputLayout);
  return nlpConcatHeadsDecodeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<NLPConcatHeadsDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t numHeads, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT nlpConcatHeadsDecodeOpNative =
      buildNLPConcatHeadsDecodeOpTFromMLIR(numHeads, outputLayout);

  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    ttnn_op_invoke::NLPConcatHeadsDecodeOpResult result =
        ttnn_op_invoke::callNLPConcatHeadsDecode(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            nlpConcatHeadsDecodeOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected NLPConcatHeadsDecodeOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     nlpConcatHeadsDecodeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<NLPConcatHeadsDecodeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t numHeads, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT nlpConcatHeadsDecodeOpNative =
      buildNLPConcatHeadsDecodeOpTFromMLIR(numHeads, outputLayout);

  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    ttnn_op_invoke::NLPConcatHeadsDecodeOpResult result =
        ttnn_op_invoke::callNLPConcatHeadsDecode(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            nlpConcatHeadsDecodeOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected NLPConcatHeadsDecodeOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(nlpConcatHeadsDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RepeatInterleaveOpT
buildRepeatInterleaveOpTFromMLIR(const unsigned int repeats, const int dim,
                                 TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RepeatInterleaveOpT repeatInterleaveOp;
  repeatInterleaveOp.repeats = repeats;
  repeatInterleaveOp.dim = dim;
  repeatInterleaveOp.out = detail::getOutputTensorRefT(outputLayout);
  return repeatInterleaveOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RepeatInterleaveOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const unsigned int repeats, const int dim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::RepeatInterleaveOpT repeatInterleaveOpNative =
      buildRepeatInterleaveOpTFromMLIR(repeats, dim, outputLayout);

  auto repeatInterleaveOpQuery = [=]() {
    ttnn_op_invoke::RepeatInterleaveOpResult result =
        ttnn_op_invoke::callRepeatInterleave(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            repeatInterleaveOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from RepeatInterleaveOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     repeatInterleaveOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RepeatInterleaveOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const unsigned int repeats, const int dim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::RepeatInterleaveOpT repeatInterleaveOpNative =
      buildRepeatInterleaveOpTFromMLIR(repeats, dim, outputLayout);

  auto repeatInterleaveOpQuery = [=]() {
    ttnn_op_invoke::RepeatInterleaveOpResult result =
        ttnn_op_invoke::callRepeatInterleave(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            repeatInterleaveOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from RepeatInterleaveOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(repeatInterleaveOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RepeatOpT
buildRepeatOpTFromMLIR(llvm::ArrayRef<int64_t> repeatDims,
                       TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RepeatOpT repeatOp;
  repeatOp.repeat_dims = {repeatDims.begin(), repeatDims.end()};
  repeatOp.out = detail::getOutputTensorRefT(outputLayout);
  return repeatOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RepeatOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> repeats, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::RepeatOpT repeatOpNative =
      buildRepeatOpTFromMLIR(repeats, outputLayout);

  auto repeatOpQuery = [=]() {
    ttnn_op_invoke::RepeatOpResult result = ttnn_op_invoke::callRepeat(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, repeatOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from RepeatOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), repeatOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RepeatOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> repeats, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::RepeatOpT repeatOpNative =
      buildRepeatOpTFromMLIR(repeats, outputLayout);

  auto repeatOpQuery = [=]() {
    ttnn_op_invoke::RepeatOpResult result =
        ttnn_op_invoke::callRepeat(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                   repeatOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from RepeatOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(repeatOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PadOpT buildPadOpTFromMLIR(llvm::ArrayRef<int32_t> padding,
                                               llvm::APFloat padValue,
                                               bool useMulticore,
                                               TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PadOpT padOp;
  for (int32_t p : padding) {
    padOp.padding.push_back(static_cast<uint32_t>(p));
  }
  padOp.value = padValue.convertToFloat();
  padOp.use_multicore = useMulticore;
  padOp.out = detail::getOutputTensorRefT(outputLayout);
  return padOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PadOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue, bool multicore,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::PadOpT padOpNative =
      buildPadOpTFromMLIR(padding, padValue, multicore, outputLayout);

  auto padOpQuery = [=]() {
    ttnn_op_invoke::PadOpResult result =
        ttnn_op_invoke::callPad(ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
                                padOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from PadOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), padOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PadOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue, bool multicore,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::PadOpT padOpNative =
      buildPadOpTFromMLIR(padding, padValue, multicore, outputLayout);

  auto padOpQuery = [=]() {
    ttnn_op_invoke::PadOpResult result =
        ttnn_op_invoke::callPad(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                padOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from PadOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(padOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::SortOpT buildSortOpTFromMLIR(int dim, bool descending,
                                                 bool stable,
                                                 TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::SortOpT sortOp;
  sortOp.dim = static_cast<int8_t>(dim);
  sortOp.descending = descending;
  sortOp.stable = stable;
  sortOp.outputs.push_back(detail::getOutputTensorRefT(outputLayout));
  return sortOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<SortOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int dim,
    bool descending, bool stable, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::SortOpT sortOpNative =
      buildSortOpTFromMLIR(dim, descending, stable, outputLayout);

  auto sortOpQuery = [=]() {
    ttnn_op_invoke::SortOpResult result =
        ttnn_op_invoke::callSort(ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
                                 sortOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from SortOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), sortOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SortOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int dim,
    bool descending, bool stable, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::SortOpT sortOpNative =
      buildSortOpTFromMLIR(dim, descending, stable, outputLayout);

  auto sortOpQuery = [=]() {
    ttnn_op_invoke::SortOpResult result =
        ttnn_op_invoke::callSort(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                 sortOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from SortOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(sortOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TopKRouterGptOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::TopKRouterGptOpT
buildTopKRouterGptOpTFromMLIR(uint32_t k, uint32_t numExperts,
                              TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::TopKRouterGptOpT topKOp;
  topKOp.k = static_cast<int32_t>(k);
  topKOp.num_experts = static_cast<int32_t>(numExperts);
  topKOp.expert_indices = detail::getOutputTensorRefT(outputLayout);
  topKOp.expert_weights = detail::getOutputTensorRefT(outputLayout);
  return topKOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<TopKRouterGptOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout, uint32_t k,
    uint32_t numExperts, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec biasSpec,
                   detail::convertToTensorSpec(device, biasShape, biasLayout));

  ::tt::target::ttnn::TopKRouterGptOpT topKOpNative =
      buildTopKRouterGptOpTFromMLIR(k, numExperts, outputLayout);

  auto topKRouterGptQuery = [=]() {
    ttnn_op_invoke::TopKRouterGptOpResult result =
        ttnn_op_invoke::callTopKRouterGpt(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, topKOpNative,
            inputSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from TopKRouterGptOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     topKRouterGptQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TopKRouterGptOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout, uint32_t k,
    uint32_t numExperts, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec biasSpec,
                   detail::convertToTensorSpec(device, biasShape, biasLayout));

  ::tt::target::ttnn::TopKRouterGptOpT topKOpNative =
      buildTopKRouterGptOpTFromMLIR(k, numExperts, outputLayout);

  auto topKRouterGptQuery = [=]() {
    ttnn_op_invoke::TopKRouterGptOpResult result =
        ttnn_op_invoke::callTopKRouterGpt(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, topKOpNative, inputSpec,
            weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from TopKRouterGptOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(topKRouterGptQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ReductionArgMaxOpT
buildArgMaxOpTFromMLIR(std::optional<int32_t> dim, bool keepDim,
                       bool useMulticore, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ReductionArgMaxOpT argMaxOp;
  if (dim.has_value()) {
    argMaxOp.dim = *dim;
  }
  argMaxOp.keep_dim = keepDim;
  argMaxOp.use_multicore = useMulticore;
  argMaxOp.out = detail::getOutputTensorRefT(outputLayout);
  return argMaxOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ArgMaxOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int32_t> dim, bool keepDim, bool multicore,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReductionArgMaxOpT argMaxOpNative =
      buildArgMaxOpTFromMLIR(dim, keepDim, multicore, outputLayout);

  auto argMaxOpQuery = [=]() {
    ttnn_op_invoke::ArgMaxOpResult result = ttnn_op_invoke::callArgMax(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, argMaxOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ArgMaxOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), argMaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ArgMaxOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                TTNNLayoutAttr inputLayout,
                                std::optional<int32_t> dim, bool keepDim,
                                bool multicore, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReductionArgMaxOpT argMaxOpNative =
      buildArgMaxOpTFromMLIR(dim, keepDim, multicore, outputLayout);

  auto argMaxOpQuery = [=]() {
    ttnn_op_invoke::ArgMaxOpResult result =
        ttnn_op_invoke::callArgMax(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                   argMaxOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from ArgMaxOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(argMaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ProdOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ReductionProdOpT
buildProdOpTFromMLIR(std::optional<int64_t> dimArg, bool keepDim,
                     TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ReductionProdOpT prodOp;
  if (dimArg.has_value()) {
    prodOp.dim_arg = *dimArg;
  }
  prodOp.keep_dim = keepDim;
  prodOp.out = detail::getOutputTensorRefT(outputLayout);
  return prodOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ProdOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int64_t> dim, bool keepDim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ReductionProdOpT prodOpNative =
      buildProdOpTFromMLIR(dim, keepDim, outputLayout);

  auto prodOpQuery = [=]() {
    ttnn_op_invoke::ProdOpResult result =
        ttnn_op_invoke::callProd(ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
                                 prodOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from ProdOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), prodOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Quantization Ops
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
template <typename OpTy>
::tt::target::ttnn::EltwiseQuantizationOpT
buildEltwiseQuantizationOpTFromMLIR(std::optional<int32_t> axis,
                                    std::optional<ttcore::DataType> outputDtype,
                                    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOp;

  if constexpr (std::is_same_v<OpTy, QuantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Quantize;
  } else if constexpr (std::is_same_v<OpTy, DequantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Dequantize;
  } else if constexpr (std::is_same_v<OpTy, RequantizeOp>) {
    eltwiseQuantizationOp.type =
        ::tt::target::ttnn::EltwiseQuantizationOpType::Requantize;
  } else {
    static_assert(ttmlir::utils::always_false(), "Unsupported OpTy");
  }

  eltwiseQuantizationOp.axis =
      axis.has_value() ? ::flatbuffers::Optional<int32_t>(axis.value())
                       : ::flatbuffers::nullopt;

  eltwiseQuantizationOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseQuantizationOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

template <typename OpTy>
llvm::Expected<OpConstraints> QuantizationOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> scaleShape, TTNNLayoutAttr scaleLayout,
    llvm::ArrayRef<int64_t> zeroPointShape, TTNNLayoutAttr zeroPointLayout,
    std::optional<int32_t> axis, std::optional<ttcore::DataType> outputDtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec scaleSpec,
      detail::convertToTensorSpec(device, scaleShape, scaleLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec zeroPointSpec,
      detail::convertToTensorSpec(device, zeroPointShape, zeroPointLayout));

  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOpNative =
      buildEltwiseQuantizationOpTFromMLIR<OpTy>(axis, outputDtype,
                                                outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    ttnn_op_invoke::EltwiseQuantizationOpResult result =
        ttnn_op_invoke::callEltwiseQuantizeDequantize(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseQuantizationOpNative, inputSpec, scaleSpec, zeroPointSpec,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
            result) &&
        "Expected ConstraintQueryResponse from callEltwiseQuantizeDequantize");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     quantizationOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> QuantizationOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> scaleShape, TTNNLayoutAttr scaleLayout,
    llvm::ArrayRef<int64_t> zeroPointShape, TTNNLayoutAttr zeroPointLayout,
    std::optional<int32_t> axis, std::optional<ttcore::DataType> outputDtype,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec scaleSpec,
      detail::convertToTensorSpec(device, scaleShape, scaleLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec zeroPointSpec,
      detail::convertToTensorSpec(device, zeroPointShape, zeroPointLayout));

  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOpNative =
      buildEltwiseQuantizationOpTFromMLIR<OpTy>(axis, outputDtype,
                                                outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    ttnn_op_invoke::EltwiseQuantizationOpResult result =
        ttnn_op_invoke::callEltwiseQuantizeDequantize(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseQuantizationOpNative, inputSpec, scaleSpec, zeroPointSpec,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from callEltwiseQuantizeDequantize");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(quantizationOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

#define INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(OP)                             \
  template ::tt::target::ttnn::EltwiseQuantizationOpT                          \
      buildEltwiseQuantizationOpTFromMLIR<OP>(std::optional<int32_t>,          \
                                              std::optional<ttcore::DataType>, \
                                              TTNNLayoutAttr);

// Explicit template instantiation for QuantizationOpModel.
template struct QuantizationOpModel<QuantizeOp>;
template struct QuantizationOpModel<DequantizeOp>;

#ifdef TTMLIR_ENABLE_OPMODEL
INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(QuantizeOp);
INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(DequantizeOp);
INSTANTIATE_BUILD_ELTWISE_QUANTIZATION(RequantizeOp);
#endif // TTMLIR_ENABLE_OPMODEL

//===----------------------------------------------------------------------===//
// RequantizeOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RequantizeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> inScaleShape, TTNNLayoutAttr inScaleLayout,
    llvm::ArrayRef<int64_t> inZeroPointShape, TTNNLayoutAttr inZeroPointLayout,
    llvm::ArrayRef<int64_t> outScaleShape, TTNNLayoutAttr outScaleLayout,
    llvm::ArrayRef<int64_t> outZeroPointShape,
    TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inScaleSpec,
      detail::convertToTensorSpec(device, inScaleShape, inScaleLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inZeroPointSpec,
      detail::convertToTensorSpec(device, inZeroPointShape, inZeroPointLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec outScaleSpec,
      detail::convertToTensorSpec(device, outScaleShape, outScaleLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec outZeroPointSpec,
                   detail::convertToTensorSpec(device, outZeroPointShape,
                                               outZeroPointLayout));

  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOpNative =
      buildEltwiseQuantizationOpTFromMLIR<RequantizeOp>(axis, outputDtype,
                                                        outputLayout);

  // Create query closure
  auto requantizeOpQuery = [=]() {
    ttnn_op_invoke::EltwiseQuantizationOpResult result =
        ttnn_op_invoke::callEltwiseRequantize(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseQuantizationOpNative, inputSpec, inScaleSpec,
            inZeroPointSpec, outScaleSpec, outZeroPointSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from callEltwiseRequantize");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     requantizeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RequantizeOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> inScaleShape, TTNNLayoutAttr inScaleLayout,
    llvm::ArrayRef<int64_t> inZeroPointShape, TTNNLayoutAttr inZeroPointLayout,
    llvm::ArrayRef<int64_t> outScaleShape, TTNNLayoutAttr outScaleLayout,
    llvm::ArrayRef<int64_t> outZeroPointShape,
    TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inScaleSpec,
      detail::convertToTensorSpec(device, inScaleShape, inScaleLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inZeroPointSpec,
      detail::convertToTensorSpec(device, inZeroPointShape, inZeroPointLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec outScaleSpec,
      detail::convertToTensorSpec(device, outScaleShape, outScaleLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec outZeroPointSpec,
                   detail::convertToTensorSpec(device, outZeroPointShape,
                                               outZeroPointLayout));

  ::tt::target::ttnn::EltwiseQuantizationOpT eltwiseQuantizationOpNative =
      buildEltwiseQuantizationOpTFromMLIR<RequantizeOp>(axis, outputDtype,
                                                        outputLayout);

  // Create query closure
  auto requantizeOpQuery = [=]() {
    ttnn_op_invoke::EltwiseQuantizationOpResult result =
        ttnn_op_invoke::callEltwiseRequantize(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseQuantizationOpNative, inputSpec, inScaleSpec,
            inZeroPointSpec, outScaleSpec, outZeroPointSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from callEltwiseRequantize");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(requantizeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::LinearOpT buildLinearOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {

  ::tt::target::ttnn::LinearOpT linearOp;

  linearOp.transpose_a = transposeA;
  linearOp.transpose_b = transposeB;

  if (activation) {
    linearOp.activation = activation->str();
  }

  if (programConfigAttr.has_value()) {
    mlir::TypeSwitch<mlir::Attribute>(*programConfigAttr)
        .Case<MatmulMultiCoreReuseProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastProgramConfigAttr,
              MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            [&](auto config) {
              linearOp.matmul_program_config.Set(toNative(config));
            });
  }
  linearOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;

  linearOp.out = detail::getOutputTensorRefT(outputLayout);

  return linearOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<LinearOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  ::tt::target::ttnn::LinearOpT linearOpNative = buildLinearOpTFromMLIR(
      transposeA, transposeB, activation, programConfigAttr,
      computeKernelConfig, outputLayout);

  // Create query closure
  auto linearOpQuery = [=]() {
    ttnn_op_invoke::LinearOpResult result = ttnn_op_invoke::callLinear(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, linearOpNative,
        inputSpecA, inputSpecB,
        biasTensor.has_value() ? std::make_optional(&biasTensor.value())
                               : std::nullopt,
        device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from LinearOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), linearOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LinearOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::Tensor> biasTensor;
  if (biasShape && biasLayout) {
    ::ttnn::TensorSpec biasSpec =
        conversion::getTensorSpec(biasShape.value(), biasLayout.value());
    biasTensor = ::tt::tt_metal::create_device_tensor(biasSpec, device);
  }

  ::tt::target::ttnn::LinearOpT linearOpNative = buildLinearOpTFromMLIR(
      transposeA, transposeB, activation, programConfigAttr,
      computeKernelConfig, outputLayout);

  // Create query closure
  auto linearOpQuery = [=]() {
    ttnn_op_invoke::LinearOpResult result = ttnn_op_invoke::callLinear(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, linearOpNative, inputSpecA,
        inputSpecB,
        biasTensor.has_value() ? std::make_optional(&biasTensor.value())
                               : std::nullopt,
        device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from LinearOp query");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(linearOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::MatmulOpT buildMatmulOpTFromMLIR(
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {

  ::tt::target::ttnn::MatmulOpT matmulOp;

  matmulOp.transpose_a = transposeA;
  matmulOp.transpose_b = transposeB;

  if (activation) {
    matmulOp.activation = activation->str();
  }

  if (programConfigAttr.has_value()) {
    mlir::TypeSwitch<mlir::Attribute>(*programConfigAttr)
        .Case<MatmulMultiCoreReuseProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastProgramConfigAttr,
              MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
              MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
            [&](auto config) {
              matmulOp.matmul_program_config.Set(toNative(config));
            });
  }
  matmulOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;

  matmulOp.out = detail::getOutputTensorRefT(outputLayout);

  return matmulOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<MatmulOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB,
    std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::MatmulOpT matmulOpNative = buildMatmulOpTFromMLIR(
      transposeA, transposeB, activation, programConfigAttr,
      computeKernelConfig, outputLayout);

  // Create query closure
  auto matmulOpQuery = [=]() {
    ttnn_op_invoke::MatmulOpResult result = ttnn_op_invoke::callMatmul(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, matmulOpNative,
        inputSpecA, inputSpecB, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from MatmulOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayoutA.getContext(), matmulOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MatmulOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB,
    std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  ::tt::target::ttnn::MatmulOpT matmulOpNative = buildMatmulOpTFromMLIR(
      transposeA, transposeB, activation, programConfigAttr,
      computeKernelConfig, outputLayout);

  // Create query closure
  auto matmulOpQuery = [=]() {
    ttnn_op_invoke::MatmulOpResult result = ttnn_op_invoke::callMatmul(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, matmulOpNative, inputSpecA,
        inputSpecB, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from MatmulOp query");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(matmulOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// DeallocateOp
//===----------------------------------------------------------------------===//

llvm::Expected<size_t>
OpModel<DeallocateOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                    TTNNLayoutAttr inputLayout, bool force) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto deallocateOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::deallocate, device, inputSpec, force);
  };

  return operation::getOpRuntime(deallocateOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// FillCacheOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::FillCacheOpT
buildFillCacheOpTFromMLIR(uint32_t batchOffset) {
  ::tt::target::ttnn::FillCacheOpT fillCacheOp;
  fillCacheOp.batch_offset = batchOffset;
  return fillCacheOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<FillCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::FillCacheOpT fillCacheOpNative =
      buildFillCacheOpTFromMLIR(batchOffset);

  auto fillCacheOpQuery = [=]() {
    ttnn_op_invoke::FillCacheOpResult result = ttnn_op_invoke::callFillCache(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, fillCacheOpNative,
        cacheSpec, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from FillCacheOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(cacheLayout.getContext(),
                                     fillCacheOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<FillCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::FillCacheOpT fillCacheOpNative =
      buildFillCacheOpTFromMLIR(batchOffset);

  auto fillCacheOpQuery = [=]() {
    ttnn_op_invoke::FillCacheOpResult result = ttnn_op_invoke::callFillCache(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, fillCacheOpNative,
        cacheSpec, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from FillCacheOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(fillCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// UpdateCacheOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<UpdateCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // TODO(#1510): modify the ttnn::update_cache to take a tensor for
  // updateIndex.
  // UpdateIndex is stored as a tensor in mlir, but the ttnn::update_cache
  // expects a scalar uint32_t. So we need to extract the scalar value from the
  // tensor which is not possible in compile time (as opposed to the workaround
  // that is implemented in runtime code in PR 1437). So we use a default value
  // of 0.
  if (updateIndexLayout.getDataType() != ttcore::DataType::UInt32) {
    return llvm::createStringError("UpdateIndex must be of type UInt32");
  }

  uint32_t updateIdx = 0; // Default to first position
  (void)updateIndexShape;
  (void)updateIndexLayout;

  auto updateCacheOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::update_cache, device, cacheSpec,
                                inputSpec, updateIdx, batchOffset,
                                /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraints(cacheLayout.getContext(),
                                     updateCacheOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<UpdateCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // TODO(#1510): modify the ttnn::update_cache to take a tensor for
  // updateIndex.
  uint32_t updateIdx = 0; // Default to first position
  (void)updateIndexShape;
  (void)updateIndexLayout;

  auto updateCacheOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::update_cache, device, cacheSpec, inputSpec,
                            updateIdx, batchOffset,
                            /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(updateCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedUpdateCacheOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PagedUpdateCacheOpT
buildPagedUpdateCacheOpTFromMLIR(bool shareCache) {
  ::tt::target::ttnn::PagedUpdateCacheOpT op;
  op.share_cache = shareCache;
  return op;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PagedUpdateCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    std::optional<llvm::ArrayRef<int64_t>> pageTableShape,
    std::optional<TTNNLayoutAttr> pageTableLayout, bool shareCache,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec updateIndexSpec,
      detail::convertToTensorSpec(device, updateIndexShape, updateIndexLayout));

  std::optional<::ttnn::TensorSpec> pageTableSpec;
  if (pageTableShape && pageTableLayout) {
    ASSIGN_OR_RETURN(
        pageTableSpec,
        detail::convertToTensorSpec(device, *pageTableShape, *pageTableLayout));
  }

  ::tt::target::ttnn::PagedUpdateCacheOpT pagedUpdateCacheOpNative =
      buildPagedUpdateCacheOpTFromMLIR(shareCache);

  auto pagedUpdateCacheOpQuery = [=]() {
    ttnn_op_invoke::PagedUpdateCacheOpResult result =
        ttnn_op_invoke::callPagedUpdateCache(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            pagedUpdateCacheOpNative, cacheSpec, inputSpec,
            std::optional<ttnn_op_invoke::TensorArg>(updateIndexSpec),
            pageTableSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*pageTableSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from PagedUpdateCacheOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(cacheLayout.getContext(),
                                     pagedUpdateCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PagedUpdateCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    std::optional<llvm::ArrayRef<int64_t>> pageTableShape,
    std::optional<TTNNLayoutAttr> pageTableLayout, bool shareCache,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec updateIndexSpec,
      detail::convertToTensorSpec(device, updateIndexShape, updateIndexLayout));

  std::optional<::ttnn::TensorSpec> pageTableSpec;
  if (pageTableShape && pageTableLayout) {
    ASSIGN_OR_RETURN(
        pageTableSpec,
        detail::convertToTensorSpec(device, *pageTableShape, *pageTableLayout));
  }

  ::tt::target::ttnn::PagedUpdateCacheOpT pagedUpdateCacheOpNative =
      buildPagedUpdateCacheOpTFromMLIR(shareCache);

  std::optional<ttnn_op_invoke::TensorArg> pageTableArg;
  if (pageTableSpec) {
    pageTableArg = *pageTableSpec;
  }

  auto pagedUpdateCacheOpQuery = [=]() {
    ttnn_op_invoke::PagedUpdateCacheOpResult result =
        ttnn_op_invoke::callPagedUpdateCache(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            pagedUpdateCacheOpNative, cacheSpec, inputSpec,
            std::optional<ttnn_op_invoke::TensorArg>(updateIndexSpec),
            pageTableSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*pageTableSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from PagedUpdateCacheOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(pagedUpdateCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedFillCacheOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PagedFillCacheOpT buildPagedFillCacheOpTFromMLIR() {
  ::tt::target::ttnn::PagedFillCacheOpT pagedFillCacheOp;

  return pagedFillCacheOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PagedFillCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchIdxShape,
    std::optional<TTNNLayoutAttr> batchIdxLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> batchIdxSpec;
  if (batchIdxShape && batchIdxLayout) {
    ASSIGN_OR_RETURN(
        batchIdxSpec,
        detail::convertToTensorSpec(device, *batchIdxShape, *batchIdxLayout));
  }

  ::tt::target::ttnn::PagedFillCacheOpT pagedFillCacheOpNative =
      buildPagedFillCacheOpTFromMLIR();

  auto pagedFillCacheOpQuery = [=]() {
    ttnn_op_invoke::PagedFillCacheOpResult result =
        ttnn_op_invoke::callPagedFillCache(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            pagedFillCacheOpNative, cacheSpec, inputSpec, pageTableSpec,
            batchIdxSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*batchIdxSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from PagedFillCacheOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(cacheLayout.getContext(),
                                     pagedFillCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PagedFillCacheOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchIdxShape,
    std::optional<TTNNLayoutAttr> batchIdxLayout, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec pageTableSpec,
      detail::convertToTensorSpec(device, pageTableShape, pageTableLayout));

  std::optional<::ttnn::TensorSpec> batchIdxSpec;
  if (batchIdxShape && batchIdxLayout) {
    ASSIGN_OR_RETURN(
        batchIdxSpec,
        detail::convertToTensorSpec(device, *batchIdxShape, *batchIdxLayout));
  }

  ::tt::target::ttnn::PagedFillCacheOpT pagedFillCacheOpNative =
      buildPagedFillCacheOpTFromMLIR();

  auto pagedFillCacheOpQuery = [=]() {
    ttnn_op_invoke::PagedFillCacheOpResult result =
        ttnn_op_invoke::callPagedFillCache(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, pagedFillCacheOpNative,
            cacheSpec, inputSpec, pageTableSpec,
            batchIdxSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*batchIdxSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from PagedFillCacheOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(pagedFillCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::Conv2dOpT buildConv2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::Conv2dOpT conv2dOp;
  conv2dOp.in_channels = in_channels;
  conv2dOp.out_channels = out_channels;
  conv2dOp.batch_size = batch_size;
  conv2dOp.input_height = input_height;
  conv2dOp.input_width = input_width;
  conv2dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  conv2dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&conv2dOp](const auto &arr) {
        conv2dOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  conv2dOp.dilation = std::vector<int32_t>(dilation.begin(), dilation.end());
  conv2dOp.groups = groups;
  conv2dOp.out = detail::getOutputTensorRefT(outputLayout);
  conv2dOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  conv2dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  conv2dOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;

  return conv2dOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<Conv2dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig, conv2dSliceConfig,
          biasLayout.has_value(), /*transpose*/ false);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig, /*transpose*/ false);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::Conv2dOpT conv2dOpNative = buildConv2dOpTFromMLIR(
      in_channels, out_channels, batch_size, input_height, input_width,
      kernel_size, stride, padding, dilation, groups, conv2dConfig,
      deviceComputeKernelConfig, conv2dSliceConfig, outputLayout);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    ttnn_op_invoke::Conv2dOpResult result = ttnn_op_invoke::callConv2d(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, conv2dOpNative,
        inputSpec, weightSpec,
        biasSpec.has_value()
            ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
            : std::nullopt,
        device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected Conv2dOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), conv2dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<Conv2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig, conv2dSliceConfig,
          biasLayout.has_value(), /*transpose*/ false);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }

  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig, /*transpose*/ false);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::Conv2dOpT conv2dOpNative = buildConv2dOpTFromMLIR(
      in_channels, out_channels, batch_size, input_height, input_width,
      kernel_size, stride, padding, dilation, groups, conv2dConfig,
      deviceComputeKernelConfig, conv2dSliceConfig, outputLayout);

  // Create query closure
  auto conv2dOpQuery = [=]() {
    ttnn_op_invoke::Conv2dOpResult result = ttnn_op_invoke::callConv2d(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, conv2dOpNative, inputSpec,
        weightSpec,
        biasSpec.has_value()
            ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
            : std::nullopt,
        device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected Conv2dOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(conv2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv3dOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::Conv3dOpT buildConv3dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_depth, uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::StringRef padding_mode,
    uint32_t groups, std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::Conv3dOpT conv3dOp;
  conv3dOp.in_channels = in_channels;
  conv3dOp.out_channels = out_channels;
  conv3dOp.batch_size = batch_size;
  conv3dOp.input_depth = input_depth;
  conv3dOp.input_height = input_height;
  conv3dOp.input_width = input_width;
  conv3dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  conv3dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  conv3dOp.padding = std::vector<int32_t>(padding.begin(), padding.end());
  conv3dOp.padding_mode = padding_mode.str();
  conv3dOp.groups = groups;
  conv3dOp.out = detail::getOutputTensorRefT(outputLayout);
  if (outputDtype.has_value() && outputDtype.value()) {
    conv3dOp.output_dtype = toNative(outputDtype.value().getValue());
  }
  conv3dOp.conv3d_config =
      (conv3dConfig.has_value() && *conv3dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv3dConfigT>(
                toNative(*conv3dConfig))
          : nullptr;
  conv3dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  auto outputMemoryConfigT = detail::getNullableMemoryConfigT(outputLayout);
  conv3dOp.memory_config =
      outputMemoryConfigT.has_value()
          ? std::make_unique<::tt::target::ttnn::MemoryConfigT>(
                *outputMemoryConfigT)
          : nullptr;

  return conv3dOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<Conv3dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_depth,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, uint32_t groups,
    llvm::StringRef padding_mode,
    std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    ASSIGN_OR_RETURN(
        biasSpec, detail::convertToTensorSpec(device, *biasShape, *biasLayout));
  }

  ::tt::target::ttnn::Conv3dOpT conv3dOpNative = buildConv3dOpTFromMLIR(
      in_channels, out_channels, batch_size, input_depth, input_height,
      input_width, kernel_size, stride, padding, padding_mode, groups,
      outputDtype, conv3dConfig, deviceComputeKernelConfig, outputLayout);

  auto conv3dOpQuery = [=]() {
    ttnn_op_invoke::Conv3dOpResult result = ttnn_op_invoke::callConv3d(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, conv3dOpNative,
        inputSpec, weightSpec,
        biasSpec.has_value()
            ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
            : std::nullopt,
        device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected Conv3dOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), conv3dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<Conv3dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_depth,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, uint32_t groups,
    llvm::StringRef padding_mode,
    std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    ASSIGN_OR_RETURN(
        biasSpec, detail::convertToTensorSpec(device, *biasShape, *biasLayout));
  }

  ::tt::target::ttnn::Conv3dOpT conv3dOpNative = buildConv3dOpTFromMLIR(
      in_channels, out_channels, batch_size, input_depth, input_height,
      input_width, kernel_size, stride, padding, padding_mode, groups,
      outputDtype, conv3dConfig, deviceComputeKernelConfig, outputLayout);

  auto conv3dOpRuntime = [=]() {
    ttnn_op_invoke::Conv3dOpResult result = ttnn_op_invoke::callConv3d(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, conv3dOpNative, inputSpec,
        weightSpec,
        biasSpec.has_value()
            ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
            : std::nullopt,
        device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected Conv3dOp runtime query to return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(conv3dOpRuntime);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::ConvTranspose2dOpT buildConvTranspose2dOpTFromMLIR(
    uint32_t in_channels, uint32_t out_channels, uint32_t batch_size,
    uint32_t input_height, uint32_t input_width,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::ConvTranspose2dOpT convTranspose2dOp;
  convTranspose2dOp.in_channels = in_channels;
  convTranspose2dOp.out_channels = out_channels;
  convTranspose2dOp.batch_size = batch_size;
  convTranspose2dOp.input_height = input_height;
  convTranspose2dOp.input_width = input_width;
  convTranspose2dOp.kernel_size =
      std::vector<int32_t>(kernel_size.begin(), kernel_size.end());
  convTranspose2dOp.stride = std::vector<int32_t>(stride.begin(), stride.end());
  convTranspose2dOp.padding =
      std::vector<int32_t>(padding.begin(), padding.end());
  convTranspose2dOp.output_padding =
      std::vector<int32_t>(output_padding.begin(), output_padding.end());
  convTranspose2dOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  convTranspose2dOp.groups = groups;
  convTranspose2dOp.out = detail::getOutputTensorRefT(outputLayout);
  convTranspose2dOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  convTranspose2dOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  convTranspose2dOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;

  return convTranspose2dOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ConvTranspose2dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> output_padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig, conv2dSliceConfig,
          biasLayout.has_value(), /*transpose*/ true);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig, /*transpose*/ true);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ConvTranspose2dOpT convTranspose2dOpNative =
      buildConvTranspose2dOpTFromMLIR(in_channels, out_channels, batch_size,
                                      input_height, input_width, kernel_size,
                                      stride, padding, output_padding, dilation,
                                      groups, conv2dConfig, conv2dSliceConfig,
                                      deviceComputeKernelConfig, outputLayout);

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    ttnn_op_invoke::ConvTranspose2dOpResult result =
        ttnn_op_invoke::callConvTranspose2d(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            convTranspose2dOpNative, inputSpec, weightSpec,
            biasSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
                : std::nullopt,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConvTranspose2dOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     convTranspose2dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ConvTranspose2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_height,
    uint32_t input_width, llvm::ArrayRef<int32_t> kernel_size,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> output_padding, llvm::ArrayRef<int32_t> dilation,
    uint32_t groups, std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig, std::nullopt,
          biasLayout.has_value(), /*transpose*/ true);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }

  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  // Prepare bias tensor if present.
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            input_height, input_width, kernel_size, stride, padding, dilation,
            groups, conv2dConfig, /*transpose*/ true);
    if (!preparedBiasExp) {
      return preparedBiasExp.takeError();
    }
    biasSpec = preparedBiasExp.get();
  }

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::ConvTranspose2dOpT convTranspose2dOpNative =
      buildConvTranspose2dOpTFromMLIR(in_channels, out_channels, batch_size,
                                      input_height, input_width, kernel_size,
                                      stride, padding, output_padding, dilation,
                                      groups, conv2dConfig, conv2dSliceConfig,
                                      deviceComputeKernelConfig, outputLayout);

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    ttnn_op_invoke::ConvTranspose2dOpResult result =
        ttnn_op_invoke::callConvTranspose2d(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, convTranspose2dOpNative,
            inputSpec, weightSpec,
            biasSpec.has_value()
                ? std::optional<ttnn_op_invoke::TensorArg>(*biasSpec)
                : std::nullopt,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected ConvTranspose2dOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(convTranspose2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConv2dWeightsOpT
buildPrepareConv2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConv2dWeightsOpT prepareConv2dWeightsOp;
  prepareConv2dWeightsOp.in_channels = inChannels;
  prepareConv2dWeightsOp.out_channels = outChannels;
  prepareConv2dWeightsOp.batch_size = batchSize;
  prepareConv2dWeightsOp.input_height = inputHeight;
  prepareConv2dWeightsOp.input_width = inputWidth;
  prepareConv2dWeightsOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConv2dWeightsOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConv2dWeightsOp](const auto &arr) {
        prepareConv2dWeightsOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConv2dWeightsOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConv2dWeightsOp.has_bias = hasBias;
  prepareConv2dWeightsOp.groups = groups;
  prepareConv2dWeightsOp.weights_format = weightsFormat.str();
  prepareConv2dWeightsOp.input_tensor_layout = toNative(inputTensorLayout);
  prepareConv2dWeightsOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConv2dWeightsOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConv2dWeightsOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConv2dWeightsOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConv2dWeightsOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConv2dWeightsOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConv2dWeightsOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConv2dWeightsOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PrepareConv2dWeightsOp>::getOpConstraints(
    TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> weightShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(weightLayout != nullptr && "Weight layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());

  ::tt::target::ttnn::PrepareConv2dWeightsOpT prepareConv2dWeightsOpNative =
      buildPrepareConv2dWeightsOpTFromMLIR(
          inputMemConfig, inputTensorLayout, weightsFormat, inChannels,
          outChannels, batchSize, inputHeight, inputWidth, kernelSize, stride,
          padding, dilation, hasBias, groups, inputDtype, outputDtype,
          conv2dConfig, deviceComputeKernelConfig, conv2dSliceConfig,
          outputLayout);

  auto prepareConv2dWeightsQuery = [=]() {
    ttnn_op_invoke::PrepareConv2dWeightsOpResult result =
        ttnn_op_invoke::callPrepareConv2dWeights(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            prepareConv2dWeightsOpNative, &weightTensor, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PrepareConv2dWeightsOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(weightLayout.getContext(),
                                     prepareConv2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConv2dBiasOpT buildPrepareConv2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConv2dBiasOpT prepareConv2dBiasOp;
  prepareConv2dBiasOp.in_channels = inChannels;
  prepareConv2dBiasOp.out_channels = outChannels;
  prepareConv2dBiasOp.batch_size = batchSize;
  prepareConv2dBiasOp.input_height = inputHeight;
  prepareConv2dBiasOp.input_width = inputWidth;
  prepareConv2dBiasOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConv2dBiasOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConv2dBiasOp](const auto &arr) {
        prepareConv2dBiasOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConv2dBiasOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConv2dBiasOp.groups = groups;
  prepareConv2dBiasOp.input_tensor_layout = toNative(inputTensorLayout);
  prepareConv2dBiasOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConv2dBiasOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConv2dBiasOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConv2dBiasOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConv2dBiasOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConv2dBiasOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConv2dBiasOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConv2dBiasOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PrepareConv2dBiasOp>::getOpConstraints(
    TTNNLayoutAttr biasLayout, llvm::ArrayRef<int64_t> biasShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(biasLayout != nullptr && "Bias layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());

  ::tt::target::ttnn::PrepareConv2dBiasOpT prepareConv2dBiasOp =
      buildPrepareConv2dBiasOpTFromMLIR(
          inputMemConfig, inputTensorLayout, inChannels, outChannels, batchSize,
          inputHeight, inputWidth, kernelSize, stride, padding, dilation,
          groups, inputDtype, outputDtype, conv2dConfig, conv2dSliceConfig,
          deviceComputeKernelConfig, outputLayout);

  auto prepareConv2dBiasQuery = [=]() {
    ttnn_op_invoke::PrepareConv2dBiasOpResult result =
        ttnn_op_invoke::callPrepareConv2dBias(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, prepareConv2dBiasOp,
            &biasTensor, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PrepareConv2dBiasOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(biasLayout.getContext(),
                                     prepareConv2dBiasQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dWeightsOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
buildPrepareConvTranspose2dWeightsOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
      prepareConvTranspose2dWeightsOp;
  prepareConvTranspose2dWeightsOp.in_channels = inChannels;
  prepareConvTranspose2dWeightsOp.out_channels = outChannels;
  prepareConvTranspose2dWeightsOp.batch_size = batchSize;
  prepareConvTranspose2dWeightsOp.input_height = inputHeight;
  prepareConvTranspose2dWeightsOp.input_width = inputWidth;
  prepareConvTranspose2dWeightsOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConvTranspose2dWeightsOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConvTranspose2dWeightsOp](const auto &arr) {
        prepareConvTranspose2dWeightsOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConvTranspose2dWeightsOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConvTranspose2dWeightsOp.has_bias = hasBias;
  prepareConvTranspose2dWeightsOp.groups = groups;
  prepareConvTranspose2dWeightsOp.weights_format = weightsFormat.str();
  prepareConvTranspose2dWeightsOp.mirror_kernel = mirrorKernel;
  prepareConvTranspose2dWeightsOp.input_tensor_layout =
      toNative(inputTensorLayout);
  prepareConvTranspose2dWeightsOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConvTranspose2dWeightsOp.output_dtype =
        toNative(outputDtype.value());
  }
  prepareConvTranspose2dWeightsOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConvTranspose2dWeightsOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConvTranspose2dWeightsOp.out =
      detail::getOutputTensorRefT(outputLayout);

  return prepareConvTranspose2dWeightsOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<PrepareConvTranspose2dWeightsOp>::getOpConstraints(
    TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> weightShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> dilation,
    bool hasBias, int32_t groups, ttcore::DataType inputDtype,
    std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(weightLayout != nullptr && "Weight layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());

  ::tt::target::ttnn::PrepareConvTranspose2dWeightsOpT
      prepareConvTranspose2dWeightsOpNative =
          buildPrepareConvTranspose2dWeightsOpTFromMLIR(
              inputMemConfig, inputTensorLayout, weightsFormat, inChannels,
              outChannels, batchSize, inputHeight, inputWidth, kernelSize,
              stride, padding, dilation, hasBias, groups, inputDtype,
              outputDtype, conv2dConfig, deviceComputeKernelConfig,
              conv2dSliceConfig, mirrorKernel, outputLayout);

  auto prepareConvTranspose2dWeightsQuery = [=]() {
    ttnn_op_invoke::PrepareConvTranspose2dWeightsOpResult result =
        ttnn_op_invoke::callPrepareConvTranspose2dWeights(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            prepareConvTranspose2dWeightsOpNative, &weightTensor, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PrepareConvTranspose2dWeightsOp constraints query to "
           "return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(weightLayout.getContext(),
                                     prepareConvTranspose2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dBiasOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
buildPrepareConvTranspose2dBiasOpTFromMLIR(
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
      prepareConvTranspose2dBiasOp;
  prepareConvTranspose2dBiasOp.in_channels = inChannels;
  prepareConvTranspose2dBiasOp.out_channels = outChannels;
  prepareConvTranspose2dBiasOp.batch_size = batchSize;
  prepareConvTranspose2dBiasOp.input_height = inputHeight;
  prepareConvTranspose2dBiasOp.input_width = inputWidth;
  prepareConvTranspose2dBiasOp.kernel_size =
      std::vector<int32_t>(kernelSize.begin(), kernelSize.end());
  prepareConvTranspose2dBiasOp.stride =
      std::vector<int32_t>(stride.begin(), stride.end());
  auto reorderedPadding = detail::reorderPool2dPadding(padding);
  std::visit(
      [&prepareConvTranspose2dBiasOp](const auto &arr) {
        prepareConvTranspose2dBiasOp.padding.assign(arr.begin(), arr.end());
      },
      reorderedPadding);
  prepareConvTranspose2dBiasOp.dilation =
      std::vector<int32_t>(dilation.begin(), dilation.end());
  prepareConvTranspose2dBiasOp.groups = groups;
  prepareConvTranspose2dBiasOp.input_tensor_layout =
      toNative(inputTensorLayout);
  prepareConvTranspose2dBiasOp.input_dtype = toNative(inputDtype);
  if (outputDtype.has_value()) {
    prepareConvTranspose2dBiasOp.output_dtype = toNative(outputDtype.value());
  }
  prepareConvTranspose2dBiasOp.input_memory_config =
      std::make_unique<::tt::target::ttnn::MemoryConfigT>(
          toNative(inputMemConfig));
  prepareConvTranspose2dBiasOp.conv2d_config =
      (conv2dConfig.has_value() && *conv2dConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dConfigT>(
                toNative(*conv2dConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.compute_config =
      (deviceComputeKernelConfig.has_value() && *deviceComputeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*deviceComputeKernelConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.conv2d_slice_config =
      (conv2dSliceConfig.has_value() && *conv2dSliceConfig)
          ? std::make_unique<::tt::target::ttnn::Conv2dSliceConfigT>(
                toNative(*conv2dSliceConfig))
          : nullptr;
  prepareConvTranspose2dBiasOp.out = detail::getOutputTensorRefT(outputLayout);

  return prepareConvTranspose2dBiasOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<PrepareConvTranspose2dBiasOp>::getOpConstraints(
    TTNNLayoutAttr biasLayout, llvm::ArrayRef<int64_t> biasShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(biasLayout != nullptr && "Bias layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());

  ::tt::target::ttnn::PrepareConvTranspose2dBiasOpT
      prepareConvTranspose2dBiasOpNative =
          buildPrepareConvTranspose2dBiasOpTFromMLIR(
              inputMemConfig, inputTensorLayout, inChannels, outChannels,
              batchSize, inputHeight, inputWidth, kernelSize, stride, padding,
              dilation, groups, inputDtype, outputDtype, conv2dConfig,
              deviceComputeKernelConfig, conv2dSliceConfig, outputLayout);

  auto prepareConvTranspose2dBiasQuery = [=]() {
    ttnn_op_invoke::PrepareConvTranspose2dBiasOpResult result =
        ttnn_op_invoke::callPrepareConvTranspose2dBias(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            prepareConvTranspose2dBiasOpNative, &biasTensor, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected PrepareConvTranspose2dBiasOp constraints query to "
           "return ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(biasLayout.getContext(),
                                     prepareConvTranspose2dBiasQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MaxPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<MaxPool2dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto maxPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, false /* deallocate_input */,
        reallocateHaloOutput, false /* return_indices */,
        ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
        configTensorsInDram.value_or(false) /* config_tensors_in_dram */);
  };

  return operation::getOpConstraints(inputLayout.getContext(), maxPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MaxPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto maxPool2DQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, false /* deallocate_input */,
        reallocateHaloOutput, false /* return_indices */,
        ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
        configTensorsInDram.value_or(false) /* config_tensors_in_dram */);
  };

  return operation::getOpRuntime(maxPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MaxPool2DWithIndices
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<MaxPool2dWithIndicesOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    bool deallocateInput, bool returnIndices,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  // When return_indices=true, tt-metal requires ROW_MAJOR layout and BFLOAT16
  auto maxPool2DWithIndicesQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, deallocateInput,
        reallocateHaloOutput, returnIndices, ::ttnn::DataType::BFLOAT16,
        ::ttnn::Layout::ROW_MAJOR, configTensorsInDram.value_or(false));
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     maxPool2DWithIndicesQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MaxPool2dWithIndicesOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    bool deallocateInput, bool returnIndices,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  // When return_indices=true, tt-metal requires ROW_MAJOR layout and BFLOAT16
  auto maxPool2DWithIndicesQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::max_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        ceilMode, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, deallocateInput,
        reallocateHaloOutput, returnIndices, ::ttnn::DataType::BFLOAT16,
        ::ttnn::Layout::ROW_MAJOR, configTensorsInDram.value_or(false));
  };

  return operation::getOpRuntime(maxPool2DWithIndicesQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AvgPool2D
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<AvgPool2dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // default values for the variables that are received by the op's invoke
  // method in tt-metal but do not exist in the op's definition in TTNNOps.td:
  bool countIncludePad = true;
  std::optional<int32_t> divisorOverride = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // Create query closure
  auto avgPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::avg_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding), ceilMode, countIncludePad,
        divisorOverride, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, computeKernelConfig,
        false /* deallocate_input */, reallocateHaloOutput,
        ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
        configTensorsInDram.value_or(false) /* config_tensors_in_dram */);
  };

  return operation::getOpConstraints(inputLayout.getContext(), avgPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<AvgPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // convert all signed integers to unsigned integers
  uint32_t batchSizeU = static_cast<uint32_t>(batchSize);
  uint32_t inputHeightU = static_cast<uint32_t>(inputHeight);

  uint32_t inputWidthU = static_cast<uint32_t>(inputWidth);

  uint32_t inputChannelsU = static_cast<uint32_t>(inputChannels);

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // default values for the variables that are received by the op's invoke
  // method in tt-metal but do not exist in the op's definition in TTNNOps.td:
  bool countIncludePad = true;
  std::optional<int32_t> divisorOverride = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // Create query closure
  auto avgPool2DQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::avg_pool2d, device, inputSpec, batchSizeU, inputHeightU,
        inputWidthU, inputChannelsU,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding), ceilMode, countIncludePad,
        divisorOverride, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, computeKernelConfig,
        false /* deallocate_input */, reallocateHaloOutput,
        ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR,
        configTensorsInDram.value_or(false) /* config_tensors_in_dram */);
  };

  return operation::getOpRuntime(avgPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GlobalAvgPool2dOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<GlobalAvgPool2dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::ttnn::DataType outputDType = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    outputDType = conversion::getDataType(dtype.value());
  } else if (outputLayout) {
    outputDType = conversion::getDataType(outputLayout.getDataType());
  }

  uint32_t batchSize = static_cast<uint32_t>(inputShape[0]);
  uint32_t inputHeight = static_cast<uint32_t>(inputShape[1]);
  uint32_t inputWidth = static_cast<uint32_t>(inputShape[2]);
  uint32_t inputChannels = static_cast<uint32_t>(inputShape[3]);
  std::optional<int32_t> divisorOverride = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;
  ::ttnn::Layout outputPageLayout =
      outputLayout ? conversion::getPageLayout(outputLayout)
                   : ::ttnn::Layout::TILE;

  // Create query closure
  auto globalAvgPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::avg_pool2d, device, inputSpec, batchSize, inputHeight,
        inputWidth, inputChannels,
        /*kernel_size=*/std::array<uint32_t, 2>{inputHeight, inputWidth},
        /*stride=*/std::array<uint32_t, 2>{1, 1},
        /*padding=*/std::array<uint32_t, 2>{0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/true, divisorOverride,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, computeKernelConfig,
        false /* deallocate_input */, true /* reallocate_halo_output */,
        outputDType, outputPageLayout, false /* config_tensors_in_dram */);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     globalAvgPool2DQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<GlobalAvgPool2dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::ttnn::DataType outputDType = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    outputDType = conversion::getDataType(dtype.value());
  } else if (outputLayout) {
    outputDType = conversion::getDataType(outputLayout.getDataType());
  }

  uint32_t batchSize = static_cast<uint32_t>(inputShape[0]);
  uint32_t inputHeight = static_cast<uint32_t>(inputShape[1]);
  uint32_t inputWidth = static_cast<uint32_t>(inputShape[2]);
  uint32_t inputChannels = static_cast<uint32_t>(inputShape[3]);
  std::optional<int32_t> divisorOverride = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;
  ::ttnn::Layout outputPageLayout =
      outputLayout ? conversion::getPageLayout(outputLayout)
                   : ::ttnn::Layout::TILE;

  // Create query closure
  auto globalAvgPool2DQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::avg_pool2d, device, inputSpec, batchSize, inputHeight,
        inputWidth, inputChannels,
        /*kernel_size=*/std::array<uint32_t, 2>{inputHeight, inputWidth},
        /*stride=*/std::array<uint32_t, 2>{1, 1},
        /*padding=*/std::array<uint32_t, 2>{0, 0},
        /*ceil_mode=*/false,
        /*count_include_pad=*/true, divisorOverride,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt /* dram_slice_config */,
        std::nullopt /* applied_shard_scheme */, computeKernelConfig,
        false /* deallocate_input */, true /* reallocate_halo_output */,
        outputDType, outputPageLayout, false /* config_tensors_in_dram */);
  };

  return operation::getOpRuntime(globalAvgPool2DQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::BatchNormInferenceOpT buildBatchNormInferenceOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::BatchNormInferenceOpT batchNormInferenceOp;
  batchNormInferenceOp.epsilon = epsilon.convertToFloat();
  batchNormInferenceOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  batchNormInferenceOp.out = detail::getOutputTensorRefT(outputLayout);
  return batchNormInferenceOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<BatchNormInferenceOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::BatchNormInferenceOpT batchNormInferenceOpNative =
      buildBatchNormInferenceOpTFromMLIR(epsilon, computeKernelConfig,
                                         outputLayout);

  auto batchNormQuery = [=]() {
    ttnn_op_invoke::BatchNormOpResult result =
        ttnn_op_invoke::callBatchNormInference(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            batchNormInferenceOpNative, inputSpec, runningMeanSpec,
            runningVarSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected BatchNormInferenceOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), batchNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<BatchNormInferenceOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::BatchNormInferenceOpT batchNormInferenceOpNative =
      buildBatchNormInferenceOpTFromMLIR(epsilon, computeKernelConfig,
                                         outputLayout);

  auto batchNormQuery = [=]() {
    ttnn_op_invoke::BatchNormOpResult result =
        ttnn_op_invoke::callBatchNormInference(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            batchNormInferenceOpNative, inputSpec, runningMeanSpec,
            runningVarSpec, weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected BatchNormInferenceOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(batchNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::BatchNormTrainingOpT buildBatchNormTrainingOpTFromMLIR(
    llvm::APFloat epsilon, llvm::APFloat momentum,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::BatchNormTrainingOpT batchNormTrainingOp;
  batchNormTrainingOp.epsilon = epsilon.convertToFloat();
  batchNormTrainingOp.momentum = momentum.convertToFloat();
  batchNormTrainingOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  batchNormTrainingOp.out = detail::getOutputTensorRefT(outputLayout);
  return batchNormTrainingOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<BatchNormTrainingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    llvm::APFloat momentum,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::BatchNormTrainingOpT batchNormTrainingOpNative =
      buildBatchNormTrainingOpTFromMLIR(epsilon, momentum, computeKernelConfig,
                                        outputLayout);

  auto batchNormQuery = [=]() {
    ttnn_op_invoke::BatchNormOpResult result =
        ttnn_op_invoke::callBatchNormTraining(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            batchNormTrainingOpNative, inputSpec, runningMeanSpec,
            runningVarSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected BatchNormTrainingOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), batchNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<BatchNormTrainingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    llvm::APFloat momentum,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> runningMeanSpec =
      detail::convertToOptionalTensorSpec(device, runningMeanShape,
                                          runningMeanLayout);
  std::optional<::ttnn::TensorSpec> runningVarSpec =
      detail::convertToOptionalTensorSpec(device, runningVarShape,
                                          runningVarLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::BatchNormTrainingOpT batchNormTrainingOpNative =
      buildBatchNormTrainingOpTFromMLIR(epsilon, momentum, computeKernelConfig,
                                        outputLayout);

  auto batchNormQuery = [=]() {
    ttnn_op_invoke::BatchNormOpResult result =
        ttnn_op_invoke::callBatchNormTraining(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            batchNormTrainingOpNative, inputSpec, runningMeanSpec,
            runningVarSpec, weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected BatchNormTrainingOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(batchNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RMSNormOpT buildRMSNormOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RMSNormOpT rmsNormOp;
  rmsNormOp.epsilon = epsilon.convertToFloat();
  rmsNormOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  rmsNormOp.out = detail::getOutputTensorRefT(outputLayout);
  return rmsNormOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RMSNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::RMSNormOpT rmsNormOpNative =
      buildRMSNormOpTFromMLIR(epsilon, computeKernelConfig, outputLayout);

  auto rmsNormQuery = [=]() {
    ttnn_op_invoke::RMSNormOpResult result = ttnn_op_invoke::callRMSNorm(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, rmsNormOpNative,
        inputSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected RMSNormOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), rmsNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RMSNormOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::RMSNormOpT rmsNormOpNative =
      buildRMSNormOpTFromMLIR(epsilon, computeKernelConfig, outputLayout);

  auto rmsNormQuery = [=]() {
    ttnn_op_invoke::RMSNormOpResult result = ttnn_op_invoke::callRMSNorm(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, rmsNormOpNative, inputSpec,
        weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RMSNormOp runtime query to return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(rmsNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RMSNormPreAllGatherOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::RMSNormPreAllGatherOpT buildRMSNormPreAllGatherOpTFromMLIR(
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    std::optional<bool> use2DCoreGrid, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::RMSNormPreAllGatherOpT rmsNormPreAllGatherOp;
  rmsNormPreAllGatherOp.use_2d_core_grid = use2DCoreGrid.value_or(false);
  rmsNormPreAllGatherOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  rmsNormPreAllGatherOp.program_config =
      (programConfig.has_value() && *programConfig)
          ? std::make_unique<
                ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfigT>(
                toNative(*programConfig))
          : nullptr;
  rmsNormPreAllGatherOp.out = detail::getOutputTensorRefT(outputLayout);
  return rmsNormPreAllGatherOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<RMSNormPreAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<ttcore::DataType> dtype, std::optional<bool> use2DCoreGrid,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> residualInputSpec =
      detail::convertToOptionalTensorSpec(device, residualInputShape,
                                          residualInputLayout);

  ::tt::target::ttnn::RMSNormPreAllGatherOpT rmsNormPreAllGatherOpNative =
      buildRMSNormPreAllGatherOpTFromMLIR(computeKernelConfig, programConfig,
                                          use2DCoreGrid, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::RMSNormPreAllGatherOpResult result =
        ttnn_op_invoke::callRMSNormPreAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            rmsNormPreAllGatherOpNative, inputSpec, residualInputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected RMSNormPreAllGatherOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RMSNormPreAllGatherOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<ttcore::DataType> dtype, std::optional<bool> use2DCoreGrid,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> residualInputSpec =
      detail::convertToOptionalTensorSpec(device, residualInputShape,
                                          residualInputLayout);

  ::tt::target::ttnn::RMSNormPreAllGatherOpT rmsNormPreAllGatherOpNative =
      buildRMSNormPreAllGatherOpTFromMLIR(computeKernelConfig, programConfig,
                                          use2DCoreGrid, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::RMSNormPreAllGatherOpResult result =
        ttnn_op_invoke::callRMSNormPreAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            rmsNormPreAllGatherOpNative, inputSpec, residualInputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RMSNormPreAllGatherOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::LayerNormOpT
buildLayerNormOpTFromMLIR(llvm::APFloat epsilon, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::LayerNormOpT layerNormOp;
  layerNormOp.epsilon = epsilon.convertToFloat();
  layerNormOp.out = detail::getOutputTensorRefT(outputLayout);
  return layerNormOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<LayerNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::LayerNormOpT layerNormOpNative =
      buildLayerNormOpTFromMLIR(epsilon, outputLayout);

  auto layerNormQuery = [=]() {
    ttnn_op_invoke::LayerNormOpResult result = ttnn_op_invoke::callLayerNorm(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, layerNormOpNative,
        inputSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected LayerNormOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), layerNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LayerNormOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::LayerNormOpT layerNormOpNative =
      buildLayerNormOpTFromMLIR(epsilon, outputLayout);

  auto layerNormQuery = [=]() {
    ttnn_op_invoke::LayerNormOpResult result = ttnn_op_invoke::callLayerNorm(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, layerNormOpNative,
        inputSpec, weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected LayerNormOp runtime query to return RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(layerNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormPreAllGatherOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::LayerNormPreAllGatherOpT
buildLayerNormPreAllGatherOpTFromMLIR(
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::LayerNormPreAllGatherOpT layerNormPreAllGatherOp;
  layerNormPreAllGatherOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  layerNormPreAllGatherOp.program_config =
      (programConfig.has_value() && *programConfig)
          ? std::make_unique<
                ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfigT>(
                toNative(*programConfig))
          : nullptr;
  layerNormPreAllGatherOp.out = detail::getOutputTensorRefT(outputLayout);
  return layerNormPreAllGatherOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<LayerNormPreAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<llvm::ArrayRef<int64_t>> recipShape,
    std::optional<TTNNLayoutAttr> recipLayout,
    std::optional<ttcore::DataType> dtype,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> residualInputSpec =
      detail::convertToOptionalTensorSpec(device, residualInputShape,
                                          residualInputLayout);
  std::optional<::ttnn::TensorSpec> recipSpec =
      detail::convertToOptionalTensorSpec(device, recipShape, recipLayout);

  ::tt::target::ttnn::LayerNormPreAllGatherOpT layerNormPreAllGatherOpNative =
      buildLayerNormPreAllGatherOpTFromMLIR(computeKernelConfig, programConfig,
                                            outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::LayerNormPreAllGatherOpResult result =
        ttnn_op_invoke::callLayerNormPreAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            layerNormPreAllGatherOpNative, inputSpec, residualInputSpec,
            recipSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected LayerNormPreAllGatherOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LayerNormPreAllGatherOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<llvm::ArrayRef<int64_t>> recipShape,
    std::optional<TTNNLayoutAttr> recipLayout,
    std::optional<ttcore::DataType> dtype,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> residualInputSpec =
      detail::convertToOptionalTensorSpec(device, residualInputShape,
                                          residualInputLayout);
  std::optional<::ttnn::TensorSpec> recipSpec =
      detail::convertToOptionalTensorSpec(device, recipShape, recipLayout);

  ::tt::target::ttnn::LayerNormPreAllGatherOpT layerNormPreAllGatherOpNative =
      buildLayerNormPreAllGatherOpTFromMLIR(computeKernelConfig, programConfig,
                                            outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::LayerNormPreAllGatherOpResult result =
        ttnn_op_invoke::callLayerNormPreAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            layerNormPreAllGatherOpNative, inputSpec, residualInputSpec,
            recipSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected LayerNormPreAllGatherOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormPostAllGatherOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::LayerNormPostAllGatherOpT
buildLayerNormPostAllGatherOpTFromMLIR(
    llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::LayerNormPostAllGatherOpT layerNormPostAllGatherOp;
  layerNormPostAllGatherOp.epsilon = epsilon.convertToFloat();
  layerNormPostAllGatherOp.compute_config =
      (computeKernelConfig.has_value() && *computeKernelConfig)
          ? std::make_unique<::tt::target::ttnn::DeviceComputeKernelConfigT>(
                toNative(*computeKernelConfig))
          : nullptr;
  layerNormPostAllGatherOp.program_config =
      (programConfig.has_value() && *programConfig)
          ? std::make_unique<
                ::tt::target::ttnn::LayerNormShardedMultiCoreProgramConfigT>(
                toNative(*programConfig))
          : nullptr;
  layerNormPostAllGatherOp.out = detail::getOutputTensorRefT(outputLayout);
  return layerNormPostAllGatherOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<LayerNormPostAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> statsShape, TTNNLayoutAttr statsLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec statsSpec,
      detail::convertToTensorSpec(device, statsShape, statsLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::LayerNormPostAllGatherOpT layerNormPostAllGatherOpNative =
      buildLayerNormPostAllGatherOpTFromMLIR(epsilon, computeKernelConfig,
                                             programConfig, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::LayerNormPostAllGatherOpResult result =
        ttnn_op_invoke::callLayerNormPostAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            layerNormPostAllGatherOpNative, inputSpec, statsSpec, weightSpec,
            biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected LayerNormPostAllGatherOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LayerNormPostAllGatherOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> statsShape, TTNNLayoutAttr statsLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    std::optional<LayerNormShardedMultiCoreProgramConfigAttr> programConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec statsSpec,
      detail::convertToTensorSpec(device, statsShape, statsLayout));

  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::LayerNormPostAllGatherOpT layerNormPostAllGatherOpNative =
      buildLayerNormPostAllGatherOpTFromMLIR(epsilon, computeKernelConfig,
                                             programConfig, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::LayerNormPostAllGatherOpResult result =
        ttnn_op_invoke::callLayerNormPostAllGather(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            layerNormPostAllGatherOpNative, inputSpec, statsSpec, weightSpec,
            biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected LayerNormPostAllGatherOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GroupNormOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::GroupNormOpT
buildGroupNormOpTFromMLIR(int64_t numGroups, llvm::APFloat epsilon,
                          TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::GroupNormOpT groupNormOp;
  groupNormOp.num_groups = numGroups;
  groupNormOp.epsilon = epsilon.convertToFloat();
  groupNormOp.out = detail::getOutputTensorRefT(outputLayout);
  return groupNormOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<GroupNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputMaskShape,
    std::optional<TTNNLayoutAttr> inputMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, int64_t numGroups,
    llvm::APFloat epsilon, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> inputMaskSpec =
      detail::convertToOptionalTensorSpec(device, inputMaskShape,
                                          inputMaskLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::GroupNormOpT groupNormOpNative =
      buildGroupNormOpTFromMLIR(numGroups, epsilon, outputLayout);

  auto groupNormQuery = [=]() {
    ttnn_op_invoke::GroupNormOpResult result = ttnn_op_invoke::callGroupNorm(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, groupNormOpNative,
        inputSpec, inputMaskSpec, weightSpec, biasSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected GroupNormOp constraints query to return "
           "ConstraintQueryResponse");
    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), groupNormQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<GroupNormOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputMaskShape,
    std::optional<TTNNLayoutAttr> inputMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, int64_t numGroups,
    llvm::APFloat epsilon, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> inputMaskSpec =
      detail::convertToOptionalTensorSpec(device, inputMaskShape,
                                          inputMaskLayout);
  std::optional<::ttnn::TensorSpec> weightSpec =
      detail::convertToOptionalTensorSpec(device, weightShape, weightLayout);
  std::optional<::ttnn::TensorSpec> biasSpec =
      detail::convertToOptionalTensorSpec(device, biasShape, biasLayout);

  ::tt::target::ttnn::GroupNormOpT groupNormOpNative =
      buildGroupNormOpTFromMLIR(numGroups, epsilon, outputLayout);

  auto groupNormQuery = [=]() {
    ttnn_op_invoke::GroupNormOpResult result = ttnn_op_invoke::callGroupNorm(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, groupNormOpNative,
        inputSpec, inputMaskSpec, weightSpec, biasSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected GroupNormOp runtime query to return "
        "RuntimeQueryResponse");
    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(groupNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampScalar
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(mlir::Attribute min,
                                                 mlir::Attribute max,
                                                 TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;
  eltwiseUnaryCompositeOp.type =
      ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampScalar;

  ::tt::target::ttnn::ClampScalarOpParamsT params;

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(min)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    params.min.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(min)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    params.min.Set(i32);
  } else {
    llvm_unreachable("Invalid clamp min attribute");
  }

  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(max)) {
    ::tt::target::ttnn::FloatingPointTypeT fp;
    fp.value = floatAttr.getValue().convertToFloat();
    params.max.Set(fp);
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(max)) {
    ::tt::target::ttnn::IntegralTypeT i32;
    i32.value = static_cast<int32_t>(intAttr.getInt());
    params.max.Set(i32);
  } else {
    llvm_unreachable("Invalid clamp max attribute");
  }

  eltwiseUnaryCompositeOp.params.Set(params);

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ClampScalarOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute min, mlir::Attribute max, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(min, max, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryCompositeOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeClampScalar(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryCompositeOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseUnaryCompositeClampScalar query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ClampScalarOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute min, mlir::Attribute max, TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeClampScalarOpTFromMLIR(min, max, outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryCompositeOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeClampScalar(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseUnaryCompositeOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnaryCompositeClampScalar "
        "query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampTensor
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::EltwiseUnaryCompositeOpT
buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOp;
  eltwiseUnaryCompositeOp.type =
      ::tt::target::ttnn::EltwiseUnaryCompositeOpType::ClampTensor;

  eltwiseUnaryCompositeOp.out = detail::getOutputTensorRefT(outputLayout);

  return eltwiseUnaryCompositeOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ClampTensorOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
    llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec minSpec,
                   detail::convertToTensorSpec(device, minShape, minLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec maxSpec,
                   detail::convertToTensorSpec(device, maxShape, maxLayout));

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryCompositeOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeClampTensor(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            eltwiseUnaryCompositeOpNative, inputSpec, minSpec, maxSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from "
           "EltwiseUnaryCompositeClampTensor query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ClampTensorOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
    llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec minSpec,
                   detail::convertToTensorSpec(device, minShape, minLayout));

  ASSIGN_OR_RETURN(::ttnn::TensorSpec maxSpec,
                   detail::convertToTensorSpec(device, maxShape, maxLayout));

  ::tt::target::ttnn::EltwiseUnaryCompositeOpT eltwiseUnaryCompositeOpNative =
      buildEltwiseUnaryCompositeClampTensorOpTFromMLIR(outputLayout);

  auto query = [=]() {
    ttnn_op_invoke::EltwiseUnaryCompositeOpResult result =
        ttnn_op_invoke::callEltwiseUnaryCompositeClampTensor(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            eltwiseUnaryCompositeOpNative, inputSpec, minSpec, maxSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EltwiseUnaryCompositeClampTensor "
        "query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::PermuteOpT
buildPermuteOpTFromMLIR(llvm::ArrayRef<int64_t> permutation,
                        llvm::APFloat padValue, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::PermuteOpT permuteOp;
  permuteOp.permutation = {permutation.begin(), permutation.end()};
  permuteOp.pad_value = padValue.convertToFloat();
  permuteOp.out = detail::getOutputTensorRefT(outputLayout);
  return permuteOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PermuteOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::PermuteOpT permuteOpNative =
      buildPermuteOpTFromMLIR(permutation, padValue, outputLayout);

  auto permuteQuery = [=]() {
    ttnn_op_invoke::PermuteOpResult result = ttnn_op_invoke::callPermute(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, permuteOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from PermuteOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), permuteQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<PermuteOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::PermuteOpT permuteOpNative =
      buildPermuteOpTFromMLIR(permutation, padValue, outputLayout);

  auto permuteQuery = [=]() {
    ttnn_op_invoke::PermuteOpResult result =
        ttnn_op_invoke::callPermute(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                    permuteOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from PermuteOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(permuteQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Upsample
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<UpsampleOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute scaleFactor, llvm::StringRef mode,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert params
  std::variant<int, std::array<int, 2>, float, std::array<float, 2>>
      convertedScaleFactor;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(scaleFactor)) {
    convertedScaleFactor = static_cast<int>(value.getSInt());
  } else if (auto tuple =
                 mlir::dyn_cast<::mlir::detail::DenseArrayAttrImpl<int32_t>>(
                     scaleFactor);
             tuple.size() == 2) {
    std::array<int, 2> arr;
    arr[0] = static_cast<int>(tuple[0]);
    arr[1] = static_cast<int>(tuple[1]);
    convertedScaleFactor = arr;
  } else {
    return llvm::createStringError("Invalid scaleFactor");
  }

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto upsampleQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::upsample, device, inputSpec,
                                convertedScaleFactor, std::string(mode),
                                detail::getNullableMemoryConfig(outputLayout),
                                /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraints(inputLayout.getContext(), upsampleQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<UpsampleOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute scaleFactor, llvm::StringRef mode,
    TTNNLayoutAttr outputLayout) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert parameters
  std::variant<int, std::array<int, 2>, float, std::array<float, 2>>
      convertedScaleFactor;
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(scaleFactor)) {
    convertedScaleFactor = static_cast<int>(value.getSInt());
  } else if (auto tuple =
                 mlir::dyn_cast<::mlir::detail::DenseArrayAttrImpl<int32_t>>(
                     scaleFactor);
             tuple.size() == 2) {
    std::array<int, 2> arr;
    arr[0] = static_cast<int>(tuple[0]);
    arr[1] = static_cast<int>(tuple[1]);
    convertedScaleFactor = arr;
  } else {
    return llvm::createStringError("Invalid scaleFactor");
  }

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto upsampleQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::upsample, device, inputSpec,
                            convertedScaleFactor, std::string(mode),
                            detail::getNullableMemoryConfig(outputLayout),
                            /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(upsampleQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmbeddingOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::EmbeddingOpT
buildEmbeddingOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EmbeddingOpT op;
  op.out = detail::getOutputTensorRefT(outputLayout);
  return op;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<EmbeddingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  ::tt::target::ttnn::EmbeddingOpT embeddingOpNative =
      buildEmbeddingOpTFromMLIR(outputLayout);

  auto embeddingOpQuery = [=]() {
    ttnn_op_invoke::EmbeddingOpResult result = ttnn_op_invoke::callEmbedding(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, embeddingOpNative,
        inputSpec, weightSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EmbeddingOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     embeddingOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<EmbeddingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  ::tt::target::ttnn::EmbeddingOpT embeddingOpNative =
      buildEmbeddingOpTFromMLIR(outputLayout);

  auto embeddingOpQuery = [=]() {
    ttnn_op_invoke::EmbeddingOpResult result = ttnn_op_invoke::callEmbedding(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, embeddingOpNative,
        inputSpec, weightSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EmbeddingOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(embeddingOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::EmbeddingBackwardOpT
buildEmbeddingBackwardOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::EmbeddingBackwardOpT op;
  op.out = detail::getOutputTensorRefT(outputLayout);
  return op;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<EmbeddingBackwardOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> inGradientShape, TTNNLayoutAttr inGradientLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inGradientSpec,
      detail::convertToTensorSpec(device, inGradientShape, inGradientLayout));

  ::tt::target::ttnn::EmbeddingBackwardOpT embeddingBackwardOpNative =
      buildEmbeddingBackwardOpTFromMLIR(outputLayout);

  auto embeddingBackwardOpQuery = [=]() {
    ttnn_op_invoke::EmbeddingBackwardOpResult result =
        ttnn_op_invoke::callEmbeddingBackward(
            ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
            embeddingBackwardOpNative, inputSpec, weightSpec, inGradientSpec,
            device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from EmbeddingBackwardOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     embeddingBackwardOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<mlir::tt::ttnn::EmbeddingBackwardOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> inGradientShape, TTNNLayoutAttr inGradientLayout,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inGradientSpec,
      detail::convertToTensorSpec(device, inGradientShape, inGradientLayout));

  ::tt::target::ttnn::EmbeddingBackwardOpT embeddingBackwardOpNative =
      buildEmbeddingBackwardOpTFromMLIR(outputLayout);

  auto embeddingBackwardOpQuery = [=]() {
    ttnn_op_invoke::EmbeddingBackwardOpResult result =
        ttnn_op_invoke::callEmbeddingBackward(
            ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
            embeddingBackwardOpNative, inputSpec, weightSpec, inGradientSpec,
            device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from EmbeddingBackwardOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(embeddingBackwardOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::GatherOpT
buildGatherOpTFromMLIR(int32_t dim, TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::GatherOpT gatherOp;
  gatherOp.dim = dim;
  gatherOp.out = detail::getOutputTensorRefT(outputLayout);
  return gatherOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<GatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout, int32_t dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indexSpec,
      detail::convertToTensorSpec(device, indexShape, indexLayout));

  ::tt::target::ttnn::GatherOpT gatherOpNative =
      buildGatherOpTFromMLIR(dim, outputLayout);

  auto gatherOpQuery = [=]() {
    ttnn_op_invoke::GatherOpResult result = ttnn_op_invoke::callGather(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, gatherOpNative,
        inputSpec, indexSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from GatherOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), gatherOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<GatherOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout, int32_t dim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indexSpec,
      detail::convertToTensorSpec(device, indexShape, indexLayout));

  ::tt::target::ttnn::GatherOpT gatherOpNative =
      buildGatherOpTFromMLIR(dim, outputLayout);

  auto gatherOpQuery = [=]() {
    ttnn_op_invoke::GatherOpResult result = ttnn_op_invoke::callGather(
        ttnn_op_invoke::CallType::QUERY_OP_RUNTIME, gatherOpNative, inputSpec,
        indexSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from GatherOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(gatherOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmptyOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::EmptyOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, mlir::tt::ttcore::DataTypeAttr dtype,
    mlir::tt::ttnn::Layout inputLayout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::DRAM_MEMORY_CONFIG;
  if (outputLayout) {
    memConfig =
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout));
  }

  auto emptyOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::empty, device, conversion::getShape(inputShape),
        conversion::getDataType(dtype.getValue()),
        conversion::getPageLayout(inputLayout), device, memConfig);
  };

  return operation::getOpConstraints(dtype.getContext(), emptyOpQuery);
#else
  return OpConstraints{};
#endif //
}

//===----------------------------------------------------------------------===//
// ArangeOp
//===----------------------------------------------------------------------===//
// sgholamiTT: There are two reasons why receiving the start, end, and step as
// attributes is better than as integers:
//   1. That is the only valid way to acquire a pointer to MLIRContext.
//   2. Using getInt() member function of ::mlir::IntegerAttr is safer and more
//      mlir idiomatic than static_cast<int64_t>(start).
llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::ArangeOp>::getOpConstraints(
    ::mlir::IntegerAttr start, ::mlir::IntegerAttr end,
    ::mlir::IntegerAttr step, std::optional<mlir::tt::ttcore::DataType> dtype,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  // ~~~~~~~~~~~~~~~~~~~~~ Note ~~~~~~~~~~~~~~~~~~~~~
  // The following default values are taken from Arrange's invoke function in
  // tt-metal/ttnn/cpp/ttnn/operations/creation/creation.hpp
  const ::tt::tt_metal::DataType defaultDtypeInMetal =
      ::tt::tt_metal::DataType::BFLOAT16;
  const ::ttnn::MemoryConfig defaultMemoryConfigInMetal =
      ::ttnn::DRAM_MEMORY_CONFIG;
  const ::ttnn::Layout defaultLayoutInMetal = ::ttnn::ROW_MAJOR_LAYOUT;
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ::tt::tt_metal::DataType dataType = defaultDtypeInMetal;
  if (dtype.has_value()) {
    dataType = conversion::getDataType(dtype.value());
  }
  ::ttnn::MemoryConfig memoryConfig = defaultMemoryConfigInMetal;
  ::ttnn::Layout layout = defaultLayoutInMetal;
  if (outputLayout) {
    memoryConfig =
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout));
    layout =
        outputLayout.isTiled() ? ::ttnn::TILE_LAYOUT : ::ttnn::ROW_MAJOR_LAYOUT;
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  auto arangeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::arange, device, start.getInt(),
                                end.getInt(), step.getInt(), dataType,
                                deviceRef, memoryConfig, layout);
  };

  return operation::getOpConstraints(start.getContext(), arangeOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<mlir::tt::ttnn::FullOp>::getOpConstraints(
    mlir::tt::ttnn::ShapeAttr shape, mlir::Attribute fillValue,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    std::optional<mlir::tt::ttnn::Layout> layout,
    mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::ttnn::MemoryConfig> metalMemConfig = std::nullopt;
  if (outputLayout) {
    metalMemConfig =
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout));
  }

  std::optional<::ttnn::DataType> metalDtype = std::nullopt;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }
  ::ttnn::Shape metalShape = conversion::getShape(shape.getShape());

  std::optional<::ttnn::Layout> metalLayout = std::nullopt;
  if (layout.has_value()) {
    metalLayout = conversion::getPageLayout(layout.value());
  }
  std::optional<std::reference_wrapper<::tt::tt_metal::distributed::MeshDevice>>
      deviceRef = *device;

  // Helper lambda to create the query with any fill value type
  auto createFullOpQuery = [=](auto convertedFillValue) {
    return [=]() {
      return QUERY_OP_CONSTRAINTS(::ttnn::full, device, metalShape,
                                  convertedFillValue, metalDtype, metalLayout,
                                  deviceRef, metalMemConfig,
                                  /*optional_output_tensor = */ std::nullopt);
    };
  };

  // The invoke function of fullOp is templated over the fill value type. That's
  // why the following code is arranged in this way.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(fillValue)) {
    int convertedFillValue = static_cast<int>(value.getInt());
    auto query = createFullOpQuery(convertedFillValue);
    return operation::getOpConstraints(fillValue.getContext(), query);
  }
  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(fillValue)) {
    float convertedFillValue = value.getValue().convertToFloat();
    auto query = createFullOpQuery(convertedFillValue);
    return operation::getOpConstraints(fillValue.getContext(), query);
  }
  return llvm::createStringError("Invalid fillValue");
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RandOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<mlir::tt::ttnn::RandOp>::getOpConstraints(
    mlir::tt::ttnn::ShapeAttr size, mlir::tt::ttcore::DataType dtype,
    mlir::tt::ttnn::Layout layout, llvm::APFloat low, llvm::APFloat high,
    uint32_t seed, mlir::tt::ttnn::TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ::ttnn::MemoryConfig metalMemConfig = ::ttnn::DRAM_MEMORY_CONFIG;
  if (outputLayout) {
    metalMemConfig =
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout));
  }

  auto randOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::rand, device, conversion::getShape(size.getShape()),
        std::ref(*device), conversion::getDataType(dtype),
        conversion::getPageLayout(layout), metalMemConfig, low.convertToFloat(),
        high.convertToFloat(), seed);
  };

  return operation::getOpConstraints(size.getContext(), randOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// DropoutOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::DropoutOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat prob, llvm::APFloat scale, uint32_t seed,
    bool usePerDeviceSeed, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  float probVal = prob.convertToFloat();
  float scaleVal = scale.convertToFloat();

  // Create query closure
  auto dropoutOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::experimental::dropout, device, inputSpec, probVal, scaleVal,
        seed, usePerDeviceSeed, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt);
  };

  return operation::getOpConstraints(inputLayout.getContext(), dropoutOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::DropoutOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat prob, llvm::APFloat scale, uint32_t seed,
    bool usePerDeviceSeed, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  float probVal = prob.convertToFloat();
  float scaleVal = scale.convertToFloat();

  // Create query closure
  auto dropoutOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::dropout, device, inputSpec,
                            probVal, scaleVal, seed, usePerDeviceSeed,
                            detail::getNullableMemoryConfig(outputLayout),
                            std::nullopt);
  };

  return operation::getOpRuntime(dropoutOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
// sgholamiTT: I decided to not promote these helper methods to conversion.hpp
// for two reasons:
//   1. There's no other clear usage for them.
//   2. Some of them are specialized for ConstantOp.

mlir::Type getElementType(mlir::ElementsAttr value) {
  if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(value)) {
    return denseAttr.getType().getElementType();
  }
  if (auto splatAttr = llvm::dyn_cast<mlir::SplatElementsAttr>(value)) {
    return splatAttr.getType().getElementType();
  }
  assert(false && "Unknown constant value attribute type");
}

::ttnn::Shape getShape(mlir::ElementsAttr value) {
  if (auto rankedTensorType =
          dyn_cast<mlir::RankedTensorType>(value.getType())) {
    // Get the shape as a vector of dimensions
    llvm::ArrayRef<int64_t> shape = rankedTensorType.getShape();
    return conversion::getShape(shape);
  }
  assert(false && "Unknown constant value attribute type");
}

::tt::tt_metal::DataType getDataType(const mlir::ElementsAttr attr) {
  ::mlir::Type elType = getElementType(attr);
  ::tt::tt_metal::DataType dtype = ::tt::tt_metal::DataType::INVALID;
  if (elType.isBF16()) {
    dtype = ::tt::tt_metal::DataType::BFLOAT16;
  } else if (elType.isF32()) {
    dtype = ::tt::tt_metal::DataType::FLOAT32;
  } else if (elType.isUnsignedInteger(32)) {
    dtype = ::tt::tt_metal::DataType::UINT32;
  } else if (elType.isUnsignedInteger(16)) {
    dtype = ::tt::tt_metal::DataType::UINT16;
  } else if (elType.isUnsignedInteger(8)) {
    dtype = ::tt::tt_metal::DataType::UINT8;
  } else if (elType.isInteger(32)) {
    dtype = ::tt::tt_metal::DataType::INT32;
  }
  assert(dtype != ::tt::tt_metal::DataType::INVALID && "Unsupported data type");
  return dtype;
}

// Helper macro to reduce repetition in type dispatch
#define DISPATCH_TYPE(TYPE_CHECK, CPP_TYPE)                                    \
  if (elType.TYPE_CHECK) {                                                     \
    auto rawDataExp = detail::getRawDataFromElementsAttr<CPP_TYPE>(value);     \
    if (!rawDataExp) {                                                         \
      return rawDataExp.takeError();                                           \
    }                                                                          \
    return func(rawDataExp.get());                                             \
  }

// Helper function to dispatch getRawDataFromElementsAttr based on element type
// (we use this technique since from_buffer op in metal is templated over the
// input vector type.)
template <typename Func>
auto dispatchGetRawData(mlir::ElementsAttr value, Func &&func)
    -> decltype(func(std::declval<std::vector<int32_t>>())) {
  // from_span<T> has template instantiations for the following types:
  // int32_t, uint8_t, uint16_t, uint32_t, bfloat16.
  // We support all of these types:
  ::mlir::Type elType = getElementType(value);
  DISPATCH_TYPE(isUnsignedInteger(8), uint8_t)
  DISPATCH_TYPE(isUnsignedInteger(16), uint16_t)
  DISPATCH_TYPE(isUnsignedInteger(32), uint32_t)
  DISPATCH_TYPE(isInteger(32), int32_t)
  DISPATCH_TYPE(isF32(), float)
  DISPATCH_TYPE(isBF16(), bfloat16)

  return llvm::createStringError(std::errc::invalid_argument,
                                 "Unsupported element type for ConstantOp");
}

#undef DISPATCH_TYPE
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<ConstantOp>::getOpConstraints(mlir::ElementsAttr value,
                                      TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::tt::tt_metal::Layout> metalLayout = std::nullopt;
  if (outputLayout) {
    metalLayout = conversion::getPageLayout(outputLayout);
  }
  auto func = [&](auto rawData) {
    auto constantOpQuery = [=]() {
      return QUERY_OP_CONSTRAINTS(
          ::ttnn::from_buffer, device, rawData, getShape(value),
          getDataType(value), device, metalLayout,
          detail::getNullableMemoryConfig(outputLayout));
    };
    return operation::getOpConstraints(value.getContext(), constantOpQuery);
  };
  return dispatchGetRawData(value, func);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::AssignOpT
buildAssignOpTFromMLIR(TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::AssignOpT assignOp;
  assignOp.output = detail::getOutputTensorRefT(outputLayout);
  return assignOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::AssignOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::AssignOpT assignOpNative =
      buildAssignOpTFromMLIR(outputLayout);

  auto assignOpQuery = [=]() {
    ttnn_op_invoke::AssignOpResult result = ttnn_op_invoke::callAssign(
        ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS, assignOpNative,
        inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from AssignOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), assignOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::AssignOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::AssignOpT assignOpNative =
      buildAssignOpTFromMLIR(outputLayout);

  auto assignOpQuery = [=]() {
    ttnn_op_invoke::AssignOpResult result =
        ttnn_op_invoke::callAssign(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                   assignOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from AssignOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(assignOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
::tt::target::ttnn::TopKOpT buildTopKOpTFromMLIR(int32_t k, int32_t dim,
                                                 bool largest, bool sorted,
                                                 TTNNLayoutAttr outputLayout) {
  ::tt::target::ttnn::TopKOpT topKOp;
  topKOp.k = k;
  topKOp.dim = dim;
  topKOp.largest = largest;
  topKOp.sorted = sorted;
  topKOp.outputs.push_back(detail::getOutputTensorRefT(outputLayout));
  return topKOp;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<TopKOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t k,
    int32_t dim, bool largest, bool sorted, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::TopKOpT topKOpNative =
      buildTopKOpTFromMLIR(k, dim, largest, sorted, outputLayout);

  auto topKQuery = [=]() {
    ttnn_op_invoke::TopKOpResult result =
        ttnn_op_invoke::callTopK(ttnn_op_invoke::CallType::QUERY_OP_CONSTRAINTS,
                                 topKOpNative, inputSpec, device);

    assert(std::holds_alternative<::ttnn::graph::ConstraintQueryResponse>(
               result) &&
           "Expected ConstraintQueryResponse from TopKOp query");

    return std::get<::ttnn::graph::ConstraintQueryResponse>(result);
  };

  return operation::getOpConstraints(inputLayout.getContext(), topKQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<TopKOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t k,
    int32_t dim, bool largest, bool sorted, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::target::ttnn::TopKOpT topKOpNative =
      buildTopKOpTFromMLIR(k, dim, largest, sorted, outputLayout);

  auto topKQuery = [=]() {
    ttnn_op_invoke::TopKOpResult result =
        ttnn_op_invoke::callTopK(ttnn_op_invoke::CallType::QUERY_OP_RUNTIME,
                                 topKOpNative, inputSpec, device);

    assert(
        std::holds_alternative<::ttnn::graph::RuntimeQueryResponse>(result) &&
        "Expected RuntimeQueryResponse from TopKOp query");

    return std::get<::ttnn::graph::RuntimeQueryResponse>(result);
  };

  return operation::getOpRuntime(topKQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SamplingOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<SamplingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputValuesShape, TTNNLayoutAttr inputValuesLayout,
    llvm::ArrayRef<int64_t> inputIndicesShape,
    TTNNLayoutAttr inputIndicesLayout, llvm::ArrayRef<int64_t> kShape,
    TTNNLayoutAttr kLayout, llvm::ArrayRef<int64_t> pShape,
    TTNNLayoutAttr pLayout, llvm::ArrayRef<int64_t> tempShape,
    TTNNLayoutAttr tempLayout, std::optional<uint32_t> seed,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // ttnn::sampling kernel expects 4D [N, C, H, W] with N*C*H==32. Runtime
  // reshapes 2D [batch, candidates] -> [1, 1, batch, candidates] before
  // dispatch; mirror that here so constraint queries see the kernel-expected
  // shape.
  llvm::SmallVector<int64_t, 4> values4D = {1, 1, inputValuesShape[0],
                                            inputValuesShape[1]};
  llvm::SmallVector<int64_t, 4> indices4D = {1, 1, inputIndicesShape[0],
                                             inputIndicesShape[1]};

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valuesSpec,
      detail::convertToTensorSpec(device, values4D, inputValuesLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indicesSpec,
      detail::convertToTensorSpec(device, indices4D, inputIndicesLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec kSpec,
                   detail::convertToTensorSpec(device, kShape, kLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec pSpec,
                   detail::convertToTensorSpec(device, pShape, pLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec tempSpec,
                   detail::convertToTensorSpec(device, tempShape, tempLayout));

  auto samplingQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::sampling, device, valuesSpec,
                                indicesSpec, kSpec, pSpec, tempSpec, seed,
                                std::nullopt, std::nullopt);
  };

  return operation::getOpConstraints(inputValuesLayout.getContext(),
                                     samplingQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SamplingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputValuesShape, TTNNLayoutAttr inputValuesLayout,
    llvm::ArrayRef<int64_t> inputIndicesShape,
    TTNNLayoutAttr inputIndicesLayout, llvm::ArrayRef<int64_t> kShape,
    TTNNLayoutAttr kLayout, llvm::ArrayRef<int64_t> pShape,
    TTNNLayoutAttr pLayout, llvm::ArrayRef<int64_t> tempShape,
    TTNNLayoutAttr tempLayout, std::optional<uint32_t> seed,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // See getOpConstraints: reshape 2D -> 4D to match runtime dispatch.
  llvm::SmallVector<int64_t, 4> values4D = {1, 1, inputValuesShape[0],
                                            inputValuesShape[1]};
  llvm::SmallVector<int64_t, 4> indices4D = {1, 1, inputIndicesShape[0],
                                             inputIndicesShape[1]};

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valuesSpec,
      detail::convertToTensorSpec(device, values4D, inputValuesLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indicesSpec,
      detail::convertToTensorSpec(device, indices4D, inputIndicesLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec kSpec,
                   detail::convertToTensorSpec(device, kShape, kLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec pSpec,
                   detail::convertToTensorSpec(device, pShape, pLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec tempSpec,
                   detail::convertToTensorSpec(device, tempShape, tempLayout));

  auto samplingQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::sampling, device, valuesSpec, indicesSpec,
                            kSpec, pSpec, tempSpec, seed, std::nullopt,
                            std::nullopt);
  };

  return operation::getOpRuntime(samplingQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MeshPartitionOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<MeshPartitionOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t dim,
    std::optional<uint32_t> clusterAxis, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto meshPartitionOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::mesh_partition, device, inputSpec, dim,
                                clusterAxis,
                                detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraints(inputLayout.getContext(),
                                     meshPartitionOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MeshPartitionOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t dim,
    std::optional<uint32_t> clusterAxis, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto meshPartitionOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::mesh_partition, device, inputSpec, dim,
                            clusterAxis,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(meshPartitionOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

} // namespace mlir::tt::ttnn::op_model
