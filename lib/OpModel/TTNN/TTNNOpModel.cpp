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
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute.hpp"
#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute_utils.hpp"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
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

// Macros to wrap overloaded functions for use with
// query_op_constraints/runtime. These create a generic lambda that forwards
// arguments, letting the compiler resolve the correct overload based on the
// actual argument types.
// clang-format off
#define WRAP_OP(op)                                                         \
  [](auto &&...args) -> decltype(op(std::forward<decltype(args)>(args)...)) {  \
    return op(std::forward<decltype(args)>(args)...);                          \
  }

#define QUERY_OP_CONSTRAINTS(op, device, ...)                                  \
  ::ttnn::graph::query_op_constraints(WRAP_OP(op), device, __VA_ARGS__)

#define QUERY_OP_CONSTRAINTS_WITH_STATE(op, device, state, ...)                \
  ::ttnn::graph::query_op_constraints_with_optional_state(WRAP_OP(op), device,  \
                                                          state, __VA_ARGS__)

#define QUERY_OP_RUNTIME(op, device, ...)                                      \
  ::ttnn::graph::query_op_runtime(WRAP_OP(op), device, __VA_ARGS__)
// clang-format on

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
    // The query can throw from the backend allocator itself (e.g. the stateful
    // override_mock_allocator_state failing to apply the accumulated live
    // records to the target L1 layout) rather than returning a failed status.
    // Surface the message and degrade to an error result so the spill manager's
    // fallback (demote-to-DRAM / handleOOM) can recover, instead of aborting.
    // The message is classified downstream in OpConstraintValidation
    // (see https://github.com/tenstorrent/tt-mlir/issues/9045): an "Out of
    // Memory" substring becomes an OOM result, anything else a backend error.
    llvm::errs() << "Exception thrown during op constraints query: " << e.what()
                 << "\n";
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        std::string("Exception thrown during op constraints query: ") +
            e.what());
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

/**
 * @brief Stateful variant of executeConstraintQuery.
 *
 * Mirrors executeConstraintQuery exactly (same ProgramCacheState +
 * disable_and_clear_program_cache + LogLevelGuard + try/catch), but the
 * callable yields a QueryOutput (response + new allocator state). Validation is
 * performed against query.response, and the whole QueryOutput is returned on
 * success.
 *
 * @param callable A callable object that performs the stateful query.
 * @return A QueryOutput if successful, or an error.
 */
template <class Callable>
llvm::Expected<::ttnn::graph::QueryOutput>
executeConstraintQueryWithState(Callable &callable) {
  ::ttnn::graph::QueryOutput query;
  try {
    auto *device = SingletonDeviceContext::getInstance().getDevice();
    ::ttnn::graph::detail::LogLevelGuard log_guard(
        spdlog::level::level_enum::off);
    ProgramCacheState pcState(device);
    device->disable_and_clear_program_cache();
    query = callable();
  } catch (const std::exception &e) {
    // The query can throw from the backend allocator itself (e.g. the stateful
    // override_mock_allocator_state failing to apply the accumulated live
    // records to the target L1 layout) rather than returning a failed status.
    // Surface the message and degrade to an error result so the spill manager's
    // fallback (demote-to-DRAM / handleOOM) can recover, instead of aborting.
    // The message is classified downstream in OpConstraintValidation
    // (see https://github.com/tenstorrent/tt-mlir/issues/9045): an "Out of
    // Memory" substring becomes an OOM result, anything else a backend error.
    llvm::errs() << "Exception thrown during op constraints query: " << e.what()
                 << "\n";
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        std::string("Exception thrown during op constraints query: ") +
            e.what());
  }

  if (query.response.status != ::ttnn::graph::ExecutionStatus::Success) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Op constraint query failed with error: " +
            query.response.error_message.value_or("<error message not set>"));
  }

  if (!query.response.output_tensor_specs.has_value() ||
      query.response.output_tensor_specs->empty()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Op constraint query missing output tensor");
  }

  return query;
}

/**
 * @brief Stateful variant of getOpConstraints.
 *
 * Mirrors getOpConstraints but runs the stateful query path: it reads
 * out.response for the resource usage + output tensor specs (identical logic to
 * getOpConstraints). The build-from-records allocations
 * (out.output_allocations) will be surfaced on OpConstraints in the
 * validation-plumbing task; the optimizer consumes those per-output records,
 * not new_state.
 *
 * @param context The MLIRContext to use for creating the TTNNLayoutAttr for the
 * output tensor.
 * @param callable A callable object that performs the stateful query.
 * @return An OpConstraints or an error.
 */
template <class Callable>
llvm::Expected<OpConstraints> getOpConstraintsWithState(MLIRContext *context,
                                                        Callable &callable) {

  llvm::Expected<::ttnn::graph::QueryOutput> query =
      executeConstraintQueryWithState<Callable>(callable);
  if (auto error = query.takeError()) {
    return error;
  }

  ::ttnn::graph::QueryOutput out = query.get();

  // The worker grid used to build interleaved output layouts is sourced from
  // the open device rather than threaded in from the IR: the two are equivalent
  // (the system desc that produced the IR's DeviceAttr is itself derived from
  // this grid), and this is the only place the value is consumed. The context
  // caches it across device open/reset, so this is a cheap lookup.
  const llvm::ArrayRef<int64_t> deviceGrid =
      SingletonDeviceContext::getInstance().getComputeGridShape();

  llvm::SmallVector<TTNNLayoutAttr> layoutAttrs;
  for (const auto &outputTensorSpec :
       out.response.output_tensor_specs.value()) {
    layoutAttrs.push_back(conversion::getLayoutAttrFromTensorSpec(
        context, outputTensorSpec, deviceGrid));
  }

  // Build-from-records: surface each output buffer's placement as a tt-mlir
  // mirror of tt-metal's AllocationRecord. The L1 spill path keeps these for
  // still-live tensors and rebuilds allocator state from them (it does not
  // thread new_state).
  llvm::SmallVector<OpModelAllocationRecord> outputAllocations;
  outputAllocations.reserve(out.output_allocations.size());
  for (const auto &record : out.output_allocations) {
    outputAllocations.push_back(
        OpModelAllocationRecord{conversion::getBufferType(record.buffer_type),
                                static_cast<uint64_t>(record.address),
                                static_cast<uint64_t>(record.size_per_bank)});
  }

  return OpConstraints(out.response.resource_usage.cb_peak_size_per_core,
                       out.response.resource_usage.l1_buffers_peak_per_core,
                       out.response.resource_usage.peak_memory_usage_per_core,
                       out.response.resource_usage.l1_output_buffer_per_core,
                       layoutAttrs, std::move(outputAllocations));
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

// Returns true if the matmul program config already carries a fused
// activation.
inline bool programCarriesFusedActivation(
    const std::optional<::ttnn::operations::matmul::MatmulProgramConfig> &pc) {
  if (!pc) {
    return false;
  }
  return std::visit(
      [](const auto &cfg) -> bool {
        using T = std::decay_t<decltype(cfg)>;
        if constexpr (
            std::is_same_v<T, ::ttnn::operations::matmul::
                                  MatmulMultiCoreReuseMultiCastProgramConfig> ||
            std::is_same_v<T,
                           ::ttnn::operations::matmul::
                               MatmulMultiCoreReuseMultiCast1DProgramConfig> ||
            std::is_same_v<
                T, ::ttnn::operations::matmul::
                       MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
            std::is_same_v<
                T,
                ::ttnn::operations::matmul::
                    MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
          return cfg.fused_activation.has_value();
        }
        return false;
      },
      *pc);
}

} // namespace detail
#endif // TTMLIR_ENABLE_OPMODEL

std::shared_ptr<MockAllocatorState>
buildInitialState(llvm::ArrayRef<OpModelAllocationRecord> liveRecords) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Build a mock allocator state even when there are no live allocations.
  // An empty state gives the same fit decision as the stateless query, but it
  // routes through the stateful (with_initial_state) query branch, which is the
  // ONLY branch that reports output_allocations. The spill path bootstraps its
  // record set from those per-op allocations; returning nullptr here would take
  // the stateless branch (no allocations reported), so the record set could
  // never seed off the first op and every subsequent query would also see an
  // empty live set -- a permanent, silent degradation to stateless behavior.
  std::vector<::tt::tt_metal::experimental::AllocationRecord> metalRecords;
  metalRecords.reserve(liveRecords.size());
  for (const OpModelAllocationRecord &record : liveRecords) {
    metalRecords.push_back(::tt::tt_metal::experimental::AllocationRecord{
        conversion::getBufferType(record.bufferType),
        static_cast<::tt::tt_metal::DeviceAddr>(record.address),
        static_cast<::tt::tt_metal::DeviceAddr>(record.sizePerBank)});
  }

  // RCA diagnostic (https://github.com/tenstorrent/tt-mlir/issues/9045
  // follow-up): the Blackhole llama crash is override_mock_allocator_state
  // failing to apply this record set to the target L1 layout. Set
  // TTMLIR_SPILL_STATE_DEBUG=1 to dump the record set built for each stateful
  // query; the last set printed before an "Exception thrown during op
  // constraints query" line is the one that failed to apply.
  if (::getenv("TTMLIR_SPILL_STATE_DEBUG")) {
    uint64_t l1Total = 0;
    for (const OpModelAllocationRecord &record : liveRecords) {
      if (record.bufferType == BufferType::L1) {
        l1Total += record.sizePerBank;
      }
    }
    llvm::errs() << "[spill-state] applying " << liveRecords.size()
                 << " live records (L1 total/bank=" << l1Total << "B):\n";
    for (const OpModelAllocationRecord &record : liveRecords) {
      llvm::errs() << "[spill-state]   bufferType="
                   << static_cast<int>(record.bufferType)
                   << " address=" << record.address
                   << " sizePerBank=" << record.sizePerBank << "\n";
    }
  }

  // The base state is a bank-config donor extracted from the open (mock)
  // device; with_allocations replaces its regions with `metalRecords`,
  // reproducing real placement/fragmentation at those addresses.
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ::tt::tt_metal::experimental::MockAllocatorState base =
      ::tt::tt_metal::experimental::extract_mock_allocator_state(*device);
  return std::make_shared<MockAllocatorState>(
      base.with_allocations(metalRecords));
#else
  return nullptr;
#endif // TTMLIR_ENABLE_OPMODEL
}

#ifdef TTMLIR_ENABLE_OPMODEL
namespace {
// Snapshot of the mock allocator state taken before a batch of stateful spill
// queries, so it can be restored exactly afterward. Single mock device per
// compile; snapshot is always paired with a restore by the spill pass.
std::optional<::tt::tt_metal::experimental::MockAllocatorState>
    g_spillAllocatorSnapshot;
} // namespace
#endif

void snapshotMockAllocatorState() {
#ifdef TTMLIR_ENABLE_OPMODEL
  auto *device = SingletonDeviceContext::getInstance().getDevice();
  if (::tt::tt_metal::experimental::get_mock_allocator(*device) == nullptr) {
    g_spillAllocatorSnapshot.reset();
    return;
  }
  g_spillAllocatorSnapshot =
      ::tt::tt_metal::experimental::extract_mock_allocator_state(*device);
#endif
}

void restoreMockAllocatorState() {
#ifdef TTMLIR_ENABLE_OPMODEL
  if (!g_spillAllocatorSnapshot.has_value()) {
    return;
  }
  // The stateful (build-from-records) query mutates the SHARED mock device's
  // allocator (override_mock_allocator_state) and does not restore it. Restore
  // the exact pre-spill snapshot so later stateless op-model queries (e.g. the
  // conv2d config search in OperationValidationAndFallback, or pool/conv
  // constraint queries) run against the same clean device main sees. A partial
  // reset (clearing allocations only) is NOT enough: residual allocator state
  // flips op-model validity (e.g. makes conv2d act_block_h_override=0
  // spuriously legal), producing wrong configs and corrupt output.
  auto *device = SingletonDeviceContext::getInstance().getDevice();
  ::tt::tt_metal::experimental::override_mock_allocator_state(
      *device, *g_spillAllocatorSnapshot);
  g_spillAllocatorSnapshot.reset();
#endif
}

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
    bool transpose, llvm::ArrayRef<int32_t> output_padding = {}) {
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

  std::array<uint32_t, 2> outputPaddingArr = {0, 0};
  if (!output_padding.empty()) {
    outputPaddingArr =
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding);
  }
  auto prepareConvTranspose2dWeightsOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_weights,
        device, weightTensor, inputSpec.memory_config(), inputSpec.layout(),
        "IOHW", in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding), outputPaddingArr,
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

//===----------------------------------------------------------------------===//
// PrepareMoEComputeW0W1WeightsOp / PrepareMoEComputeW2WeightsOp
//===----------------------------------------------------------------------===//

// tt-metal exposes no single weight-prep entry, so the packer + bf4 quantize +
// bank-permuted memory config are composed here and traced by
// query_op_constraints to derive the packed output spec. The runtime prepare
// ops compose the identical sequence. Biases arrive (L, E, intermediate) /
// (L, E, hidden) — the moe_compute verifier enforces that, so they're forwarded
// verbatim.
static ::ttnn::Tensor
moeComputePackW0W1(const ::ttnn::Tensor &w0, const ::ttnn::Tensor &w1,
                   std::optional<::ttnn::Tensor> b0,
                   std::optional<::ttnn::Tensor> b1, uint32_t hiddenSize,
                   uint32_t intermediateSize, ::ttnn::MeshDevice *device) {
  uint32_t L = w0.logical_shape()[0];
  uint32_t E = w0.logical_shape()[1];
  bool hasBias = b0.has_value();
  ::ttnn::Tensor packed =
      hasBias ? ::ttnn::experimental::prepare_w0_w1_tensor_with_bias(
                    w0, w1, *b0, *b1, L, E, hiddenSize, intermediateSize)
              : ::ttnn::experimental::prepare_w0_w1_tensor_for_moe_compute(
                    w0, w1, L, E, hiddenSize, intermediateSize);
  return ::ttnn::experimental::quantize_weights_via_host(
      packed, ::tt::tt_metal::DataType::BFLOAT4_B,
      ::ttnn::experimental::get_weight_mem_configs(device, L, E, hiddenSize,
                                                   intermediateSize, hasBias)
          .w0_w1);
}

static ::ttnn::Tensor moeComputePackW2(const ::ttnn::Tensor &w2,
                                       std::optional<::ttnn::Tensor> b2,
                                       uint32_t hiddenSize,
                                       uint32_t intermediateSize,
                                       ::ttnn::MeshDevice *device) {
  uint32_t L = w2.logical_shape()[0];
  uint32_t E = w2.logical_shape()[1];
  bool hasBias = b2.has_value();
  ::ttnn::Tensor packed =
      hasBias ? ::ttnn::experimental::prepare_w2_tensor_with_bias(
                    w2, *b2, L, E, intermediateSize, hiddenSize)
              : ::ttnn::experimental::prepare_w2_tensor_for_moe_compute(
                    w2, L, E, intermediateSize, hiddenSize);
  return ::ttnn::experimental::quantize_weights_via_host(
      packed, ::tt::tt_metal::DataType::BFLOAT4_B,
      ::ttnn::experimental::get_weight_mem_configs(device, L, E, hiddenSize,
                                                   intermediateSize, hasBias)
          .w2);
}

// Constraint-query closure shared by the spec getter and getOpConstraints.
static auto makePrepareMoEComputeW0W1WeightsQuery(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  return [=]() {
    ::ttnn::TensorSpec w0Spec = conversion::getTensorSpec(w0Shape, w0Layout);
    ::ttnn::TensorSpec w1Spec = conversion::getTensorSpec(w1Shape, w1Layout);
    std::optional<::ttnn::TensorSpec> b0Spec;
    if (bias0Shape && bias0Layout) {
      b0Spec = conversion::getTensorSpec(*bias0Shape, *bias0Layout);
    }
    std::optional<::ttnn::TensorSpec> b1Spec;
    if (bias1Shape && bias1Layout) {
      b1Spec = conversion::getTensorSpec(*bias1Shape, *bias1Layout);
    }
    return ::ttnn::graph::query_op_constraints(
        WRAP_OP(moeComputePackW0W1), device, w0Spec, w1Spec, b0Spec, b1Spec,
        hiddenSize, intermediateSize, device);
  };
}

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW0W1WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  auto query = makePrepareMoEComputeW0W1WeightsQuery(
      w0Shape, w0Layout, w1Shape, w1Layout, bias0Shape, bias0Layout, bias1Shape,
      bias1Layout, hiddenSize, intermediateSize);
  auto output = operation::executeConstraintQuery(query);
  if (!output) {
    return output.takeError();
  }
  assert(output.get().output_tensor_specs.has_value() &&
         !output.get().output_tensor_specs->empty());
  return output.get().output_tensor_specs.value()[0];
}

static auto makePrepareMoEComputeW2WeightsQuery(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  return [=]() {
    ::ttnn::TensorSpec w2Spec = conversion::getTensorSpec(w2Shape, w2Layout);
    std::optional<::ttnn::TensorSpec> b2Spec;
    if (bias2Shape && bias2Layout) {
      b2Spec = conversion::getTensorSpec(*bias2Shape, *bias2Layout);
    }
    return ::ttnn::graph::query_op_constraints(
        WRAP_OP(moeComputePackW2), device, w2Spec, b2Spec, hiddenSize,
        intermediateSize, device);
  };
}

llvm::Expected<::ttnn::TensorSpec>
getPrepareMoEComputeW2WeightsOpOutputTensorSpec(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  auto query = makePrepareMoEComputeW2WeightsQuery(
      w2Shape, w2Layout, bias2Shape, bias2Layout, hiddenSize, intermediateSize);
  auto output = operation::executeConstraintQuery(query);
  if (!output) {
    return output.takeError();
  }
  assert(output.get().output_tensor_specs.has_value() &&
         !output.get().output_tensor_specs->empty());
  return output.get().output_tensor_specs.value()[0];
}

// Query tt-metal for the moe_compute tilize-drain core (the single L1 core the
// fused kernel allocates its expert indices/scores CBs against) and return it
// as a single-core CoreRangeSetAttr. The core depends on the device's DRAM-bank
// to worker assignment + arch, so this must run against an initialized device.
CoreRangeSetAttr computeMoeTilizeDrainCoreRangeSet(
    ::mlir::MLIRContext *context, uint32_t outputHeightShardDim,
    uint32_t hiddenSize, CoreRangeSetAttr muxCoreRangeSet) {
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // data-parallel cores = largest divisor of hidden_tiles <= 4 (mirrors the
  // device op and the former runtime drain-core query).
  uint32_t hiddenTiles = hiddenSize / 32;
  uint32_t numDataParallelCores = 1;
  for (uint32_t d = 4; d >= 1; --d) {
    if (hiddenTiles % d == 0) {
      numDataParallelCores = d;
      break;
    }
  }

  ::tt::tt_metal::CoreRangeSet muxCrs =
      conversion::getCoreRangeSet(muxCoreRangeSet);
  ::ttnn::CoreCoord drainCore = ::ttnn::experimental::get_moe_tilize_drain_core(
      device, outputHeightShardDim, numDataParallelCores, hiddenSize, muxCrs);

  ::tt::tt_metal::CoreRangeSet drainCrs(
      ::tt::tt_metal::CoreRange(drainCore, drainCore));
  return conversion::getCoreRangeSet(context, drainCrs);
}

#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints>
OpModel<PrepareMoEComputeW0W1WeightsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  return getOpConstraintsWithState(
      w0Shape, w0Layout, w1Shape, w1Layout, bias0Shape, bias0Layout, bias1Shape,
      bias1Layout, hiddenSize, intermediateSize, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareMoEComputeW0W1WeightsOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> w0Shape, TTNNLayoutAttr w0Layout,
    llvm::ArrayRef<int64_t> w1Shape, TTNNLayoutAttr w1Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias0Shape,
    std::optional<TTNNLayoutAttr> bias0Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias1Shape,
    std::optional<TTNNLayoutAttr> bias1Layout, uint32_t hiddenSize,
    uint32_t intermediateSize, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto query = [=]() {
    ::ttnn::TensorSpec w0Spec = conversion::getTensorSpec(w0Shape, w0Layout);
    ::ttnn::TensorSpec w1Spec = conversion::getTensorSpec(w1Shape, w1Layout);
    std::optional<::ttnn::TensorSpec> b0Spec;
    if (bias0Shape && bias0Layout) {
      b0Spec = conversion::getTensorSpec(*bias0Shape, *bias0Layout);
    }
    std::optional<::ttnn::TensorSpec> b1Spec;
    if (bias1Shape && bias1Layout) {
      b1Spec = conversion::getTensorSpec(*bias1Shape, *bias1Layout);
    }
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        moeComputePackW0W1, device, initialStateOpt, w0Spec, w1Spec, b0Spec,
        b1Spec, hiddenSize, intermediateSize, device);
  };
  return operation::getOpConstraintsWithState(w0Layout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<OpConstraints>
OpModel<PrepareMoEComputeW2WeightsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize) {
  return getOpConstraintsWithState(w2Shape, w2Layout, bias2Shape, bias2Layout,
                                   hiddenSize, intermediateSize,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareMoEComputeW2WeightsOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> w2Shape, TTNNLayoutAttr w2Layout,
    std::optional<llvm::ArrayRef<int64_t>> bias2Shape,
    std::optional<TTNNLayoutAttr> bias2Layout, uint32_t hiddenSize,
    uint32_t intermediateSize, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto query = [=]() {
    ::ttnn::TensorSpec w2Spec = conversion::getTensorSpec(w2Shape, w2Layout);
    std::optional<::ttnn::TensorSpec> b2Spec;
    if (bias2Shape && bias2Layout) {
      b2Spec = conversion::getTensorSpec(*bias2Shape, *bias2Layout);
    }
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        moeComputePackW2, device, initialStateOpt, w2Spec, b2Spec, hiddenSize,
        intermediateSize, device);
  };
  return operation::getOpConstraintsWithState(w2Layout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Unary Eltwise Ops
//===----------------------------------------------------------------------===//

// Cache-facing stateless entry. Its signature must stay exactly as the op-model
// cache's getOrCompute invokes it (by function pointer), so it cannot grow
// params; it forwards to the shared stateful body with a null state. The null
// path runs query_op_constraints_with_optional_state(nullopt), which metal
// dispatches straight to the stateless query.
template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseOpModel<OpTy>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

// Shared body. Stateful entry (L1 spill path); bypasses the op-model cache.
template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL

  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        detail::getOpSymbol<OpTy>(), device, initialStateOpt, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
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

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec,
        detail::getNullableMemoryConfig(outputLayout));
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
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

template <typename OpTy>
llvm::Expected<OpConstraints>
UnaryEltwiseWithFastApproxModeOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  bool fastApproxMode = true;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        detail::getOpSymbol<OpTy>(), device, initialStateOpt, inputSpec,
        fastApproxMode, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
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

  bool fastApproxMode = true;

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, fastApproxMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for UnaryEltwiseOpModel.
template struct UnaryEltwiseOpModel<ReluOp>;
template struct UnaryEltwiseOpModel<Relu6Op>;
template struct UnaryEltwiseOpModel<HardsigmoidOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<SqrtOp>;
template struct UnaryEltwiseOpModel<SinOp>;
template struct UnaryEltwiseOpModel<AbsOp>;
template struct UnaryEltwiseOpModel<CosOp>;
template struct UnaryEltwiseOpModel<TanhOp>;
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
template struct UnaryEltwiseOpModel<CbrtOp>;
template struct UnaryEltwiseOpModel<BitwiseNotOp>;
template struct UnaryEltwiseOpModel<SiluOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<MishOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<Log1pOp>;
template struct UnaryEltwiseOpModel<Expm1Op>;
template struct UnaryEltwiseWithFastApproxModeOpModel<RsqrtOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ErfOp>;
template struct UnaryEltwiseOpModel<ErfcOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<ExpOp>;
template struct UnaryEltwiseWithFastApproxModeOpModel<GeluOp>;

//===----------------------------------------------------------------------===//
// SigmoidOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<SigmoidOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                     TTNNLayoutAttr inputLayout,
                                     TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<SigmoidOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Add default parameters
  int32_t vectorMode =
      static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  auto sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::sigmoid, device, initialStateOpt, inputSpec, vectorMode,
        sigmoidMode, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
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

  // Add default parameters
  int32_t vectorMode =
      static_cast<int32_t>(::ttnn::operations::unary::VecMode::RC);
  auto sigmoidMode = ::ttnn::operations::unary::SigmoidMode::ACCURATE;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::sigmoid, device, inputSpec, vectorMode,
                            sigmoidMode,
                            detail::getNullableMemoryConfig(outputLayout));
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
  return getOpConstraintsWithState(inputShape, inputLayout, slope, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<LeakyReluOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::APFloat slope, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::leaky_relu, device, initialStateOpt, inputSpec,
        slope.convertToFloat(), detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  auto leakyReluOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::leaky_relu, device, inputSpec,
                            slope.convertToFloat(),
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(leakyReluOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Binary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryEltwiseOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr opDtypeAttr) {
  return getOpConstraintsWithState(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, outputLayout, opDtypeAttr,
                                   /*initialState=*/nullptr);
}

template <typename OpTy>
llvm::Expected<OpConstraints>
BinaryEltwiseOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr opDtypeAttr,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  if (!outputDType && opDtypeAttr) {
    outputDType = conversion::getDataType(opDtypeAttr.getValue());
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        detail::getOpSymbol<OpTy>(), device, initialStateOpt, inputSpecA,
        inputSpecB, outputDType, outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(), query);
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

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB, outputDType,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<OpConstraints> BinaryCompositeOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr opDtypeAttr) {
  return getOpConstraintsWithState(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, outputLayout, opDtypeAttr,
                                   /*initialState=*/nullptr);
}

template <typename OpTy>
llvm::Expected<OpConstraints>
BinaryCompositeOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, ttcore::DataTypeAttr /*opDtypeAttr*/,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(detail::getOpSymbol<OpTy>(), device,
                                           initialStateOpt, inputSpecA,
                                           inputSpecB, outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(), query);
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

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

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

//===----------------------------------------------------------------------===//
// GeluBackwardOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<GeluBackwardOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::string approximate, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, approximate, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<GeluBackwardOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::string approximate, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::gelu_bw, device, initialStateOpt, inputSpecA,
        inputSpecB, approximate, outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(), query);
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
llvm::Expected<OpConstraints> OpModel<PowScalarOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute exponent, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, exponent,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<PowScalarOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute exponent, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Helper lambda to create the query with any exponent value type.
  auto powScalarQuery = [=](auto convertedExponent) {
    return [=]() {
      return QUERY_OP_CONSTRAINTS_WITH_STATE(
          ::ttnn::pow, device, initialStateOpt, inputSpec, convertedExponent,
          detail::getNullableMemoryConfig(outputLayout));
    };
  };

  // The invoke function of PowScalarOp is templated over the exponent value
  // type. That's why the following code is arranged in this way.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(exponent)) {
    int32_t convertedExponent = static_cast<int32_t>(value.getInt());
    auto query = powScalarQuery(convertedExponent);
    return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                                query);
  }
  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(exponent)) {
    float convertedExponent = value.getValue().convertToFloat();
    auto query = powScalarQuery(convertedExponent);
    return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                                query);
  }
  return llvm::createStringError("Invalid exponent");
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

  // Helper lambda to create the query with any exponent value type.
  auto powScalarQuery = [=](auto convertedExponent) {
    return [=]() {
      return QUERY_OP_RUNTIME(::ttnn::pow, device, inputSpec, convertedExponent,
                              detail::getNullableMemoryConfig(outputLayout));
    };
  };

  // The invoke function of PowScalarOp is templated over the exponent value
  // type. That's why the following code is arranged in this way.
  if (auto value = mlir::dyn_cast<mlir::IntegerAttr>(exponent)) {
    int32_t convertedExponent = static_cast<int32_t>(value.getInt());
    auto query = powScalarQuery(convertedExponent);
    return operation::getOpRuntime(query);
  }
  if (auto value = mlir::dyn_cast<mlir::FloatAttr>(exponent)) {
    float convertedExponent = value.getValue().convertToFloat();
    auto query = powScalarQuery(convertedExponent);
    return operation::getOpRuntime(query);
  }

  return llvm::createStringError("Invalid exponent");
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Ternary Eltwise Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> TernaryEltwiseOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, inputShapeC, inputLayoutC,
                                   outputLayout, /*initialState=*/nullptr);
}

template <typename OpTy>
llvm::Expected<OpConstraints>
TernaryEltwiseOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    llvm::ArrayRef<int64_t> inputShapeC, TTNNLayoutAttr inputLayoutC,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        detail::getOpSymbol<OpTy>(), device, initialStateOpt, inputSpecA,
        inputSpecB, inputSpecC, outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(), query);
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

  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(detail::getOpSymbol<OpTy>(), device,
                                           inputSpecA, inputSpecB, inputSpecC,
                                           outputMemoryConfig);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for TernaryEltwiseOpModel.
template struct TernaryEltwiseOpModel<WhereOp>;

//===----------------------------------------------------------------------===//
// Reduction Ops
//===----------------------------------------------------------------------===//

template <typename OpTy>
llvm::Expected<OpConstraints> ReductionOpModel<OpTy>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dimArg, keepDim,
                                   outputLayout, /*initialState=*/nullptr);
}

template <typename OpTy>
llvm::Expected<OpConstraints> ReductionOpModel<OpTy>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttsl::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        detail::getOpSymbol<OpTy>(), device, initialStateOpt, inputSpec,
        dimArgConverted, keepDim, detail::getNullableMemoryConfig(outputLayout),
        /*compute_kernel_config=*/std::nullopt,
        /*scalar=*/1.0f, /*correction=*/true,
        /*sub_core_grids=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

template <typename OpTy>
llvm::Expected<size_t> ReductionOpModel<OpTy>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> dimArg, bool keepDim,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttsl::SmallVector<int>> dimArgConverted;
  if (dimArg) {
    dimArgConverted =
        conversion::convertLLVMSmallVecToTTNNSmallVec(dimArg.value());
  } else {
    dimArgConverted = std::nullopt;
  }

  // Create query closure
  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, dimArgConverted,
        keepDim, detail::getNullableMemoryConfig(outputLayout),
        /*compute_kernel_config=*/std::nullopt,
        /*scalar=*/1.0f, /*correction=*/true,
        /*sub_core_grids=*/std::nullopt);
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
llvm::Expected<OpConstraints> OpModel<SoftmaxOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dimArg,
                                   numericStable, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<SoftmaxOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto softmaxOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::softmax, device, initialStateOpt, inputSpec, dimArg,
        detail::getNullableMemoryConfig(outputLayout),
        std::nullopt, // compute_kernel_config,
        numericStable);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              softmaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<SoftmaxOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dimArg, bool numericStable, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto softmaxOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::softmax, device, inputSpec, dimArg,
                            detail::getNullableMemoryConfig(outputLayout),
                            std::nullopt, // compute_kernel_config,
                            numericStable);
  };

  return operation::getOpRuntime(softmaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ScatterOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout,
    llvm::ArrayRef<int64_t> sourceShape, TTNNLayoutAttr sourceLayout,
    int32_t dim, std::optional<ttcore::ReduceTypeAttr> optReduction,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      inputShape, inputLayout, indexShape, indexLayout, sourceShape,
      sourceLayout, dim, optReduction, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ScatterOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout,
    llvm::ArrayRef<int64_t> sourceShape, TTNNLayoutAttr sourceLayout,
    int32_t dim, std::optional<ttcore::ReduceTypeAttr> optReduction,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  // Convert optReduction to ScatterReductionType enum
  auto optReductionType = conversion::getScatterReductionType(optReduction);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  //  Create query closure
  auto scatterOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::scatter, device, initialStateOpt, inputSpec, dim, indexSpec,
        sourceSpec, detail::getNullableMemoryConfig(outputLayout),
        optReductionType,
        /* sub_core_grid */ std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              scatterOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ScatterOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout,
    llvm::ArrayRef<int64_t> sourceShape, TTNNLayoutAttr sourceLayout,
    int32_t dim, std::optional<ttcore::ReduceTypeAttr> optReduction,
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

  auto optReductionType = conversion::getScatterReductionType(optReduction);

  //  Create query closure
  auto scatterOpRuntimeQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::scatter, device, inputSpec, dim, indexSpec, sourceSpec,
        detail::getNullableMemoryConfig(outputLayout), optReductionType,
        /* sub_core_grid */ std::nullopt);
  };

  return operation::getOpRuntime(scatterOpRuntimeQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ReshapeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, outputShape,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ReshapeOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> outputShape, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto reshapeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::reshape, device, initialStateOpt, inputSpec,
        conversion::getShape(outputShape),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              reshapeOpQuery);
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

  // Create query closure
  auto reshapeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::reshape, device, inputSpec,
                            conversion::getShape(outputShape),
                            detail::getNullableMemoryConfig(outputLayout));
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
  return getOpConstraintsWithState(inputShape, inputLayout, begins, ends, step,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<SliceStaticOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> begins, llvm::ArrayRef<int64_t> ends,
    llvm::ArrayRef<int64_t> step, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto sliceOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::slice, device, initialStateOpt, inputSpec, beginsSpan, endsSpan,
        stepSpan, detail::getNullableMemoryConfig(outputLayout), std::nullopt,
        std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              sliceOpQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, beginsShape,
                                   beginsLayout, endsShape, endsLayout, step,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<SliceDynamicOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> beginsShape, TTNNLayoutAttr beginsLayout,
    llvm::ArrayRef<int64_t> endsShape, TTNNLayoutAttr endsLayout,
    std::optional<llvm::SmallVector<int64_t>> step, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure to make a call to the static version of the op:
  auto sliceOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::slice, device, initialStateOpt, inputSpec, beginsVec, endsVec,
        stepVec, detail::getNullableMemoryConfig(outputLayout), outputSpec,
        padValue);
  };
  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              sliceOpQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, dtype, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<BitcastConvertOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto bitcastOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::bitcast, device, initialStateOpt, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              bitcastOpQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, dtype, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<TypecastOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    ttcore::DataTypeAttr dtype, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto typecastOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::typecast, device, initialStateOpt, inputSpec,
        conversion::getDataType(dtype.getValue()),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              typecastOpQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, outputDtype,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ToLayoutOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto toLayoutOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::to_layout, device, initialStateOpt, inputSpec,
        conversion::getPageLayout(outputLayout.getLayout()), dtype,
        detail::getNullableMemoryConfig(outputLayout));
  };
  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              toLayoutOpQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<ToMemoryConfigOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto toMemoryConfigOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::to_memory_config, device, initialStateOpt, inputSpec,
        conversion::getMemoryConfig(MemoryConfigAttr::get(outputLayout)));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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
llvm::Expected<OpConstraints> OpModel<ConcatOp>::getOpConstraints(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShapes, inputLayouts, dim, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ConcatOp>::getOpConstraintsWithState(
    std::vector<llvm::ArrayRef<int64_t>> inputShapes,
    std::vector<TTNNLayoutAttr> inputLayouts, const int dim,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto concatOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::concat, device, initialStateOpt, inputSpecs, dim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayouts[0].getContext(),
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

  // Create query closure
  auto concatOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::concat, device, inputSpecs, dim,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(concatOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<TransposeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dim0, const int dim1, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim0, dim1,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<TransposeOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int dim0, const int dim1, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto transposeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transpose, device, initialStateOpt, inputSpec,
        static_cast<int64_t>(dim0), static_cast<int64_t>(dim1),
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  auto transposeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::transpose, device, inputSpec,
                            static_cast<int64_t>(dim0),
                            static_cast<int64_t>(dim1),
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(transposeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// CumSumOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<CumSumOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int32_t dim, std::optional<ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim, dtype,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<CumSumOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int32_t dim, std::optional<ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::DataType> ttnnDtype = std::nullopt;
  if (dtype) {
    ttnnDtype = conversion::getDataType(*dtype);
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto cumSumOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::cumsum, device, initialStateOpt, inputSpec, dim, ttnnDtype,
        false, std::nullopt, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              cumSumOpQuery);
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

  std::optional<::ttnn::DataType> ttnnDtype = std::nullopt;
  if (dtype) {
    ttnnDtype = conversion::getDataType(*dtype);
  }

  // Create query closure
  auto cumSumOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::cumsum, device, inputSpec, dim, ttnnDtype,
                            false, std::nullopt,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(cumSumOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// CumProdOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<CumProdOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int32_t dim, std::optional<ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim, dtype,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<CumProdOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const int32_t dim, std::optional<ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::DataType> ttnnDtype = std::nullopt;
  if (dtype) {
    ttnnDtype = conversion::getDataType(*dtype);
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto cumProdOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::cumprod, device, initialStateOpt, inputSpec, dim, ttnnDtype,
        /*reverse_order=*/false,
        /*optional_out=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              cumProdOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<CumProdOp>::getOpRuntime(llvm::ArrayRef<int64_t> inputShape,
                                 TTNNLayoutAttr inputLayout, const int32_t dim,
                                 std::optional<ttcore::DataType> dtype,
                                 TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::DataType> ttnnDtype = std::nullopt;
  if (dtype) {
    ttnnDtype = conversion::getDataType(*dtype);
  }

  auto cumProdOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::cumprod, device, inputSpec, dim, ttnnDtype,
                            /*reverse_order=*/false,
                            /*optional_out=*/std::nullopt,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(cumProdOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConcatenateHeadsOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<ConcatenateHeadsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<ConcatenateHeadsOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto concatenateHeadsOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::concatenate_heads, device, initialStateOpt,
        inputSpec, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  auto concatenateHeadsOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::transformer::concatenate_heads, device,
                            inputSpec,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(concatenateHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//
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
  return getOpConstraintsWithState(
      queryShape, queryLayout, keyShape, keyLayout, valueShape, valueLayout,
      isCausal, attentionMaskShape, attentionMaskLayout, curPosTensorShape,
      curPosTensorLayout, attentionSinkShape, attentionSinkLayout, scale,
      programConfig, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionDecodeOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;
  std::optional<uint32_t> slidingWindowSize = std::nullopt;

  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  const std::vector<uint32_t> curPosEmpty = {};

  auto sdpaProgramConfigConverted =
      conversion::getSDPAProgramConfig(programConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::scaled_dot_product_attention_decode, device,
        initialStateOpt, querySpec, keySpec, valueSpec, isCausal,
        attentionMaskSpec, curPosEmpty, curPosTensorSpec, attentionSinkSpec,
        scaleFloat, slidingWindowSize,
        detail::getNullableMemoryConfig(outputLayout),
        sdpaProgramConfigConverted,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(
      queryLayout.getContext(), scaledDotProductAttentionDecodeOpQuery);
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  // The current position information is required for this op. It can either be
  // passed as a tensor or as a uint vector. The uint vector is not wrapped in a
  // std::optional so we must pass an empty vector.
  const std::vector<uint32_t> curPosEmpty = {};
  auto scaledDotProductAttentionDecodeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::scaled_dot_product_attention_decode, device,
        querySpec, keySpec, valueSpec, isCausal, attentionMaskSpec, curPosEmpty,
        curPosTensorSpec, attentionSinkSpec, scaleFloat,
        /*slidingWindowSize=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(scaledDotProductAttentionDecodeOpQuery);

#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedScaledDotProductAttentionDecodeOp
//===----------------------------------------------------------------------===//

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
  return getOpConstraintsWithState(
      queryShape, queryLayout, keyShape, keyLayout, valueShape, valueLayout,
      pageTableShape, pageTableLayout, isCausal, attentionMaskShape,
      attentionMaskLayout, curPosTensorShape, curPosTensorLayout,
      attentionSinkShape, attentionSinkLayout, scale, slidingWindowSize,
      programConfig, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PagedScaledDotProductAttentionDecodeOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      sdpaProgramConfig = conversion::getSDPAProgramConfig(programConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto pagedScaledDotProductAttentionDecodeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::paged_scaled_dot_product_attention_decode, device,
        initialStateOpt, querySpec, keySpec, valueSpec, pageTableSpec, isCausal,
        attentionMaskSpec, curPosTensorSpec, attentionSinkSpec, scaleFloat,
        slidingWindowSize, detail::getNullableMemoryConfig(outputLayout),
        sdpaProgramConfig,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      sdpaProgramConfig = conversion::getSDPAProgramConfig(programConfig);

  auto pagedScaledDotProductAttentionDecodeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::paged_scaled_dot_product_attention_decode, device,
        querySpec, keySpec, valueSpec, pageTableSpec, isCausal,
        attentionMaskSpec, curPosTensorSpec, attentionSinkSpec, scaleFloat,
        slidingWindowSize, detail::getNullableMemoryConfig(outputLayout),
        sdpaProgramConfig,
        /*compute_kernel_config=*/std::nullopt);
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
static ::ttnn::operations::transformer::SDPAProgramConfig
getPagedFlashMlaDecodeProgramConfig(
    ::tt::tt_metal::distributed::MeshDevice *device) {
  ::ttnn::operations::transformer::SDPAProgramConfig programConfig{};
  programConfig.k_chunk_size = 32;
  programConfig.compute_with_storage_grid_size =
      device->compute_with_storage_grid_size();
  return programConfig;
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
  return getOpConstraintsWithState(
      queryShape, queryLayout, keyShape, keyLayout, valueShape, valueLayout,
      headDimV, pageTableShape, pageTableLayout, isCausal, attentionMaskShape,
      attentionMaskLayout, curPosTensorShape, curPosTensorLayout,
      attentionSinkShape, attentionSinkLayout, scale, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PagedFlashMultiLatentAttentionDecodeOp>::getOpConstraintsWithState(
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
    std::optional<llvm::APFloat> scale, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = getPagedFlashMlaDecodeProgramConfig(device);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto pagedFlashMlaDecodeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::paged_flash_multi_latent_attention_decode, device,
        initialStateOpt, querySpec, keySpec, valueSpec, headDimV, pageTableSpec,
        isCausal, attentionMaskSpec, curPosTensorSpec, attentionSinkSpec,
        scaleFloat,
        /*slidingWindowSize=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout), programConfig,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(queryLayout.getContext(),
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      programConfig = getPagedFlashMlaDecodeProgramConfig(device);

  auto pagedFlashMlaDecodeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::paged_flash_multi_latent_attention_decode, device,
        querySpec, keySpec, valueSpec, headDimV, pageTableSpec, isCausal,
        attentionMaskSpec, curPosTensorSpec, attentionSinkSpec, scaleFloat,
        /*slidingWindowSize=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout), programConfig,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(pagedFlashMlaDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ChunkedScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<ChunkedScaledDotProductAttentionOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    llvm::ArrayRef<int64_t> chunkStartIdxShape,
    TTNNLayoutAttr chunkStartIdxLayout, std::optional<llvm::APFloat> scale,
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
  ASSIGN_OR_RETURN(::ttnn::TensorSpec chunkStartIdxSpec,
                   detail::convertToTensorSpec(device, chunkStartIdxShape,
                                               chunkStartIdxLayout));

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      sdpaProgramConfig = conversion::getSDPAProgramConfig(programConfig);

  auto chunkedScaledDotProductAttentionOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::transformer::chunked_scaled_dot_product_attention, device,
        querySpec, keySpec, valueSpec, pageTableSpec, chunkStartIdxSpec,
        scaleFloat, detail::getNullableMemoryConfig(outputLayout),
        sdpaProgramConfig,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraints(queryLayout.getContext(),
                                     chunkedScaledDotProductAttentionOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t>
OpModel<ChunkedScaledDotProductAttentionOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    llvm::ArrayRef<int64_t> chunkStartIdxShape,
    TTNNLayoutAttr chunkStartIdxLayout, std::optional<llvm::APFloat> scale,
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
  ASSIGN_OR_RETURN(::ttnn::TensorSpec chunkStartIdxSpec,
                   detail::convertToTensorSpec(device, chunkStartIdxShape,
                                               chunkStartIdxLayout));

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;
  std::optional<::ttnn::operations::transformer::SDPAProgramConfig>
      sdpaProgramConfig = conversion::getSDPAProgramConfig(programConfig);

  auto chunkedScaledDotProductAttentionOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::chunked_scaled_dot_product_attention, device,
        querySpec, keySpec, valueSpec, pageTableSpec, chunkStartIdxSpec,
        scaleFloat, detail::getNullableMemoryConfig(outputLayout),
        sdpaProgramConfig,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(chunkedScaledDotProductAttentionOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ScaledDotProductAttentionOp
//===----------------------------------------------------------------------===//

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
  return getOpConstraintsWithState(
      queryShape, queryLayout, keyShape, keyLayout, valueShape, valueLayout,
      attentionMaskShape, attentionMaskLayout, attentionSinkShape,
      attentionSinkLayout, isCausal, scale, slidingWindowSize, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<ScaledDotProductAttentionOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    llvm::ArrayRef<int64_t> valueShape, TTNNLayoutAttr valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionSinkShape,
    std::optional<TTNNLayoutAttr> attentionSinkLayout, bool isCausal,
    std::optional<llvm::APFloat> scale,
    std::optional<uint32_t> slidingWindowSize, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto scaledDotProductAttentionOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::scaled_dot_product_attention, device,
        initialStateOpt, querySpec, keySpec, valueSpec, attentionMaskSpec,
        isCausal, scaleFloat, slidingWindowSize,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt, attentionSinkSpec);
  };

  return operation::getOpConstraintsWithState(queryLayout.getContext(),
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

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  auto scaledDotProductAttentionOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::scaled_dot_product_attention, device, querySpec,
        keySpec, valueSpec, attentionMaskSpec, isCausal, scaleFloat,
        slidingWindowSize, detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt, attentionSinkSpec);
  };

  return operation::getOpRuntime(scaledDotProductAttentionOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// FlashMlaPrefillOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<FlashMlaPrefillOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    std::optional<llvm::ArrayRef<int64_t>> valueShape,
    std::optional<TTNNLayoutAttr> valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout, uint32_t headDimV,
    bool isCausal, std::optional<llvm::APFloat> scale,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      queryShape, queryLayout, keyShape, keyLayout, valueShape, valueLayout,
      attentionMaskShape, attentionMaskLayout, headDimV, isCausal, scale,
      outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<FlashMlaPrefillOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    std::optional<llvm::ArrayRef<int64_t>> valueShape,
    std::optional<TTNNLayoutAttr> valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout, uint32_t headDimV,
    bool isCausal, std::optional<llvm::APFloat> scale,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));

  std::optional<::ttnn::TensorSpec> valueSpec =
      detail::convertToOptionalTensorSpec(device, valueShape, valueLayout);
  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto flashMlaPrefillOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::flash_mla_prefill, device, initialStateOpt,
        querySpec, keySpec, headDimV, valueSpec, attentionMaskSpec, isCausal,
        scaleFloat, detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(queryLayout.getContext(),
                                              flashMlaPrefillOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<FlashMlaPrefillOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> queryShape, TTNNLayoutAttr queryLayout,
    llvm::ArrayRef<int64_t> keyShape, TTNNLayoutAttr keyLayout,
    std::optional<llvm::ArrayRef<int64_t>> valueShape,
    std::optional<TTNNLayoutAttr> valueLayout,
    std::optional<llvm::ArrayRef<int64_t>> attentionMaskShape,
    std::optional<TTNNLayoutAttr> attentionMaskLayout, uint32_t headDimV,
    bool isCausal, std::optional<llvm::APFloat> scale,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec querySpec,
      detail::convertToTensorSpec(device, queryShape, queryLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec keySpec,
                   detail::convertToTensorSpec(device, keyShape, keyLayout));

  std::optional<::ttnn::TensorSpec> valueSpec =
      detail::convertToOptionalTensorSpec(device, valueShape, valueLayout);
  std::optional<::ttnn::TensorSpec> attentionMaskSpec =
      detail::convertToOptionalTensorSpec(device, attentionMaskShape,
                                          attentionMaskLayout);

  std::optional<float> scaleFloat =
      scale ? std::make_optional(scale.value().convertToFloat()) : std::nullopt;

  auto flashMlaPrefillOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::transformer::flash_mla_prefill, device,
                            querySpec, keySpec, headDimV, valueSpec,
                            attentionMaskSpec, isCausal, scaleFloat,
                            detail::getNullableMemoryConfig(outputLayout),
                            /*program_config=*/std::nullopt,
                            /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpRuntime(flashMlaPrefillOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===-----------------------------------------------------------------------===//
// RotaryEmbeddingLlamaOp
// ===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RotaryEmbeddingLlamaOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    llvm::ArrayRef<int64_t> transMatShape, TTNNLayoutAttr transMatLayout,
    bool isDecodeMode, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, cosShape, cosLayout,
                                   sinShape, sinLayout, transMatShape,
                                   transMatLayout, isDecodeMode, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<RotaryEmbeddingLlamaOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    llvm::ArrayRef<int64_t> transMatShape, TTNNLayoutAttr transMatLayout,
    bool isDecodeMode, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::rotary_embedding_llama, device, initialStateOpt,
        inputSpec, cosSpec, sinSpec, transMatSpec, isDecodeMode,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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
    bool isDecodeMode, TTNNLayoutAttr outputLayout) {
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

  // Create query closure
  auto rotaryEmbeddingLlamaOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::rotary_embedding_llama,
                            device, inputSpec, cosSpec, sinSpec, transMatSpec,
                            isDecodeMode,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(rotaryEmbeddingLlamaOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RotaryEmbeddingOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RotaryEmbeddingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    std::optional<uint32_t> tokenIndex, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, cosShape, cosLayout,
                                   sinShape, sinLayout, tokenIndex,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<RotaryEmbeddingOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    std::optional<uint32_t> tokenIndex, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto rotaryEmbeddingOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::rotary_embedding, device, initialStateOpt,
        inputSpec, cosSpec, sinSpec, tokenIndex,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              rotaryEmbeddingOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RotaryEmbeddingOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> cosShape, TTNNLayoutAttr cosLayout,
    llvm::ArrayRef<int64_t> sinShape, TTNNLayoutAttr sinLayout,
    std::optional<uint32_t> tokenIndex, TTNNLayoutAttr outputLayout) {
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

  // Create query closure
  auto rotaryEmbeddingOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::rotary_embedding, device,
                            inputSpec, cosSpec, sinSpec, tokenIndex,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(rotaryEmbeddingOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPCreateQKVHeadsDecodeOp
//===----------------------------------------------------------------------===//
llvm::Expected<op_model::OpConstraints>
OpModel<NLPCreateQKVHeadsDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
    std::optional<TTNNLayoutAttr> batchOffsetLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, std::optional<bool> overlapQKCoregrid,
    std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, batchOffsetShape,
                                   batchOffsetLayout, numHeads, numKVHeads,
                                   overlapQKCoregrid, sliceSize, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<op_model::OpConstraints>
OpModel<NLPCreateQKVHeadsDecodeOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchOffsetShape,
    std::optional<TTNNLayoutAttr> batchOffsetLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, std::optional<bool> overlapQKCoregrid,
    std::optional<uint32_t> sliceSize, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  std::optional<std::array<::ttnn::Tensor, 3>> optionalOutputTensors =
      std::nullopt;
  auto nlpCreateQKVHeadsDecode = [&]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::nlp_create_qkv_heads_decode, device,
        initialStateOpt, inputSpec, numHeads, numKVHeads, optionalOutputTensors,
        std::optional<const bool>(overlapQKCoregrid), batchOffsetSpec,
        sliceSize, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  std::optional<std::array<::ttnn::Tensor, 3>> optionalOutputTensors =
      std::nullopt;
  auto nlpCreateQKVHeadsDecode = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::experimental::nlp_create_qkv_heads_decode, device, inputSpec,
        numHeads, numKVHeads, optionalOutputTensors,
        std::optional<const bool>(overlapQKCoregrid), batchOffsetSpec,
        sliceSize, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(nlpCreateQKVHeadsDecode);
#else
  return llvm::createStringError("Not implemented");
#endif
}

//===----------------------------------------------------------------------===//
// SplitQueryKeyValueAndSplitHeadsOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<SplitQueryKeyValueAndSplitHeadsOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
    std::optional<TTNNLayoutAttr> inputKVLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      inputShape, inputLayout, inputKVShape, inputKVLayout, numHeads,
      numKVHeads, transposeKey, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<SplitQueryKeyValueAndSplitHeadsOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputKVShape,
    std::optional<TTNNLayoutAttr> inputKVLayout, uint32_t numHeads,
    std::optional<uint32_t> numKVHeads, bool transposeKey,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto splitQueryKeyValueAndSplitHeadsOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::transformer::split_query_key_value_and_split_heads, device,
        initialStateOpt, inputSpec, inputKVSpec, numHeads, numKVHeads,
        transposeKey, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(
      inputLayout.getContext(), splitQueryKeyValueAndSplitHeadsOpQuery);
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

  // Create query closure
  auto splitQueryKeyValueAndSplitHeadsOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::transformer::split_query_key_value_and_split_heads, device,
        inputSpec, inputKVSpec, numHeads, numKVHeads, transposeKey,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(splitQueryKeyValueAndSplitHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints>
OpModel<NLPConcatHeadsOp>::getOpConstraints(llvm::ArrayRef<int64_t> inputShape,
                                            TTNNLayoutAttr inputLayout,
                                            TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<NLPConcatHeadsOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto nlpConcatHeadsOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::nlp_concat_heads, device, initialStateOpt,
        inputSpec, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  auto nlpConcatHeadsOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::nlp_concat_heads, device,
                            inputSpec,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(nlpConcatHeadsOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// NLPConcatHeadsDecodeOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<NLPConcatHeadsDecodeOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t numHeads, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, numHeads,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<NLPConcatHeadsDecodeOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t numHeads, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // tt-metal's nlp_concat_heads_decode infers on_subcoregrids from the input
  // shard grid: if the CoreRangeSet has multiple ranges or doesn't start at
  // (0,0), it sets on_subcoregrids=true and requires sub_core_grids.
  // Compute sub_core_grids from the input layout so the subcoregrids path
  // doesn't crash on a nullopt dereference in compute_output_specs.
  std::optional<::tt::tt_metal::CoreRangeSet> subCoreGrids = std::nullopt;
  if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout() &&
      isShardedMemoryLayout(inputLayout.getMemLayout().getValue())) {
    auto coreRangeSet = conversion::getCoreRangeSet(inputLayout);
    auto ranges = coreRangeSet.ranges();
    if (ranges.size() != 1 ||
        ranges[0].start_coord != ::tt::tt_metal::CoreCoord{0, 0}) {
      subCoreGrids = coreRangeSet;
    }
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::nlp_concat_heads_decode, device, initialStateOpt,
        inputSpec, numHeads, detail::getNullableMemoryConfig(outputLayout),
        std::optional<::tt::tt_metal::Tensor>(std::nullopt), subCoreGrids);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Pass sub_core_grids when the input shard grid would trigger subcoregrids
  // (see getOpConstraints above for rationale).
  std::optional<::tt::tt_metal::CoreRangeSet> subCoreGrids = std::nullopt;
  if (inputLayout.hasL1BufferType() && inputLayout.getMemLayout() &&
      isShardedMemoryLayout(inputLayout.getMemLayout().getValue())) {
    auto coreRangeSet = conversion::getCoreRangeSet(inputLayout);
    auto ranges = coreRangeSet.ranges();
    if (ranges.size() != 1 ||
        ranges[0].start_coord != ::tt::tt_metal::CoreCoord{0, 0}) {
      subCoreGrids = coreRangeSet;
    }
  }

  // Create query closure
  auto nlpConcatHeadsDecodeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::experimental::nlp_concat_heads_decode, device, inputSpec,
        numHeads, detail::getNullableMemoryConfig(outputLayout),
        std::optional<::tt::tt_metal::Tensor>(std::nullopt), subCoreGrids);
  };

  return operation::getOpRuntime(nlpConcatHeadsDecodeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RepeatInterleaveOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RepeatInterleaveOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const unsigned int repeats, const int dim, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, repeats, dim,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<RepeatInterleaveOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    const unsigned int repeats, const int dim, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto repeatInterleaveOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::repeat_interleave, device, initialStateOpt, inputSpec, repeats,
        dim, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Create query closure
  auto repeatInterleaveOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::repeat_interleave, device, inputSpec,
                            repeats, dim,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(repeatInterleaveOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<RepeatOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> repeats, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, repeats,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<RepeatOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> repeats, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Convert repeats to ttnn::Shape
  ::ttnn::Shape repeatShape = conversion::getShape(repeats);

  // Convert output layout to memory config
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Convert Shape to SmallVector<uint32_t> to use overload with memory_config
  ::ttsl::SmallVector<uint32_t> repeatVec(repeatShape.cbegin(),
                                          repeatShape.cend());

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto repeatOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(::ttnn::repeat, device,
                                           initialStateOpt, inputSpec,
                                           repeatVec, outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              repeatOpQuery);
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

  // Convert repeats to SmallVector<uint32_t> to use overload with memory_config
  ::ttsl::SmallVector<uint32_t> repeatVec;
  repeatVec.reserve(repeats.size());
  for (int64_t r : repeats) {
    repeatVec.push_back(static_cast<uint32_t>(r));
  }

  // Convert output layout to memory config
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto repeatOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::repeat, device, inputSpec, repeatVec,
                            outputMemoryConfig);
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
/**
 * @brief Converts padding array to PadSpecDim format for TTNN operations.
 *
 * @param padding Array of padding values in [before0, after0, before1, after1,
 * ...] format
 * @return SmallVector of PadSpecDim objects
 */
static ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim>
convertPadding(llvm::ArrayRef<int32_t> padding) {
  ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> paddingSpec;
  // Reserve space to avoid memory reallocations
  paddingSpec.reserve((padding.size() + 1) / 2);

  constexpr int32_t defaultPadValue = 0;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int32_t before = padding[i];
    int32_t after = (i + 1 < padding.size()) ? padding[i + 1] : defaultPadValue;

    assert(before >= 0 && after >= 0 && "Padding values must be non-negative");

    paddingSpec.emplace_back(static_cast<uint32_t>(before),
                             static_cast<uint32_t>(after));
  }
  return paddingSpec;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<PadOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue, bool multicore,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, padding, padValue,
                                   multicore, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<PadOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int32_t> padding, llvm::APFloat padValue, bool multicore,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Convert padding to PadSpecDim format
  auto paddingSpec = convertPadding(padding);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto padOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::pad, device, initialStateOpt, inputSpec, paddingSpec,
        padValue.convertToFloat(), multicore,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              padOpQuery);
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

  // Convert padding to PadSpecDim format
  auto paddingSpec = convertPadding(padding);

  // Create query closure
  auto padOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::pad, device, inputSpec, paddingSpec,
                            padValue.convertToFloat(), multicore,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(padOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<SortOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int dim,
    bool descending, bool stable, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim, descending,
                                   stable, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<SortOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int dim,
    bool descending, bool stable, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto sortOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::sort, device, initialStateOpt, inputSpec, dim, descending,
        stable, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              sortOpQuery);
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

  // Create query closure
  auto sortOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::sort, device, inputSpec,
                            static_cast<int8_t>(dim), descending, stable,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(sortOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TopKRouterGptOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<TopKRouterGptOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout, uint32_t k,
    uint32_t numExperts, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, weightShape,
                                   weightLayout, biasShape, biasLayout, k,
                                   numExperts, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<TopKRouterGptOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> biasShape, TTNNLayoutAttr biasLayout, uint32_t k,
    uint32_t numExperts, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto topKRouterGptQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::topk_router_gpt, device, initialStateOpt,
        inputSpec, weightSpec, biasSpec, k, numExperts);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  auto topKRouterGptQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::topk_router_gpt, device,
                            inputSpec, weightSpec, biasSpec, k, numExperts);
  };

  return operation::getOpRuntime(topKRouterGptQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ArgMaxOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int32_t> dim, bool keepDim, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim, keepDim,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ArgMaxOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int32_t> dim, bool keepDim, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto argMaxOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::argmax, device, initialStateOpt, inputSpec, dim, keepDim,
        std::nullopt, detail::getNullableMemoryConfig(outputLayout),
        std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              argMaxOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<ArgMaxOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int32_t> dim, bool keepDim, TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto argMaxOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::argmax, device, inputSpec, dim, keepDim, std::nullopt,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt);
  };

  return operation::getOpRuntime(argMaxOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ProdOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ProdOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int64_t> dim, bool keepDim, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, dim, keepDim,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ProdOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<int64_t> dim, bool keepDim, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto prodOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::prod, device, initialStateOpt, inputSpec, dim, keepDim,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              prodOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Quantization Ops
//===----------------------------------------------------------------------===//

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

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    return ::ttnn::graph::query_op_constraints(
        detail::getOpSymbol<OpTy>(), device, inputSpec, scaleSpec,
        zeroPointSpec, axis, outputDType, outputMemoryConfig);
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

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto quantizationOpQuery = [=]() {
    return ::ttnn::graph::query_op_runtime(
        detail::getOpSymbol<OpTy>(), device, inputSpec, scaleSpec,
        zeroPointSpec, axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpRuntime(quantizationOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

// Explicit template instantiation for QuantizationOpModel.
template struct QuantizationOpModel<QuantizeOp>;
template struct QuantizationOpModel<DequantizeOp>;

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
  return getOpConstraintsWithState(
      inputShape, inputLayout, inScaleShape, inScaleLayout, inZeroPointShape,
      inZeroPointLayout, outScaleShape, outScaleLayout, outZeroPointShape,
      outZeroPointLayout, axis, outputDtype, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<RequantizeOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> inScaleShape, TTNNLayoutAttr inScaleLayout,
    llvm::ArrayRef<int64_t> inZeroPointShape, TTNNLayoutAttr inZeroPointLayout,
    llvm::ArrayRef<int64_t> outScaleShape, TTNNLayoutAttr outScaleLayout,
    llvm::ArrayRef<int64_t> outZeroPointShape,
    TTNNLayoutAttr outZeroPointLayout, std::optional<int32_t> axis,
    std::optional<ttcore::DataType> outputDtype, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure

  auto requantizeOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::requantize, device, initialStateOpt, inputSpec, inScaleSpec,
        inZeroPointSpec, outScaleSpec, outZeroPointSpec, axis, outputDType,
        outputMemoryConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  // Use the explicit outputDtype parameter if provided, otherwise infer from
  // layout
  std::optional<::tt::tt_metal::DataType> outputDType;
  if (outputDtype.has_value()) {
    outputDType = conversion::getDataType(outputDtype.value());
  } else {
    outputDType = detail::getNullableDataType(outputLayout);
  }
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto requantizeOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::requantize, device, inputSpec, inScaleSpec,
                            inZeroPointSpec, outScaleSpec, outZeroPointSpec,
                            axis, outputDType, outputMemoryConfig);
  };

  return operation::getOpRuntime(requantizeOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<LinearOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
  return getOpConstraintsWithState(
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, biasShape,
      biasLayout, outputLayout, transposeA, transposeB, activation,
      programConfigAttr, computeKernelConfig, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<LinearOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB, std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    const MockAllocatorState *initialState) {
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

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Convert program config attribute
  auto programConfig =
      programConfigAttr ? conversion::getMatmulProgramConfig(*programConfigAttr)
                        : std::nullopt;

  std::optional<std::string> activationStr;
  if (activation && !detail::programCarriesFusedActivation(programConfig)) {
    activationStr = activation->str();
  }

  std::optional<::ttnn::DeviceComputeKernelConfig>
      computeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(computeKernelConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto linearOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::linear, device, initialStateOpt, inputSpecA, inputSpecB,
        biasTensor, transposeA, transposeB, outputMemoryConfig, outputDType,
        programConfig, activationStr, computeKernelConfigConverted,
        /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(),
                                              linearOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<LinearOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, TTNNLayoutAttr outputLayout,
    bool transposeA, bool transposeB) {
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

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto linearOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::linear, device, inputSpecA, inputSpecB, biasTensor, transposeA,
        transposeB, outputMemoryConfig, outputDType,
        /*program_config=*/std::nullopt,
        /*activation=*/std::nullopt, /*compute_kernel_config=*/std::nullopt,
        /*core_grid=*/std::nullopt, /*output_tile=*/std::nullopt,
        /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  };

  return operation::getOpRuntime(linearOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
// Cache-facing stateless entry; signature fixed for getOrCompute. Forwards to
// the shared stateful body with a null state (metal dispatches nullopt to the
// stateless query).
llvm::Expected<OpConstraints> OpModel<MatmulOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB,
    std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
  return getOpConstraintsWithState(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, outputLayout, transposeA,
                                   transposeB, activation, programConfigAttr,
                                   computeKernelConfig,
                                   /*initialState=*/nullptr);
}

// Shared body. Stateful entry (L1 spill path); bypasses the op-model cache.
llvm::Expected<OpConstraints> OpModel<MatmulOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB,
    std::optional<llvm::StringRef> activation,
    std::optional<mlir::Attribute> programConfigAttr,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Convert program config attribute
  auto programConfig =
      programConfigAttr ? conversion::getMatmulProgramConfig(*programConfigAttr)
                        : std::nullopt;

  std::optional<std::string> activationStr;
  if (activation && !detail::programCarriesFusedActivation(programConfig)) {
    activationStr = activation->str();
  }

  std::optional<::ttnn::DeviceComputeKernelConfig>
      computeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(computeKernelConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto matmulOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::matmul, device, initialStateOpt, inputSpecA, inputSpecB,
        transposeA, transposeB, outputMemoryConfig, outputDType, programConfig,
        activationStr, computeKernelConfigConverted, /*core_grid=*/std::nullopt,
        /*output_tile=*/std::nullopt, /*optional_output_tensor=*/std::nullopt,
        /*global_cb=*/std::nullopt, /*sub_device_id=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayoutA.getContext(),
                                              matmulOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<MatmulOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShapeA, TTNNLayoutAttr inputLayoutA,
    llvm::ArrayRef<int64_t> inputShapeB, TTNNLayoutAttr inputLayoutB,
    TTNNLayoutAttr outputLayout, bool transposeA, bool transposeB) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecA,
      detail::convertToTensorSpec(device, inputShapeA, inputLayoutA));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpecB,
      detail::convertToTensorSpec(device, inputShapeB, inputLayoutB));

  std::optional<::tt::tt_metal::DataType> outputDType =
      detail::getNullableDataType(outputLayout);
  std::optional<::tt::tt_metal::MemoryConfig> outputMemoryConfig =
      detail::getNullableMemoryConfig(outputLayout);

  // Create query closure
  auto matmulOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::matmul, device, inputSpecA, inputSpecB,
                            transposeA, transposeB, outputMemoryConfig,
                            outputDType,
                            /*program_config=*/std::nullopt,
                            /*activation=*/std::nullopt,
                            /*compute_kernel_config=*/std::nullopt,
                            /*core_grid=*/std::nullopt,
                            /*output_tile=*/std::nullopt,
                            /*optional_output_tensor=*/std::nullopt,
                            /*global_cb=*/std::nullopt,
                            /*sub_device_id=*/std::nullopt);
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

llvm::Expected<OpConstraints> OpModel<FillCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(cacheShape, cacheLayout, inputShape,
                                   inputLayout, batchOffset, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<FillCacheOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec cacheSpec,
      detail::convertToTensorSpec(device, cacheShape, cacheLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto fillCacheOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(::ttnn::fill_cache, device,
                                           initialStateOpt, cacheSpec,
                                           inputSpec, batchOffset);
  };

  return operation::getOpConstraintsWithState(cacheLayout.getContext(),
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

  auto fillCacheOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::fill_cache, device, cacheSpec, inputSpec,
                            batchOffset);
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
  return getOpConstraintsWithState(
      cacheShape, cacheLayout, inputShape, inputLayout, updateIndexShape,
      updateIndexLayout, batchOffset, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<UpdateCacheOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    uint32_t batchOffset, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto updateCacheOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::update_cache, device, initialStateOpt, cacheSpec, inputSpec,
        updateIdx, batchOffset,
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(cacheLayout.getContext(),
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
llvm::Expected<OpConstraints> OpModel<PagedUpdateCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    std::optional<llvm::ArrayRef<int64_t>> pageTableShape,
    std::optional<TTNNLayoutAttr> pageTableLayout, bool shareCache,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      cacheShape, cacheLayout, inputShape, inputLayout, updateIndexShape,
      updateIndexLayout, pageTableShape, pageTableLayout, shareCache,
      outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PagedUpdateCacheOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> updateIndexShape, TTNNLayoutAttr updateIndexLayout,
    std::optional<llvm::ArrayRef<int64_t>> pageTableShape,
    std::optional<TTNNLayoutAttr> pageTableLayout, bool shareCache,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  std::vector<uint32_t> emptyUpdateIndex = {};
  auto pagedUpdateCacheOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::paged_update_cache, device, initialStateOpt,
        cacheSpec, inputSpec, emptyUpdateIndex, updateIndexSpec, shareCache,
        pageTableSpec,
        /*batch_offset=*/0,
        /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(cacheLayout.getContext(),
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

  std::vector<uint32_t> emptyUpdateIndex = {};
  auto pagedUpdateCacheOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::paged_update_cache, device,
                            cacheSpec, inputSpec, emptyUpdateIndex,
                            updateIndexSpec, shareCache, pageTableSpec, 0,
                            std::nullopt, std::nullopt);
  };

  return operation::getOpRuntime(pagedUpdateCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PagedFillCacheOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<PagedFillCacheOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchIdxShape,
    std::optional<TTNNLayoutAttr> batchIdxLayout, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(cacheShape, cacheLayout, inputShape,
                                   inputLayout, pageTableShape, pageTableLayout,
                                   batchIdxShape, batchIdxLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PagedFillCacheOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> cacheShape, TTNNLayoutAttr cacheLayout,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> pageTableShape, TTNNLayoutAttr pageTableLayout,
    std::optional<llvm::ArrayRef<int64_t>> batchIdxShape,
    std::optional<TTNNLayoutAttr> batchIdxLayout, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto pagedFillCacheOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::paged_fill_cache, device, initialStateOpt,
        cacheSpec, inputSpec, pageTableSpec, batchIdxSpec,
        /*batch_offset=*/0,
        /*compute_kernel_config=*/std::nullopt, /*mesh_coords=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(cacheLayout.getContext(),
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

  auto pagedFillCacheOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::experimental::paged_fill_cache, device,
                            cacheSpec, inputSpec, pageTableSpec, batchIdxSpec,
                            /*batch_offset=*/0,
                            /*compute_kernel_config=*/std::nullopt,
                            /*mesh_coords=*/std::nullopt);
  };

  return operation::getOpRuntime(pagedFillCacheOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv2dOp
//===----------------------------------------------------------------------===//
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
  return getOpConstraintsWithState(
      inputShape, inputLayout, weightShape, weightLayout, biasShape, biasLayout,
      in_channels, out_channels, batch_size, input_height, input_width,
      kernel_size, stride, padding, dilation, groups, conv2dConfig,
      deviceComputeKernelConfig, conv2dSliceConfig, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<Conv2dOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);

  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto conv2dOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::conv2d, device, initialStateOpt, inputSpec, weightSpec, device,
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout), sliceConfigConverted,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              conv2dOpQuery);
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  auto conv2dConfigConverted = conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);
  // Create query closure
  auto conv2dOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::conv2d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout), sliceConfigConverted,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpRuntime(conv2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv1dOp
//===----------------------------------------------------------------------===//
#ifdef TTMLIR_ENABLE_OPMODEL
namespace {
// Builds the 2D-equivalent parameters `ttnn::conv1d` uses internally, for the
// shared conv2d weight/bias prepare step:
//   input   (N, L_in, C) -> (N, 1, L_in, C)
//   weight  (O, C/G, K)  -> (O, C/G, 1, K)  ("OIHW")
//   kernel/stride/dilation gain a leading 1; padding [pL, pR] ->
//     {top=0, left=pL, bottom=0, right=pR} (the order reorderPool2dPadding
//     consumes for a 4-element padding).
struct Conv1dConv2dPrepParams {
  llvm::SmallVector<int64_t, 4> inputShape;
  llvm::SmallVector<int64_t, 4> weightShape;
  llvm::SmallVector<int32_t, 2> kernelSize;
  llvm::SmallVector<int32_t, 2> stride;
  llvm::SmallVector<int32_t, 4> padding;
  llvm::SmallVector<int32_t, 2> dilation;
};

Conv1dConv2dPrepParams
getConv1dConv2dPrepParams(llvm::ArrayRef<int64_t> inputShape,
                          llvm::ArrayRef<int64_t> weightShape,
                          uint32_t kernel_size, uint32_t stride,
                          llvm::ArrayRef<int32_t> padding, uint32_t dilation) {
  return Conv1dConv2dPrepParams{
      /*inputShape=*/{inputShape[0], 1, inputShape[1], inputShape[2]},
      /*weightShape=*/{weightShape[0], weightShape[1], 1, weightShape[2]},
      /*kernelSize=*/{1, static_cast<int32_t>(kernel_size)},
      /*stride=*/{1, static_cast<int32_t>(stride)},
      /*padding=*/{0, padding[0], 0, padding[1]},
      /*dilation=*/{1, static_cast<int32_t>(dilation)}};
}

// `ttnn::conv1d` forces L1_FULL slicing when no slice config is provided;
// return that effective config so the weight prepare and the query match
// runtime.
Conv2dSliceConfigAttr getConv1dEffectiveSliceConfig(
    mlir::MLIRContext *ctx, std::optional<Conv2dSliceConfigAttr> sliceConfig) {
  if (sliceConfig && *sliceConfig) {
    return *sliceConfig;
  }
  return Conv2dSliceConfigAttr::get(ctx, Conv2dSliceType::L1Full,
                                    /*num_slices=*/0);
}
} // namespace
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<Conv1dOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_length,
    uint32_t kernel_size, uint32_t stride, llvm::ArrayRef<int32_t> padding,
    uint32_t dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  Conv1dConv2dPrepParams prep = getConv1dConv2dPrepParams(
      inputShape, weightShape, kernel_size, stride, padding, dilation);
  Conv2dSliceConfigAttr sliceConfig = getConv1dEffectiveSliceConfig(
      inputLayout.getContext(), conv2dSliceConfig);

  // Prepare weight tensor first (shared conv2d prepare helper).
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          prep.inputShape, inputLayout, prep.weightShape, weightLayout,
          in_channels, out_channels, batch_size, /*input_height=*/1,
          /*input_width=*/input_length, prep.kernelSize, prep.stride,
          prep.padding, prep.dilation, groups, conv2dConfig, sliceConfig,
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
            prep.inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            /*input_height=*/1, /*input_width=*/input_length, prep.kernelSize,
            prep.stride, prep.padding, prep.dilation, groups, conv2dConfig,
            /*transpose*/ false);
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);
  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(sliceConfig);

  // conv1d padding is std::variant<std::array<uint32_t, 2>, uint32_t>; use the
  // (left, right) array alternative, matching the runtime op.
  std::variant<std::array<uint32_t, 2>, uint32_t> conv1dPadding =
      conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding);

  auto conv1dOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(
        ::ttnn::conv1d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_length, kernel_size, stride,
        conv1dPadding, dilation, groups, outputDtype, biasSpec,
        conv2dConfigConverted, deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout), sliceConfigConverted,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpConstraints(inputLayout.getContext(), conv1dOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<Conv1dOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t in_channels,
    uint32_t out_channels, uint32_t batch_size, uint32_t input_length,
    uint32_t kernel_size, uint32_t stride, llvm::ArrayRef<int32_t> padding,
    uint32_t dilation, uint32_t groups,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig,
    TTNNLayoutAttr outputLayout) {
#ifdef TTMLIR_ENABLE_OPMODEL
  Conv1dConv2dPrepParams prep = getConv1dConv2dPrepParams(
      inputShape, weightShape, kernel_size, stride, padding, dilation);
  Conv2dSliceConfigAttr sliceConfig = getConv1dEffectiveSliceConfig(
      inputLayout.getContext(), conv2dSliceConfig);

  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          prep.inputShape, inputLayout, prep.weightShape, weightLayout,
          in_channels, out_channels, batch_size, /*input_height=*/1,
          /*input_width=*/input_length, prep.kernelSize, prep.stride,
          prep.padding, prep.dilation, groups, conv2dConfig, sliceConfig,
          biasLayout.has_value(), /*transpose*/ false);
  if (!preparedWeightExp) {
    return preparedWeightExp.takeError();
  }
  ::ttnn::TensorSpec weightSpec = preparedWeightExp.get();

  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    llvm::Expected<::ttnn::TensorSpec> preparedBiasExp =
        getPrepareConv2dBiasOpOutputTensorSpec(
            prep.inputShape, inputLayout, *biasShape, *biasLayout,
            weightSpec.data_type(), in_channels, out_channels, batch_size,
            /*input_height=*/1, /*input_width=*/input_length, prep.kernelSize,
            prep.stride, prep.padding, prep.dilation, groups, conv2dConfig,
            /*transpose*/ false);
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);
  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);
  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(sliceConfig);

  std::variant<std::array<uint32_t, 2>, uint32_t> conv1dPadding =
      conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding);

  auto conv1dOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::conv1d, device, inputSpec, weightSpec, device, in_channels,
        out_channels, batch_size, input_length, kernel_size, stride,
        conv1dPadding, dilation, groups, outputDtype, biasSpec,
        conv2dConfigConverted, deviceComputeKernelConfigConverted,
        detail::getNullableMemoryConfig(outputLayout), sliceConfigConverted,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpRuntime(conv1dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Conv3dOp
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_OPMODEL
namespace {

struct Conv3dSpecs {
  ::ttnn::TensorSpec inputSpec;
  ::ttnn::TensorSpec weightSpec;
  std::optional<::ttnn::TensorSpec> biasSpec;
  std::optional<::ttnn::experimental::prim::Conv3dConfig> config;
  ::tt::tt_metal::DataType dtype;
  uint32_t outputChannels;
  std::array<uint32_t, 3> kernelSize;
  std::array<uint32_t, 3> stride;
  std::array<uint32_t, 3> padding;
  std::string paddingMode;
  uint32_t groups;
  std::optional<::ttnn::DeviceComputeKernelConfig> deviceComputeKernelConfig;
};

llvm::Expected<Conv3dSpecs> prepareConv3dSpecs(
    ::tt::tt_metal::distributed::MeshDevice *device,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, uint32_t out_channels,
    llvm::ArrayRef<int32_t> kernel_size, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::StringRef padding_mode,
    uint32_t groups, std::optional<ttcore::DataTypeAttr> outputDtype,
    std::optional<Conv3dConfigAttr> conv3dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {

  // Convert input layout to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Convert weight layout to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  // Convert bias if present
  std::optional<::ttnn::TensorSpec> biasSpec;
  if (biasShape && biasLayout) {
    ASSIGN_OR_RETURN(
        biasSpec, detail::convertToTensorSpec(device, *biasShape, *biasLayout));
  }

  std::optional<::ttnn::experimental::prim::Conv3dConfig> config;

  // Apply Conv3dConfig overrides if provided
  if (conv3dConfig.has_value()) {
    config.emplace();
    if (conv3dConfig->getWeightsDtype()) {
      config->weights_dtype =
          conversion::getDataType(*conv3dConfig->getWeightsDtype());
    }
    if (conv3dConfig->getTOutBlock()) {
      config->T_out_block = *conv3dConfig->getTOutBlock();
    }
    if (conv3dConfig->getWOutBlock()) {
      config->W_out_block = *conv3dConfig->getWOutBlock();
    }
    if (conv3dConfig->getHOutBlock()) {
      config->H_out_block = *conv3dConfig->getHOutBlock();
    }
    if (conv3dConfig->getCOutBlock()) {
      config->C_out_block = *conv3dConfig->getCOutBlock();
    }
    if (conv3dConfig->getCInBlock()) {
      config->C_in_block = *conv3dConfig->getCInBlock();
    }
    config->compute_with_storage_grid_size =
        device->compute_with_storage_grid_size();
  }

  // Get output dtype in this order: explicit outputDtype → outputLayout →
  // BFLOAT16
  std::optional<::tt::tt_metal::DataType> dtype;
  if (outputDtype.has_value() && outputDtype.value()) {
    dtype = conversion::getDataType(outputDtype.value().getValue());
  }
  if (!dtype) {
    dtype = detail::getNullableDataType(outputLayout);
  }

  std::optional<::ttnn::DeviceComputeKernelConfig>
      deviceComputeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig);

  return Conv3dSpecs{
      inputSpec,
      weightSpec,
      biasSpec,
      config,
      dtype.value_or(::tt::tt_metal::DataType::BFLOAT16),
      out_channels,
      conversion::convertLLVMArrayRefToStdArray<uint32_t, 3>(kernel_size),
      conversion::convertLLVMArrayRefToStdArray<uint32_t, 3>(stride),
      conversion::convertLLVMArrayRefToStdArray<uint32_t, 3>(padding),
      padding_mode.str(),
      groups,
      deviceComputeKernelConfigConverted};
}
} // namespace
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
  return getOpConstraintsWithState(
      inputShape, inputLayout, weightShape, weightLayout, biasShape, biasLayout,
      in_channels, out_channels, batch_size, input_depth, input_height,
      input_width, kernel_size, stride, padding, groups, padding_mode,
      outputDtype, conv3dConfig, deviceComputeKernelConfig, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<Conv3dOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  auto specsExp = prepareConv3dSpecs(
      device, inputShape, inputLayout, weightShape, weightLayout, biasShape,
      biasLayout, out_channels, kernel_size, stride, padding, padding_mode,
      groups, outputDtype, conv3dConfig, deviceComputeKernelConfig,
      outputLayout);
  if (!specsExp) {
    return specsExp.takeError();
  }
  auto specs = specsExp.get();

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto conv3dOpQuery = [=, &specs]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::experimental::conv3d, device, initialStateOpt, specs.inputSpec,
        specs.weightSpec,
        std::optional<::tt::tt_metal::distributed::MeshDevice *>(device),
        specs.biasSpec, specs.config, specs.dtype, specs.outputChannels,
        specs.kernelSize, specs.stride, specs.padding,
        std::array<uint32_t, 3>{1, 1, 1}, specs.paddingMode, specs.groups,
        detail::getNullableMemoryConfig(outputLayout),
        specs.deviceComputeKernelConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              conv3dOpQuery);
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

  auto specsExp = prepareConv3dSpecs(
      device, inputShape, inputLayout, weightShape, weightLayout, biasShape,
      biasLayout, out_channels, kernel_size, stride, padding, padding_mode,
      groups, outputDtype, conv3dConfig, deviceComputeKernelConfig,
      outputLayout);
  if (!specsExp) {
    return specsExp.takeError();
  }
  auto specs = specsExp.get();

  auto conv3dOpRuntime = [=, &specs]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::experimental::conv3d, device, specs.inputSpec, specs.weightSpec,
        std::optional<::tt::tt_metal::distributed::MeshDevice *>(device),
        specs.biasSpec, specs.config, specs.dtype, specs.outputChannels,
        specs.kernelSize, specs.stride, specs.padding,
        std::array<uint32_t, 3>{1, 1, 1}, specs.paddingMode, specs.groups,
        detail::getNullableMemoryConfig(outputLayout),
        specs.deviceComputeKernelConfig);
  };

  return operation::getOpRuntime(conv3dOpRuntime);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ConvTranspose2dOp
//===----------------------------------------------------------------------===//
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
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      inputShape, inputLayout, weightShape, weightLayout, biasShape, biasLayout,
      in_channels, out_channels, batch_size, input_height, input_width,
      kernel_size, stride, padding, output_padding, dilation, groups,
      conv2dConfig, conv2dSliceConfig, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<ConvTranspose2dOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  // Prepare weight tensor first.
  llvm::Expected<::ttnn::TensorSpec> preparedWeightExp =
      getPrepareConv2dWeightsOpOutputTensorSpec(
          inputShape, inputLayout, weightShape, weightLayout, in_channels,
          out_channels, batch_size, input_height, input_width, kernel_size,
          stride, padding, dilation, groups, conv2dConfig, conv2dSliceConfig,
          biasLayout.has_value(), /*transpose*/ true, output_padding);
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::Conv2dSliceConfig> conv2dSliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::conv_transpose2d, device, initialStateOpt, inputSpec,
        weightSpec, device, in_channels, out_channels, batch_size, input_height,
        input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        /* compute_config */ std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        conv2dSliceConfigConverted,
        /*mirror_kernel=*/true,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  std::optional<::tt::tt_metal::DataType> outputDtype =
      detail::getNullableDataType(outputLayout);

  std::optional<::ttnn::Conv2dConfig> conv2dConfigConverted =
      conversion::getConv2dConfig(conv2dConfig);
  std::optional<::ttnn::Conv2dSliceConfig> conv2dSliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);

  // Create query closure
  auto convTranspose2dOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::conv_transpose2d, device, inputSpec, weightSpec, device,
        in_channels, out_channels, batch_size, input_height, input_width,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernel_size),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, outputDtype, biasSpec, conv2dConfigConverted,
        /* compute_config */ std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        conv2dSliceConfigConverted,
        /*mirror_kernel=*/true,
        /*return_output_dim=*/false,
        /*return_weights_and_bias=*/false);
  };

  return operation::getOpRuntime(convTranspose2dOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dWeightsOp
//===----------------------------------------------------------------------===//

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
  return getOpConstraintsWithState(
      weightLayout, weightShape, inputMemConfig, inputTensorLayout,
      weightsFormat, inChannels, outChannels, batchSize, inputHeight,
      inputWidth, kernelSize, stride, padding, dilation, hasBias, groups,
      inputDtype, outputDtype, conv2dConfig, deviceComputeKernelConfig,
      conv2dSliceConfig, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareConv2dWeightsOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(weightLayout != nullptr && "Weight layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfigConverted =
      conversion::getConv2dSliceConfig(conv2dSliceConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto prepareConv2dWeightsQuery = [=]() {
    return ::ttnn::graph::query_op_constraints_with_optional_state(
        &::ttnn::operations::conv::conv2d::prepare_conv_weights, device,
        initialStateOpt, weightTensor,
        conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), weightsFormat.str(),
        inChannels, outChannels, batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig),
        sliceConfigConverted);
  };

  return operation::getOpConstraintsWithState(weightLayout.getContext(),
                                              prepareConv2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<PrepareConv2dBiasOp>::getOpConstraints(
    TTNNLayoutAttr biasLayout, llvm::ArrayRef<int64_t> biasShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      biasLayout, biasShape, inputMemConfig, inputTensorLayout, inChannels,
      outChannels, batchSize, inputHeight, inputWidth, kernelSize, stride,
      padding, dilation, groups, inputDtype, outputDtype, conv2dConfig,
      deviceComputeKernelConfig, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareConv2dBiasOp>::getOpConstraintsWithState(
    TTNNLayoutAttr biasLayout, llvm::ArrayRef<int64_t> biasShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    int32_t inChannels, int32_t outChannels, int32_t batchSize,
    int32_t inputHeight, int32_t inputWidth, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(biasLayout != nullptr && "Weight layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }

  std::optional<::ttnn::Conv2dSliceConfig> sliceConfig = std::nullopt;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto prepareConv2dWeightsQuery = [=]() {
    return ::ttnn::graph::query_op_constraints_with_optional_state(
        &::ttnn::operations::conv::conv2d::prepare_conv_bias, device,
        initialStateOpt, biasTensor,
        conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), inChannels, outChannels,
        batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig),
        sliceConfig);
  };

  return operation::getOpConstraintsWithState(biasLayout.getContext(),
                                              prepareConv2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dWeightsOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<PrepareConvTranspose2dWeightsOp>::getOpConstraints(
    TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> weightShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, bool hasBias, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      weightLayout, weightShape, inputMemConfig, inputTensorLayout,
      weightsFormat, inChannels, outChannels, batchSize, inputHeight,
      inputWidth, kernelSize, stride, padding, output_padding, dilation,
      hasBias, groups, inputDtype, outputDtype, conv2dConfig,
      deviceComputeKernelConfig, conv2dSliceConfig, mirrorKernel, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareConvTranspose2dWeightsOp>::getOpConstraintsWithState(
    TTNNLayoutAttr weightLayout, llvm::ArrayRef<int64_t> weightShape,
    MemoryConfigAttr inputMemConfig, ::mlir::tt::ttnn::Layout inputTensorLayout,
    llvm::StringRef weightsFormat, int32_t inChannels, int32_t outChannels,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    llvm::ArrayRef<int32_t> kernelSize, llvm::ArrayRef<int32_t> stride,
    llvm::ArrayRef<int32_t> padding, llvm::ArrayRef<int32_t> output_padding,
    llvm::ArrayRef<int32_t> dilation, bool hasBias, int32_t groups,
    ttcore::DataType inputDtype, std::optional<ttcore::DataType> outputDtype,
    std::optional<Conv2dConfigAttr> conv2dConfig,
    std::optional<DeviceComputeKernelConfigAttr> deviceComputeKernelConfig,
    std::optional<Conv2dSliceConfigAttr> conv2dSliceConfig, bool mirrorKernel,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(weightLayout != nullptr && "Weight layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor weightTensor =
      createMetalHostTensor(weightShape, weightLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto prepareConvTranspose2dWeightsQuery = [=]() {
    return ::ttnn::graph::query_op_constraints_with_optional_state(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_weights,
        device, initialStateOpt, weightTensor,
        conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), weightsFormat.str(),
        inChannels, outChannels, batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(output_padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        hasBias, groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig),
        conversion::getConv2dSliceConfig(conv2dSliceConfig), mirrorKernel);
  };

  return operation::getOpConstraintsWithState(
      weightLayout.getContext(), prepareConvTranspose2dWeightsQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// PrepareConvTranspose2dBiasOp
//===----------------------------------------------------------------------===//

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
  return getOpConstraintsWithState(
      biasLayout, biasShape, inputMemConfig, inputTensorLayout, inChannels,
      outChannels, batchSize, inputHeight, inputWidth, kernelSize, stride,
      padding, dilation, groups, inputDtype, outputDtype, conv2dConfig,
      deviceComputeKernelConfig, conv2dSliceConfig, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<PrepareConvTranspose2dBiasOp>::getOpConstraintsWithState(
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
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();
  assert(device != nullptr && "Device is nullptr");
  assert(biasLayout != nullptr && "Bias layout is nullptr");

  // TODO(#4043): Move this to tt-metal side.
  ::tt::tt_metal::Tensor biasTensor =
      createMetalHostTensor(biasShape, biasLayout.getDataType());
  // Read output data type from output layout (if present) or from outputDtype.
  std::optional<::tt::tt_metal::DataType> convertedOutputDtype = std::nullopt;
  if (outputLayout) {
    convertedOutputDtype = conversion::getDataType(outputLayout.getDataType());
  } else if (outputDtype.has_value()) {
    convertedOutputDtype = conversion::getDataType(outputDtype.value());
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto prepareConvTranspose2dBiasQuery = [=]() {
    return ::ttnn::graph::query_op_constraints_with_optional_state(
        &::ttnn::operations::conv::conv_transpose2d::
            prepare_conv_transpose2d_bias,
        device, initialStateOpt, biasTensor,
        conversion::getMemoryConfig(inputMemConfig),
        conversion::getPageLayout(inputTensorLayout), inChannels, outChannels,
        batchSize, inputHeight, inputWidth,
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(kernelSize),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(stride),
        detail::reorderPool2dPadding(padding),
        conversion::convertLLVMArrayRefToStdArray<uint32_t, 2>(dilation),
        groups, device, conversion::getDataType(inputDtype),
        convertedOutputDtype, conversion::getConv2dConfig(conv2dConfig),
        conversion::getDeviceComputeKernelConfig(deviceComputeKernelConfig),
        conversion::getConv2dSliceConfig(conv2dSliceConfig));
  };

  return operation::getOpConstraintsWithState(biasLayout.getContext(),
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
  return getOpConstraintsWithState(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      reallocateHaloOutput, configTensorsInDram, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<MaxPool2dOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto maxPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::max_pool2d, device, initialStateOpt, inputSpec, batchSizeU,
        inputHeightU, inputWidthU, inputChannelsU,
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

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              maxPool2DQuery);
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
  return getOpConstraintsWithState(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      reallocateHaloOutput, deallocateInput, returnIndices, configTensorsInDram,
      outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<MaxPool2dWithIndicesOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    bool deallocateInput, bool returnIndices,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  // When return_indices=true, tt-metal requires ROW_MAJOR layout and BFLOAT16
  auto maxPool2DWithIndicesQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::max_pool2d, device, initialStateOpt, inputSpec, batchSizeU,
        inputHeightU, inputWidthU, inputChannelsU,
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

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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
  return getOpConstraintsWithState(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      reallocateHaloOutput, configTensorsInDram, outputLayout,
      /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<AvgPool2dOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    int32_t batchSize, int32_t inputHeight, int32_t inputWidth,
    int32_t inputChannels, llvm::ArrayRef<int32_t> kernelSize,
    llvm::ArrayRef<int32_t> stride, llvm::ArrayRef<int32_t> padding,
    llvm::ArrayRef<int32_t> dilation, bool ceilMode, bool reallocateHaloOutput,
    std::optional<bool> configTensorsInDram, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto avgPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::avg_pool2d, device, initialStateOpt, inputSpec, batchSizeU,
        inputHeightU, inputWidthU, inputChannelsU,
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

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              avgPool2DQuery);
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
  return getOpConstraintsWithState(inputShape, inputLayout, dtype, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<GlobalAvgPool2dOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> dtype,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto globalAvgPool2DQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::avg_pool2d, device, initialStateOpt, inputSpec, batchSize,
        inputHeight, inputWidth, inputChannels,
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

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      inputShape, inputLayout, runningMeanShape, runningMeanLayout,
      runningVarShape, runningVarLayout, weightShape, weightLayout, biasShape,
      biasLayout, epsilon, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<BatchNormInferenceOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // For inference, training is false and momentum is 0.1 (default)
  bool training = false;
  float momentum = 0.1f;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto batchNormQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::batch_norm, device, initialStateOpt, inputSpec, runningMeanSpec,
        runningVarSpec, training, epsilon.convertToFloat(), momentum,
        weightSpec, biasSpec, outputSpec,
        detail::getNullableMemoryConfig(outputLayout), computeKernelConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              batchNormQuery);
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
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // For inference, training is false and momentum is 0.1 (default)
  bool training = false;
  float momentum = 0.1f;

  // Create query closure
  auto batchNormQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::batch_norm, device, inputSpec, runningMeanSpec, runningVarSpec,
        training, epsilon.convertToFloat(), momentum, weightSpec, biasSpec,
        outputSpec, detail::getNullableMemoryConfig(outputLayout),
        computeKernelConfig);
  };

  return operation::getOpRuntime(batchNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

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
    llvm::APFloat momentum, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(
      inputShape, inputLayout, runningMeanShape, runningMeanLayout,
      runningVarShape, runningVarLayout, weightShape, weightLayout, biasShape,
      biasLayout, epsilon, momentum, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<BatchNormTrainingOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningMeanShape,
    std::optional<TTNNLayoutAttr> runningMeanLayout,
    std::optional<llvm::ArrayRef<int64_t>> runningVarShape,
    std::optional<TTNNLayoutAttr> runningVarLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    llvm::APFloat momentum, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // For training mode
  bool training = true;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto batchNormQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::batch_norm, device, initialStateOpt, inputSpec, runningMeanSpec,
        runningVarSpec, training, epsilon.convertToFloat(),
        momentum.convertToFloat(), weightSpec, biasSpec, outputSpec,
        detail::getNullableMemoryConfig(outputLayout), computeKernelConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              batchNormQuery);
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
    llvm::APFloat momentum, TTNNLayoutAttr outputLayout) {
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
  // The following arguments are received by the invoke method of batch norm but
  // they don't exist in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> outputSpec = std::nullopt;
  std::optional<::ttnn::DeviceComputeKernelConfig> computeKernelConfig =
      std::nullopt;

  // For training mode
  bool training = true;

  // Create query closure
  auto batchNormQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::batch_norm, device, inputSpec, runningMeanSpec, runningVarSpec,
        training, epsilon.convertToFloat(), momentum.convertToFloat(),
        weightSpec, biasSpec, outputSpec,
        detail::getNullableMemoryConfig(outputLayout), computeKernelConfig);
  };

  return operation::getOpRuntime(batchNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RMSNormOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RMSNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig) {
  return getOpConstraintsWithState(inputShape, inputLayout, weightShape,
                                   weightLayout, biasShape, biasLayout, epsilon,
                                   outputLayout, computeKernelConfig,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<RMSNormOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout,
    std::optional<DeviceComputeKernelConfigAttr> computeKernelConfig,
    const MockAllocatorState *initialState) {
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

  // This information is not available in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig>
      computeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(computeKernelConfig);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto rmsNormQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::rms_norm, device, initialStateOpt, inputSpec,
        epsilon.convertToFloat(), weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt, computeKernelConfigConverted);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              rmsNormQuery);
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

  // This information is not available in the op's definition in TTNNOps.td:
  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig>
      computeKernelConfigConverted =
          conversion::getDeviceComputeKernelConfig(computeKernelConfig);

  // Create query closure
  auto rmsNormQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::rms_norm, device, inputSpec, epsilon.convertToFloat(),
        weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt, computeKernelConfigConverted);
  };

  return operation::getOpRuntime(rmsNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// RMSNormPreAllGatherOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<RMSNormPreAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<ttcore::DataType> dtype, std::optional<bool> use2DCoreGrid,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, residualInputShape,
                                   residualInputLayout, dtype, use2DCoreGrid,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<RMSNormPreAllGatherOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<ttcore::DataType> dtype, std::optional<bool> use2DCoreGrid,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<::ttnn::TensorSpec> residualInputSpec =
      detail::convertToOptionalTensorSpec(device, residualInputShape,
                                          residualInputLayout);

  ::ttnn::DataType metalDtype = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::rms_norm_pre_all_gather, device, initialStateOpt, inputSpec,
        /*dtype=*/metalDtype,
        /*residual_input_tensor=*/residualInputSpec,
        /*compute_kernel_config=*/std::nullopt,
        /*program_config=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*use_2d_core_grid=*/use2DCoreGrid);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);

#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<RMSNormPreAllGatherOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<ttcore::DataType> dtype, std::optional<bool> use2DCoreGrid,
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

  ::ttnn::DataType metalDtype = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }

  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::rms_norm_pre_all_gather, device, inputSpec,
        /*dtype=*/metalDtype,
        /*residual_input_tensor=*/residualInputSpec,
        /*compute_kernel_config=*/std::nullopt,
        /*program_config=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*use_2d_core_grid=*/use2DCoreGrid);
  };
  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<LayerNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, weightShape,
                                   weightLayout, biasShape, biasLayout, epsilon,
                                   outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<LayerNormOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto layerNormQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::layer_norm, device, initialStateOpt, inputSpec,
        epsilon.convertToFloat(), weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt, /*recip_tensor=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              layerNormQuery);
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

  std::optional<::ttnn::TensorSpec> residualInputSpec = std::nullopt;

  // Create query closure
  auto layerNormQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::layer_norm, device, inputSpec, epsilon.convertToFloat(),
        weightSpec, biasSpec, residualInputSpec,
        detail::getNullableMemoryConfig(outputLayout),
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt, /*recip_tensor=*/std::nullopt);
  };

  return operation::getOpRuntime(layerNormQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormPreAllGatherOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<LayerNormPreAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<llvm::ArrayRef<int64_t>> recipShape,
    std::optional<TTNNLayoutAttr> recipLayout,
    std::optional<ttcore::DataType> dtype, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, residualInputShape,
                                   residualInputLayout, recipShape, recipLayout,
                                   dtype, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<LayerNormPreAllGatherOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> residualInputShape,
    std::optional<TTNNLayoutAttr> residualInputLayout,
    std::optional<llvm::ArrayRef<int64_t>> recipShape,
    std::optional<TTNNLayoutAttr> recipLayout,
    std::optional<ttcore::DataType> dtype, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  ::ttnn::DataType metalDtype = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::layer_norm_pre_all_gather, device, initialStateOpt, inputSpec,
        /*dtype=*/metalDtype,
        /*residual_input_tensor=*/residualInputSpec,
        /*compute_kernel_config=*/std::nullopt,
        /*program_config=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*recip_tensor=*/recipSpec);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
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
    std::optional<ttcore::DataType> dtype, TTNNLayoutAttr outputLayout) {
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

  ::ttnn::DataType metalDtype = ::ttnn::DataType::BFLOAT16;
  if (dtype.has_value()) {
    metalDtype = conversion::getDataType(dtype.value());
  }

  auto query = [=]() {
    return ::ttnn::graph::query_op_runtime(
        ::ttnn::layer_norm_pre_all_gather, device, inputSpec,
        /*dtype=*/metalDtype,
        /*residual_input_tensor=*/residualInputSpec,
        /*compute_kernel_config=*/std::nullopt,
        /*program_config=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*recip_tensor=*/recipSpec);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// LayerNormPostAllGatherOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<LayerNormPostAllGatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> statsShape, TTNNLayoutAttr statsLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, statsShape,
                                   statsLayout, weightShape, weightLayout,
                                   biasShape, biasLayout, epsilon, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<LayerNormPostAllGatherOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> statsShape, TTNNLayoutAttr statsLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, llvm::APFloat epsilon,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto query = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::layer_norm_post_all_gather, device, initialStateOpt, inputSpec,
        statsSpec, epsilon.convertToFloat(), weightSpec, biasSpec,
        detail::getNullableMemoryConfig(outputLayout),
        /*compute_kernel_config=*/std::nullopt,
        /*program_config=*/std::nullopt,
        /*dtype=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(), query);
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

  auto query = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::layer_norm_post_all_gather, device,
                            inputSpec, statsSpec, epsilon.convertToFloat(),
                            weightSpec, biasSpec,
                            detail::getNullableMemoryConfig(outputLayout),
                            /*compute_kernel_config=*/std::nullopt,
                            /*program_config=*/std::nullopt,
                            /*dtype=*/std::nullopt);
  };

  return operation::getOpRuntime(query);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GroupNormOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<GroupNormOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputMaskShape,
    std::optional<TTNNLayoutAttr> inputMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, int64_t numGroups,
    llvm::APFloat epsilon, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, inputMaskShape,
                                   inputMaskLayout, weightShape, weightLayout,
                                   biasShape, biasLayout, numGroups, epsilon,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<GroupNormOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<llvm::ArrayRef<int64_t>> inputMaskShape,
    std::optional<TTNNLayoutAttr> inputMaskLayout,
    std::optional<llvm::ArrayRef<int64_t>> weightShape,
    std::optional<TTNNLayoutAttr> weightLayout,
    std::optional<llvm::ArrayRef<int64_t>> biasShape,
    std::optional<TTNNLayoutAttr> biasLayout, int64_t numGroups,
    llvm::APFloat epsilon, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
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

  int numGroupsInt = static_cast<int>(numGroups);
  float epsilonFloat = epsilon.convertToFloat();

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto groupNormQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::group_norm, device, initialStateOpt, inputSpec, numGroupsInt,
        epsilonFloat, inputMaskSpec, weightSpec, biasSpec,
        /*reciprocals=*/std::nullopt,
        detail::getNullableMemoryConfig(outputLayout),
        /*dtype=*/std::nullopt,
        /*core_grid=*/std::nullopt,
        /*inplace=*/std::nullopt,
        /*output_layout=*/std::nullopt,
        /*num_out_blocks=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt,
        /*negative_mask=*/std::nullopt,
        /*use_welford=*/false);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              groupNormQuery);
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

  int numGroupsInt = static_cast<int>(numGroups);
  float epsilonFloat = epsilon.convertToFloat();

  auto groupNormQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::group_norm, device, inputSpec, numGroupsInt,
                            epsilonFloat, inputMaskSpec, weightSpec, biasSpec,
                            /*reciprocals=*/std::nullopt,
                            detail::getNullableMemoryConfig(outputLayout),
                            /*dtype=*/std::nullopt,
                            /*core_grid=*/std::nullopt,
                            /*inplace=*/std::nullopt,
                            /*output_layout=*/std::nullopt,
                            /*num_out_blocks=*/std::nullopt,
                            /*compute_kernel_config=*/std::nullopt,
                            /*negative_mask=*/std::nullopt,
                            /*use_welford=*/false);
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
/// Convert a clamp min/max mlir::Attribute (F32Attr or I32Attr) to the
/// std::variant tt-metal expects, mirroring the runtime's NumberType dispatch.
static std::optional<std::variant<float, int32_t>>
clampAttrToVariant(mlir::Attribute attr) {
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    return static_cast<int32_t>(intAttr.getValue().getSExtValue());
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
    return static_cast<float>(floatAttr.getValueAsDouble());
  }
  return std::nullopt;
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<ClampScalarOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute min, mlir::Attribute max, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, min, max,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ClampScalarOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute min, mlir::Attribute max, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {

#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  auto memConfig = detail::getNullableMemoryConfig(outputLayout);
  auto minVariant = clampAttrToVariant(min);
  auto maxVariant = clampAttrToVariant(max);

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto clampScalarQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(::ttnn::clamp, device,
                                           initialStateOpt, inputSpec,
                                           minVariant, maxVariant, memConfig);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              clampScalarQuery);
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

  auto memConfig = detail::getNullableMemoryConfig(outputLayout);
  auto minVariant = clampAttrToVariant(min);
  auto maxVariant = clampAttrToVariant(max);

  auto clampScalarQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::clamp, device, inputSpec, minVariant,
                            maxVariant, memConfig);
  };

  return operation::getOpRuntime(clampScalarQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// ClampTensor
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<ClampTensorOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
    llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, minShape, minLayout,
                                   maxShape, maxLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ClampTensorOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> minShape, TTNNLayoutAttr minLayout,
    llvm::ArrayRef<int64_t> maxShape, TTNNLayoutAttr maxLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto clampTensorQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::clamp, device, initialStateOpt, inputSpec, minSpec, maxSpec,
        detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              clampTensorQuery);
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

  // Create query closure
  auto clampTensorQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::clamp, device, inputSpec, minSpec, maxSpec,
                            detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpRuntime(clampTensorQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// Permute
//===----------------------------------------------------------------------===//
llvm::Expected<OpConstraints> OpModel<PermuteOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, permutation,
                                   padValue, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<PermuteOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> permutation, llvm::APFloat padValue,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert permutations of TTNN_PermuteOp to dims of ttnn::permute
  ::ttsl::SmallVector<int64_t> dims(permutation.size());
  std::copy(permutation.begin(), permutation.end(), dims.begin());

  float defaultedPadValue = padValue.convertToFloat();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto permuteQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::permute, device, initialStateOpt, inputSpec, dims,
        detail::getNullableMemoryConfig(outputLayout), defaultedPadValue);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              permuteQuery);
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

  // Convert permutations of TTNN_PermuteOp to dims of ttnn::permute
  ::ttsl::SmallVector<int64_t> dims(permutation.size());
  std::copy(permutation.begin(), permutation.end(), dims.begin());

  // Convert float
  float defaultedPadValue = padValue.convertToFloat();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  // Create query closure
  auto permuteQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::permute, device, inputSpec, dims,
                            detail::getNullableMemoryConfig(outputLayout),
                            defaultedPadValue);
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
  return getOpConstraintsWithState(inputShape, inputLayout, scaleFactor, mode,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<UpsampleOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    mlir::Attribute scaleFactor, llvm::StringRef mode,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {

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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto upsampleQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::upsample, device, initialStateOpt, inputSpec,
        convertedScaleFactor, std::string(mode),
        detail::getNullableMemoryConfig(outputLayout),
        /*compute_kernel_config=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              upsampleQuery);
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
struct EmbeddingOpArgs {
  ::ttnn::TensorSpec inputSpec;
  ::ttnn::TensorSpec weightSpec;
};

llvm::Expected<EmbeddingOpArgs> getEmbeddingOpArgs(
    ::tt::tt_metal::distributed::MeshDevice *device,
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout) {
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec weightSpec,
      detail::convertToTensorSpec(device, weightShape, weightLayout));

  return EmbeddingOpArgs{inputSpec, weightSpec};
}
#endif // TTMLIR_ENABLE_OPMODEL

llvm::Expected<OpConstraints> OpModel<EmbeddingOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, weightShape,
                                   weightLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<EmbeddingOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  llvm::Expected<EmbeddingOpArgs> embeddingOpArgsExp = getEmbeddingOpArgs(
      device, inputShape, inputLayout, weightShape, weightLayout);
  if (!embeddingOpArgsExp) {
    return embeddingOpArgsExp.takeError();
  }
  EmbeddingOpArgs &embeddingOpArgs = embeddingOpArgsExp.get();

  // sgholamiTT: For the following arguments, I tried to follow the same pattern
  // as in the runtime/embedding.cpp. Subject to change in the future.
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout =
      outputLayout ? (outputLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT)
                   : (weightLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT);
  auto embeddingsType = ::ttnn::prim::EmbeddingsType::GENERIC;
  std::optional<::ttnn::DataType> dtype =
      outputLayout ? std::make_optional(
                         conversion::getDataType(outputLayout.getDataType()))
                   : std::nullopt;

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto embeddingOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::embedding, device, initialStateOpt, embeddingOpArgs.inputSpec,
        embeddingOpArgs.weightSpec, padToken, layout, embeddingsType, dtype,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  llvm::Expected<EmbeddingOpArgs> embeddingOpArgsExp = getEmbeddingOpArgs(
      device, inputShape, inputLayout, weightShape, weightLayout);
  if (!embeddingOpArgsExp) {
    return embeddingOpArgsExp.takeError();
  }
  EmbeddingOpArgs &embeddingOpArgs = embeddingOpArgsExp.get();

  // sgholamiTT: For the following arguments, I tried to follow the same pattern
  // as in the runtime/embedding.cpp. Subject to change in the future.
  std::optional<int> padToken = std::nullopt;
  ::ttnn::Layout layout =
      outputLayout ? (outputLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT)
                   : (weightLayout.isTiled() ? ::ttnn::TILE_LAYOUT
                                             : ::ttnn::ROW_MAJOR_LAYOUT);
  auto embeddingsType = ::ttnn::prim::EmbeddingsType::GENERIC;
  std::optional<::ttnn::DataType> dtype =
      outputLayout ? std::make_optional(
                         conversion::getDataType(outputLayout.getDataType()))
                   : std::nullopt;

  auto embeddingOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::embedding, device, embeddingOpArgs.inputSpec,
        embeddingOpArgs.weightSpec, padToken, layout, embeddingsType, dtype,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt);
  };

  return operation::getOpRuntime(embeddingOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// EmbeddingBackwardOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<EmbeddingBackwardOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> inGradientShape, TTNNLayoutAttr inGradientLayout,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, weightShape,
                                   weightLayout, inGradientShape,
                                   inGradientLayout, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<EmbeddingBackwardOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> weightShape, TTNNLayoutAttr weightLayout,
    llvm::ArrayRef<int64_t> inGradientShape, TTNNLayoutAttr inGradientLayout,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
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

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto embeddingBackwardOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::embedding_bw, device, initialStateOpt, inputSpec, weightSpec,
        inGradientSpec,
        /*dtype*/ std::nullopt, detail::getNullableMemoryConfig(outputLayout),
        /*optional_output_tensor*/ std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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

  auto embeddingBackwardOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::embedding_bw, device, inputSpec, weightSpec, inGradientSpec,
        /*dtype*/ std::nullopt, detail::getNullableMemoryConfig(outputLayout),
        /*optional_output_tensor*/ std::nullopt);
  };

  return operation::getOpRuntime(embeddingBackwardOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<GatherOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout, int32_t dim,
    TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, indexShape,
                                   indexLayout, dim, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<GatherOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    llvm::ArrayRef<int64_t> indexShape, TTNNLayoutAttr indexLayout, int32_t dim,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec indexSpec,
      detail::convertToTensorSpec(device, indexShape, indexLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto gatherOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::gather, device, initialStateOpt, inputSpec,
        static_cast<int8_t>(dim), indexSpec,
        /*sparse_grad=*/false, detail::getNullableMemoryConfig(outputLayout),
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              gatherOpQuery);
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

  auto gatherOpQuery = [=]() {
    return QUERY_OP_RUNTIME(
        ::ttnn::gather, device, inputSpec, static_cast<int8_t>(dim), indexSpec,
        /*sparse_grad=*/false, detail::getNullableMemoryConfig(outputLayout),
        /*optional_output_tensor=*/std::nullopt,
        /*sub_core_grids=*/std::nullopt);
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
    return QUERY_OP_CONSTRAINTS(::ttnn::arange, device, start.getSInt(),
                                end.getSInt(), step.getSInt(), dataType,
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
  return getOpConstraintsWithState(value, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<ConstantOp>::getOpConstraintsWithState(
    mlir::ElementsAttr value, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  std::optional<::tt::tt_metal::Layout> metalLayout = std::nullopt;
  if (outputLayout) {
    metalLayout = conversion::getPageLayout(outputLayout);
  }
  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;
  auto func = [&](auto rawData) {
    auto constantOpQuery = [=]() {
      return QUERY_OP_CONSTRAINTS_WITH_STATE(
          ::ttnn::from_buffer, device, initialStateOpt, rawData,
          getShape(value), getDataType(value), device, metalLayout,
          detail::getNullableMemoryConfig(outputLayout));
    };
    return operation::getOpConstraintsWithState(value.getContext(),
                                                constantOpQuery);
  };
  return dispatchGetRawData(value, func);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints>
OpModel<mlir::tt::ttnn::AssignOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> outputDtype) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::tt_metal::MemoryConfig metalMemConfig =
      conversion::getMemoryConfig(MemoryConfigAttr::get(inputLayout));

  // Convert optional output dtype
  std::optional<::tt::tt_metal::DataType> metalOutputDtype = std::nullopt;
  if (outputDtype.has_value()) {
    metalOutputDtype = conversion::getDataType(outputDtype.value());
  }
  // Create query closure
  auto assignOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS(::ttnn::assign, device, inputSpec,
                                metalMemConfig, metalOutputDtype,
                                std::nullopt /*optionalOutputTensor*/);
  };

  return operation::getOpConstraints(inputLayout.getContext(), assignOpQuery);
#else
  return OpConstraints{};
#endif // TTMLIR_ENABLE_OPMODEL
}

llvm::Expected<size_t> OpModel<mlir::tt::ttnn::AssignOp>::getOpRuntime(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout,
    std::optional<mlir::tt::ttcore::DataType> outputDtype) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  ::tt::tt_metal::MemoryConfig metalMemConfig =
      conversion::getMemoryConfig(inputLayout);

  // Convert optional output dtype
  std::optional<::tt::tt_metal::DataType> metalOutputDtype = std::nullopt;
  if (outputDtype.has_value()) {
    metalOutputDtype = conversion::getDataType(outputDtype.value());
  }

  // Create query closure
  auto assignOpQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::assign, device, inputSpec, metalMemConfig,
                            metalOutputDtype,
                            std::nullopt /*optionalOutputTensor*/);
  };

  return operation::getOpRuntime(assignOpQuery);
#else
  return llvm::createStringError("Not Implemented");
#endif // TTMLIR_ENABLE_OPMODEL
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

llvm::Expected<OpConstraints> OpModel<TopKOp>::getOpConstraints(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t k,
    int32_t dim, bool largest, bool sorted, TTNNLayoutAttr outputLayout) {
  return getOpConstraintsWithState(inputShape, inputLayout, k, dim, largest,
                                   sorted, outputLayout,
                                   /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<TopKOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t k,
    int32_t dim, bool largest, bool sorted, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto topKQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::topk, device, initialStateOpt, inputSpec,
        static_cast<uint32_t>(k), static_cast<int8_t>(dim), largest, sorted,
        detail::getNullableMemoryConfig(outputLayout), std::nullopt,
        std::nullopt, std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
                                              topKQuery);
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

  // Create query closure
  auto topKQuery = [=]() {
    return QUERY_OP_RUNTIME(::ttnn::topk, device, inputSpec,
                            static_cast<uint32_t>(k), static_cast<int8_t>(dim),
                            largest, sorted,
                            detail::getNullableMemoryConfig(outputLayout),
                            std::nullopt, std::nullopt, std::nullopt);
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
  return getOpConstraintsWithState(
      inputValuesShape, inputValuesLayout, inputIndicesShape,
      inputIndicesLayout, kShape, kLayout, pShape, pLayout, tempShape,
      tempLayout, seed, outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints> OpModel<SamplingOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputValuesShape, TTNNLayoutAttr inputValuesLayout,
    llvm::ArrayRef<int64_t> inputIndicesShape,
    TTNNLayoutAttr inputIndicesLayout, llvm::ArrayRef<int64_t> kShape,
    TTNNLayoutAttr kLayout, llvm::ArrayRef<int64_t> pShape,
    TTNNLayoutAttr pLayout, llvm::ArrayRef<int64_t> tempShape,
    TTNNLayoutAttr tempLayout, std::optional<uint32_t> seed,
    TTNNLayoutAttr outputLayout, const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // OpModel queries happen before the workaround pass (where
  // SamplingOpRank2RewritePattern unsqueezes rank-2 inputs to the kernel-true
  // rank-4 form). Pad to rank-4 here so the constraint query sees the shape
  // ttnn::sampling actually accepts.
  llvm::SmallVector<int64_t, 4> values4D, indices4D;
  llvm::ArrayRef<int64_t> valuesQueryShape = inputValuesShape;
  llvm::ArrayRef<int64_t> indicesQueryShape = inputIndicesShape;
  if (inputValuesShape.size() == 2) {
    values4D = {1, 1, inputValuesShape[0], inputValuesShape[1]};
    valuesQueryShape = values4D;
  }
  if (inputIndicesShape.size() == 2) {
    indices4D = {1, 1, inputIndicesShape[0], inputIndicesShape[1]};
    indicesQueryShape = indices4D;
  }

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valuesSpec,
      detail::convertToTensorSpec(device, valuesQueryShape, inputValuesLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec indicesSpec,
                   detail::convertToTensorSpec(device, indicesQueryShape,
                                               inputIndicesLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec kSpec,
                   detail::convertToTensorSpec(device, kShape, kLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec pSpec,
                   detail::convertToTensorSpec(device, pShape, pLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec tempSpec,
                   detail::convertToTensorSpec(device, tempShape, tempLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  auto samplingQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::sampling, device, initialStateOpt, valuesSpec, indicesSpec,
        kSpec, pSpec, tempSpec, seed, std::nullopt, std::nullopt);
  };

  return operation::getOpConstraintsWithState(inputValuesLayout.getContext(),
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

  // See getOpConstraints: rank-2 IR is padded to rank-4 for the kernel query
  // because the workaround pass runs after OpModel queries.
  llvm::SmallVector<int64_t, 4> values4D, indices4D;
  llvm::ArrayRef<int64_t> valuesQueryShape = inputValuesShape;
  llvm::ArrayRef<int64_t> indicesQueryShape = inputIndicesShape;
  if (inputValuesShape.size() == 2) {
    values4D = {1, 1, inputValuesShape[0], inputValuesShape[1]};
    valuesQueryShape = values4D;
  }
  if (inputIndicesShape.size() == 2) {
    indices4D = {1, 1, inputIndicesShape[0], inputIndicesShape[1]};
    indicesQueryShape = indices4D;
  }

  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec valuesSpec,
      detail::convertToTensorSpec(device, valuesQueryShape, inputValuesLayout));
  ASSIGN_OR_RETURN(::ttnn::TensorSpec indicesSpec,
                   detail::convertToTensorSpec(device, indicesQueryShape,
                                               inputIndicesLayout));
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
  return getOpConstraintsWithState(inputShape, inputLayout, dim, clusterAxis,
                                   outputLayout, /*initialState=*/nullptr);
}

llvm::Expected<OpConstraints>
OpModel<MeshPartitionOp>::getOpConstraintsWithState(
    llvm::ArrayRef<int64_t> inputShape, TTNNLayoutAttr inputLayout, int32_t dim,
    std::optional<uint32_t> clusterAxis, TTNNLayoutAttr outputLayout,
    const MockAllocatorState *initialState) {
#ifdef TTMLIR_ENABLE_OPMODEL
  ::tt::tt_metal::distributed::MeshDevice *device =
      SingletonDeviceContext::getInstance().getDevice();

  // Convert input tensor to TensorSpec
  ASSIGN_OR_RETURN(
      ::ttnn::TensorSpec inputSpec,
      detail::convertToTensorSpec(device, inputShape, inputLayout));

  std::optional<MockAllocatorState> initialStateOpt =
      initialState ? std::optional<MockAllocatorState>(*initialState)
                   : std::nullopt;

  // Create query closure
  auto meshPartitionOpQuery = [=]() {
    return QUERY_OP_CONSTRAINTS_WITH_STATE(
        ::ttnn::mesh_partition, device, initialStateOpt, inputSpec, dim,
        clusterAxis, detail::getNullableMemoryConfig(outputLayout));
  };

  return operation::getOpConstraintsWithState(inputLayout.getContext(),
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
