// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_UTILS_H
#define TT_RUNTIME_UTILS_H

#include <cstdlib>
#include <memory>
#include <numeric>
#include <type_traits>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/Common/types_generated.h"
#pragma clang diagnostic pop

// Forward declarations
namespace tt::runtime {
enum class DispatchCoreType;
} // namespace tt::runtime

namespace tt::runtime::utils {

namespace detail {

template <typename FromTy, typename ToTy>
inline void handleEquallySignedIntegerBufferCast(const FromTy *oldBuffer,
                                                 ToTy *newBuffer,
                                                 int64_t numElements) {
  static_assert(
      (std::is_signed<FromTy>::value && std::is_signed<ToTy>::value) ||
          (std::is_unsigned<FromTy>::value && std::is_unsigned<ToTy>::value),
      "Source and destination types must be of the same signedness");

  static_assert(!std::is_same<FromTy, ToTy>::value,
                "Source and destination types are the same! Please use "
                "std::memcpy(newBuffer, oldBuffer, numElements * "
                "sizeof(FromTy)) instead.");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  constexpr ToTy toTyMin = std::numeric_limits<ToTy>::min();
  if constexpr (sizeof(ToTy) >= sizeof(FromTy)) {
    for (int64_t i = 0; i < numElements; i++) {
      newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
    }
  } else {
    for (int64_t i = 0; i < numElements; i++) {
      if (oldBuffer[i] < static_cast<FromTy>(toTyMin)) {
        newBuffer[i] = toTyMin;
      } else if (oldBuffer[i] > static_cast<FromTy>(toTyMax)) {
        newBuffer[i] = toTyMax;
      } else {
        newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleUnsignedToSignedIntegerBufferCast(const FromTy *oldBuffer,
                                                    ToTy *newBuffer,
                                                    int64_t numElements) {
  static_assert(std::is_unsigned<FromTy>::value,
                "Source type must be an unsigned integer type");
  static_assert(std::is_signed<ToTy>::value,
                "Destination type must be a signed integer type");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  if constexpr (sizeof(ToTy) > sizeof(FromTy)) {
    for (int64_t i = 0; i < numElements; i++) {
      newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
    }
  } else { // unsigned FromTy has larger or equal bitwidth than signed ToTy
    for (int64_t i = 0; i < numElements; i++) {
      if (oldBuffer[i] > static_cast<FromTy>(toTyMax)) {
        newBuffer[i] = toTyMax;
      } else {
        newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleSignedToUnsignedIntegerBufferCast(const FromTy *oldBuffer,
                                                    ToTy *newBuffer,
                                                    int64_t numElements) {
  static_assert(std::is_signed<FromTy>::value,
                "Source type must be a signed integer type");
  static_assert(std::is_unsigned<ToTy>::value,
                "Destination type must be an unsigned integer type");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  constexpr ToTy toTyMin = std::numeric_limits<ToTy>::min();
  if constexpr (sizeof(ToTy) >= sizeof(FromTy)) {
    for (int64_t i = 0; i < numElements; i++) {
      // when the signed and unsigned type have the same bitwidth then
      // it is impossible for the signed type to be beyond the maximum
      // unsigned value
      if (oldBuffer[i] < static_cast<FromTy>(toTyMin)) {
        newBuffer[i] = toTyMin;
      } else {
        newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
      }
    }
  } else { // signed FromTy has larger or equal bitwidth than unsigned ToTy
    for (int64_t i = 0; i < numElements; i++) {
      if (oldBuffer[i] < static_cast<FromTy>(toTyMin)) {
        newBuffer[i] = toTyMin;
      } else if (oldBuffer[i] > static_cast<FromTy>(toTyMax)) {
        newBuffer[i] = toTyMax;
      } else {
        newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleIntegerBufferCast(const FromTy *oldBuffer, ToTy *newBuffer,
                                    int64_t numElements) {
  assert(oldBuffer && newBuffer && "Buffer pointers must not be null");
  static_assert(std::is_integral<FromTy>::value,
                "Source type must be an integer type");
  static_assert(std::is_integral<ToTy>::value,
                "Destination type must be an integer type");

  if constexpr ((std::is_signed<FromTy>::value &&
                 std::is_signed<ToTy>::value) ||
                (std::is_unsigned<FromTy>::value &&
                 std::is_unsigned<ToTy>::value)) {
    detail::handleEquallySignedIntegerBufferCast<FromTy, ToTy>(
        oldBuffer, newBuffer, numElements);
  } else if constexpr (std::is_unsigned<FromTy>::value &&
                       std::is_signed<ToTy>::value) {
    detail::handleUnsignedToSignedIntegerBufferCast<FromTy, ToTy>(
        oldBuffer, newBuffer, numElements);
  } else if constexpr (std::is_signed<FromTy>::value &&
                       std::is_unsigned<ToTy>::value) {
    detail::handleSignedToUnsignedIntegerBufferCast<FromTy, ToTy>(
        oldBuffer, newBuffer, numElements);
  } else { // Both are unsigned
    throw std::runtime_error("Unhandled integer buffer cast case");
  }
}

void handleBFloat16ToBool(const uint16_t *oldBuffer, bool *newBuffer,
                          int64_t numElements);

void handleFloat16ToBFloat16(const uint16_t *oldBuffer, uint16_t *newBuffer,
                             int64_t numElements);

void handleBFloat16ToFloat16(const uint16_t *oldBuffer, uint16_t *newBuffer,
                             int64_t numElements);

} // namespace detail

std::string getMlirHome();

std::shared_ptr<void> mallocShared(const size_t size);
std::shared_ptr<void> callocShared(const size_t size);

template <typename T>
inline std::shared_ptr<const void> unsafeBorrowShared(const T *ptr) {
  return std::shared_ptr<const void>(static_cast<const void *>(ptr),
                                     [](const void *) {});
}

template <typename T>
inline std::shared_ptr<void> unsafeBorrowShared(T *ptr) {
  return std::shared_ptr<void>(static_cast<void *>(ptr), [](void *) {});
}

::tt::target::DispatchCoreType
fromRuntimeDispatchCoreType(::tt::runtime::DispatchCoreType dispatchCoreType);

::tt::runtime::DispatchCoreType
toRuntimeDispatchCoreType(::tt::target::DispatchCoreType dispatchCoreType);

std::uint32_t dataTypeElementSize(::tt::target::DataType dataType);

bool isSupportedDataType(::tt::target::DataType dataType);

::tt::target::DataType
getUnsupportedDataTypeAlias(::tt::target::DataType unsupportedDataType);

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline std::vector<uint32_t> calculateStride(const std::vector<T> &shape) {
  // Scalar case:
  // For empty shape, return empty stride
  if (shape.empty()) {
    return {};
  }

  // For non-empty shape, calculate stride
  std::vector<uint32_t> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}

template <typename Iter>
auto product(const Iter begin, const Iter end) ->
    typename std::iterator_traits<Iter>::value_type {
  using ValueType = typename std::iterator_traits<Iter>::value_type;
  return std::accumulate(begin, end, static_cast<ValueType>(1),
                         std::multiplies<ValueType>());
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
T alignUp(const T val, const T alignment) {
  assert(alignment > 0);
  return ((val + alignment - 1) / alignment) * alignment;
}

void handleBufferCast(const void *oldBuffer, void *newBuffer,
                      target::DataType oldDataType,
                      target::DataType newDataType, int64_t numElements);

} // namespace tt::runtime::utils

#endif
