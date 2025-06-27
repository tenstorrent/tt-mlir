// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_UTILS_H
#define TT_RUNTIME_UTILS_H

#include <memory>
#include <type_traits>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/Common/types_generated.h"
#pragma clang diagnostic pop

namespace tt::runtime::utils {

inline std::shared_ptr<void> malloc_shared(size_t size) {
  return std::shared_ptr<void>(std::malloc(size), std::free);
}

template <typename T>
inline std::shared_ptr<const void> unsafe_borrow_shared(const T *ptr) {
  return std::shared_ptr<const void>(static_cast<const void *>(ptr),
                                     [](const void *) {});
}

template <typename T>
inline std::shared_ptr<void> unsafe_borrow_shared(T *ptr) {
  return std::shared_ptr<void>(static_cast<void *>(ptr), [](void *) {});
}

inline std::uint32_t dataTypeElementSize(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float64:
  case ::tt::target::DataType::Int64:
  case ::tt::target::DataType::UInt64:
    return 8;
  case ::tt::target::DataType::Float32:
  case ::tt::target::DataType::UInt32:
  case ::tt::target::DataType::Int32:
    return 4;
  case ::tt::target::DataType::Float16:
  case ::tt::target::DataType::BFloat16:
  case ::tt::target::DataType::UInt16:
  case ::tt::target::DataType::Int16:
    return 2;
  case ::tt::target::DataType::UInt8:
  case ::tt::target::DataType::Int8:
  case ::tt::target::DataType::Bool:
    return 1;
  default:
    assert(false && "Unsupported element size for data type");
    return 0;
  }
}

inline std::int64_t tileRowAlignment(::tt::target::DataType dataType) {
  std::int64_t numAlignElems = 32;
  return dataTypeElementSize(dataType) * numAlignElems;
}

inline std::int64_t tileAlignment(::tt::target::DataType dataType) {
  std::int64_t numAlignRows = 32;
  return tileRowAlignment(dataType) * numAlignRows;
}

inline bool isSupportedDataType(::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
  case ::tt::target::DataType::Float16:
  case ::tt::target::DataType::BFloat16:
  case ::tt::target::DataType::BFP_Float8:
  case ::tt::target::DataType::BFP_BFloat8:
  case ::tt::target::DataType::BFP_Float4:
  case ::tt::target::DataType::BFP_BFloat4:
  case ::tt::target::DataType::BFP_Float2:
  case ::tt::target::DataType::BFP_BFloat2:
  case ::tt::target::DataType::UInt32:
  case ::tt::target::DataType::UInt16:
  case ::tt::target::DataType::UInt8:
  case ::tt::target::DataType::Int32:
    return true;
  case ::tt::target::DataType::Float64:
  case ::tt::target::DataType::Int64:
  case ::tt::target::DataType::UInt64:
  case ::tt::target::DataType::Int16:
  case ::tt::target::DataType::Int8:
  case ::tt::target::DataType::Bool:
    return false;
  }
}

inline ::tt::target::DataType
getUnsupportedDataTypeAlias(::tt::target::DataType unsupportedDataType) {
  switch (unsupportedDataType) {
  case ::tt::target::DataType::Float64:
    return ::tt::target::DataType::Float32;
  case ::tt::target::DataType::Int64:
    return ::tt::target::DataType::Int32;
  case ::tt::target::DataType::UInt64:
    return ::tt::target::DataType::UInt32;
  case ::tt::target::DataType::Int16:
    return ::tt::target::DataType::UInt16;
  case ::tt::target::DataType::Int8:
    return ::tt::target::DataType::UInt8;
  case ::tt::target::DataType::Bool:
    return ::tt::target::DataType::BFloat16;
  default:
    throw std::runtime_error(
        "The data type: " +
        std::string(target::EnumNameDataType(unsupportedDataType)) +
        " is either supported and thus needs no alias OR it is not supported "
        "and is not accounted for in this function (that would be a bug).");
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
inline std::vector<uint32_t> calculateStride(const std::vector<T> &shape) {
  assert(!shape.empty());
  std::vector<uint32_t> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <typename T>
T alignUp(T ptr, T alignment) {
  return (ptr + alignment - 1) & ~(alignment - 1);
}

namespace detail {

template <typename FromTy, typename ToTy>
inline void handleUncheckedBufferCast(const FromTy *old_buffer,
                                      ToTy *new_buffer, int64_t num_elements) {
  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
  }
}

template <typename FromTy, typename ToTy>
inline void handleEquallySignedIntegerBufferCast(const FromTy *old_buffer,
                                                 ToTy *new_buffer,
                                                 int64_t num_elements) {
  static_assert(
      (std::is_signed<FromTy>::value && std::is_signed<ToTy>::value) ||
          (std::is_unsigned<FromTy>::value && std::is_unsigned<ToTy>::value),
      "Source and destination types must be of the same signedness");

  static_assert(!std::is_same<FromTy, ToTy>::value,
                "Source and destination types are the same! Please use "
                "std::memcpy(new_buffer, old_buffer, num_elements * "
                "sizeof(FromTy)) instead.");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  constexpr ToTy toTyMin = std::numeric_limits<ToTy>::min();
  if constexpr (sizeof(ToTy) >= sizeof(FromTy)) {
    for (int64_t i = 0; i < num_elements; i++) {
      new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
    }
  } else {
    for (int64_t i = 0; i < num_elements; i++) {
      if (old_buffer[i] < static_cast<FromTy>(toTyMin)) {
        new_buffer[i] = toTyMin;
      } else if (old_buffer[i] > static_cast<FromTy>(toTyMax)) {
        new_buffer[i] = toTyMax;
      } else {
        new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleUnsignedToSignedIntegerBufferCast(const FromTy *old_buffer,
                                                    ToTy *new_buffer,
                                                    int64_t num_elements) {
  static_assert(std::is_unsigned<FromTy>::value,
                "Source type must be an unsigned integer type");
  static_assert(std::is_signed<ToTy>::value,
                "Destination type must be a signed integer type");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  if constexpr (sizeof(ToTy) > sizeof(FromTy)) {
    for (int64_t i = 0; i < num_elements; i++) {
      new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
    }
  } else { // unsigned FromTy has larger or equal bitwidth than signed ToTy
    for (int64_t i = 0; i < num_elements; i++) {
      if (old_buffer[i] > static_cast<FromTy>(toTyMax)) {
        new_buffer[i] = toTyMax;
      } else {
        new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleSignedToUnsignedIntegerBufferCast(const FromTy *old_buffer,
                                                    ToTy *new_buffer,
                                                    int64_t num_elements) {
  static_assert(std::is_signed<FromTy>::value,
                "Source type must be a signed integer type");
  static_assert(std::is_unsigned<ToTy>::value,
                "Destination type must be an unsigned integer type");

  constexpr ToTy toTyMax = std::numeric_limits<ToTy>::max();
  constexpr ToTy toTyMin = std::numeric_limits<ToTy>::min();
  if constexpr (sizeof(ToTy) >= sizeof(FromTy)) {
    for (int64_t i = 0; i < num_elements; i++) {
      // when the signed and unsigned type have the same bitwidth then
      // it is impossible for the signed type to be beyond the maximum
      // unsigned value
      if (old_buffer[i] < static_cast<FromTy>(toTyMin)) {
        new_buffer[i] = toTyMin;
      } else {
        new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
      }
    }
  } else { // signed FromTy has larger or equal bitwidth than unsigned ToTy
    for (int64_t i = 0; i < num_elements; i++) {
      if (old_buffer[i] < static_cast<FromTy>(toTyMin)) {
        new_buffer[i] = toTyMin;
      } else if (old_buffer[i] > static_cast<FromTy>(toTyMax)) {
        new_buffer[i] = toTyMax;
      } else {
        new_buffer[i] = static_cast<ToTy>(old_buffer[i]);
      }
    }
  }
}

template <typename FromTy, typename ToTy>
inline void handleIntegerBufferCast(const FromTy *old_buffer, ToTy *new_buffer,
                                    int64_t num_elements) {
  assert(old_buffer && new_buffer && "Buffer pointers must not be null");
  static_assert(std::is_integral<FromTy>::value,
                "Source type must be an integer type");
  static_assert(std::is_integral<ToTy>::value,
                "Destination type must be an integer type");

  if constexpr ((std::is_signed<FromTy>::value &&
                 std::is_signed<ToTy>::value) ||
                (std::is_unsigned<FromTy>::value &&
                 std::is_unsigned<ToTy>::value)) {
    detail::handleEquallySignedIntegerBufferCast<FromTy, ToTy>(
        old_buffer, new_buffer, num_elements);
  } else if constexpr (std::is_unsigned<FromTy>::value &&
                       std::is_signed<ToTy>::value) {
    detail::handleUnsignedToSignedIntegerBufferCast<FromTy, ToTy>(
        old_buffer, new_buffer, num_elements);
  } else if constexpr (std::is_signed<FromTy>::value &&
                       std::is_unsigned<ToTy>::value) {
    detail::handleSignedToUnsignedIntegerBufferCast<FromTy, ToTy>(
        old_buffer, new_buffer, num_elements);
  } else { // Both are unsigned
    throw std::runtime_error("Unhandled integer buffer cast case");
  }
}

inline void handleBFloat16ToBool(const uint16_t *old_buffer, bool *new_buffer,
                                 int64_t num_elements) {
  assert(old_buffer && new_buffer && "Buffer pointers must not be null");
  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] =
        old_buffer[i] !=
        0; // 0 in bfloat16 is also 00000000 00000000, just as in uint16_t
  }
}

inline void handleBoolToBFloat16(const bool *old_buffer, uint16_t *new_buffer,
                                 int64_t num_elements) {
  assert(old_buffer && new_buffer && "Buffer pointers must not be null");
  assert(sizeof(bool) == 1 && "bool must be 1 byte");

  for (int i = 0; i < num_elements; i++) {
    new_buffer[i] = old_buffer[i]
                        ? 0x3f80
                        : 0; // 0x3f80 is the bfloat16 representation of 1.0
  }
}
} // namespace detail

inline void handleBufferCast(const void *old_buffer, void *new_buffer,
                             target::DataType oldDataType,
                             target::DataType newDataType,
                             int64_t num_elements) {
  if (!old_buffer || !new_buffer) {
    throw std::runtime_error("Buffer pointers must not be null");
  }
  if (oldDataType == newDataType) {
    std::memcpy(new_buffer, old_buffer,
                num_elements * dataTypeElementSize(oldDataType));
    return;
  }

  if (oldDataType == tt::target::DataType::Int64 &&
      newDataType == tt::target::DataType::Int32) {
    detail::handleIntegerBufferCast<int64_t, int32_t>(
        static_cast<const int64_t *>(old_buffer),
        static_cast<int32_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Int32 &&
             newDataType == tt::target::DataType::Int64) {
    detail::handleIntegerBufferCast<int32_t, int64_t>(
        static_cast<const int32_t *>(old_buffer),
        static_cast<int64_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::UInt64 &&
             newDataType == tt::target::DataType::UInt32) {
    detail::handleIntegerBufferCast<uint64_t, uint32_t>(
        static_cast<const uint64_t *>(old_buffer),
        static_cast<uint32_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::UInt32 &&
             newDataType == tt::target::DataType::UInt64) {
    detail::handleIntegerBufferCast<uint32_t, uint64_t>(
        static_cast<const uint32_t *>(old_buffer),
        static_cast<uint64_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Int16 &&
             newDataType == tt::target::DataType::UInt16) {
    detail::handleIntegerBufferCast<int16_t, uint16_t>(
        static_cast<const int16_t *>(old_buffer),
        static_cast<uint16_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::UInt16 &&
             newDataType == tt::target::DataType::Int16) {
    detail::handleIntegerBufferCast<uint16_t, int16_t>(
        static_cast<const uint16_t *>(old_buffer),
        static_cast<int16_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Int8 &&
             newDataType == tt::target::DataType::UInt8) {
    detail::handleIntegerBufferCast<int8_t, uint8_t>(
        static_cast<const int8_t *>(old_buffer),
        static_cast<uint8_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::UInt8 &&
             newDataType == tt::target::DataType::Int8) {
    detail::handleIntegerBufferCast<uint8_t, int8_t>(
        static_cast<const uint8_t *>(old_buffer),
        static_cast<int8_t *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Float32 &&
             newDataType == tt::target::DataType::Float64) {
    detail::handleUncheckedBufferCast<float, double>(
        static_cast<const float *>(old_buffer),
        static_cast<double *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Float64 &&
             newDataType == tt::target::DataType::Float32) {
    detail::handleUncheckedBufferCast<double, float>(
        static_cast<const double *>(old_buffer),
        static_cast<float *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::BFloat16 &&
             newDataType == tt::target::DataType::Bool) {
    detail::handleBFloat16ToBool(static_cast<const uint16_t *>(old_buffer),
                                 static_cast<bool *>(new_buffer), num_elements);
  } else if (oldDataType == tt::target::DataType::Bool &&
             newDataType == tt::target::DataType::BFloat16) {
    detail::handleBoolToBFloat16(static_cast<const bool *>(old_buffer),
                                 static_cast<uint16_t *>(new_buffer),
                                 num_elements);
  } else {
    throw std::runtime_error(
        "Unhandled buffer cast case: From " +
        std::string(target::EnumNameDataType(oldDataType)) + " to " +
        std::string(target::EnumNameDataType(newDataType)));
  }
}

} // namespace tt::runtime::utils

#endif
