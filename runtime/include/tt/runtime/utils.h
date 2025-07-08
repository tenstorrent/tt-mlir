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
  case ::tt::target::DataType::Float16:
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
  case ::tt::target::DataType::Float16:
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
inline void handleUncheckedBufferCast(const FromTy *oldBuffer, ToTy *newBuffer,
                                      int64_t numElements) {
  for (int i = 0; i < numElements; i++) {
    newBuffer[i] = static_cast<ToTy>(oldBuffer[i]);
  }
}

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

inline void handleBFloat16ToBool(const uint16_t *oldBuffer, bool *newBuffer,
                                 int64_t numElements) {
  assert(oldBuffer && newBuffer && "Buffer pointers must not be null");
  for (int i = 0; i < numElements; i++) {
    newBuffer[i] =
        oldBuffer[i] !=
        0; // 0 in bfloat16 is also 00000000 00000000, just as in uint16_t
  }
}

inline void handleBoolToBFloat16(const bool *oldBuffer, uint16_t *newBuffer,
                                 int64_t numElements) {
  assert(oldBuffer && newBuffer && "Buffer pointers must not be null");
  assert(sizeof(bool) == 1 && "bool must be 1 byte");

  for (int i = 0; i < numElements; i++) {
    newBuffer[i] = oldBuffer[i]
                       ? 0x3f80
                       : 0; // 0x3f80 is the bfloat16 representation of 1.0
  }
}

inline void handleFloat16ToBFloat16(const uint16_t *oldBuffer,
                                    uint16_t *newBuffer, int64_t numElements) {
  assert(oldBuffer && newBuffer && "Buffer pointers must not be null");

  for (int64_t i = 0; i < numElements; ++i) {
    uint16_t f16 = oldBuffer[i];

    // Decompose float16
    uint16_t sign = (f16 >> 15) & 0x1;
    uint16_t exponent = (f16 >> 10) & 0x1F;
    uint16_t mantissa = f16 & 0x3FF;

    uint32_t f32Bits = 0;

    if (exponent == 0) {
      if (mantissa == 0) {
        // Zero
        f32Bits = static_cast<uint32_t>(sign) << 31;
      } else {
        // Subnormal float16 → normalize to float32 subnormal
        // Shift mantissa left until leading 1
        int shift = 0;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          ++shift;
        }
        mantissa &= 0x3FF; // Remove the leading 1
        exponent = 1;
        exponent -= shift;

        int32_t exp32 = static_cast<int32_t>(exponent) - 15 + 127;
        f32Bits = (static_cast<uint32_t>(sign) << 31) |
                  (static_cast<uint32_t>(exp32) << 23) |
                  (static_cast<uint32_t>(mantissa) << 13);
      }
    } else if (exponent == 0x1F) {
      // Inf or NaN
      f32Bits = (static_cast<uint32_t>(sign) << 31) | (0xFF << 23) |
                (static_cast<uint32_t>(mantissa) << 13);
    } else {
      // Normalized number
      int32_t exp32 = static_cast<int32_t>(exponent) - 15 + 127;
      f32Bits = (static_cast<uint32_t>(sign) << 31) |
                (static_cast<uint32_t>(exp32) << 23) |
                (static_cast<uint32_t>(mantissa) << 13);
    }

    // Convert float32 bits to bfloat16 with rounding-to-nearest-even
    uint32_t lsb = (f32Bits >> 16) & 1;
    uint32_t roundingBias = 0x7FFF + lsb;
    f32Bits += roundingBias;

    uint16_t bfloat16 = static_cast<uint16_t>(f32Bits >> 16);
    newBuffer[i] = bfloat16;
  }
}

inline void handleBFloat16ToFloat16(const uint16_t *oldBuffer,
                                    uint16_t *newBuffer, int64_t numElements) {
  assert(oldBuffer && newBuffer && "Buffer pointers must not be null");
  for (int64_t i = 0; i < numElements; i++) {
    uint16_t bf16Bits = oldBuffer[i];

    // Extract components from bfloat16
    uint16_t sign = (bf16Bits >> 15) & 0x1;
    uint16_t exponent = (bf16Bits >> 7) & 0xFF;
    uint16_t mantissa = bf16Bits & 0x7F;

    // Convert to float16
    if (exponent == 0 && mantissa == 0) {
      // Zero
      newBuffer[i] = sign << 15;
    } else if (exponent == 0xFF) {
      // Infinity or NaN
      newBuffer[i] = (sign << 15) | 0x7C00 | ((mantissa != 0) ? 0x200 : 0);
    } else {
      // Normal numbers
      // Adjust exponent bias: bfloat16 bias is 127, float16 bias is 15
      int32_t newExponent = exponent - 127 + 15;

      if (newExponent <= 0) {
        // Underflow to zero
        newBuffer[i] = sign << 15;
      } else if (newExponent >= 31) {
        // Overflow to infinity
        newBuffer[i] = (sign << 15) | 0x7C00;
      } else {
        // Extend mantissa from 7 bits to 10 bits
        uint16_t extendedMantissa = mantissa << 3;
        newBuffer[i] = (sign << 15) | (newExponent << 10) | extendedMantissa;
      }
    }
  }
}

} // namespace detail

inline void handleBufferCast(const void *oldBuffer, void *newBuffer,
                             target::DataType oldDataType,
                             target::DataType newDataType,
                             int64_t numElements) {
  if (!oldBuffer || !newBuffer) {
    throw std::runtime_error("Buffer pointers must not be null");
  }
  if (oldDataType == newDataType) {
    std::memcpy(newBuffer, oldBuffer,
                numElements * dataTypeElementSize(oldDataType));
    return;
  }

  if (oldDataType == tt::target::DataType::Int64 &&
      newDataType == tt::target::DataType::Int32) {
    detail::handleIntegerBufferCast<int64_t, int32_t>(
        static_cast<const int64_t *>(oldBuffer),
        static_cast<int32_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::Int32 &&
             newDataType == tt::target::DataType::Int64) {
    detail::handleIntegerBufferCast<int32_t, int64_t>(
        static_cast<const int32_t *>(oldBuffer),
        static_cast<int64_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::UInt64 &&
             newDataType == tt::target::DataType::UInt32) {
    detail::handleIntegerBufferCast<uint64_t, uint32_t>(
        static_cast<const uint64_t *>(oldBuffer),
        static_cast<uint32_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::UInt32 &&
             newDataType == tt::target::DataType::UInt64) {
    detail::handleIntegerBufferCast<uint32_t, uint64_t>(
        static_cast<const uint32_t *>(oldBuffer),
        static_cast<uint64_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::Int16 &&
             newDataType == tt::target::DataType::UInt16) {
    detail::handleIntegerBufferCast<int16_t, uint16_t>(
        static_cast<const int16_t *>(oldBuffer),
        static_cast<uint16_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::UInt16 &&
             newDataType == tt::target::DataType::Int16) {
    detail::handleIntegerBufferCast<uint16_t, int16_t>(
        static_cast<const uint16_t *>(oldBuffer),
        static_cast<int16_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::Int8 &&
             newDataType == tt::target::DataType::UInt8) {
    detail::handleIntegerBufferCast<int8_t, uint8_t>(
        static_cast<const int8_t *>(oldBuffer),
        static_cast<uint8_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::UInt8 &&
             newDataType == tt::target::DataType::Int8) {
    detail::handleIntegerBufferCast<uint8_t, int8_t>(
        static_cast<const uint8_t *>(oldBuffer),
        static_cast<int8_t *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::Float32 &&
             newDataType == tt::target::DataType::Float64) {
    detail::handleUncheckedBufferCast<float, double>(
        static_cast<const float *>(oldBuffer), static_cast<double *>(newBuffer),
        numElements);
  } else if (oldDataType == tt::target::DataType::Float64 &&
             newDataType == tt::target::DataType::Float32) {
    detail::handleUncheckedBufferCast<double, float>(
        static_cast<const double *>(oldBuffer), static_cast<float *>(newBuffer),
        numElements);
  } else if (oldDataType == tt::target::DataType::BFloat16 &&
             newDataType == tt::target::DataType::Bool) {
    detail::handleBFloat16ToBool(static_cast<const uint16_t *>(oldBuffer),
                                 static_cast<bool *>(newBuffer), numElements);
  } else if (oldDataType == tt::target::DataType::Bool &&
             newDataType == tt::target::DataType::BFloat16) {
    detail::handleBoolToBFloat16(static_cast<const bool *>(oldBuffer),
                                 static_cast<uint16_t *>(newBuffer),
                                 numElements);
  } else if (oldDataType == tt::target::DataType::Float16 &&
             newDataType == tt::target::DataType::BFloat16) {
    detail::handleFloat16ToBFloat16(static_cast<const uint16_t *>(oldBuffer),
                                    static_cast<uint16_t *>(newBuffer),
                                    numElements);
  } else if (oldDataType == tt::target::DataType::BFloat16 &&
             newDataType == tt::target::DataType::Float16) {
    detail::handleBFloat16ToFloat16(static_cast<const uint16_t *>(oldBuffer),
                                    static_cast<uint16_t *>(newBuffer),
                                    numElements);
  } else {
    throw std::runtime_error(
        "Unhandled buffer cast case: From " +
        std::string(target::EnumNameDataType(oldDataType)) + " to " +
        std::string(target::EnumNameDataType(newDataType)));
  }
}

} // namespace tt::runtime::utils

#endif
