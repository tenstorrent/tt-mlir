// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/utils.h"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

template <typename FromType, typename ToType>
ToType calculateExpected(FromType value) {
  static_assert(std::is_integral_v<FromType>, "FromType must be integral");
  static_assert(std::is_integral_v<ToType>, "ToType must be integral");

  // Handle same type conversion (no-op)
  if constexpr (std::is_same_v<FromType, ToType>) {
    return value;
  }

  // Handle sign conversion cases
  if constexpr (std::is_signed_v<FromType> != std::is_signed_v<ToType>) {
    // Signed to unsigned conversion
    if constexpr (std::is_signed_v<FromType> && std::is_unsigned_v<ToType>) {
      // Negative values become 0 (or min value for unsigned, which is 0)
      if (value < 0) {
        return std::numeric_limits<ToType>::min(); // 0 for unsigned
      }
      // Check if positive value exceeds unsigned max
      if (static_cast<std::make_unsigned_t<FromType>>(value) >
          std::numeric_limits<ToType>::max()) {
        return std::numeric_limits<ToType>::max();
      }
      return static_cast<ToType>(value);
    }
    // Unsigned to signed conversion
    else if constexpr (std::is_unsigned_v<FromType> &&
                       std::is_signed_v<ToType>) {
      // Check if unsigned value exceeds signed max
      if (value > static_cast<std::make_unsigned_t<ToType>>(
                      std::numeric_limits<ToType>::max())) {
        return std::numeric_limits<ToType>::max();
      }
      return static_cast<ToType>(value);
    }
  }

  // Same signedness conversions (both signed or both unsigned)
  else {
    // Use safe comparison to avoid issues with different sized types
    using CommonType = std::common_type_t<FromType, ToType>;

    // Check lower bound
    if (static_cast<CommonType>(value) <
        static_cast<CommonType>(std::numeric_limits<ToType>::min())) {
      return std::numeric_limits<ToType>::min();
    }

    // Check upper bound
    if (static_cast<CommonType>(value) >
        static_cast<CommonType>(std::numeric_limits<ToType>::max())) {
      return std::numeric_limits<ToType>::max();
    }

    return static_cast<ToType>(value);
  }
}

template <typename From, typename To, size_t NumElements>
struct IntegerCastParam {
  using FromType = From;
  using ToType = To;
  static constexpr size_t numElements = NumElements;
  static constexpr bool isNarrowing = sizeof(From) > sizeof(To);
  static constexpr bool isSignChange =
      std::is_signed_v<From> != std::is_signed_v<To>;
};

template <typename T>
class IntegerBufferCast : public ::testing::Test {};

using TestParamCombinations = ::testing::Types<
    IntegerCastParam<int64_t, int32_t, 3>,
    IntegerCastParam<int64_t, int16_t, 3>, IntegerCastParam<int64_t, int8_t, 3>,
    IntegerCastParam<int64_t, uint64_t, 3>,
    IntegerCastParam<int64_t, uint32_t, 3>,
    IntegerCastParam<int64_t, uint16_t, 3>,
    IntegerCastParam<int64_t, uint8_t, 3>,
    IntegerCastParam<int32_t, int64_t, 3>,
    IntegerCastParam<int32_t, int16_t, 3>, IntegerCastParam<int32_t, int8_t, 3>,
    IntegerCastParam<int32_t, uint64_t, 3>,
    IntegerCastParam<int32_t, uint32_t, 3>,
    IntegerCastParam<int32_t, uint16_t, 3>,
    IntegerCastParam<int32_t, uint8_t, 3>,
    IntegerCastParam<int16_t, int64_t, 3>,
    IntegerCastParam<int16_t, int32_t, 3>, IntegerCastParam<int16_t, int8_t, 3>,
    IntegerCastParam<int16_t, uint64_t, 3>,
    IntegerCastParam<int16_t, uint32_t, 3>,
    IntegerCastParam<int16_t, uint16_t, 3>,
    IntegerCastParam<int16_t, uint8_t, 3>, IntegerCastParam<int8_t, int64_t, 3>,
    IntegerCastParam<int8_t, int32_t, 3>, IntegerCastParam<int8_t, int16_t, 3>,
    IntegerCastParam<int8_t, uint64_t, 3>,
    IntegerCastParam<int8_t, uint32_t, 3>,
    IntegerCastParam<int8_t, uint16_t, 3>>;

TYPED_TEST_SUITE(IntegerBufferCast, TestParamCombinations);

TYPED_TEST(IntegerBufferCast, BasicTests) {
  using FromType = typename TypeParam::FromType;
  using ToType = typename TypeParam::ToType;
  constexpr size_t numElements = TypeParam::numElements;

  FromType old_buffer[numElements];
  ToType new_buffer[numElements];

  for (size_t i = 0; i < numElements; ++i) {
    if (i == 0) {
      old_buffer[i] = std::numeric_limits<FromType>::min();
    } else if (i == numElements - 1) {
      old_buffer[i] = std::numeric_limits<FromType>::max();
    } else {
      old_buffer[i] = static_cast<FromType>(i);
    }
    new_buffer[i] = ToType{0};
  }

  tt::runtime::utils::detail::handleIntegerBufferCast(old_buffer, new_buffer,
                                                      numElements);

  for (size_t i = 0; i < numElements; ++i) {
    ToType expected = calculateExpected<FromType, ToType>(old_buffer[i]);
    EXPECT_EQ(new_buffer[i], expected)
        << "Element " << i << " failed for " << typeid(FromType).name() << "["
        << numElements << "] -> " << typeid(ToType).name();
  }
}
