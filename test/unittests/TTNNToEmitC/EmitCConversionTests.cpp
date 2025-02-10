// // SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// //
// // SPDX-License-Identifier: Apache-2.0

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/MLIRContext.h"
// #include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"
// #include "gtest/gtest.h"

// using namespace mlir;
// using namespace tt::ttnn_to_emitc;

// class EmitCConversionTest : public ::testing::Test {
// protected:
//   EmitCConversionTest() : context(), builder(&context) {}

//   MLIRContext context;
//   Builder builder;
// };

// // Test EmitCSerializer for basic types
// TEST_F(EmitCConversionTest, BasicTypesSerialization) {
//   // Test integer serialization
//   EXPECT_EQ(EmitCSerializer<int32_t>::convert(42), "42");
//   EXPECT_EQ(EmitCSerializer<int64_t>::convert(-123), "-123");
//   EXPECT_EQ(EmitCSerializer<uint32_t>::convert(1000), "1000");
//   EXPECT_EQ(EmitCSerializer<uint64_t>::convert(9999), "9999");

//   // Test floating point serialization
//   EXPECT_EQ(EmitCSerializer<float>::convert(3.14f), std::to_string(3.14f));
//   EXPECT_EQ(EmitCSerializer<double>::convert(2.718), std::to_string(2.718));

//   // Test boolean serialization
//   EXPECT_EQ(EmitCSerializer<bool>::convert(true), "true");
//   EXPECT_EQ(EmitCSerializer<bool>::convert(false), "false");

//   // Test string serialization
//   EXPECT_EQ(EmitCSerializer<std::string>::convert("test"), "\"test\"");
// }

// // Test EmitCSerializer for container types
// TEST_F(EmitCConversionTest, ContainerTypesSerialization) {
//   EXPECT_EQ(EmitCSerializer<std::array<int32_t, 3>>::value,
//   "::std::array<int32_t, 3>");

//   EXPECT_EQ(EmitCSerializer<std::vector<float>>::value,
//   "::std::vector<float>");
// }

// // Test EmitCTypeConverter for integral types
// TEST_F(EmitCConversionTest, IntegralTypeConversion) {
//   // Test integer attribute conversion
//   auto intAttr = builder.getI32IntegerAttr(42);
//   auto result = EmitCTypeConverter<int32_t>::convert(intAttr);
//   ASSERT_TRUE(result);
//   EXPECT_EQ(result, 42);

//   // Test conversion from different integer types
//   EXPECT_EQ(EmitCTypeConverter<int32_t>::convert(int64_t{123}), 123);
//   EXPECT_EQ(EmitCTypeConverter<uint32_t>::convert(uint16_t{456}), 456u);
// }

// // Test EmitCTypeConverter for floating point types
// TEST_F(EmitCConversionTest, FloatingPointTypeConversion) {
//   // Test float attribute conversion
//   auto floatAttr = builder.getF32FloatAttr(3.14f);
//   auto result = EmitCTypeConverter<float>::convert(floatAttr);
//   EXPECT_FLOAT_EQ(result, 3.14f);

//   // Test conversion between different float types
//   EXPECT_FLOAT_EQ(EmitCTypeConverter<float>::convert(double{2.718}), 2.718f);
// }

// // Test EmitCTypeConverter for array types
// TEST_F(EmitCConversionTest, ArrayTypeConversion) {
//   // Test array attribute conversion
//   std::vector<Attribute> elements = {builder.getI32IntegerAttr(1),
//                                      builder.getI32IntegerAttr(2),
//                                      builder.getI32IntegerAttr(3)};
//   auto arrayAttr = builder.getArrayAttr(elements);

//   auto result = EmitCTypeConverter<std::vector<int32_t>>::convert(arrayAttr);
//   EXPECT_EQ(result.size(), 3u);
//   EXPECT_EQ(result[0], 1);
//   EXPECT_EQ(result[1], 2);
//   EXPECT_EQ(result[2], 3);
// }

// // Test EmitCTypeConverter for variant types
// TEST_F(EmitCConversionTest, VariantTypeConversion) {
//   using VariantType = std::variant<int32_t, float>;

//   // Test integer variant conversion
//   auto intAttr = builder.getI32IntegerAttr(42);
//   auto intResult = EmitCTypeConverter<VariantType>::convert(intAttr);
//   EXPECT_EQ(std::get<int32_t>(intResult), 42);

//   // Test float variant conversion
//   auto floatAttr = builder.getF32FloatAttr(3.14f);
//   auto floatResult = EmitCTypeConverter<VariantType>::convert(floatAttr);
//   EXPECT_FLOAT_EQ(std::get<float>(floatResult), 3.14f);
// }

// TEST_F(EmitCConversionTest, CompositeVariantConversion) {
//   using VariantType =
//       std::variant<std::vector<std::array<int32_t, 3>>, std::array<int32_t,
//       3>>;

//   // Test vector of arrays variant conversion
//   std::vector<Attribute> elements = {
//       builder.getArrayAttr({builder.getI32IntegerAttr(1),
//                             builder.getI32IntegerAttr(2),
//                             builder.getI32IntegerAttr(3)}),
//       builder.getArrayAttr({builder.getI32IntegerAttr(4),
//                             builder.getI32IntegerAttr(5),
//                             builder.getI32IntegerAttr(6)})};
//   auto vectorAttr = builder.getArrayAttr(elements);
//   auto vectorResult = EmitCTypeConverter<VariantType>::convert(vectorAttr);
//   EXPECT_EQ(
//       (std::get<std::vector<std::array<int32_t, 3>>>(vectorResult).size()),
//       2u);
//   EXPECT_EQ((std::get<std::vector<std::array<int32_t, 3>>>(vectorResult)[0]),
//             (std::array<int32_t, 3>{1, 2, 3}));
//   EXPECT_EQ((std::get<std::vector<std::array<int32_t, 3>>>(vectorResult)[1]),
//             (std::array<int32_t, 3>{4, 5, 6}));

//   // Test array variant conversion
//   auto arrayAttr = builder.getArrayAttr({builder.getI32IntegerAttr(7),
//                                          builder.getI32IntegerAttr(8),
//                                          builder.getI32IntegerAttr(9)});
//   auto arrayResult = EmitCTypeConverter<VariantType>::convert(arrayAttr);
//   EXPECT_EQ((std::get<std::array<int32_t, 3>>(arrayResult)),
//             (std::array<int32_t, 3>{7, 8, 9}));
// }
