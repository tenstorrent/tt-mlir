// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace mlir {
namespace tt {
namespace ttnn_to_emitc {

class EmitCConversionTest : public ::testing::Test {
protected:
  EmitCConversionTest() : context(), builder(&context) {}

  mlir::MLIRContext context;
  mlir::Builder builder;
};

TEST(TypeNameTest, PrimitiveTypes) {
  EXPECT_EQ(TypeNameV<int32_t>, "int32_t");
  EXPECT_EQ(TypeNameV<int64_t>, "int64_t");
  EXPECT_EQ(TypeNameV<uint32_t>, "uint32_t");
  EXPECT_EQ(TypeNameV<uint64_t>, "uint64_t");
  EXPECT_EQ(TypeNameV<float>, "float");
  EXPECT_EQ(TypeNameV<double>, "double");
  EXPECT_EQ(TypeNameV<bool>, "bool");
  EXPECT_EQ(TypeNameV<std::string>, "::std::string");
}

TEST(TypeNameTest, ArrayTypes) {
  EXPECT_EQ((TypeNameV<std::array<int32_t, 4>>), "::std::array<int32_t, 4>");
  EXPECT_EQ((TypeNameV<std::array<std::string, 2>>),
            "::std::array<::std::string, 2>");
}

TEST(TypeNameTest, VectorTypes) {
  EXPECT_EQ(TypeNameV<std::vector<int32_t>>, "::std::vector<int32_t>");
  EXPECT_EQ(TypeNameV<std::vector<std::string>>,
            "::std::vector<::std::string>");
  EXPECT_EQ(TypeNameV<std::vector<std::vector<int32_t>>>,
            "::std::vector<::std::vector<int32_t>>");
  EXPECT_EQ((TypeNameV<std::vector<std::array<std::vector<std::string>, 3>>>),
            "::std::vector<::std::array<::std::vector<::std::string>, 3>>");
}

TEST(TypeNameTest, SmallVectorTypes) {
  EXPECT_EQ(TypeNameV<::ttnn::SmallVector<bool>>, "::ttnn::SmallVector<bool>");
  EXPECT_EQ(TypeNameV<::ttnn::SmallVector<std::string>>,
            "::ttnn::SmallVector<::std::string>");
  EXPECT_EQ(TypeNameV<::ttnn::SmallVector<::ttnn::SmallVector<uint64_t>>>,
            "::ttnn::SmallVector<::ttnn::SmallVector<uint64_t>>");
  EXPECT_EQ(
      (TypeNameV<::ttnn::SmallVector<std::array<std::vector<std::string>, 3>>>),
      "::ttnn::SmallVector<::std::array<::std::vector<::std::string>, 3>>");
}

TEST_F(EmitCConversionTest, ConvertI32IntegerAttr) {
  mlir::IntegerAttr int32Attr = builder.getI32IntegerAttr(42);
  std::string converted = EmitCTypeConverter<int32_t>::convert(int32Attr);
  EXPECT_EQ(converted, "42");

  mlir::Attribute int32AsAttribute = int32Attr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<int32_t>::convert(int32AsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "42");
}

TEST_F(EmitCConversionTest, ConvertAPInt) {
  mlir::APInt apInt32(32, 37);
  std::string converted = EmitCTypeConverter<uint32_t>::convert(apInt32);
  EXPECT_EQ(converted, "37");

  mlir::APInt apInt64(64, 17);
  converted = EmitCTypeConverter<int64_t>::convert(apInt64);
  EXPECT_EQ(converted, "17");
}

TEST_F(EmitCConversionTest, ConvertCIntType) {
  int32_t i32Val = 42;
  std::string converted = EmitCTypeConverter<int32_t>::convert(i32Val);
  EXPECT_EQ(converted, "42");

  converted = EmitCTypeConverter<uint64_t>::convert(i32Val);
  EXPECT_EQ(converted, "42");
}

TEST_F(EmitCConversionTest, IntegralExpectedFailure) {
  mlir::Attribute emptyAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<int32_t>::convert(emptyAttr);
  EXPECT_FALSE(maybeConverted);

  mlir::FloatAttr floatAttr = builder.getF32FloatAttr(42.0);
  maybeConverted = EmitCTypeConverter<int32_t>::convert(floatAttr);
  EXPECT_FALSE(maybeConverted);
}

TEST_F(EmitCConversionTest, ConvertF32FloatAttr) {
  mlir::FloatAttr floatAttr = builder.getF32FloatAttr(42.0);
  std::string converted = EmitCTypeConverter<float>::convert(floatAttr);
  EXPECT_EQ(converted, "42.000000");

  mlir::Attribute floatAsAttribute = floatAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<float>::convert(floatAsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "42.000000");
}

TEST_F(EmitCConversionTest, ConvertF64FloatAttr) {
  mlir::FloatAttr floatAttr = builder.getF64FloatAttr(42.0);
  std::string converted = EmitCTypeConverter<double>::convert(floatAttr);
  EXPECT_EQ(converted, "42.000000");

  mlir::Attribute floatAsAttribute = floatAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<double>::convert(floatAsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "42.000000");
}

TEST_F(EmitCConversionTest, ConvertAPFloat) {
  mlir::APFloat apFloat32(42.0);
  std::string converted = EmitCTypeConverter<float>::convert(apFloat32);
  EXPECT_EQ(converted, "42.000000");

  mlir::APFloat apFloat64(42.0);
  converted = EmitCTypeConverter<double>::convert(apFloat64);
  EXPECT_EQ(converted, "42.000000");
}

TEST_F(EmitCConversionTest, ConvertCFPType) {
  float f32Val = 42.0;
  std::string converted = EmitCTypeConverter<float>::convert(f32Val);
  EXPECT_EQ(converted, "42.000000");

  converted = EmitCTypeConverter<double>::convert(f32Val);
  EXPECT_EQ(converted, "42.000000");
}

TEST_F(EmitCConversionTest, FloatingPointExpectedFailure) {
  mlir::Attribute emptyAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<float>::convert(emptyAttr);
  EXPECT_FALSE(maybeConverted);

  mlir::IntegerAttr int32Attr = builder.getI32IntegerAttr(42);
  maybeConverted = EmitCTypeConverter<float>::convert(int32Attr);
  EXPECT_FALSE(maybeConverted);
}

TEST_F(EmitCConversionTest, ConvertArrayAttrToStdVector) {
  mlir::ArrayAttr arrayAttr = builder.getArrayAttr({
      builder.getI32IntegerAttr(1),
      builder.getI32IntegerAttr(2),
      builder.getI32IntegerAttr(3),
  });
  std::string converted =
      EmitCTypeConverter<std::vector<int32_t>>::convert(arrayAttr);
  EXPECT_EQ(converted, "::std::vector<int32_t>{1, 2, 3}");

  mlir::Attribute arrayAsAttribute = arrayAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<std::vector<int32_t>>::convert(arrayAsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "::std::vector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseBoolArrayAttrToStdVector) {
  mlir::DenseBoolArrayAttr denseArrayAttr =
      builder.getDenseBoolArrayAttr({true, false, true});
  std::string converted =
      EmitCTypeConverter<std::vector<bool>>::convert(denseArrayAttr);
  EXPECT_EQ(converted, "::std::vector<bool>{true, false, true}");
}

TEST_F(EmitCConversionTest, ConvertDenseI32ArrayAttrToStdVector) {
  mlir::DenseI32ArrayAttr denseArrayAttr =
      builder.getDenseI32ArrayAttr({1, 2, 3});
  std::string converted =
      EmitCTypeConverter<std::vector<int32_t>>::convert(denseArrayAttr);
  EXPECT_EQ(converted, "::std::vector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseF32ArrayAttrToStdVector) {
  mlir::DenseF32ArrayAttr denseArrayAttr =
      builder.getDenseF32ArrayAttr({1.0, 2.0, 3.0});
  std::string converted =
      EmitCTypeConverter<std::vector<float>>::convert(denseArrayAttr);
  EXPECT_EQ(converted, "::std::vector<float>{1.000000, 2.000000, 3.000000}");
}

TEST_F(EmitCConversionTest, ConvertDenseIntElementsAttrToStdVector) {
  mlir::DenseIntElementsAttr denseElementsAttr =
      builder.getI32TensorAttr({1, 2, 3});
  std::string converted =
      EmitCTypeConverter<std::vector<int32_t>>::convert(denseElementsAttr);
  EXPECT_EQ(converted, "::std::vector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertArrayRefToStdVector) {
  std::vector<uint32_t> vec = {1, 2, 3};
  std::string converted =
      EmitCTypeConverter<std::vector<int32_t>>::convert(llvm::ArrayRef(vec));
  EXPECT_EQ(converted, "::std::vector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertArrayAttrToTtnnSmallVector) {
  mlir::ArrayAttr arrayAttr = builder.getArrayAttr({
      builder.getI32IntegerAttr(1),
      builder.getI32IntegerAttr(2),
      builder.getI32IntegerAttr(3),
  });
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<int32_t>>::convert(arrayAttr);
  EXPECT_EQ(converted, "::ttnn::SmallVector<int32_t>{1, 2, 3}");

  mlir::Attribute arrayAsAttribute = arrayAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<::ttnn::SmallVector<int32_t>>::convert(
          arrayAsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "::ttnn::SmallVector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseBoolArrayAttrToTtnnSmallVector) {
  mlir::DenseBoolArrayAttr denseArrayAttr =
      builder.getDenseBoolArrayAttr({true, false, true});
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<bool>>::convert(denseArrayAttr);
  EXPECT_EQ(converted, "::ttnn::SmallVector<bool>{true, false, true}");
}

TEST_F(EmitCConversionTest, ConvertDenseI32ArrayAttrToTtnnSmallVector) {
  mlir::DenseI32ArrayAttr denseArrayAttr =
      builder.getDenseI32ArrayAttr({1, 2, 3});
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<int32_t>>::convert(denseArrayAttr);
  EXPECT_EQ(converted, "::ttnn::SmallVector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseF32ArrayAttrToTtnnSmallVector) {
  mlir::DenseF32ArrayAttr denseArrayAttr =
      builder.getDenseF32ArrayAttr({1.0, 2.0, 3.0});
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<float>>::convert(denseArrayAttr);
  EXPECT_EQ(converted,
            "::ttnn::SmallVector<float>{1.000000, 2.000000, 3.000000}");
}

TEST_F(EmitCConversionTest, ConvertDenseIntElementsAttrToTtnnSmallVector) {
  mlir::DenseIntElementsAttr denseElementsAttr =
      builder.getI32TensorAttr({1, 2, 3});
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<int32_t>>::convert(
          denseElementsAttr);
  EXPECT_EQ(converted, "::ttnn::SmallVector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertArrayRefToTtnnSmallVector) {
  std::vector<uint32_t> vec = {1, 2, 3};
  std::string converted =
      EmitCTypeConverter<::ttnn::SmallVector<int32_t>>::convert(
          llvm::ArrayRef(vec));
  EXPECT_EQ(converted, "::ttnn::SmallVector<int32_t>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertArrayAttrToStdArray) {
  mlir::ArrayAttr arrayAttr = builder.getArrayAttr({
      builder.getI32IntegerAttr(1),
      builder.getI32IntegerAttr(2),
      builder.getI32IntegerAttr(3),
  });
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<int32_t, 3>>::convert(arrayAttr);
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<int32_t, 3>{1, 2, 3}");

  mlir::Attribute arrayAsAttribute = arrayAttr;
  std::optional<std::string> maybeConverted =
      EmitCTypeConverter<std::array<int32_t, 3>>::convert(arrayAsAttribute);
  ASSERT_TRUE(maybeConverted);
  EXPECT_EQ(*maybeConverted, "::std::array<int32_t, 3>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseBoolArrayAttrToStdArray) {
  mlir::DenseBoolArrayAttr denseArrayAttr =
      builder.getDenseBoolArrayAttr({true, false, true});
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<bool, 3>>::convert(denseArrayAttr);
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<bool, 3>{true, false, true}");
}

TEST_F(EmitCConversionTest, ConvertDenseI32ArrayAttrToStdArray) {
  mlir::DenseI32ArrayAttr denseArrayAttr =
      builder.getDenseI32ArrayAttr({1, 2, 3});
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<int32_t, 3>>::convert(denseArrayAttr);
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<int32_t, 3>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertDenseF32ArrayAttrToStdArray) {
  mlir::DenseF32ArrayAttr denseArrayAttr =
      builder.getDenseF32ArrayAttr({1.0, 2.0, 3.0});
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<float, 3>>::convert(denseArrayAttr);
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<float, 3>{1.000000, 2.000000, 3.000000}");
}

TEST_F(EmitCConversionTest, ConvertDenseIntElementsAttrToStdArray) {
  mlir::DenseIntElementsAttr denseElementsAttr =
      builder.getI32TensorAttr({1, 2, 3});
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<int32_t, 3>>::convert(denseElementsAttr);
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<int32_t, 3>{1, 2, 3}");
}

TEST_F(EmitCConversionTest, ConvertArrayRefToStdArray) {
  std::vector<uint32_t> vec = {1, 2, 3};
  std::optional<std::string> converted =
      EmitCTypeConverter<std::array<int32_t, 3>>::convert(llvm::ArrayRef(vec));
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, "::std::array<int32_t, 3>{1, 2, 3}");
}

} // namespace ttnn_to_emitc
} // namespace tt
} // namespace mlir
