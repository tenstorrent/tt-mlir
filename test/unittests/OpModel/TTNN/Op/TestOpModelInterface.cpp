// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include <cstddef>
#include <llvm-gtest/gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <optional>
#include <string>

namespace mlir::tt::ttnn {

class OpModelBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override { context.loadDialect<TTNNDialect>(); }
  void TearDown() override {}

  // helper function to create layout
  TTNNLayoutAttr CreateLayout(llvm::ArrayRef<int64_t> tensorShape,
                              BufferType bufferType,
                              TensorMemoryLayout tensorMemoryLayout,
                              ArrayRef<int64_t> gridShape = {8, 8}) {
    return TTNNLayoutAttr::get(
        &context, tensorShape, TileType::get(&context, builder.getBF16Type()),
        bufferType, GridAttr::get(&context, gridShape),
        TensorMemoryLayoutAttr::get(&context, tensorMemoryLayout));
  }

  // helper function to extract op data and call into get op constraints
  std::optional<
      std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
                 std::optional<std::string>>>
  getOpConstraints(Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;

    // create input layouts
    auto numOperand = op->getNumOperands();
    // some ops have multiple operands
    auto limit = (numOperand > 1) ? numOperand - 1 : numOperand;
    for (size_t i = 0; i < limit; i++) {
      auto operand = op->getOperand(i);
      auto inputShape =
          mlir::cast<RankedTensorType>(operand.getType()).getShape();
      auto inputLayout = CreateLayout(inputShape, BufferType::L1,
                                      TensorMemoryLayout::Interleaved);
      inputs.push_back(inputLayout);
    }

    // create output layout
    auto output = op->getResult(0);
    auto outputShape =
        mlir::cast<RankedTensorType>(output.getType()).getShape();
    auto outputLayout = CreateLayout(outputShape, BufferType::L1,
                                     TensorMemoryLayout::Interleaved);

    // call op model interface - getOpConstraints()
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      auto constraints = backend.getOpConstraints(inputs, outputLayout);
      return constraints;
    }
    return std::nullopt;
  }
};

TEST_F(OpModelBase, ReluInterface) {
  // create ReluOp
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  Type elementType = builder.getBF16Type();
  RankedTensorType rankedTensorType =
      RankedTensorType::get(tensorShape, elementType);

  auto input = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  auto output = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);

  auto relu = builder.create<ReluOp>(builder.getUnknownLoc(), output.getType(),
                                     ::mlir::ValueRange{input, output});
  // test ReluOp interface
  auto value = getOpConstraints(relu.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      auto l1Usage = l1.value();
      EXPECT_GT(std::get<0>(l1Usage), 0);
      EXPECT_GT(std::get<1>(l1Usage), 0);
      EXPECT_GT(std::get<2>(l1Usage), 0);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}
TEST_F(OpModelBase, SoftmaxInterface) {
  // create SoftmaxOp
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  Type elementType = builder.getBF16Type();
  RankedTensorType rankedTensorType =
      RankedTensorType::get(tensorShape, elementType);

  auto input = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  auto output = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);

  auto softmax = builder.create<SoftmaxOp>(builder.getUnknownLoc(),
                                           output.getType(), input, -1);
  // test SoftmaxOp interface
  auto value = getOpConstraints(softmax.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      auto l1Usage = l1.value();
      EXPECT_GT(std::get<0>(l1Usage), 0);
      EXPECT_GT(std::get<1>(l1Usage), 0);
      EXPECT_GT(std::get<2>(l1Usage), 0);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

TEST_F(OpModelBase, AddInterface) {
  // create AddOp
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  Type elementType = builder.getBF16Type();
  RankedTensorType rankedTensorType =
      RankedTensorType::get(tensorShape, elementType);

  auto input1 = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  auto input2 = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  auto output = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorType, nullptr,
      ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);

  auto add = builder.create<AddOp>(builder.getUnknownLoc(), output.getType(),
                                   ::mlir::ValueRange{input1, input2, output});
  // test AddOp interface
  auto value = getOpConstraints(add.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      auto l1Usage = l1.value();
      EXPECT_GT(std::get<0>(l1Usage), 0);
      EXPECT_GT(std::get<1>(l1Usage), 0);
      EXPECT_GT(std::get<2>(l1Usage), 0);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

TEST_F(OpModelBase, MatmulInterface) {
  // create MatmulOp
  llvm::ArrayRef<int64_t> tensorShapeA = {2048, 1024};
  llvm::ArrayRef<int64_t> tensorShapeB = {1024, 2048};
  llvm::ArrayRef<int64_t> tensorShapeO = {2048, 2048};
  Type elementType = builder.getBF16Type();
  RankedTensorType rankedTensorTypeA =
      RankedTensorType::get(tensorShapeA, elementType);
  RankedTensorType rankedTensorTypeB =
      RankedTensorType::get(tensorShapeB, elementType);
  RankedTensorType rankedTensorTypeO =
      RankedTensorType::get(tensorShapeO, elementType);
  auto inputA = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorTypeA, nullptr,
      ShapeAttr::get(&context, tensorShapeA), nullptr, nullptr, nullptr);
  auto inputB = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorTypeB, nullptr,
      ShapeAttr::get(&context, tensorShapeB), nullptr, nullptr, nullptr);
  auto output = builder.create<EmptyOp>(
      builder.getUnknownLoc(), rankedTensorTypeO, nullptr,
      ShapeAttr::get(&context, tensorShapeO), nullptr, nullptr, nullptr);

  auto matmul =
      builder.create<MatmulOp>(builder.getUnknownLoc(), output.getType(),
                               ::mlir::ValueRange{inputA, inputB, output});
  // test MatmulOp interface
  auto value = getOpConstraints(matmul.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      auto l1Usage = l1.value();
      EXPECT_GT(std::get<0>(l1Usage), 0);
      EXPECT_GT(std::get<1>(l1Usage), 0);
      EXPECT_GT(std::get<2>(l1Usage), 0);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

} // namespace mlir::tt::ttnn