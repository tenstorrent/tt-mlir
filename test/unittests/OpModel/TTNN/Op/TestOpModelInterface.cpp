// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "gtest/gtest.h"

#include <optional>

namespace mlir::tt::ttnn {

class OpModelBase : public OpModelFixture {
public:
  // helper function to extract op data and call into get op constraints
  std::optional<
      std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
                 std::optional<std::string>>>
  getOpConstraints(Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;

    // TODO(odjuricic): check for DPS explicitly.
    // create input layouts
    auto numOperand = op->getNumOperands();
    // some ops have multiple operands
    auto limit = (numOperand > 1) ? numOperand - 1 : numOperand;
    for (size_t i = 0; i < limit; i++) {
      auto operand = op->getOperand(i);
      auto inputShape =
          mlir::cast<RankedTensorType>(operand.getType()).getShape();
      auto inputLayout = CreateTiledLayout(inputShape, BufferType::L1,
                                           TensorMemoryLayout::Interleaved);
      inputs.push_back(inputLayout);
    }

    // create output layout
    auto output = op->getResult(0);
    auto outputShape =
        mlir::cast<RankedTensorType>(output.getType()).getShape();
    auto outputLayout = CreateTiledLayout(outputShape, BufferType::L1,
                                          TensorMemoryLayout::Interleaved);

    // call op model interface - getOpConstraints()
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      auto constraints = backend.getOpConstraints(inputs, outputLayout);
      return constraints;
    }
    return std::nullopt;
  }

  auto getFakeDeviceAttr() {
    auto deviceIdx = mlir::getAffineConstantExpr(0, &context);
    auto shardOffset = mlir::getAffineConstantExpr(0, &context);
    auto d0 = mlir::getAffineDimExpr(0, &context); // d0
    auto d1 = mlir::getAffineDimExpr(1, &context); // d1
    auto map3 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1}, &context);
    auto map4 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1, shardOffset},
        &context);
    auto workerGrid = GridAttr::get(&context, gridShapeHwN300, map3);

    return DeviceAttr::get(&context, workerGrid, map4, map4, {1}, {0});
  }
};

TEST_F(OpModelBase, ReluInterface) {
  // create ReluOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
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
  relu->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test ReluOp interface
  auto value = getOpConstraints(relu.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 8192);
      EXPECT_EQ(peak_size, 4096);
      EXPECT_EQ(output_size, 4096);
    } else {
      FAIL() << "Missing L1 constraints; Error="
             << std::get<2>(constraints).value() << std::endl;
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}
TEST_F(OpModelBase, SoftmaxInterface) {
  // create SoftmaxOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
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
  softmax->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test SoftmaxOp interface
  auto value = getOpConstraints(softmax.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 137216);
      EXPECT_EQ(peak_size, 4096);
      EXPECT_EQ(output_size, 4096);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

TEST_F(OpModelBase, AddInterface) {
  // create AddOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
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
  add->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test AddOp interface
  auto value = getOpConstraints(add.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 12288);
      EXPECT_EQ(peak_size, 4096);
      EXPECT_EQ(output_size, 4096);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

TEST_F(OpModelBase, MatmulInterface) {
  // create MatmulOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};
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
  matmul->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test MatmulOp interface
  auto value = getOpConstraints(matmul.getOperation());
  if (value.has_value()) {
    auto constraints = value.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 786432);
      EXPECT_EQ(peak_size, 151552);
      EXPECT_EQ(output_size, 151552);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }
}

} // namespace mlir::tt::ttnn
