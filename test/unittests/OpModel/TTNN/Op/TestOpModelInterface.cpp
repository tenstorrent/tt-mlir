// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <llvm/Support/Error.h>
#include <optional>

namespace mlir::tt::ttnn {

class OpModelBase : public OpModelFixture {
public:
  std::optional<
      std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
                 std::optional<std::string>>>
  getOpConstraints(Operation *op) {
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      return backend.getOpConstraints(getInputLayouts(op), getOutputLayout(op));
    }
    return std::nullopt;
  }

  llvm::Expected<size_t> getOpRuntime(Operation *op) {
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      return backend.getOpRuntime(getInputLayouts(op), getOutputLayout(op));
    }
    return llvm::createStringError("Could not cast op to OpModel");
  }

  std::vector<TTNNLayoutAttr> getInputLayouts(Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;

    // TODO(odjuricic): check for DPS explicitly.
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
    return inputs;
  }

  mlir::tt::ttnn::TTNNLayoutAttr getOutputLayout(Operation *op) {
    auto output = op->getResult(0);
    auto outputShape =
        mlir::cast<RankedTensorType>(output.getType()).getShape();
    return CreateTiledLayout(outputShape, BufferType::L1,
                             TensorMemoryLayout::Interleaved);
  }

  mlir::tt::DeviceAttr getFakeDeviceAttr() {
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

  mlir::Value createEmptyTensor(llvm::ArrayRef<int64_t> tensorShape) {
    Type elementType = builder.getBF16Type();
    RankedTensorType rankedTensorType =
        RankedTensorType::get(tensorShape, elementType);
    return builder.create<OnesOp>(builder.getUnknownLoc(), rankedTensorType,
                                  ShapeAttr::get(&context, tensorShape),
                                  nullptr, nullptr, nullptr, nullptr);
  }
};

TEST_F(OpModelBase, ReluInterface) {
  // create ReluOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto output = createEmptyTensor(tensorShape);

  auto relu = builder.create<ReluOp>(builder.getUnknownLoc(), output.getType(),
                                     ::mlir::ValueRange{input, output});
  relu->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test ReluOp interface
  auto constraintsOpt = getOpConstraints(relu.getOperation());
  if (constraintsOpt.has_value()) {
    auto constraints = constraintsOpt.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 8192);
      EXPECT_EQ(peak_size, 2048);
      EXPECT_EQ(output_size, 2048);
    } else {
      FAIL() << "Missing L1 constraints; Error="
             << std::get<2>(constraints).value() << std::endl;
    }
  } else {
    FAIL() << "Failed to cast ReluOp to OpModel";
  }

  auto runtimeExp = getOpRuntime(relu.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}
TEST_F(OpModelBase, SoftmaxInterface) {
  // create SoftmaxOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto output = createEmptyTensor(tensorShape);

  auto softmax = builder.create<SoftmaxOp>(builder.getUnknownLoc(),
                                           output.getType(), input, -1);
  softmax->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test SoftmaxOp interface
  auto constraintsOpt = getOpConstraints(softmax.getOperation());
  if (constraintsOpt.has_value()) {
    auto constraints = constraintsOpt.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 137216);
      EXPECT_EQ(peak_size, 2048);
      EXPECT_EQ(output_size, 2048);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast SoftmaxOp to OpModel";
  }

  auto runtimeExp = getOpRuntime(softmax.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, AddInterface) {
  // create AddOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto output = createEmptyTensor(tensorShape);

  auto add = builder.create<AddOp>(builder.getUnknownLoc(), output.getType(),
                                   ::mlir::ValueRange{input1, input2, output});
  add->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test AddOp interface
  auto constraintsOpt = getOpConstraints(add.getOperation());
  if (constraintsOpt.has_value()) {
    auto constraints = constraintsOpt.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 12288);
      EXPECT_EQ(peak_size, 2048);
      EXPECT_EQ(output_size, 2048);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast AddOp to OpModel";
  }

  auto runtimeExp = getOpRuntime(add.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, MatmulInterface) {
  // create MatmulOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto output = createEmptyTensor(tensorShapeO);

  auto matmul =
      builder.create<MatmulOp>(builder.getUnknownLoc(), output.getType(),
                               ::mlir::ValueRange{inputA, inputB, output});
  matmul->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test MatmulOp interface
  auto constraintsOpt = getOpConstraints(matmul.getOperation());
  if (constraintsOpt.has_value()) {
    auto constraints = constraintsOpt.value();
    EXPECT_EQ(std::get<bool>(constraints), true);
    auto l1 = std::get<1>(constraints);
    if (l1.has_value()) {
      const auto &[cb_size, peak_size, output_size] = l1.value();
      EXPECT_EQ(cb_size, 786432);
      EXPECT_EQ(peak_size, 131072);
      EXPECT_EQ(output_size, 131072);
    } else {
      FAIL() << "Missing L1 constraints";
    }
  } else {
    FAIL() << "Failed to cast MatmulOp to OpModel";
  }

  auto runtimeExp = getOpRuntime(matmul.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

} // namespace mlir::tt::ttnn
