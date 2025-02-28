// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

class OpModelBase : public OpModelFixture {
public:
  llvm::Expected<std::tuple<size_t, size_t, size_t>>
  getOpConstraints(Operation *op) {
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      return backend.getOpConstraints(getInputLayouts(op), getOutputLayout(op));
    }
    return llvm::createStringError("Could not cast op to OpModel");
  }

  llvm::Expected<size_t> getOpRuntime(Operation *op) {
    if (OpModel backend = dyn_cast<OpModel>(op)) {
      return backend.getOpRuntime(getInputLayouts(op), getOutputLayout(op));
    }
    return llvm::createStringError("Could not cast op to OpModel");
  }

  std::vector<TTNNLayoutAttr> getInputLayouts(Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;

    auto limit = op->getNumOperands();
    if (isa<DestinationStyleOpInterface>(op)) {
      limit--;
    }

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

  mlir::RankedTensorType createRankedTensorType(llvm::ArrayRef<int64_t> shape) {
    Type elementType = builder.getBF16Type();
    RankedTensorType rankedTensorType =
        RankedTensorType::get(shape, elementType);
    return rankedTensorType;
  }

  mlir::Value createEmptyTensor(llvm::ArrayRef<int64_t> tensorShape) {
    RankedTensorType rankedTensorType = createRankedTensorType(tensorShape);
    return builder.create<OnesOp>(builder.getUnknownLoc(), rankedTensorType,
                                  ShapeAttr::get(&context, tensorShape),
                                  nullptr, nullptr, nullptr, nullptr);
  }
};

TEST_F(OpModelBase, ReluInterface) {
  // create ReluOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto relu = builder.create<ReluOp>(builder.getUnknownLoc(), outputType,
                                     ::mlir::ValueRange{input});
  relu->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test ReluOp interface
  auto constraintsExp = getOpConstraints(relu.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 8192);
    EXPECT_EQ(peak_size, 2048);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
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
  auto output = createRankedTensorType(tensorShape);

  auto softmax =
      builder.create<SoftmaxOp>(builder.getUnknownLoc(), output, input, -1);
  softmax->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test SoftmaxOp interface
  auto constraintsExp = getOpConstraints(softmax.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 137216);
    EXPECT_EQ(peak_size, 2048);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
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
  auto outputType = createRankedTensorType(tensorShape);

  auto add = builder.create<AddOp>(builder.getUnknownLoc(), outputType,
                                   ::mlir::ValueRange{input1, input2});
  add->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test AddOp interface
  auto constraintsExp = getOpConstraints(add.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 12288);
    EXPECT_EQ(peak_size, 2048);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
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
  auto outputType = createRankedTensorType(tensorShapeO);

  auto matmul = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                         ::mlir::ValueRange{inputA, inputB});
  matmul->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test MatmulOp interface
  auto constraintsExp = getOpConstraints(matmul.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 786432);
    EXPECT_EQ(peak_size, 131072);
    EXPECT_EQ(output_size, 131072);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(matmul.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, MeanInterface) {
  // create MeanOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 1024};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  auto mean = builder.create<MeanOp>(builder.getUnknownLoc(), output.getType(),
                                     ::mlir::ValueRange{input});
  mean->setAttr(DeviceAttr::name, getFakeDeviceAttr());
  mean.setKeepDim(true);
  mean.setDimArgAttr(builder.getArrayAttr(
      llvm::SmallVector<mlir::Attribute>{builder.getI64IntegerAttr(1)}));

  // test mean Op interface
  auto constraintsExp = getOpConstraints(mean.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 12288);
    EXPECT_EQ(peak_size, 2048);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(mean.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, reshapeOp) {
  // create ReshapeOp
  llvm::SmallVector<int64_t> tensorShapeA = {64, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {64 * 4, 1024 / 4};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  auto reshape = builder.create<ReshapeOp>(
      builder.getUnknownLoc(), output.getType(), ::mlir::ValueRange{input});
  reshape->setAttr(DeviceAttr::name, getFakeDeviceAttr());
  reshape.setShapeAttr(builder.getArrayAttr(llvm::SmallVector<mlir::Attribute>{
      builder.getI64IntegerAttr(64 * 4), builder.getI64IntegerAttr(1024 / 4)}));

  // test mean Op interface
  auto constraintsExp = getOpConstraints(reshape.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 262144);
    EXPECT_EQ(peak_size, 4096);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(reshape.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

} // namespace mlir::tt::ttnn
