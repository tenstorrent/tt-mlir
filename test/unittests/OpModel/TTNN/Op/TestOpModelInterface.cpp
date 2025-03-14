// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "../lib/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/AffineExpr.h"
#include "gtest/gtest.h"

#include <cstdint>

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
      if (!operand || !mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
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

TEST_F(OpModelBase, ReluOpInterface) {
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
TEST_F(OpModelBase, SoftmaxOpInterface) {
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

TEST_F(OpModelBase, AddOpInterface) {
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

TEST_F(OpModelBase, MultiplyOpInterface) {
  // create MultiplyOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto output = createEmptyTensor(tensorShape);

  auto multiply =
      builder.create<MultiplyOp>(builder.getUnknownLoc(), output.getType(),
                                 ::mlir::ValueRange{input1, input2});
  multiply->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test MultiplyOp interface
  auto constraintsExp = getOpConstraints(multiply.getOperation());
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

  auto runtimeExp = getOpRuntime(multiply.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, MatmulOpInterface) {
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

TEST_F(OpModelBase, MeanOpInterface) {
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

TEST_F(OpModelBase, ReshapeOpInterface) {
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

  // test reshape Op interface
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

TEST_F(OpModelBase, toLayoutOp) {
  llvm::SmallVector<int64_t> tensorShape = {64, 1024};
  RankedTensorType rankedTensorType = createRankedTensorType(tensorShape);
  auto tensor = builder.create<OnesOp>(
      builder.getUnknownLoc(), rankedTensorType,
      ShapeAttr::get(&context, tensorShape), nullptr,
      LayoutAttr::get(&context, Layout::RowMajor), nullptr, nullptr);

  DeviceAttr deviceAttr = getFakeDeviceAttr();
  // Need to pass a GetDeviceOp to make sure the layout change happens on the
  // device
  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1));
  ToLayoutOp toLayout = builder.create<ToLayoutOp>(
      builder.getUnknownLoc(), tensor.getType(), tensor, Layout::Tile, nullptr,
      nullptr, deviceOp);
  toLayout->setAttr(DeviceAttr::name, deviceAttr);

  // Manually create the operand layouts for calling the backend to make sure
  // the layouts are propagated all the way
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAMRowMajor =
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAMTiled =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  OpModel backend = dyn_cast<OpModel>(toLayout.getOperation());
  if (!backend) {
    FAIL() << "Could not cast op to OpModel";
  }

  auto constraintsExp = backend.getOpConstraints(
      std::vector{layoutDRAMRowMajor}, layoutDRAMTiled);
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 131072);
    EXPECT_EQ(peak_size, 0);
    EXPECT_EQ(output_size, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp =
      backend.getOpRuntime(std::vector{layoutDRAMRowMajor}, layoutDRAMTiled);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, transposeOp) {
  // create TransposeOp
  llvm::SmallVector<int64_t> tensorShapeA = {64, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {1024, 64};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  auto transpose = builder.create<TransposeOp>(builder.getUnknownLoc(),
                                               output.getType(), input, 0, 1);
  transpose->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  // test transpose Op interface
  auto constraintsExp = getOpConstraints(transpose.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 8192);
    EXPECT_EQ(peak_size, 2048);
    EXPECT_EQ(output_size, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(transpose.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, typecastOp) {
  // create TransposeOp
  llvm::SmallVector<int64_t> tensorShape = {64, 1024};

  RankedTensorType rankedTensorTypeBF16 =
      RankedTensorType::get(tensorShape, builder.getBF16Type());

  auto input =
      builder.create<OnesOp>(builder.getUnknownLoc(), rankedTensorTypeBF16,
                             ShapeAttr::get(&context, tensorShape),
                             DataTypeAttr::get(&context, DataType::BFloat16),
                             nullptr, nullptr, nullptr);
  RankedTensorType rankedTensorTypeF32 =
      RankedTensorType::get(tensorShape, builder.getF32Type());

  auto typecast = builder.create<TypecastOp>(
      builder.getUnknownLoc(), rankedTensorTypeF32, input, DataType::Float32);
  typecast->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  auto constraintsExp = getOpConstraints(typecast.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cb_size, peak_size, output_size] = l1;
    EXPECT_EQ(cb_size, 12288);
    EXPECT_EQ(peak_size, 4096);
    EXPECT_EQ(output_size, 4096);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(typecast.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, Conv2dInterface) {
  // create Conv2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {1, 1, 1568, 64};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  auto input = createEmptyTensor(inputShape);
  auto weight = createEmptyTensor(weightShape);
  auto outputType = createRankedTensorType(outputShape);

  DeviceAttr deviceAttr = getFakeDeviceAttr();
  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(deviceAttr),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, nullptr);

  conv2d->setAttr(DeviceAttr::name, deviceAttr);

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  // test Conv2dOp interface
  auto constraintsExp = getOpConstraints(conv2d.getOperation());
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  if (!constraintsExp) {
    std::string error = llvm::toString(constraintsExp.takeError());
    EXPECT_TRUE(error.find("Mismatch!! L1 Allocation Pre Op") !=
                std::string::npos);
  }

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(conv2d.getOperation());
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

} // namespace mlir::tt::ttnn
