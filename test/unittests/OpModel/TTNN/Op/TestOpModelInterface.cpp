// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

#include <cstdint>

namespace mlir::tt::ttnn {

class OpModelBase : public OpModelFixture {
public:
  llvm::Expected<op_model::ttnn::OpConstraints>
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
      auto operandType = mlir::cast<RankedTensorType>(operand.getType());
      if (operandType.getEncoding()) {
        inputs.push_back(mlir::cast<mlir::tt::ttnn::TTNNLayoutAttr>(
            operandType.getEncoding()));
        continue;
      }

      auto inputShape = operandType.getShape();
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

  mlir::RankedTensorType
  createRankedTensorType(llvm::ArrayRef<int64_t> shape,
                         mlir::Type elementType = nullptr,
                         TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    RankedTensorType rankedTensorType =
        RankedTensorType::get(shape, elementType, layout);
    return rankedTensorType;
  }

  mlir::Value createEmptyTensor(llvm::ArrayRef<int64_t> tensorShape,
                                mlir::Type elementType = nullptr,
                                TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    RankedTensorType rankedTensorType =
        createRankedTensorType(tensorShape, elementType, layout);
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

  // test ReluOp interface
  auto constraintsExp = getOpConstraints(relu.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 8192);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

TEST_F(OpModelBase, ReluOpInterfaceNullOutput) {
  // create ReluOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto relu = builder.create<ReluOp>(builder.getUnknownLoc(), outputType,
                                     ::mlir::ValueRange{input});

  // test ReluOp interface
  OpModel backend = dyn_cast<OpModel>(relu.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(relu), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 8192);
  EXPECT_EQ(peakSize, 2048);
  EXPECT_EQ(outputSize, 2048);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedL1TensorMemoryLayout());
}

TEST_F(OpModelBase, SqrtOpInterface) {
  // create SqrtOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sqrt = builder.create<SqrtOp>(builder.getUnknownLoc(), outputType,
                                     ::mlir::ValueRange{input});

  // test SqrtOp interface
  auto constraintsExp = getOpConstraints(sqrt.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 8192);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(sqrt.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, SigmoidOpInterface) {
  // create SigmoidOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sigmoid = builder.create<SigmoidOp>(builder.getUnknownLoc(), outputType,
                                           ::mlir::ValueRange{input});

  // test SigmoidOp interface
  auto constraintsExp = getOpConstraints(sigmoid.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 8192);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(sigmoid.getOperation());
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

  // test SoftmaxOp interface
  auto constraintsExp = getOpConstraints(softmax.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 137216);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

  // test AddOp interface
  auto constraintsExp = getOpConstraints(add.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12288);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

TEST_F(OpModelBase, AddOpInterfaceNullOutput) {
  // create AddOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto add = builder.create<AddOp>(builder.getUnknownLoc(), outputType,
                                   ::mlir::ValueRange{input1, input2});

  // test AddOp interface
  OpModel backend = dyn_cast<OpModel>(add.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(add), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 12288);
  EXPECT_EQ(peakSize, 2048);
  EXPECT_EQ(outputSize, 2048);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedL1TensorMemoryLayout());
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

  // test MultiplyOp interface
  auto constraintsExp = getOpConstraints(multiply.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12288);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

  // test MatmulOp interface
  auto constraintsExp = getOpConstraints(matmul.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 786432);
    EXPECT_EQ(peakSize, 131072);
    EXPECT_EQ(outputSize, 131072);
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

TEST_F(OpModelBase, MatmulOpInterfaceNullOutput) {
  // create MatmulOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto outputType = createRankedTensorType(tensorShapeO);

  auto matmul = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                         ::mlir::ValueRange{inputA, inputB});

  // test MatmulOp interface
  OpModel backend = dyn_cast<OpModel>(matmul.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(matmul), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 786432);
  EXPECT_EQ(peakSize, 0);
  EXPECT_EQ(outputSize, 0);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedDRAMTensorMemoryLayout());
}

TEST_F(OpModelBase, MeanOpInterface) {
  // create MeanOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 1024};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  auto mean = builder.create<MeanOp>(builder.getUnknownLoc(), output.getType(),
                                     ::mlir::ValueRange{input});
  mean.setKeepDim(true);
  mean.setDimArgAttr(builder.getArrayAttr(
      llvm::SmallVector<mlir::Attribute>{builder.getI64IntegerAttr(1)}));

  // test mean Op interface
  auto constraintsExp = getOpConstraints(mean.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12288);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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
  reshape.setShapeAttr(builder.getArrayAttr(llvm::SmallVector<mlir::Attribute>{
      builder.getI64IntegerAttr(64 * 4), builder.getI64IntegerAttr(1024 / 4)}));

  // test reshape Op interface
  auto constraintsExp = getOpConstraints(reshape.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 5120);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

  // Need to pass a GetDeviceOp to make sure the layout change happens on the
  // device
  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
      ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));
  ToLayoutOp toLayout = builder.create<ToLayoutOp>(
      builder.getUnknownLoc(), tensor.getType(), tensor, Layout::Tile, nullptr,
      nullptr, deviceOp);

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
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 131072);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
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

  // test transpose Op interface
  auto constraintsExp = getOpConstraints(transpose.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 8192);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
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

  auto constraintsExp = getOpConstraints(typecast.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12288);
    EXPECT_EQ(peakSize, 4096);
    EXPECT_EQ(outputSize, 4096);
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
  llvm::SmallVector<int64_t> weightShape = {64, 3, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  auto input = createEmptyTensor(inputShape);
  Type weightElementType = builder.getBF16Type();
  auto weightLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, weightShape, weightElementType,
      mlir::tt::ttnn::BufferType::SystemMemory, GridAttr::get(&context, 2));
  auto weight = createEmptyTensor(weightShape, weightElementType, weightLayout);
  auto outputType = createRankedTensorType(outputShape);

  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
      ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(),         // Location
      outputType,                      // Output type
      input,                           // Input tensor
      weight,                          // Weight tensor
      nullptr,                         // Bias tensor (optional)
      deviceOp,                        // Device operation
      3,                               // Input channels
      64,                              // Output channels
      1,                               // Batch size
      224,                             // Input height
      224,                             // Input width
      llvm::ArrayRef<int32_t>({7, 7}), // Kernel size [H, W]
      llvm::ArrayRef<int32_t>({2, 2}), // Stride [H, W]
      llvm::ArrayRef<int32_t>({3, 3}), // Padding [H, W]
      llvm::ArrayRef<int32_t>({1, 1}), // Dilation [H, W]
      1,                               // Groups
      nullptr,                         // Conv2dConfig (optional)
      nullptr                          // ComputeKernelConfig (optional)
  );

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  // test Conv2dOp interface
  auto constraintsExp = getOpConstraints(conv2d.getOperation());
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 229440);
  EXPECT_EQ(peakSize, 190568);
  EXPECT_EQ(outputSize, 26624);

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

TEST_F(OpModelBase, Conv2dInterfaceNullOutput) {
  // create Conv2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {64, 3, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  auto input = createEmptyTensor(inputShape);
  Type weightElementType = builder.getBF16Type();
  auto weightLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, weightShape, weightElementType,
      mlir::tt::ttnn::BufferType::SystemMemory, GridAttr::get(&context, 2));
  auto weight = createEmptyTensor(weightShape, weightElementType, weightLayout);
  auto outputType = createRankedTensorType(outputShape);

  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
      ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(),         // Location
      outputType,                      // Output type
      input,                           // Input tensor
      weight,                          // Weight tensor
      nullptr,                         // Bias tensor (optional)
      deviceOp,                        // Device operation
      3,                               // Input channels
      64,                              // Output channels
      1,                               // Batch size
      224,                             // Input height
      224,                             // Input width
      llvm::ArrayRef<int32_t>({7, 7}), // Kernel size [H, W]
      llvm::ArrayRef<int32_t>({2, 2}), // Stride [H, W]
      llvm::ArrayRef<int32_t>({3, 3}), // Padding [H, W]
      llvm::ArrayRef<int32_t>({1, 1}), // Dilation [H, W]
      1,                               // Groups
      nullptr,                         // Conv2dConfig (optional)
      nullptr                          // ComputeKernelConfig (optional)
  );

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  // test Conv2dOp interface
  OpModel backend = dyn_cast<OpModel>(conv2d.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d), OpConfig(/*outputLayout=*/nullptr));
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 229440);
  EXPECT_EQ(peakSize, 190568);
  EXPECT_EQ(outputSize, 28672);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasShardedL1TensorMemoryLayout());
  EXPECT_EQ(outputLayout.getMemLayout().getValue(),
            TensorMemoryLayout::HeightSharded);
}

TEST_F(OpModelBase, PrepareConv2dWeightsOutput) {
  // create Conv2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {64, 3, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  Type elemetType = builder.getBF16Type();

  auto inputLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, inputShape, elemetType, mlir::tt::ttnn::BufferType::DRAM,
      GridAttr::get(&context, 2),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, weightShape, elemetType,
      mlir::tt::ttnn::BufferType::SystemMemory, GridAttr::get(&context, 2));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);

  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
      ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, nullptr, nullptr);

  auto preparedWeightOutput =
      mlir::tt::op_model::ttnn::getPreparedConv2dWeightsOutputTensor(&conv2d);

  auto preparedShape = preparedWeightOutput.getShape();
  llvm::SmallVector<int64_t> expectedShape = {1, 1, 147, 64};

  EXPECT_EQ(preparedShape.size(), expectedShape.size());
  for (size_t i = 0; i < preparedShape.size(); i++) {
    EXPECT_EQ(preparedShape[i], expectedShape[i]);
  }
}

TEST_F(OpModelBase, Conv2dInterfaceConfigs) {
  // create Conv2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {64, 3, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  Type elemetType = builder.getBF16Type();

  auto inputLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, inputShape, elemetType, mlir::tt::ttnn::BufferType::DRAM,
      GridAttr::get(&context, 2),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, weightShape, elemetType,
      mlir::tt::ttnn::BufferType::SystemMemory, GridAttr::get(&context, 2));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);

  GetDeviceOp deviceOp = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      ttnn::MeshShapeAttr::get(builder.getContext(), 1, 1),
      ttnn::MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, nullptr, nullptr);

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  // Will fail due to assertion at
  // tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:156 "Conv2d
  // supports Height, Block or Width Sharded Layouts but got
  // TensorMemoryLayout::INTERLEAVED"
  auto badConvConfig = Conv2dConfigAttr::get(
      &context, /*dtype=*/DataType::BFloat16,
      /*weights_dtype=*/DataType::BFloat16,
      /*activation=*/StringAttr::get(&context, ""),
      /*deallocate_activation=*/BoolAttr::get(&context, false),
      /*reallocate_halo_output=*/BoolAttr::get(&context, true),
      /*act_block_h_override=*/0, /*act_block_w_div=*/1,
      /*reshard_if_not_optimal=*/BoolAttr::get(&context, false),
      /*override_sharding_config=*/BoolAttr::get(&context, false),
      /*shard_layout=*/TensorMemoryLayout::Interleaved,
      /*core_grid=*/ttnn::CoreRangeSetAttr(),
      /*transpose_shards=*/BoolAttr::get(&context, false),
      /*output_layout=*/Layout::Tile,
      /*preprocess_weights_on_device=*/BoolAttr::get(&context, false),
      /*always_preprocess_weights=*/BoolAttr::get(&context, false),
      /*enable_act_double_buffer=*/BoolAttr::get(&context, false),
      /*enable_weights_double_buffer=*/BoolAttr::get(&context, false),
      /*enable_split_reader=*/BoolAttr::get(&context, false),
      /*enable_subblock_padding=*/BoolAttr::get(&context, false));

  OpModel backend = dyn_cast<OpModel>(conv2d.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d),
      OpConfig(getOutputLayout(conv2d), badConvConfig));
  ASSERT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp =
      backend.getOpRuntime(getInputLayouts(conv2d),
                           OpConfig(getOutputLayout(conv2d), badConvConfig));
  ASSERT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto goodConvConfig = Conv2dConfigAttr::get(
      &context, /*dtype=*/DataType::BFloat16,
      /*weights_dtype=*/DataType::BFloat16,
      /*activation=*/StringAttr::get(&context, ""),
      /*deallocate_activation=*/BoolAttr::get(&context, false),
      /*reallocate_halo_output=*/BoolAttr::get(&context, true),
      /*act_block_h_override=*/0, /*act_block_w_div=*/1,
      /*reshard_if_not_optimal=*/BoolAttr::get(&context, false),
      /*override_sharding_config=*/BoolAttr::get(&context, false),
      /*shard_layout=*/std::nullopt,
      /*core_grid=*/ttnn::CoreRangeSetAttr(),
      /*transpose_shards=*/BoolAttr::get(&context, false),
      /*output_layout=*/Layout::Tile,
      /*preprocess_weights_on_device=*/BoolAttr::get(&context, false),
      /*always_preprocess_weights=*/BoolAttr::get(&context, false),
      /*enable_act_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_weights_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_split_reader=*/BoolAttr::get(&context, false),
      /*enable_subblock_padding=*/BoolAttr::get(&context, false));

  constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d),
      OpConfig(getOutputLayout(conv2d), goodConvConfig));
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cb_size, peak_size, output_size, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cb_size, 69696);
  EXPECT_EQ(peak_size, 88400);
  EXPECT_EQ(output_size, 26624);

  // Device hangs otherwise.
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();

  runtimeExp =
      backend.getOpRuntime(getInputLayouts(conv2d),
                           OpConfig(getOutputLayout(conv2d), goodConvConfig));
  ASSERT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

TEST_F(OpModelBase, maxPool2DOp) {
  // TODO(2976): Some of these test cases return L1 interleaved row major
  // tensors which triggers an assertion in TTNNLayoutAttr. Will be reenabled
  // when the linked issue is fixed
  GTEST_SKIP();

  // Create maxPool2DOp with flattened input tensor
  llvm::SmallVector<int64_t> tensorShapeA = {1, 1, 128 * 128, 32};
  llvm::SmallVector<int64_t> tensorShapeO = {1, 1, 64 * 64, 32};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  // Input params
  int32_t batchSize = 1;
  int32_t inputHeight = 128;
  int32_t inputWidth = 128;
  int32_t numChannels = 32;

  // Pooling params
  int32_t kernelHeight = 2;
  int32_t kernelWidth = 2;
  int32_t strideHeight = 2;
  int32_t strideWidth = 2;
  int32_t dilationHeight = 1;
  int32_t dilationWidth = 1;
  int32_t paddingHeight = 0;
  int32_t paddingWidth = 0;
  MemoryConfigAttr memoryConfigAttr = nullptr;
  TensorMemoryLayoutAttr appliedShardScheme = nullptr;
  bool ceilMode = false;
  bool inPlaceHalo = false;

  llvm::SmallVector<int32_t, 2> kernelSize = {kernelHeight, kernelWidth};
  llvm::SmallVector<int32_t, 2> stride = {strideHeight, strideWidth};
  llvm::SmallVector<int32_t, 2> padding = {paddingHeight, paddingWidth};
  llvm::SmallVector<int32_t, 2> dilation = {dilationHeight, dilationWidth};

  auto maxPool2DOp = builder.create<MaxPool2dOp>(
      builder.getUnknownLoc(), output.getType(), input, batchSize, inputHeight,
      inputWidth, numChannels, kernelSize, stride, padding, dilation,
      memoryConfigAttr, appliedShardScheme, ceilMode, inPlaceHalo);
  maxPool2DOp->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  constexpr int32_t numRuns = 10;
  for (int i = 0; i < numRuns; i++) {
    op_model::ttnn::SingletonDeviceContext::resetInstance();
    auto constraintsExp = getOpConstraints(maxPool2DOp.getOperation());
    if (!constraintsExp) {
      FAIL() << "Missing L1 constraints; Error="
             << llvm::toString(constraintsExp.takeError()) << std::endl;
    }
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  }
  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(maxPool2DOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, clampScalarOp) {
  // Create ClampScalarOp with flattened input tensor
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  // Input params
  float minVal = 0.0f;
  float maxVal = 5.0f;

  // Convert float values to APFloat objects
  llvm::APFloat minValAPF(minVal);
  llvm::APFloat maxValAPF(maxVal);

  ClampScalarOp clampScalarOp = builder.create<ClampScalarOp>(
      builder.getUnknownLoc(), outputType, input, minValAPF, maxValAPF);
  clampScalarOp->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto constraintsExp = getOpConstraints(clampScalarOp.getOperation());
  if (!constraintsExp) {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_GT(peakSize, 0);
  EXPECT_GT(outputSize, 0);

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(clampScalarOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, permuteOp) {
  llvm::SmallVector<int64_t> inputShape = {4, 64, 128, 256};
  llvm::SmallVector<int64_t> outputShape = {4, 256, 64, 128};

  auto input = createEmptyTensor(inputShape);
  auto outputType = createRankedTensorType(outputShape);

  PermuteOp permuteOp = builder.create<PermuteOp>(
      builder.getUnknownLoc(), outputType, input,
      llvm::ArrayRef<int64_t>({0, 3, 1, 2}), nullptr, llvm::APFloat(0.0f));
  permuteOp->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto constraintsExp = getOpConstraints(permuteOp.getOperation());
  if (!constraintsExp) {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_GT(peakSize, 0);
  EXPECT_GT(outputSize, 0);

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(permuteOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, upsampleOp) {
  // Create UpsampleOp with flattened input tensor
  llvm::SmallVector<int64_t> inputShape = {2, 128, 16, 8};
  llvm::SmallVector<int64_t> outputShape = {2, 256, 32, 8};
  int scaleFactor = 2;
  std::string mode = "nearest";

  // ttnn::upsample requires input tensor layout to be RowMajor
  // Meanwhile L1 RowMajor does not work, see
  // https://github.com/tenstorrent/tt-mlir/issues/2976
  auto input = createEmptyTensor(
      inputShape, builder.getBF16Type(),
      CreateRowMajorLayout(inputShape, mlir::tt::ttnn::BufferType::DRAM,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved));
  auto outputType = createRankedTensorType(
      outputShape, builder.getBF16Type(),
      CreateRowMajorLayout(outputShape, mlir::tt::ttnn::BufferType::DRAM,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved));

  // Convert to Attr
  mlir::IntegerAttr scaleFactorAttr = builder.getI32IntegerAttr(scaleFactor);
  mlir::StringAttr modeAttr = builder.getStringAttr(mode);

  UpsampleOp upsampleOp =
      builder.create<UpsampleOp>(builder.getUnknownLoc(), outputType, input,
                                 scaleFactorAttr, modeAttr, nullptr);
  upsampleOp->setAttr(DeviceAttr::name, getFakeDeviceAttr());

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  // getOutputLayout() hardcodes L1, so we cannot use it
  OpModel backend = dyn_cast<OpModel>(upsampleOp.getOperation());
  auto constraintsExp =
      backend.getOpConstraints(getInputLayouts(upsampleOp), OpConfig());
  if (!constraintsExp) {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_EQ(peakSize, 0);
  EXPECT_EQ(outputSize, 0);

  op_model::ttnn::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(upsampleOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, SubtractOpInterface) {
  // create SubtractOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sub = builder.create<SubtractOp>(builder.getUnknownLoc(), outputType,
                                        ::mlir::ValueRange{input1, input2});

  // test SubtractOp interface
  auto constraintsExp = getOpConstraints(sub.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12288);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(sub.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, SubtractOpInterfaceNullOutput) {
  // create SubtractOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sub = builder.create<SubtractOp>(builder.getUnknownLoc(), outputType,
                                        ::mlir::ValueRange{input1, input2});

  // test SubtractOp interface
  OpModel backend = dyn_cast<OpModel>(sub.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(sub), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 12288);
  EXPECT_EQ(peakSize, 2048);
  EXPECT_EQ(outputSize, 2048);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedL1TensorMemoryLayout());
}

} // namespace mlir::tt::ttnn
