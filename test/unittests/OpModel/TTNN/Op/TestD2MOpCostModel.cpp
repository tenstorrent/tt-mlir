// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/D2MOpCostModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "gtest/gtest.h"

#include <vector>

using namespace mlir::tt::ttnn;

namespace {

class D2MOpCostModelTest : public OpModelFixture {
public:
  mlir::RankedTensorType
  createRankedTensorType(llvm::ArrayRef<int64_t> shape,
                         mlir::Type elementType = nullptr,
                         TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    return mlir::RankedTensorType::get(shape, elementType, layout);
  }

  mlir::Value createEmptyTensor(llvm::ArrayRef<int64_t> tensorShape,
                                mlir::Type elementType = nullptr,
                                TTNNLayoutAttr layout = nullptr) {
    if (!elementType) {
      elementType = builder.getBF16Type();
    }
    auto rankedTensorType =
        createRankedTensorType(tensorShape, elementType, layout);
    return builder.create<OnesOp>(
        builder.getUnknownLoc(), rankedTensorType, nullptr,
        ShapeAttr::get(&context, tensorShape), nullptr, nullptr, nullptr);
  }

  std::vector<TTNNLayoutAttr> getInputLayoutsFromOperands(mlir::Operation *op) {
    std::vector<TTNNLayoutAttr> inputs;
    for (mlir::Value operand : op->getOperands()) {
      auto type = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (type && type.getEncoding()) {
        inputs.push_back(mlir::cast<TTNNLayoutAttr>(type.getEncoding()));
      }
    }
    return inputs;
  }

  OpConfig getOutputConfig(mlir::Operation *op) {
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    TTNNLayoutAttr outputLayout =
        resultType.getEncoding()
            ? mlir::cast<TTNNLayoutAttr>(resultType.getEncoding())
            : CreateTiledLayout(resultType.getShape(), BufferType::L1,
                                TensorMemoryLayout::Interleaved);
    return OpConfig(outputLayout);
  }
};

TEST_F(D2MOpCostModelTest, AddOp) {
  llvm::SmallVector<int64_t> shape = {64, 64};
  auto layout =
      CreateTiledLayout(shape, BufferType::L1, TensorMemoryLayout::Interleaved);
  auto lhs = createEmptyTensor(shape, builder.getBF16Type(), layout);
  auto rhs = createEmptyTensor(shape, builder.getBF16Type(), layout);
  auto outputType =
      createRankedTensorType(shape, builder.getBF16Type(), layout);
  auto addOp = builder.create<AddOp>(builder.getUnknownLoc(), outputType,
                                     mlir::ValueRange{lhs, rhs});

  std::vector<TTNNLayoutAttr> inputs = getInputLayoutsFromOperands(addOp);
  OpConfig opConfig = getOutputConfig(addOp);
  mlir::Operation *op = addOp.getOperation();

  auto constraints = estimateOpConstraints(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(constraints)) << "estimateOpConstraints failed";
  EXPECT_EQ(constraints->cbL1PeakSize, 0u);
  EXPECT_GT(constraints->tensorL1PeakSize, 0u);
  EXPECT_GT(constraints->peakL1MemorySize, 0u);
  EXPECT_GT(constraints->outputL1BufferSize, 0u);
  EXPECT_GE(constraints->peakL1MemorySize, constraints->outputL1BufferSize);

  auto runtime = estimateOpRuntime(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(runtime)) << "estimateOpRuntime failed";
  EXPECT_EQ(*runtime, 0u);
}

TEST_F(D2MOpCostModelTest, ReluOp) {
  llvm::SmallVector<int64_t> shape = {64, 64};
  auto layout =
      CreateTiledLayout(shape, BufferType::L1, TensorMemoryLayout::Interleaved);
  auto input = createEmptyTensor(shape, builder.getBF16Type(), layout);
  auto outputType =
      createRankedTensorType(shape, builder.getBF16Type(), layout);
  auto reluOp = builder.create<ReluOp>(builder.getUnknownLoc(), outputType,
                                       mlir::ValueRange{input});

  std::vector<TTNNLayoutAttr> inputs = getInputLayoutsFromOperands(reluOp);
  OpConfig opConfig = getOutputConfig(reluOp);
  mlir::Operation *op = reluOp.getOperation();

  auto constraints = estimateOpConstraints(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(constraints)) << "estimateOpConstraints failed";
  EXPECT_EQ(constraints->cbL1PeakSize, 0u);
  EXPECT_GT(constraints->tensorL1PeakSize, 0u);
  EXPECT_GT(constraints->peakL1MemorySize, 0u);
  EXPECT_GT(constraints->outputL1BufferSize, 0u);
  EXPECT_GE(constraints->peakL1MemorySize, constraints->outputL1BufferSize);

  auto runtime = estimateOpRuntime(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(runtime)) << "estimateOpRuntime failed";
  EXPECT_EQ(*runtime, 0u);
}

TEST_F(D2MOpCostModelTest, SumOp) {
  llvm::SmallVector<int64_t> inputShape = {64, 64};
  llvm::SmallVector<int64_t> outputShape = {64, 1}; // sum over dim 1, keep_dim
  auto layoutIn = CreateTiledLayout(inputShape, BufferType::L1,
                                    TensorMemoryLayout::Interleaved);
  auto layoutOut = CreateTiledLayout(outputShape, BufferType::L1,
                                     TensorMemoryLayout::Interleaved);
  auto input = createEmptyTensor(inputShape, builder.getBF16Type(), layoutIn);
  auto outputType =
      createRankedTensorType(outputShape, builder.getBF16Type(), layoutOut);
  auto sumOp = builder.create<SumOp>(
      builder.getUnknownLoc(), outputType, input, /*keep_dim=*/true,
      builder.getArrayAttr(
          llvm::SmallVector<mlir::Attribute>{builder.getI64IntegerAttr(1)}));

  std::vector<TTNNLayoutAttr> inputs = getInputLayoutsFromOperands(sumOp);
  OpConfig opConfig = getOutputConfig(sumOp);
  mlir::Operation *op = sumOp.getOperation();

  auto constraints = estimateOpConstraints(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(constraints)) << "estimateOpConstraints failed";
  EXPECT_EQ(constraints->cbL1PeakSize, 0u);
  EXPECT_GT(constraints->tensorL1PeakSize, 0u);
  EXPECT_GT(constraints->peakL1MemorySize, 0u);
  EXPECT_GT(constraints->outputL1BufferSize, 0u);
  EXPECT_GE(constraints->peakL1MemorySize, constraints->outputL1BufferSize);

  auto runtime = estimateOpRuntime(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(runtime)) << "estimateOpRuntime failed";
  EXPECT_EQ(*runtime, 0u);
}

TEST_F(D2MOpCostModelTest, MatmulOp) {
  llvm::SmallVector<int64_t> shapeA = {64, 64};
  llvm::SmallVector<int64_t> shapeB = {64, 64};
  llvm::SmallVector<int64_t> shapeO = {64, 64};
  auto layoutA = CreateTiledLayout(shapeA, BufferType::L1,
                                   TensorMemoryLayout::Interleaved);
  auto layoutB = CreateTiledLayout(shapeB, BufferType::L1,
                                   TensorMemoryLayout::Interleaved);
  auto layoutO = CreateTiledLayout(shapeO, BufferType::L1,
                                   TensorMemoryLayout::Interleaved);
  auto inputA = createEmptyTensor(shapeA, builder.getBF16Type(), layoutA);
  auto inputB = createEmptyTensor(shapeB, builder.getBF16Type(), layoutB);
  auto outputType =
      createRankedTensorType(shapeO, builder.getBF16Type(), layoutO);
  auto matmulOp = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                           mlir::ValueRange{inputA, inputB});

  std::vector<TTNNLayoutAttr> inputs = getInputLayoutsFromOperands(matmulOp);
  OpConfig opConfig = getOutputConfig(matmulOp);
  mlir::Operation *op = matmulOp.getOperation();

  auto constraints = estimateOpConstraints(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(constraints)) << "estimateOpConstraints failed";
  EXPECT_EQ(constraints->cbL1PeakSize, 0u);
  EXPECT_GT(constraints->tensorL1PeakSize, 0u);
  EXPECT_GT(constraints->peakL1MemorySize, 0u);
  EXPECT_GT(constraints->outputL1BufferSize, 0u);
  EXPECT_GE(constraints->peakL1MemorySize, constraints->outputL1BufferSize);

  auto runtime = estimateOpRuntime(op, inputs, opConfig);
  ASSERT_TRUE(static_cast<bool>(runtime)) << "estimateOpRuntime failed";
  EXPECT_EQ(*runtime, 0u);
}

} // namespace
