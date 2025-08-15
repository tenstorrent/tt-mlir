// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include "ttmlir/OpModel/TTNN/TTNNOpsModelCache.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::ttnn {

class OpModelBase : public OpModelFixture {
public:
  llvm::Expected<op_model::OpConstraints> getOpConstraints(Operation *op) {
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
        inputs.push_back(mlir::cast<TTNNLayoutAttr>(operandType.getEncoding()));
        continue;
      }

      auto inputShape = operandType.getShape();
      auto inputLayout = CreateTiledLayout(inputShape, BufferType::L1,
                                           TensorMemoryLayout::Interleaved);
      inputs.push_back(inputLayout);
    }
    return inputs;
  }

  TTNNLayoutAttr getOutputLayout(Operation *op) {
    auto output = op->getResult(0);
    auto outputShape =
        mlir::cast<RankedTensorType>(output.getType()).getShape();
    return CreateTiledLayout(outputShape, BufferType::L1,
                             TensorMemoryLayout::Interleaved);
  }

  ttcore::DeviceAttr getFakeDeviceAttr() {
    auto deviceIdx = mlir::getAffineConstantExpr(0, &context);
    auto shardOffset = mlir::getAffineConstantExpr(0, &context);
    auto d0 = mlir::getAffineDimExpr(0, &context); // d0
    auto d1 = mlir::getAffineDimExpr(1, &context); // d1
    auto map3 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1}, &context);
    auto map4 = mlir::AffineMap::get(
        /*dimCount=*/2, /*symbolCount=*/0, {deviceIdx, d0, d1, shardOffset},
        &context);
    auto workerGrid = ttcore::GridAttr::get(&context, gridShapeHwN300, map3);

    return ttcore::DeviceAttr::get(&context, workerGrid, map4, map4, {1}, {0});
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
struct ExpectedResult {
  bool expectedLegal = false;
  size_t expectedCbSize = 0;
  size_t expectedPeakSize = 0;
  size_t expectedOutputSize = 0;
};
struct UnaryOpTestParams {
  std::string testName;
  std::function<Operation *(OpBuilder &, Location, Type, ValueRange)> createOp;
  ExpectedResult expectedResult;
};

// Create a fixture for parameterized tests
class UnaryOpModelTest
    : public OpModelBase,
      public ::testing::WithParamInterface<UnaryOpTestParams> {
protected:
  void SetUp() override {
    OpModelBase::SetUp();
    params = GetParam();
  }

  UnaryOpTestParams params;
};

// Test case for normal operation
TEST_P(UnaryOpModelTest, TestOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);
  Operation *op = params.createOp(builder, builder.getUnknownLoc(), outputType,
                                  mlir::ValueRange{input});
  // Test constraints
  auto constraintsExp = getOpConstraints(op);
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, params.expectedResult.expectedCbSize);
    EXPECT_EQ(peakSize, params.expectedResult.expectedPeakSize);
    EXPECT_EQ(outputSize, params.expectedResult.expectedOutputSize);
  } else {
    FAIL() << "Missing L1 constraints for " << params.testName
           << "; Error=" << llvm::toString(constraintsExp.takeError());
  }

  // Test runtime
  auto runtimeExp = getOpRuntime(op);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << "Runtime test failed for " << params.testName
           << "; Error=" << llvm::toString(runtimeExp.takeError());
  }
}

// Test case for null output layout
TEST_P(UnaryOpModelTest, TestOpInterfaceNullOutput) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);
  Operation *op = params.createOp(builder, builder.getUnknownLoc(), outputType,
                                  mlir::ValueRange{input});
  // Test constraints with null output
  OpModel backend = dyn_cast<OpModel>(op);
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(op), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_EQ(static_cast<bool>(constraintsExp),
            params.expectedResult.expectedLegal);

  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, params.expectedResult.expectedCbSize);
  EXPECT_EQ(peakSize, params.expectedResult.expectedPeakSize);
  EXPECT_EQ(outputSize, params.expectedResult.expectedOutputSize);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedL1TensorMemoryLayout());
}

const ExpectedResult expected{true, 8192, 2048, 2048};
const ExpectedResult cbrtExpected{true, 12288, 6144, 2048};

//===---------------------------------------------------------===
const auto createRelu = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<ReluOp>(loc, type, ops).getOperation();
};
const auto createSin = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<SinOp>(loc, type, ops).getOperation();
};
const auto createCos = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<CosOp>(loc, type, ops).getOperation();
};
const auto createExp = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<ExpOp>(loc, type, ops).getOperation();
};
const auto createTanh = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<TanhOp>(loc, type, ops).getOperation();
};
const auto createLog = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<LogOp>(loc, type, ops).getOperation();
};
const auto createAbs = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<AbsOp>(loc, type, ops).getOperation();
};
const auto createCeil = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<CeilOp>(loc, type, ops).getOperation();
};
const auto createSign = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<SignOp>(loc, type, ops).getOperation();
};
const auto createErf = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<ErfOp>(loc, type, ops).getOperation();
};
const auto createErfc = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<ErfcOp>(loc, type, ops).getOperation();
};
const auto createFloor = [](OpBuilder &b, Location loc, Type type,
                            ValueRange ops) {
  return b.create<FloorOp>(loc, type, ops).getOperation();
};
const auto createGelu = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<GeluOp>(loc, type, ops).getOperation();
};
const auto createIsFinite = [](OpBuilder &b, Location loc, Type type,
                               ValueRange ops) {
  return b.create<IsFiniteOp>(loc, type, ops).getOperation();
};
const auto createLogicalNot = [](OpBuilder &b, Location loc, Type type,
                                 ValueRange ops) {
  return b.create<LogicalNotOp>(loc, type, ops).getOperation();
};
const auto createNeg = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<NegOp>(loc, type, ops).getOperation();
};
const auto createTan = [](OpBuilder &b, Location loc, Type type,
                          ValueRange ops) {
  return b.create<TanOp>(loc, type, ops).getOperation();
};
const auto createAtan = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<AtanOp>(loc, type, ops).getOperation();
};
const auto createRsqrt = [](OpBuilder &b, Location loc, Type type,
                            ValueRange ops) {
  return b.create<RsqrtOp>(loc, type, ops).getOperation();
};
const auto createLog1p = [](OpBuilder &b, Location loc, Type type,
                            ValueRange ops) {
  return b.create<Log1pOp>(loc, type, ops).getOperation();
};
const auto createExpm1 = [](OpBuilder &b, Location loc, Type type,
                            ValueRange ops) {
  return b.create<Expm1Op>(loc, type, ops).getOperation();
};
const auto createReciprocal = [](OpBuilder &b, Location loc, Type type,
                                 ValueRange ops) {
  return b.create<ReciprocalOp>(loc, type, ops).getOperation();
};
const auto createCbrt = [](OpBuilder &b, Location loc, Type type,
                           ValueRange ops) {
  return b.create<CbrtOp>(loc, type, ops).getOperation();
};

//===---------------------------------------------------------===
// Define the test parameters
const std::vector<UnaryOpTestParams> unaryOpTestParams = {
    {"Relu", createRelu, expected},
    {"Sin", createSin, expected},
    {"Cos", createCos, expected},
    {"Exp", createExp, expected},
    {"Tanh", createTanh, expected},
    {"Log", createLog, expected},
    {"Abs", createAbs, expected},
    {"Ceil", createCeil, expected},
    {"Sign", createSign, expected},
    {"Erf", createErf, expected},
    {"Erfc", createErfc, expected},
    {"Floor", createFloor, expected},
    {"Reciprocal", createReciprocal, expected},
    {"Cbrt", createCbrt, cbrtExpected},
    {"Gelu", createGelu, expected},
    {"IsFinite", createIsFinite, expected},
    {"LogicalNot", createLogicalNot, expected},
    {"Neg", createNeg, expected},
    {"Tan", createTan, expected},
    {"Atan", createAtan, expected},
    {"Rsqrt", createRsqrt, expected},
    {"Log1p", createLog1p, expected},
    {"Expm1", createExpm1, expected}};

// Instantiate the test suite
INSTANTIATE_TEST_SUITE_P(
    UnaryOpModelTests, UnaryOpModelTest, ::testing::ValuesIn(unaryOpTestParams),
    [](const testing::TestParamInfo<UnaryOpTestParams> &info) {
      return info.param.testName;
    });

//===---------------------------------------------------------===
struct BinaryOpTestParams {
  std::string testName;
  std::function<Operation *(OpBuilder &, Location, Type, ValueRange)> createOp;
  ExpectedResult expectedResult;
};

class BinaryOpModelTest
    : public OpModelBase,
      public ::testing::WithParamInterface<BinaryOpTestParams> {
protected:
  void SetUp() override {
    OpModelBase::SetUp();
    params = GetParam();
  }

  BinaryOpTestParams params;
};

// Test case for normal operation
TEST_P(BinaryOpModelTest, TestOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);
  Operation *op = params.createOp(builder, builder.getUnknownLoc(), outputType,
                                  mlir::ValueRange{input1, input2});
  // Test constraints
  auto constraintsExp = getOpConstraints(op);
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, params.expectedResult.expectedCbSize);
    EXPECT_EQ(peakSize, params.expectedResult.expectedPeakSize);
    EXPECT_EQ(outputSize, params.expectedResult.expectedOutputSize);
  } else {
    FAIL() << "Missing L1 constraints for " << params.testName
           << "; Error=" << llvm::toString(constraintsExp.takeError());
  }

  // Test runtime
  auto runtimeExp = getOpRuntime(op);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << "Runtime test failed for " << params.testName
           << "; Error=" << llvm::toString(runtimeExp.takeError());
  }
}

// Test case for null output layout
TEST_P(BinaryOpModelTest, TestOpInterfaceNullOutput) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);
  Operation *op = params.createOp(builder, builder.getUnknownLoc(), outputType,
                                  mlir::ValueRange{input1, input2});
  // Test constraints with null output
  OpModel backend = dyn_cast<OpModel>(op);
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(op), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_EQ(static_cast<bool>(constraintsExp),
            params.expectedResult.expectedLegal);

  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, params.expectedResult.expectedCbSize);
  EXPECT_EQ(peakSize, params.expectedResult.expectedPeakSize);
  EXPECT_EQ(outputSize, params.expectedResult.expectedOutputSize);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedL1TensorMemoryLayout());
}

// The default expected result for binary operations:
const ExpectedResult binaryExpected{true, 12288, 2048, 2048};
// Some binary ops (such as divide, logicalOr, etc.) require extra circular
// buffer memory which is captured via the following expected values:
const ExpectedResult binaryExpected_extraCb2048{true, 12288 + 2048, 2048, 2048};
const ExpectedResult binaryExpected_extraCb4096{true, 12288 + 4096, 2048, 2048};

//===---------------------------------------------------------===
// Lambda functions for creating binary operations
const auto createAdd = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<AddOp>(l, t, r).getOperation();
};
const auto createSubtract = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<SubtractOp>(l, t, r).getOperation();
};
const auto createMultiply = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<MultiplyOp>(l, t, r).getOperation();
};
const auto createDivide = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<DivideOp>(l, t, r).getOperation();
};
const auto createEqual = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<EqualOp>(l, t, r).getOperation();
};
const auto createNotEqual = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<NotEqualOp>(l, t, r).getOperation();
};
const auto createGE = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<GreaterEqualOp>(l, t, r).getOperation();
};
const auto createGT = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<GreaterThanOp>(l, t, r).getOperation();
};
const auto createLE = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<LessEqualOp>(l, t, r).getOperation();
};
const auto createLT = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<LessThanOp>(l, t, r).getOperation();
};
const auto createAnd = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<LogicalAndOp>(l, t, r).getOperation();
};
const auto createOr = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<LogicalOrOp>(l, t, r).getOperation();
};
const auto createXor = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<LogicalXorOp>(l, t, r).getOperation();
};
const auto createMax = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<MaximumOp>(l, t, r).getOperation();
};
const auto createMin = [](OpBuilder &b, Location l, Type t, ValueRange r) {
  return b.create<MinimumOp>(l, t, r).getOperation();
};
//===---------------------------------------------------------===

// Define the test parameters for binary operations
const std::vector<BinaryOpTestParams> binaryOpTestParams = {
    {"Add", createAdd, binaryExpected},
    {"Subtract", createSubtract, binaryExpected},
    {"Multiply", createMultiply, binaryExpected},
    {"Divide", createDivide, binaryExpected_extraCb2048},
    {"Equal", createEqual, binaryExpected},
    {"NotEqual", createNotEqual, binaryExpected},
    {"GreaterEqual", createGE, binaryExpected},
    {"GreaterThan", createGT, binaryExpected},
    {"LessEqual", createLE, binaryExpected},
    {"LessThan", createLT, binaryExpected},
    {"LogicalAnd", createAnd, binaryExpected_extraCb4096},
    {"LogicalOr", createOr, binaryExpected_extraCb4096},
    {"LogicalXor", createXor, binaryExpected_extraCb4096},
    {"Maximum", createMax, binaryExpected},
    {"Minimum", createMin, binaryExpected}};

// Instantiate the test suite
INSTANTIATE_TEST_SUITE_P(
    BinaryOpModelTests, BinaryOpModelTest,
    ::testing::ValuesIn(binaryOpTestParams),
    [](const testing::TestParamInfo<BinaryOpTestParams> &info) {
      return info.param.testName;
    });

// Separate test for BitwiseNot with integer data types
TEST_F(OpModelBase, BitwiseNotOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  // Create TTNNLayoutAttr with Int32 data type manually
  // (CreateTiledLayout only supports FloatType, not IntegerType)
  auto int32DataType = ttcore::DataType::Int32;
  auto tileType = ttcore::TileType::get(&context, {32, 32}, int32DataType);
  auto bufferType = BufferType::L1;
  auto grid = ttcore::GridAttr::get(&context, {8, 8});
  auto memLayout =
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved);

  auto int32Layout = TTNNLayoutAttr::get(&context, tensorShape, tileType,
                                         bufferType, grid, memLayout);

  // Create tensors with the proper Int32 layout
  auto intType = builder.getIntegerType(32);
  auto inputType = createRankedTensorType(tensorShape, intType, int32Layout);
  auto outputType = createRankedTensorType(tensorShape, intType, int32Layout);

  // Create input tensor using OnesOp with Int32 layout
  auto input = builder.create<OnesOp>(builder.getUnknownLoc(), inputType,
                                      ShapeAttr::get(&context, tensorShape),
                                      nullptr, nullptr, nullptr, nullptr);

  auto bitwiseNot = builder.create<BitwiseNotOp>(
      builder.getUnknownLoc(), outputType, ::mlir::ValueRange{input});

  // Test BitwiseNot interface
  auto constraintsExp = getOpConstraints(bitwiseNot.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 16384);
    EXPECT_EQ(peakSize, 4096);
    EXPECT_EQ(outputSize, 4096);
  } else {
    FAIL() << "Missing L1 constraints for BitwiseNot; Error="
           << llvm::toString(constraintsExp.takeError());
  }

  auto runtimeExp = getOpRuntime(bitwiseNot.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << "Runtime test failed for BitwiseNot; Error="
           << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, SqrtOpInterface) {
  // create SqrtOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sqrt = builder.create<SqrtOp>(builder.getUnknownLoc(), outputType,
                                     mlir::ValueRange{input});

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
                                           mlir::ValueRange{input});

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

TEST_F(OpModelBase, LinearOpInterface) {
  // create LinearOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> biasShape = {2048, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto bias = createEmptyTensor(biasShape);
  auto outputType = createRankedTensorType(tensorShapeO);

  auto linear =
      builder.create<LinearOp>(builder.getUnknownLoc(), outputType,
                               mlir::ValueRange{inputA, inputB, bias});

  // test LinearOp interface
  auto constraintsExp = getOpConstraints(linear.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 262144);
    EXPECT_EQ(peakSize, 262144);
    EXPECT_EQ(outputSize, 131072);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(linear.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, LinearOpInterfaceNullOutput) {
  // create LinearOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> biasShape = {2048, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto bias = createEmptyTensor(biasShape);
  auto outputType = createRankedTensorType(tensorShapeO);

  auto linear =
      builder.create<LinearOp>(builder.getUnknownLoc(), outputType,
                               mlir::ValueRange{inputA, inputB, bias});

  // test LinearOp interface
  OpModel backend = dyn_cast<OpModel>(linear.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(linear), OpConfig(/*outputLayout=*/nullptr));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 262144);
  EXPECT_EQ(peakSize, 0);
  EXPECT_EQ(outputSize, 0);

  ASSERT_TRUE(outputLayout);
  EXPECT_EQ(outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(outputLayout.hasInterleavedDRAMTensorMemoryLayout());
  op_model::SingletonDeviceContext::resetInstance();
}

TEST_F(OpModelBase, LinearOpInterfacePartialOutput) {
  // create LinearOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> biasShape = {2048, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto bias = createEmptyTensor(biasShape);
  auto outputType = createRankedTensorType(tensorShapeO);

  auto outputLayout = CreateTiledLayout(tensorShapeO, BufferType::L1,
                                        TensorMemoryLayout::BlockSharded)
                          .withIgnorePhysicalLayout(true);
  auto linear =
      builder.create<LinearOp>(builder.getUnknownLoc(), outputType,
                               mlir::ValueRange{inputA, inputB, bias});

  // test LinearOp interface
  OpModel backend = dyn_cast<OpModel>(linear.getOperation());
  auto constraintsExp =
      backend.getOpConstraints(getInputLayouts(linear), OpConfig(outputLayout));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  auto constraints = constraintsExp.get();
  EXPECT_EQ(constraints.cbL1PeakSize, 131072);
  EXPECT_EQ(constraints.tensorL1PeakSize, 262144);
  EXPECT_EQ(constraints.outputL1BufferSize, 131072);

  ASSERT_TRUE(constraints.outputLayout);
  EXPECT_EQ(constraints.outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(constraints.outputLayout.hasShardedL1TensorMemoryLayout());
  EXPECT_TRUE(constraints.outputLayout.getGrid());
  op_model::SingletonDeviceContext::resetInstance();
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
                                         mlir::ValueRange{inputA, inputB});

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
                                         mlir::ValueRange{inputA, inputB});

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

TEST_F(OpModelBase, MatmulOpInterfacePartialOutput) {
  // create MatmulOp
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeB = {1024, 2048};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 2048};

  auto inputA = createEmptyTensor(tensorShapeA);
  auto inputB = createEmptyTensor(tensorShapeB);
  auto outputType = createRankedTensorType(tensorShapeO);

  auto outputLayout = CreateTiledLayout(tensorShapeO, BufferType::L1,
                                        TensorMemoryLayout::BlockSharded)
                          .withIgnorePhysicalLayout(true);
  auto matmul = builder.create<MatmulOp>(builder.getUnknownLoc(), outputType,
                                         mlir::ValueRange{inputA, inputB});

  // test MatmulOp interface
  OpModel backend = dyn_cast<OpModel>(matmul.getOperation());
  auto constraintsExp =
      backend.getOpConstraints(getInputLayouts(matmul), OpConfig(outputLayout));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  auto constraints = constraintsExp.get();
  EXPECT_EQ(constraints.cbL1PeakSize, 262144);
  EXPECT_EQ(constraints.tensorL1PeakSize, 524288);
  EXPECT_EQ(constraints.outputL1BufferSize, 524288);

  ASSERT_TRUE(constraints.outputLayout);
  EXPECT_EQ(constraints.outputLayout.getLayout(), Layout::Tile);
  EXPECT_TRUE(constraints.outputLayout.hasShardedL1TensorMemoryLayout());
  EXPECT_TRUE(constraints.outputLayout.getGrid());
}

// Forward declarations
using OpConstraintsFn =
    llvm::Expected<op_model::OpConstraints> (OpModelBase::*)(mlir::Operation *);
using OpRuntimeFn = llvm::Expected<size_t> (OpModelBase::*)(mlir::Operation *);

// Helper function to test reduction operations (Mean/Sum/etc.)
template <typename OpType>
void testReductionOp(OpModelBase *testFixture, mlir::OpBuilder &builder,
                     mlir::Value input, mlir::Type outputType,
                     int64_t expectedCbSize, int64_t expectedPeakSize,
                     int64_t expectedOutputSize,
                     OpConstraintsFn getOpConstraintsFn,
                     OpRuntimeFn getOpRuntimeFn) {
  // Create the reduction operation
  auto op = builder.create<OpType>(builder.getUnknownLoc(), outputType,
                                   mlir::ValueRange{input});
  op.setKeepDim(true);
  op.setDimArgAttr(builder.getArrayAttr(
      llvm::SmallVector<mlir::Attribute>{builder.getI64IntegerAttr(1)}));

  // Test operation constraints
  auto constraintsExp = (testFixture->*getOpConstraintsFn)(op.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  // Test operation runtime
  auto runtimeExp = (testFixture->*getOpRuntimeFn)(op.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, MeanOpInterface) {
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 1024};
  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  testReductionOp<MeanOp>(
      this, builder, input, output.getType(), /*expectedCbSize=*/12288,
      /*expectedPeakSize=*/2048, /*expectedOutputSize=*/2048,
      &OpModelBase::getOpConstraints, &OpModelBase::getOpRuntime);
}

TEST_F(OpModelBase, SumOpInterface) {
  llvm::SmallVector<int64_t> tensorShapeA = {2048, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {2048, 1024};
  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  testReductionOp<SumOp>(
      this, builder, input, output.getType(), /*expectedCbSize=*/12288,
      /*expectedPeakSize=*/2048, /*expectedOutputSize=*/2048,
      &OpModelBase::getOpConstraints, &OpModelBase::getOpRuntime);

  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();
}

TEST_F(OpModelBase, ReshapeOpInterface) {
  // create ReshapeOp
  llvm::SmallVector<int64_t> tensorShapeA = {64, 1024};
  llvm::SmallVector<int64_t> tensorShapeO = {64 * 4, 1024 / 4};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();
  auto reshape = builder.create<ReshapeOp>(
      builder.getUnknownLoc(), output.getType(), mlir::ValueRange{input});
  reshape.setShapeAttr(builder.getArrayAttr(llvm::SmallVector<mlir::Attribute>{
      builder.getI64IntegerAttr(64 * 4), builder.getI64IntegerAttr(1024 / 4)}));

  // Clear the caches to ensure we are testing the cache functionality just for
  // this op:
  opConstraintsCache().clear();
  opRuntimeCache().clear();

  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();
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
  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();

  auto cachedConstraintsExp = getOpConstraints(reshape.getOperation());
  if (cachedConstraintsExp) {
    auto l1 = cachedConstraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 5120);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();

  auto cachedRuntimeExp = getOpRuntime(reshape.getOperation());
  if (cachedRuntimeExp) {
    EXPECT_TRUE(cachedRuntimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }

  EXPECT_EQ(opConstraintsCache().getStats().hits, 1);
  EXPECT_EQ(opConstraintsCache().getStats().misses, 1);

  EXPECT_EQ(opRuntimeCache().getStats().hits, 1);
  EXPECT_EQ(opRuntimeCache().getStats().misses, 1);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  op_model::SingletonDeviceContext::resetInstance();
}

TEST_F(OpModelBase, SliceOpInterface) {
  // create SliceOp
  llvm::SmallVector<int64_t> tensorShapeA = {1, 56, 56, 96};
  llvm::SmallVector<int64_t> tensorShapeO = {1, 28, 56, 95};

  auto input = createEmptyTensor(tensorShapeA);
  auto output = createEmptyTensor(tensorShapeO);

  llvm::SmallVector<int64_t> beginsArray = {0, 0, 0, 0};
  llvm::SmallVector<int64_t> endsArray = {1, 56, 56, 95};
  llvm::SmallVector<int64_t> stepArray = {1, 2, 1, 1};

  auto sliceOp = builder.create<SliceOp>(
      builder.getUnknownLoc(), output.getType(), mlir::ValueRange{input});

  sliceOp.setBeginsAttr(builder.getI64ArrayAttr(beginsArray));
  sliceOp.setEndsAttr(builder.getI64ArrayAttr(endsArray));
  sliceOp.setStepAttr(builder.getI64ArrayAttr(stepArray));

  // test SliceOp interface
  auto constraintsExp = getOpConstraints(sliceOp.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(sliceOp.getOperation());
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

  ToLayoutOp toLayout =
      builder.create<ToLayoutOp>(builder.getUnknownLoc(), tensor.getType(),
                                 tensor, Layout::Tile, nullptr, nullptr);

  // Manually create the operand layouts for calling the backend to make sure
  // the layouts are propagated all the way
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

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

TEST_F(OpModelBase, toMemoryConfigOp) {
  llvm::SmallVector<int64_t> tensorShape = {1, 8, 64, 1024};
  const TTNNLayoutAttr inputLayout_L1Tiled = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayout_DRAMRowMajor = CreateRowMajorLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  MemoryConfigAttr memoryConfig = MemoryConfigAttr::get(
      &context, outputLayout_DRAMRowMajor.getMemLayout(),
      BufferTypeAttr::get(&context, outputLayout_DRAMRowMajor.getBufferType()),
      std::nullopt);
  auto inputTensor =
      createEmptyTensor(tensorShape, nullptr, inputLayout_L1Tiled);

  ToMemoryConfigOp toMemoryConfig = builder.create<ToMemoryConfigOp>(
      builder.getUnknownLoc(), inputTensor.getType(), inputTensor,
      memoryConfig);

  OpModel backend = dyn_cast<OpModel>(toMemoryConfig.getOperation());
  if (!backend) {
    FAIL() << "Could not cast op to OpModel";
  }

  auto constraintsExp =
      backend.getOpConstraints(std::vector{inputLayout_L1Tiled}, OpConfig());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_GT(cbSize, 0);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp =
      backend.getOpRuntime(std::vector{inputLayout_L1Tiled}, OpConfig());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, concatOp) {
  // create concat op
  llvm::SmallVector<int64_t> tensorShape1 = {1, 2, 8, 32};
  llvm::SmallVector<int64_t> tensorShape2 = {1, 2, 16, 32};
  llvm::SmallVector<int64_t> tensorShape3 = {1, 2, 64, 32};
  llvm::SmallVector<int64_t> tensorShapeO = {1, 2, 88, 32};

  mlir::Value inputTensor1 = createEmptyTensor(tensorShape1);
  mlir::Value inputTensor2 = createEmptyTensor(tensorShape2);
  mlir::Value inputTensor3 = createEmptyTensor(tensorShape3);
  mlir::Value output = createEmptyTensor(tensorShapeO);

  auto concatOp = builder.create<ConcatOp>(
      builder.getUnknownLoc(), output.getType(),
      mlir::ValueRange{inputTensor1, inputTensor2, inputTensor3}, 2, nullptr);

  // test concat Op interface
  auto constraintsExp = getOpConstraints(concatOp.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto &[cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(concatOp.getOperation());
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
  // create TypecastOp
  llvm::SmallVector<int64_t> tensorShape = {64, 1024};

  RankedTensorType rankedTensorTypeBF16 =
      RankedTensorType::get(tensorShape, builder.getBF16Type());

  auto input = builder.create<OnesOp>(
      builder.getUnknownLoc(), rankedTensorTypeBF16,
      ShapeAttr::get(&context, tensorShape),
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::BFloat16), nullptr,
      nullptr, nullptr);
  RankedTensorType rankedTensorTypeF32 =
      RankedTensorType::get(tensorShape, builder.getF32Type());

  auto typecast =
      builder.create<TypecastOp>(builder.getUnknownLoc(), rankedTensorTypeF32,
                                 input, ttcore::DataType::Float32);

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
  auto weightLayout = TTNNLayoutAttr::get(
      &context, weightShape, weightElementType, BufferType::SystemMemory,
      ttcore::GridAttr::get(&context));
  auto weight = createEmptyTensor(weightShape, weightElementType, weightLayout);
  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

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
      outputDtype,                     // OutputDtype
      nullptr,                         // Conv2dConfig (optional)
      nullptr                          // ComputeKernelConfig (optional)
  );

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  // test Conv2dOp interface
  auto constraintsExp = getOpConstraints(conv2d.getOperation());
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 229440);
  EXPECT_EQ(peakSize, 190572);
  EXPECT_EQ(outputSize, 26624);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

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
  auto weightLayout = TTNNLayoutAttr::get(
      &context, weightShape, weightElementType, BufferType::SystemMemory,
      ttcore::GridAttr::get(&context));
  auto weight = createEmptyTensor(weightShape, weightElementType, weightLayout);
  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

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
      outputDtype,                     // OutputDtype
      nullptr,                         // Conv2dConfig (optional)
      nullptr                          // ComputeKernelConfig (optional)
  );

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  // test Conv2dOp interface
  OpModel backend = dyn_cast<OpModel>(conv2d.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d), OpConfig(/*outputLayout=*/nullptr));
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 229440);
  EXPECT_EQ(peakSize, 190572);
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

  auto inputLayout = TTNNLayoutAttr::get(
      &context, inputShape, elemetType, BufferType::DRAM,
      ttcore::GridAttr::get(&context),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = TTNNLayoutAttr::get(&context, weightShape, elemetType,
                                          BufferType::SystemMemory,
                                          ttcore::GridAttr::get(&context));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, outputDtype, nullptr, nullptr);

  auto preparedWeightOutput =
      op_model::getPreparedConv2dWeightsOutputTensor(&conv2d);

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

  auto inputLayout = TTNNLayoutAttr::get(
      &context, inputShape, elemetType, BufferType::DRAM,
      ttcore::GridAttr::get(&context),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = TTNNLayoutAttr::get(&context, weightShape, elemetType,
                                          BufferType::SystemMemory,
                                          ttcore::GridAttr::get(&context));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, outputDtype, nullptr, nullptr);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  // Will fail due to assertion at
  // tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:156 "Conv2d
  // supports Height, Block or Width Sharded Layouts but got
  // TensorMemoryLayout::INTERLEAVED"
  auto badConvConfig = Conv2dConfigAttr::get(
      &context,
      /*weights_dtype=*/ttcore::DataType::BFloat16,
      /*activation=*/StringAttr::get(&context, ""),
      /*deallocate_activation=*/BoolAttr::get(&context, false),
      /*reallocate_halo_output=*/BoolAttr::get(&context, true),
      /*act_block_h_override=*/0, /*act_block_w_div=*/1,
      /*reshard_if_not_optimal=*/BoolAttr::get(&context, false),
      /*override_sharding_config=*/BoolAttr::get(&context, false),
      /*shard_layout=*/TensorMemoryLayout::Interleaved,
      /*core_grid=*/CoreRangeSetAttr(),
      /*transpose_shards=*/BoolAttr::get(&context, false),
      /*output_layout=*/Layout::Tile,
      /*enable_act_double_buffer=*/BoolAttr::get(&context, false),
      /*enable_weights_double_buffer=*/BoolAttr::get(&context, false),
      /*enable_split_reader=*/BoolAttr::get(&context, false),
      /*in_place=*/BoolAttr::get(&context, false));

  OpModel backend = dyn_cast<OpModel>(conv2d.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d),
      OpConfig(getOutputLayout(conv2d),
               Conv2dAttrs{badConvConfig, std::nullopt}));
  ASSERT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp =
      backend.getOpRuntime(getInputLayouts(conv2d),
                           OpConfig(getOutputLayout(conv2d),
                                    Conv2dAttrs{badConvConfig, std::nullopt}));
  ASSERT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  auto goodConvConfig = Conv2dConfigAttr::get(
      &context,
      /*weights_dtype=*/ttcore::DataType::BFloat16,
      /*activation=*/StringAttr::get(&context, ""),
      /*deallocate_activation=*/BoolAttr::get(&context, false),
      /*reallocate_halo_output=*/BoolAttr::get(&context, true),
      /*act_block_h_override=*/0, /*act_block_w_div=*/1,
      /*reshard_if_not_optimal=*/BoolAttr::get(&context, false),
      /*override_sharding_config=*/BoolAttr::get(&context, false),
      /*shard_layout=*/std::nullopt,
      /*core_grid=*/CoreRangeSetAttr(),
      /*transpose_shards=*/BoolAttr::get(&context, false),
      /*output_layout=*/Layout::Tile,
      /*enable_act_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_weights_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_split_reader=*/BoolAttr::get(&context, false),
      /*in_place=*/BoolAttr::get(&context, false));

  constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d),
      OpConfig(getOutputLayout(conv2d),
               Conv2dAttrs{goodConvConfig, std::nullopt}));
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cb_size, peak_size, output_size, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cb_size, 69696);
  EXPECT_EQ(peak_size, 88400);
  EXPECT_EQ(output_size, 26624);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  runtimeExp =
      backend.getOpRuntime(getInputLayouts(conv2d),
                           OpConfig(getOutputLayout(conv2d),
                                    Conv2dAttrs{goodConvConfig, std::nullopt}));
  ASSERT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

TEST_F(OpModelBase, conv2dInterfaceComputeKernelConfig) {
  // create Conv2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {64, 3, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  Type elemetType = builder.getBF16Type();

  auto inputLayout = TTNNLayoutAttr::get(
      &context, inputShape, elemetType, BufferType::DRAM,
      ttcore::GridAttr::get(&context, 2),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = TTNNLayoutAttr::get(&context, weightShape, elemetType,
                                          BufferType::SystemMemory,
                                          ttcore::GridAttr::get(&context, 2));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

  Conv2dOp conv2d = builder.create<Conv2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({1, 1}), 1, outputDtype, nullptr, nullptr);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  OpModel backend = dyn_cast<OpModel>(conv2d.getOperation());

  Conv2dConfigAttr configAttr = Conv2dConfigAttr::get(&context);
  DeviceComputeKernelConfigAttr deviceConfigAttr =
      DeviceComputeKernelConfigAttr::get(
          &context, MathFidelity::HiFi2, BoolAttr::get(&context, true),
          BoolAttr::get(&context, false), BoolAttr::get(&context, true),
          BoolAttr::get(&context, true));
  Conv2dAttrs opConfigAttrs = Conv2dAttrs{
      configAttr, deviceConfigAttr}; // passes both configs to backend.

  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(conv2d),
      OpConfig(getOutputLayout(conv2d), opConfigAttrs));

  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cb_size, peak_size, output_size, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cb_size, 65600);
  EXPECT_EQ(peak_size, 88400);
  EXPECT_EQ(output_size, 26624);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp =
      backend.getOpRuntime(getInputLayouts(conv2d),
                           OpConfig(getOutputLayout(conv2d), opConfigAttrs));
  ASSERT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

TEST_F(OpModelBase, ConvTranspose2dInterfaceConfigs) {
  // create ConvTranspose2dOp
  llvm::SmallVector<int64_t> inputShape = {1, 1, 50176, 3};
  llvm::SmallVector<int64_t> weightShape = {3, 64, 7, 7};
  llvm::SmallVector<int64_t> outputShape = {1, 1, 12544, 64};

  Type elemetType = builder.getBF16Type();

  auto inputLayout = TTNNLayoutAttr::get(
      &context, inputShape, elemetType, BufferType::DRAM,
      ttcore::GridAttr::get(&context),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));
  auto input = createEmptyTensor(inputShape, elemetType, inputLayout);

  auto weightLayout = TTNNLayoutAttr::get(&context, weightShape, elemetType,
                                          BufferType::SystemMemory,
                                          ttcore::GridAttr::get(&context));
  auto weight = createEmptyTensor(weightShape, elemetType, weightLayout);

  auto outputType = createRankedTensorType(outputShape);
  auto outputDtype = ttcore::DataTypeAttr::get(
      &context, ttcore::elementTypeToDataType(outputType.getElementType()));

  GetDeviceOp deviceOp = builder.create<GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<DeviceType>(),
      MeshShapeAttr::get(builder.getContext(), 1, 1),
      MeshOffsetAttr::get(builder.getContext(), 0, 0));

  ConvTranspose2dOp convTranspose2d = builder.create<ConvTranspose2dOp>(
      builder.getUnknownLoc(), outputType, input, weight, nullptr, deviceOp, 3,
      64, 1, 224, 224, llvm::ArrayRef<int32_t>({7, 7}),
      llvm::ArrayRef<int32_t>({2, 2}), llvm::ArrayRef<int32_t>({3, 3}),
      llvm::ArrayRef<int32_t>({0, 0}), llvm::ArrayRef<int32_t>({1, 1}), 1,
      outputDtype, nullptr, nullptr);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  auto goodConvConfig = Conv2dConfigAttr::get(
      &context,
      /*weights_dtype=*/ttcore::DataType::BFloat16,
      /*activation=*/StringAttr::get(&context, ""),
      /*deallocate_activation=*/BoolAttr::get(&context, false),
      /*reallocate_halo_output=*/BoolAttr::get(&context, true),
      /*act_block_h_override=*/0, /*act_block_w_div=*/1,
      /*reshard_if_not_optimal=*/BoolAttr::get(&context, false),
      /*override_sharding_config=*/BoolAttr::get(&context, false),
      /*shard_layout=*/std::nullopt,
      /*core_grid=*/CoreRangeSetAttr(),
      /*transpose_shards=*/BoolAttr::get(&context, false),
      /*output_layout=*/Layout::Tile,
      /*enable_act_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_weights_double_buffer=*/BoolAttr::get(&context, true),
      /*enable_split_reader=*/BoolAttr::get(&context, false),
      /*in_place=*/BoolAttr::get(&context, false));

  OpModel backend = dyn_cast<OpModel>(convTranspose2d.getOperation());
  auto constraintsExp = backend.getOpConstraints(
      getInputLayouts(convTranspose2d),
      OpConfig(getOutputLayout(convTranspose2d),
               Conv2dAttrs{goodConvConfig, std::nullopt}));
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  const auto &[cb_size, peak_size, output_size, outputLayout] =
      constraintsExp.get();
  EXPECT_GT(cb_size, 0);
  EXPECT_GT(peak_size, 0);
  EXPECT_GT(output_size, 0);

  // Device hangs otherwise.
  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp =
      backend.getOpRuntime(getInputLayouts(convTranspose2d),
                           OpConfig(getOutputLayout(convTranspose2d),
                                    Conv2dAttrs{goodConvConfig, std::nullopt}));
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
  maxPool2DOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  constexpr int32_t numRuns = 10;
  for (int i = 0; i < numRuns; i++) {
    op_model::SingletonDeviceContext::resetInstance();
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
  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(maxPool2DOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, LeakyReluOp) {
  // Create LeakyReluOp with flattened input tensor
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  // Input param
  float slope = 0.01f;

  // Convert float value to APFloat object
  llvm::APFloat slopeAPF(slope);

  LeakyReluOp leakyReluOp = builder.create<LeakyReluOp>(
      builder.getUnknownLoc(), outputType, input, slopeAPF);
  leakyReluOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  op_model::SingletonDeviceContext::resetInstance();

  auto constraintsExp = getOpConstraints(leakyReluOp.getOperation());
  if (!constraintsExp) {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
  const auto &[cbSize, peakSize, outputSize, outputLayout] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 8192);
  EXPECT_EQ(peakSize, 2048);
  EXPECT_EQ(outputSize, 2048);

  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(leakyReluOp.getOperation());
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
  clampScalarOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  op_model::SingletonDeviceContext::resetInstance();

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

  op_model::SingletonDeviceContext::resetInstance();

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
  permuteOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  op_model::SingletonDeviceContext::resetInstance();

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

  op_model::SingletonDeviceContext::resetInstance();

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

  // upsample requires input tensor layout to be RowMajor
  // Meanwhile L1 RowMajor does not work, see
  // https://github.com/tenstorrent/tt-mlir/issues/2976
  auto input =
      createEmptyTensor(inputShape, builder.getBF16Type(),
                        CreateRowMajorLayout(inputShape, BufferType::DRAM,
                                             TensorMemoryLayout::Interleaved));
  auto outputType = createRankedTensorType(
      outputShape, builder.getBF16Type(),
      CreateRowMajorLayout(outputShape, BufferType::DRAM,
                           TensorMemoryLayout::Interleaved));

  // Convert to Attr
  mlir::IntegerAttr scaleFactorAttr = builder.getSI32IntegerAttr(scaleFactor);
  mlir::StringAttr modeAttr = builder.getStringAttr(mode);

  UpsampleOp upsampleOp =
      builder.create<UpsampleOp>(builder.getUnknownLoc(), outputType, input,
                                 scaleFactorAttr, modeAttr, nullptr);
  upsampleOp->setAttr(ttcore::DeviceAttr::name, getFakeDeviceAttr());

  op_model::SingletonDeviceContext::resetInstance();

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

  op_model::SingletonDeviceContext::resetInstance();

  auto runtimeExp = getOpRuntime(upsampleOp.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, EmbeddingOpInterface) {
  // Create input tensors for embedding op
  // [batch_size, seq_len]
  llvm::SmallVector<int64_t> inputShape = {workerCoresN300, 1024};
  // [vocab_size, hidden_size]
  llvm::SmallVector<int64_t> weightShape = {256, 128};
  // [batch_size, seq_len, hidden_size]
  llvm::SmallVector<int64_t> outputShape = {workerCoresN300, 1024, 128};

  auto input = createEmptyTensor(inputShape);
  auto weight = createEmptyTensor(weightShape);
  auto outputType = createRankedTensorType(outputShape);

  // Create EmbeddingOp
  auto embedding = builder.create<EmbeddingOp>(
      builder.getUnknownLoc(), outputType, mlir::ValueRange{input, weight});

  // Test EmbeddingOp interface constraints
  auto constraintsExp = getOpConstraints(embedding.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 16384);
    EXPECT_EQ(peakSize, 525312);
    EXPECT_EQ(outputSize, 262144);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  // Test EmbeddingOp runtime
  auto runtimeExp = getOpRuntime(embedding.getOperation());
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, EmbeddingOpNullOutputLayout) {
  // Create input tensors for embedding op
  // [batch_size, seq_len]
  llvm::SmallVector<int64_t> inputShape = {workerCoresN300, 1024};
  // [vocab_size, hidden_size]
  llvm::SmallVector<int64_t> weightShape = {256, 128};
  // [batch_size, seq_len, hidden_size]
  llvm::SmallVector<int64_t> outputShape = {workerCoresN300, 1024, 128};

  auto input = createEmptyTensor(inputShape);
  auto weight = createEmptyTensor(weightShape);
  auto outputType = createRankedTensorType(outputShape);

  // Create EmbeddingOp
  auto embedding = builder.create<EmbeddingOp>(
      builder.getUnknownLoc(), outputType, ::mlir::ValueRange{input, weight});

  // Test EmbeddingOp interface constraints
  auto constraintsExp = embedding.getOpConstraints(
      getInputLayouts(embedding), OpConfig(/*outputLayout=*/nullptr));
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 16384);
    EXPECT_EQ(peakSize, 525312);
    EXPECT_EQ(outputSize, 262144);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  // Test EmbeddingOp runtime
  auto runtimeExp = embedding.getOpRuntime(getInputLayouts(embedding),
                                           OpConfig(/*outputLayout=*/nullptr));
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, EmbeddingBackwardOp) {
  llvm::SmallVector<int64_t> inputShape = {2, 1024};
  llvm::SmallVector<int64_t> weightShape = {3200, 4096};
  llvm::SmallVector<int64_t> inGradientShape = {1, 1, 2048, 4096};

  auto input =
      createEmptyTensor(inputShape, builder.getBF16Type(),
                        CreateRowMajorLayout(inputShape, BufferType::DRAM,
                                             TensorMemoryLayout::Interleaved));
  auto weight =
      createEmptyTensor(weightShape, builder.getBF16Type(),
                        CreateRowMajorLayout(weightShape, BufferType::DRAM,
                                             TensorMemoryLayout::Interleaved));
  auto inGradient = createEmptyTensor(inGradientShape);
  auto outputType = createRankedTensorType(
      inGradientShape, builder.getBF16Type(),
      CreateTiledLayout(inGradientShape, BufferType::L1,
                        TensorMemoryLayout::Interleaved));

  auto embeddingBackward = builder.create<EmbeddingBackwardOp>(
      builder.getUnknownLoc(), outputType,
      ::mlir::ValueRange{input, weight, inGradient});

  auto constraintsExp = getOpConstraints(embeddingBackward.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 12400);
    EXPECT_EQ(peakSize, 409600);
    EXPECT_EQ(outputSize, 409600);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(embeddingBackward.getOperation());
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, CacheOpConstraintsTest) {
  opConstraintsCache().clear();
  opRuntimeCache().clear();

  // create SubtractOp
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);

  auto sub = builder.create<SubtractOp>(builder.getUnknownLoc(), outputType,
                                        mlir::ValueRange{input1, input2});

  // test SubtractOp interface
  auto constraintsExp = getOpConstraints(sub.getOperation());
  ASSERT_TRUE(static_cast<bool>(constraintsExp));
  auto stats = opConstraintsCache().getStats();
  EXPECT_EQ(stats.hits, 0);
  EXPECT_EQ(stats.misses, 1);

  // If getOpConstraints is called again, it will always produce the same result
  // without calculation:
  auto cachedConstraintsExp = getOpConstraints(sub.getOperation());
  ASSERT_TRUE(static_cast<bool>(cachedConstraintsExp));

  stats = opConstraintsCache().getStats();
  EXPECT_EQ(stats.hits, 1);
  EXPECT_EQ(stats.misses, 1);

  // Clear the cache and verify that it is empty:
  opConstraintsCache().clear();
  stats = opConstraintsCache().getStats();
  EXPECT_EQ(stats.hits, 0);
  EXPECT_EQ(stats.misses, 0);
  EXPECT_TRUE(opConstraintsCache().empty());

  // test the runtime cache:
  auto runtimeExp = getOpRuntime(sub.getOperation());
  ASSERT_TRUE(static_cast<bool>(runtimeExp));
  auto statsRuntime = opRuntimeCache().getStats();
  EXPECT_EQ(statsRuntime.hits, 0);
  EXPECT_EQ(statsRuntime.misses, 1);

  // If getOpRuntime is called again, it will always produce the same result
  // without calculation:
  auto cachedRuntimeExp = getOpRuntime(sub.getOperation());
  ASSERT_TRUE(static_cast<bool>(cachedRuntimeExp));

  statsRuntime = opRuntimeCache().getStats();
  EXPECT_EQ(statsRuntime.hits, 1);
  EXPECT_EQ(statsRuntime.misses, 1);

  // Clear the cache and verify that it is empty:
  opRuntimeCache().clear();
  statsRuntime = opRuntimeCache().getStats();
  EXPECT_EQ(statsRuntime.hits, 0);
  EXPECT_EQ(statsRuntime.misses, 0);
  EXPECT_TRUE(opRuntimeCache().empty());
}

TEST_F(OpModelBase, CacheOpConstraintsMissesTest) {
  opConstraintsCache().clear();
  opRuntimeCache().clear();

  // create Two add ops with different input sizes:
  llvm::SmallVector<int64_t> tensorShape1 = {workerCoresN300, 1024};
  auto input1 = createEmptyTensor(tensorShape1);
  auto input2 = createEmptyTensor(tensorShape1);
  auto outputType1 = createRankedTensorType(tensorShape1);
  auto add1 = builder.create<AddOp>(builder.getUnknownLoc(), outputType1,
                                    mlir::ValueRange{input1, input2});

  llvm::SmallVector<int64_t> tensorShape2 = {workerCoresN300, 512};
  auto input3 = createEmptyTensor(tensorShape2);
  auto input4 = createEmptyTensor(tensorShape2);
  auto outputType2 = createRankedTensorType(tensorShape2);
  auto add2 = builder.create<AddOp>(builder.getUnknownLoc(), outputType2,
                                    mlir::ValueRange{input3, input4});

  // test AddOp interface
  auto constraintsExp1 = getOpConstraints(add1.getOperation());
  ASSERT_TRUE(static_cast<bool>(constraintsExp1));
  auto stats1 = opConstraintsCache().getStats();
  EXPECT_EQ(stats1.hits, 0);
  EXPECT_EQ(stats1.misses, 1);

  auto constraintsExp2 = getOpConstraints(add2.getOperation());
  ASSERT_TRUE(static_cast<bool>(constraintsExp2));
  auto stats2 = opConstraintsCache().getStats();
  EXPECT_EQ(stats2.hits, 0);
  // The input sizes are different, so it should be a miss:
  EXPECT_EQ(stats2.misses, 2);
}

TEST_F(OpModelBase, WhereOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto input1 = createEmptyTensor(tensorShape);
  auto input2 = createEmptyTensor(tensorShape);
  auto input3 = createEmptyTensor(tensorShape);
  auto outputType = createRankedTensorType(tensorShape);
  auto where =
      builder.create<WhereOp>(builder.getUnknownLoc(), outputType,
                              mlir::ValueRange{input1, input2, input3});

  // test WhereOp interface
  auto constraintsExp = getOpConstraints(where.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 16384);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }

  auto runtimeExp = getOpRuntime(where.getOperation());
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    FAIL() << llvm::toString(runtimeExp.takeError());
  }
}

TEST_F(OpModelBase, EmptyOpInterface) {
  // create EmptyOp with full TTNN layout attributes
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};

  // Create a proper TTNN layout for the tensor
  auto layout = CreateTiledLayout(tensorShape, BufferType::L1,
                                  TensorMemoryLayout::Interleaved);
  RankedTensorType inputType =
      createRankedTensorType(tensorShape, builder.getBF16Type(), layout);

  // Cast to get TTNN layout attributes
  RankedTensorType inputTensorType = mlir::cast<RankedTensorType>(inputType);
  ttnn::TTNNLayoutAttr ttnnLayoutAttr =
      mlir::cast<ttnn::TTNNLayoutAttr>(inputTensorType.getEncoding());

  // Create memory config attribute
  ttnn::MemoryConfigAttr memoryConfigAttr = ttnn::MemoryConfigAttr::get(
      &context, ttnnLayoutAttr.getMemLayout(),
      ttnn::BufferTypeAttr::get(&context, ttnnLayoutAttr.getBufferType()),
      std::nullopt); // No sharding for this test

  // Create a device value (required for EmptyOp)
  auto device = builder.create<ttnn::GetDeviceOp>(
      builder.getUnknownLoc(), builder.getType<ttnn::DeviceType>(),
      ttnn::MeshShapeAttr::get(&context, 1, 1),
      ttnn::MeshOffsetAttr::get(&context, 0, 0));

  // Create the EmptyOp with all required parameters
  auto empty = builder.create<ttnn::EmptyOp>(
      builder.getUnknownLoc(), inputType,
      ttnn::ShapeAttr::get(&context, inputTensorType.getShape()),
      ttcore::DataTypeAttr::get(&context, ttnnLayoutAttr.getDataType()),
      ttnn::LayoutAttr::get(&context, ttnnLayoutAttr.getLayout()), device,
      memoryConfigAttr);

  // test EmptyOp interface
  auto constraintsExp = getOpConstraints(empty.getOperation());
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 0);
    EXPECT_EQ(peakSize, 2048);
    EXPECT_EQ(outputSize, 2048);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
}

TEST_F(OpModelBase, ArangeOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 5};
  auto layout = CreateRowMajorLayout(tensorShape, ttnn::BufferType::DRAM,
                                     ttnn::TensorMemoryLayout::Interleaved);
  auto resultType =
      createRankedTensorType(tensorShape, builder.getBF16Type(), layout);

  // Create ArangeOp with IntegerAttr parameters
  auto startAttr = builder.getI64IntegerAttr(0);
  auto endAttr = builder.getI64IntegerAttr(10);
  auto stepAttr = builder.getI64IntegerAttr(2);

  auto arange =
      builder.create<ArangeOp>(builder.getUnknownLoc(), resultType, startAttr,
                               endAttr, stepAttr, /*dtype=*/nullptr,
                               /*device=*/nullptr, /*memoryConfig=*/nullptr);

  // test ArangeOp interface
  auto backend = dyn_cast<OpModel>(arange.getOperation());
  auto constraintsExp =
      backend.getOpConstraints(getInputLayouts(arange), OpConfig(nullptr));
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 0);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExp.takeError()) << std::endl;
  }
}

struct NamedFullOpTestParams {
  std::string testName;
  std::function<Operation *(OpBuilder &, Location, Type, ttnn::ShapeAttr)>
      createOp;
  ExpectedResult expectedResult;
};

class NamedFullOpModelTest
    : public OpModelBase,
      public ::testing::WithParamInterface<NamedFullOpTestParams> {
protected:
  void SetUp() override {
    OpModelBase::SetUp();
    params = GetParam();
  }

  NamedFullOpTestParams params;
};

// Test case for namedFull operations
TEST_P(NamedFullOpModelTest, TestOpInterface) {
  llvm::SmallVector<int64_t> tensorShapeO = {workerCoresN300, 2048};
  auto layout = CreateTiledLayout(tensorShapeO, BufferType::L1,
                                  TensorMemoryLayout::BlockSharded);
  auto outputType =
      createRankedTensorType(tensorShapeO, builder.getBF16Type(), layout);
  Operation *op = params.createOp(builder, builder.getUnknownLoc(), outputType,
                                  ttnn::ShapeAttr::get(&context, tensorShapeO));
  auto backend = dyn_cast<OpModel>(op);
  auto constraintsExp =
      backend.getOpConstraints(getInputLayouts(op), OpConfig(nullptr));
  if (constraintsExp) {
    auto l1 = constraintsExp.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, params.expectedResult.expectedCbSize);
    EXPECT_EQ(peakSize, params.expectedResult.expectedPeakSize);
    EXPECT_EQ(outputSize, params.expectedResult.expectedOutputSize);
  } else {
    FAIL() << "Missing L1 constraints for " << params.testName
           << "; Error=" << llvm::toString(constraintsExp.takeError());
  }
}

const auto createZeros = [](OpBuilder &b, Location loc, Type type,
                            ttnn::ShapeAttr shape) {
  return b
      .create<ZerosOp>(loc, type, shape, /*dtype=*/nullptr, /*layout=*/nullptr,
                       /*device=*/nullptr, /*memoryConfig=*/nullptr)
      .getOperation();
};
const auto createOnes = [](OpBuilder &b, Location loc, Type type,
                           ttnn::ShapeAttr shape) {
  return b
      .create<OnesOp>(loc, type, shape, /*dtype=*/nullptr, /*layout=*/nullptr,
                      /*device=*/nullptr, /*memoryConfig=*/nullptr)
      .getOperation();
};

const ExpectedResult namedFullExpected{true, 0, 0, 0};

const std::vector<NamedFullOpTestParams> namedFullOpTestParams = {
    {"Zeros", createZeros, namedFullExpected},
    {"Ones", createOnes, namedFullExpected}};

INSTANTIATE_TEST_SUITE_P(
    NamedFullOpModelTests, NamedFullOpModelTest,
    ::testing::ValuesIn(namedFullOpTestParams),
    [](const testing::TestParamInfo<NamedFullOpTestParams> &info) {
      return info.param.testName;
    });

TEST_F(OpModelBase, FullOpInterface) {
  llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  auto layout = CreateTiledLayout(tensorShape, BufferType::L1,
                                  TensorMemoryLayout::Interleaved);
  auto outputType =
      createRankedTensorType(tensorShape, builder.getBF16Type(), layout);
  auto fullInt =
      builder.create<FullOp>(builder.getUnknownLoc(), outputType,
                             ttnn::ShapeAttr::get(&context, tensorShape),
                             builder.getI32IntegerAttr(42),
                             /*dtype=*/nullptr, /*layout=*/nullptr,
                             /*device=*/nullptr, /*memoryConfig=*/nullptr);

  // test FullOp interface with int fill value:
  auto backendI = dyn_cast<OpModel>(fullInt.getOperation());
  auto constraintsExpI =
      backendI.getOpConstraints(getInputLayouts(fullInt), OpConfig(nullptr));
  if (constraintsExpI) {
    auto l1 = constraintsExpI.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 0);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExpI.takeError()) << std::endl;
  }

  // test FullOp interface with float fill value:
  auto fullF = builder.create<FullOp>(
      builder.getUnknownLoc(), outputType,
      ttnn::ShapeAttr::get(&context, tensorShape), builder.getF32FloatAttr(0.5),
      /*dtype=*/nullptr, /*layout=*/nullptr,
      /*device=*/nullptr, /*memoryConfig=*/nullptr);
  auto backendF = dyn_cast<OpModel>(fullF.getOperation());
  auto constraintsExpF =
      backendF.getOpConstraints(getInputLayouts(fullF), OpConfig(nullptr));
  if (constraintsExpF) {
    auto l1 = constraintsExpF.get();
    const auto [cbSize, peakSize, outputSize, outputLayout] = l1;
    EXPECT_EQ(cbSize, 0);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    FAIL() << "Missing L1 constraints; Error="
           << llvm::toString(constraintsExpF.takeError()) << std::endl;
  }
}
} // namespace mlir::tt::ttnn
