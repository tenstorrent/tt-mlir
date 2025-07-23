// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <tuple>

namespace mlir::tt::op_model::ttnn {

class OpModelTest : public OpModelFixture {};

namespace detail {
namespace {
struct TestTensor {
  llvm::SmallVector<int64_t> shape;
  mlir::tt::ttnn::TensorMemoryLayout layout;
  mlir::tt::ttnn::BufferType bufferType;
  std::optional<llvm::SmallVector<int64_t>> virtualGrid = std::nullopt;
};

struct ExpectedResult {
  bool expectedLegal = false;
  size_t expectedCbSize = 0;
  size_t expectedPeakSize = 0;
  size_t expectedOutputSize = 0;
};
} // namespace

const TestTensor interleavedN300X1024Dram = {
    {OpModelFixture::workerCoresN300, 1024},
    mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
    mlir::tt::ttnn::BufferType::DRAM};
const TestTensor interleavedN300X1024L1 = {
    {OpModelFixture::workerCoresN300, 1024},
    mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
    mlir::tt::ttnn::BufferType::L1};

const TestTensor interleaved2048X2048Dram = {
    {2048, 2048},
    mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
    mlir::tt::ttnn::BufferType::DRAM,
    llvm::SmallVector<int64_t>{8, 8}};

const TestTensor inerleaved2048X2048L1 = {
    {2048, 2048},
    mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
    mlir::tt::ttnn::BufferType::L1,
    llvm::SmallVector<int64_t>{8, 8}};
} // namespace detail

// ==== Unary Eltwise Ops Starts ====
enum class UnaryEltwiseOpType {
  Relu,
  Sqrt,
  Sigmoid,
  Sin,
  Cos,
  Exp,
  Tanh,
  Log,
  Reciprocal
};

class OpModelUnaryEltwiseParam : public OpModelTest,
                                 public testing::WithParamInterface<
                                     std::tuple<UnaryEltwiseOpType,
                                                detail::TestTensor, // input
                                                detail::TestTensor, // output
                                                detail::ExpectedResult>> {
protected:
  std::map<UnaryEltwiseOpType,
           std::function<llvm::Expected<size_t>(
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>>
      runtimeMap = {
          {UnaryEltwiseOpType::Relu, ReluOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Sqrt, SqrtOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Sigmoid, SigmoidOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Sin, SinOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Cos, CosOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Exp, ExpOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Tanh, TanhOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Log, LogOpInterface::getOpRuntime},
          {UnaryEltwiseOpType::Reciprocal, ReciprocalOpInterface::getOpRuntime},
      };
  std::map<UnaryEltwiseOpType,
           std::function<llvm::Expected<op_model::ttnn::OpConstraints>(
               ttcore::GridAttr, llvm::ArrayRef<int64_t>,
               mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
               mlir::tt::ttnn::TTNNLayoutAttr)>>
      constraintsMap = {
          {UnaryEltwiseOpType::Relu, ReluOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Sqrt, SqrtOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Sigmoid, SigmoidOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Sin, SinOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Cos, CosOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Exp, ExpOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Tanh, TanhOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Log, LogOpInterface::getOpConstraints},
          {UnaryEltwiseOpType::Reciprocal,
           ReciprocalOpInterface::getOpConstraints},
      };
  void RunTest() {
    auto params = GetParam();
    const auto opType = std::get<0>(params);
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<1>(params);
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<2>(params);
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = std::get<3>(params);

    const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = constraintsMap.at(opType)(
        CreateWorkerGrid(), inputShape, inputLayout, outputShape, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();
      EXPECT_EQ(cbSize, expectedCbSize);
      EXPECT_EQ(peakSize, expectedPeakSize);
      EXPECT_EQ(outputSize, expectedOutputSize);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBack);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = runtimeMap.at(opType)(inputShape, inputLayout,
                                            outputShape, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelUnaryEltwiseParam, UnaryOp) { RunTest(); }

const std::initializer_list<
    std::tuple<detail::TestTensor, detail::TestTensor, detail::ExpectedResult>>
    unaryEltwiseParams = {
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 8192, 0, 0}),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 8192, 2048, 2048}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 8192, 0, 0}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 8192, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{
                {14 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{
                {14 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 0, 14 * 32 * 32 * 2,
                                   14 * 32 * 32 * 2}),
        std::make_tuple(
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{
                {14 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{
                {14 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{false})};

::testing::internal::ParamGenerator<
    std::tuple<UnaryEltwiseOpType, detail::TestTensor, detail::TestTensor,
               detail::ExpectedResult>>
generateBinaryEltwiseParams(
    UnaryEltwiseOpType opType,
    std::initializer_list<std::tuple<detail::TestTensor, detail::TestTensor,
                                     detail::ExpectedResult>>
        values) {
  std::vector<std::tuple<UnaryEltwiseOpType, detail::TestTensor,
                         detail::TestTensor, detail::ExpectedResult>>
      newValues;
  for (const auto &v : values) {
    newValues.emplace_back(std::make_tuple(opType, std::get<0>(v),
                                           std::get<1>(v), std::get<2>(v)));
  }
  return ::testing::ValuesIn(newValues);
}

INSTANTIATE_TEST_SUITE_P(ReluTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Relu,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SqrtTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Sqrt,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SigmoidTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(
                             UnaryEltwiseOpType::Sigmoid, unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SinTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Sin,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(CosTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Cos,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ExpTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Exp,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(TanhTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Tanh,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogTests, OpModelUnaryEltwiseParam,
                         generateBinaryEltwiseParams(UnaryEltwiseOpType::Log,
                                                     unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    ReciprocalTests, OpModelUnaryEltwiseParam,
    generateBinaryEltwiseParams(UnaryEltwiseOpType::Reciprocal,
                                unaryEltwiseParams));

// ==== Unary Eltwise Ops Ends ====

enum class ReductionOpType { Sum, Mean };

class OpModelReductionParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<ReductionOpType,    // operation type
                     detail::TestTensor, // input
                     detail::TestTensor, // output
                     std::optional<llvm::SmallVector<int64_t>>, // dim arg
                     bool,                                      // keep dim
                     detail::ExpectedResult>> {
protected:
  using ReductionOpConstraintsFunc =
      std::function<llvm::Expected<OpConstraints>(
          ttcore::GridAttr, llvm::ArrayRef<int64_t>,
          mlir::tt::ttnn::TTNNLayoutAttr,
          std::optional<llvm::ArrayRef<int64_t>>, bool,
          mlir::tt::ttnn::TTNNLayoutAttr)>;

  using ReductionOpRuntimeFunc = std::function<llvm::Expected<size_t>(
      llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
      std::optional<llvm::ArrayRef<int64_t>>, bool,
      mlir::tt::ttnn::TTNNLayoutAttr)>;

  std::map<ReductionOpType,
           std::pair<ReductionOpConstraintsFunc, ReductionOpRuntimeFunc>>
      opMap = {
          {ReductionOpType::Sum,
           {SumOpInterface::getOpConstraints, SumOpInterface::getOpRuntime}},
          {ReductionOpType::Mean,
           {MeanOpInterface::getOpConstraints, MeanOpInterface::getOpRuntime}}};
};

TEST_P(OpModelReductionParam, Reduction) {
  auto params = GetParam();
  const ReductionOpType opType = std::get<0>(params);
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<1>(params);

  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<2>(params);
  const auto dimArg = std::get<3>(params);
  const auto keepDim = std::get<4>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<5>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();
  auto constraintsExp = OpModel<OpTy>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, dimArg, keepDim,
      outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp =
      runtimeFunc(inputShape, inputLayout, dimArg, keepDim, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

// Helper function to generate test parameters for reduction operations
template <ReductionOpType OpType>
static auto generateReductionParams() {
  return ::testing::Values(
      std::make_tuple(OpType, detail::interleavedN300X1024Dram,
                      detail::interleavedN300X1024Dram,
                      llvm::SmallVector<int64_t>{1}, true,
                      detail::ExpectedResult{true, 12288, 0, 0}),
      std::make_tuple(OpType, detail::interleavedN300X1024Dram,
                      detail::interleavedN300X1024Dram,
                      llvm::SmallVector<int64_t>{1, 2}, false,
                      detail::ExpectedResult{false, 0, 0, 0}),
      std::make_tuple(OpType, detail::interleavedN300X1024Dram,
                      detail::interleavedN300X1024Dram,
                      llvm::SmallVector<int64_t>{1, 0}, false,
                      detail::ExpectedResult{true, 12288, 0, 0}),
      std::make_tuple(OpType, detail::interleavedN300X1024L1,
                      detail::interleavedN300X1024Dram,
                      llvm::SmallVector<int64_t>{1}, false,
                      detail::ExpectedResult{true, 12288, 0, 0}));
}

INSTANTIATE_TEST_SUITE_P(SumTests, OpModelReductionParam,
                         generateReductionParams<ReductionOpType::Sum>());

INSTANTIATE_TEST_SUITE_P(MeanTests, OpModelReductionParam,
                         generateReductionParams<ReductionOpType::Mean>());

TEST_F(OpModelTest, SoftmaxInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  std::vector<std::tuple<mlir::tt::ttnn::TTNNLayoutAttr,
                         mlir::tt::ttnn::TTNNLayoutAttr>>
      layout_combinations = {{inputLayout_dram, inputLayout_dram},
                             {inputLayout_l1, inputLayout_dram},
                             {inputLayout_dram, inputLayout_l1},
                             {inputLayout_l1, inputLayout_l1}};
  for (const auto &[input_layout, output_layout] : layout_combinations) {
    auto runtimeExp = SoftmaxOpInterface::getOpRuntime(
        tensorShape, input_layout, -1, tensorShape, output_layout);
    EXPECT_TRUE(static_cast<bool>(runtimeExp));
    EXPECT_TRUE(runtimeExp.get() > 0);
  }
}

TEST_F(OpModelTest, Reshape) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAM =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutL1 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = ReshapeOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, {workerCoresN300 * 4, 256},
      layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 5120);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  auto runtimeExp = ReshapeOpInterface::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  constraintsExp = ReshapeOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, {workerCoresN300 * 4, 256},
      layoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 5120);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  runtimeExp = ReshapeOpInterface::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutL1);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();
}

TEST_F(OpModelTest, Slice) {
  const llvm::SmallVector<int64_t> inputTensorShape = {1, 56, 56, 96};
  const llvm::SmallVector<int64_t> outputTensorShape = {1, 28, 56, 95};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAM =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  llvm::SmallVector<int64_t> begins = {0, 0, 0, 0};
  llvm::SmallVector<int64_t> ends = {1, 56, 56, 95};
  llvm::SmallVector<int64_t> step = {1, 2, 1, 1};

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = SliceOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, layoutDRAM, begins, ends, step,
      outputTensorShape, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp =
      SliceOpInterface::getOpRuntime(inputTensorShape, layoutDRAM, begins, ends,
                                     step, outputTensorShape, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, ToLayout) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAMTiled =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAMRowMajor =
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutL1RowMajorHS =
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                           mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = ToLayoutOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 131072);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayout);

  auto runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = ToLayoutOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutL1RowMajorHS);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutL1RowMajorHS);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());

  constraintsExp = ToLayoutOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 131072);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayout);

  runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, ToMemoryConfig) {
  const llvm::SmallVector<int64_t> tensorShape = {1, 8, 64, 128};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutL1Tiled =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayoutDRAMTiled =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::MemoryConfigAttr memoryConfig =
      mlir::tt::ttnn::MemoryConfigAttr::get(
          &context, outputLayoutDRAMTiled.getMemLayout(),
          mlir::tt::ttnn::BufferTypeAttr::get(
              &context, outputLayoutDRAMTiled.getBufferType()),
          std::nullopt /*shardSpec*/);
  auto constraintsExp = ToMemoryConfigOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutL1Tiled, memoryConfig,
      tensorShape, outputLayoutDRAMTiled);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = ToMemoryConfigOpInterface::getOpRuntime(
      tensorShape, inputLayoutL1Tiled, memoryConfig, tensorShape,
      outputLayoutDRAMTiled);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  auto coreRangeSetAttr = ::mlir::tt::ttnn::CoreRangeSetAttr::get(
      &context,
      ::llvm::ArrayRef<mlir::tt::ttnn::CoreRangeAttr>{
          ::mlir::tt::ttnn::CoreRangeAttr::get(
              &context, ::mlir::tt::ttnn::CoreCoordAttr::get(&context, 0, 0),
              ::mlir::tt::ttnn::CoreCoordAttr::get(&context, 7, 0))});
  mlir::tt::ttnn::ShardSpecAttr shardSpec = mlir::tt::ttnn::ShardSpecAttr::get(
      &context, coreRangeSetAttr,
      ::mlir::tt::ttnn::ShapeAttr::get(&context, {64, 128}),
      ::mlir::tt::ttnn::ShardOrientationAttr::get(
          &context, ::mlir::tt::ttnn::ShardOrientation::RowMajor),
      ::mlir::tt::ttnn::ShardModeAttr::get(
          &context, ::mlir::tt::ttnn::ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayoutL1Tiled =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  memoryConfig = mlir::tt::ttnn::MemoryConfigAttr::get(
      &context, outputLayoutL1Tiled.getMemLayout(),
      mlir::tt::ttnn::BufferTypeAttr::get(&context,
                                          outputLayoutL1Tiled.getBufferType()),
      shardSpec);
  constraintsExp = ToMemoryConfigOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutL1Tiled, memoryConfig,
      tensorShape, outputLayoutL1Tiled);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 16384);
  EXPECT_EQ(opCstr.outputL1BufferSize, 16384);

  runtimeExp = ToMemoryConfigOpInterface::getOpRuntime(
      tensorShape, inputLayoutL1Tiled, memoryConfig, tensorShape,
      outputLayoutL1Tiled);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Concat) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAM =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutL1Interleaved =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  auto constraintsExp = ConcatOpInterface::getOpConstraints(
      CreateWorkerGrid(), {inputTensorShape, inputTensorShape},
      {layoutL1Interleaved, layoutL1Interleaved}, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 4096);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = ConcatOpInterface::getOpRuntime(
      {inputTensorShape, inputTensorShape},
      {layoutL1Interleaved, layoutL1Interleaved}, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Transpose) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutDRAM =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutL1Interleaved =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr layoutL1WSharded =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = TransposeOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = TransposeOpInterface::getOpRuntime(tensorShape, layoutDRAM,
                                                       0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TransposeOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, 1, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  runtimeExp = TransposeOpInterface::getOpRuntime(tensorShape, layoutDRAM, 0, 1,
                                                  layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TransposeOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 0, 1,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = TransposeOpInterface::getOpRuntime(
      tensorShape, layoutL1Interleaved, 0, 1, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

TEST_F(OpModelTest, SoftmaxSharded) {
  const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_hs, -2, tensorShape,
      inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_hs, -2, tensorShape,
      inputLayout_l1_i);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_i, -2, tensorShape,
      inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  auto runtimeExp = SoftmaxOpInterface::getOpRuntime(
      tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Typecast) {
  const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutDRAMIBF16 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutL1HSBF16 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutDRAMIF32 = CreateTiledLayout(
      tensorShape, mlir::tt::ttnn::BufferType::DRAM,
      mlir::tt::ttnn::TensorMemoryLayout::Interleaved, std::nullopt,
      GetPhysicalGridSize(), builder.getF32Type());
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = TypecastOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      tensorShape, inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 12288);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = TypecastOpInterface::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      tensorShape, inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TypecastOpInterface::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      tensorShape, inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());
  runtimeExp = TypecastOpInterface::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      tensorShape, inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

// ==== Binary Eltwise Ops Starts ====
enum class BinaryEltwiseOpType {
  Add,
  Mul,
  Subtract,
  Maximum,
  Minimum,
  Divide,
  Equal,
  NotEqual,
  GreaterEqual,
  GreaterThan,
  LessEqual,
  LessThan,
  LogicalAnd,
  LogicalOr,
  LogicalXor,
  // Use the following value when the op is not yet known:
  Unknown
};

struct BinaryEltwiseParam {
  BinaryEltwiseOpType opType;
  detail::TestTensor inputA;
  detail::TestTensor inputB;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};

class OpModelBinaryEltwiseParam
    : public OpModelTest,
      public testing::WithParamInterface<BinaryEltwiseParam> {

protected:
  std::map<BinaryEltwiseOpType,
           std::function<llvm::Expected<size_t>(
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>>
      // clang-format off
      runtimeMap = {
          {BinaryEltwiseOpType::Add,            AddOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Mul,            MultiplyOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Subtract,       SubtractOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Maximum,        MaximumOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Minimum,        MinimumOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Divide,         DivideOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::Equal,          EqualOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::NotEqual,       NotEqualOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::GreaterEqual,   GreaterEqualOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::GreaterThan,    GreaterThanOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::LessEqual,      LessEqualOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::LessThan,       LessThanOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::LogicalAnd,     LogicalAndOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::LogicalOr,      LogicalOrOpInterface::getOpRuntime},
          {BinaryEltwiseOpType::LogicalXor,     LogicalXorOpInterface::getOpRuntime}};
  std::map<BinaryEltwiseOpType,
           std::function<llvm::Expected<OpConstraints>(
               ttcore::GridAttr, llvm::ArrayRef<int64_t>,
               mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
               mlir::tt::ttnn::TTNNLayoutAttr, llvm::ArrayRef<int64_t>,
               mlir::tt::ttnn::TTNNLayoutAttr)>>
      constraintsMap = {
          {BinaryEltwiseOpType::Add,            AddOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Mul,            MultiplyOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Subtract,       SubtractOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Maximum,        MaximumOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Minimum,        MinimumOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Divide,         DivideOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::Equal,          EqualOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::NotEqual,       NotEqualOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::GreaterEqual,   GreaterEqualOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::GreaterThan,    GreaterThanOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::LessEqual,      LessEqualOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::LessThan,       LessThanOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::LogicalAnd,     LogicalAndOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::LogicalOr,      LogicalOrOpInterface::getOpConstraints},
          {BinaryEltwiseOpType::LogicalXor,     LogicalXorOpInterface::getOpConstraints}};
  // clang-format on
  void RunTest() {
    const auto opType = GetParam().opType;
    const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
                inputVirtualGridA] = GetParam().inputA;
    const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
                inputVirtualGridB] = GetParam().inputB;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = GetParam().expectedResult;

    const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
        inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
    const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
        inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
    const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = constraintsMap[opType](
        CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB,
        inputLayoutB, outputShape, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();
      EXPECT_EQ(cbSize, expectedCbSize);
      EXPECT_EQ(peakSize, expectedPeakSize);
      EXPECT_EQ(outputSize, expectedOutputSize);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBack);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    llvm::Expected<size_t> runtimeExp =
        runtimeMap[opType](inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
                           outputShape, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelBinaryEltwiseParam, BinaryOp) { RunTest(); }

const std::initializer_list<BinaryEltwiseParam> binaryEltwiseParams = {
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024Dram,
     detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
     detail::ExpectedResult{false, 0, 0, 0}}, // incompatible dimensions at
                                              // the input
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {BinaryEltwiseOpType::Unknown, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {BinaryEltwiseOpType::Unknown,
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        mlir::tt::ttnn::BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        mlir::tt::ttnn::BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        mlir::tt::ttnn::BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::ExpectedResult{true, 4096, 262144, 262144}},
    {BinaryEltwiseOpType::Unknown,
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        mlir::tt::ttnn::BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        mlir::tt::ttnn::BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        mlir::tt::ttnn::BufferType::DRAM},
     detail::ExpectedResult{true, 8192, 0, 0}},
    {BinaryEltwiseOpType::Unknown,
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        mlir::tt::ttnn::BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        mlir::tt::ttnn::BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        mlir::tt::ttnn::BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::ExpectedResult{true, 8192, 262144, 262144}}};

::testing::internal::ParamGenerator<BinaryEltwiseParam>
generateBinaryEltwiseParams(BinaryEltwiseOpType opType,
                            std::initializer_list<BinaryEltwiseParam> values,
                            std::size_t extraCbRequirement = 0) {
  // The expected size of the circular buffer is the same for most binary ops,
  // but some of them (such as Divide, LogicalOr and LogicalXor) extra memory is
  // required due to the op's implementation.
  std::vector<BinaryEltwiseParam> newValues;
  for (const auto &v : values) {
    newValues.emplace_back(v);
    // Update the op type from Unknown to the actual op type:
    newValues.back().opType = opType;
    if (extraCbRequirement > 0) {
      newValues.back().expectedResult.expectedCbSize += extraCbRequirement;
    }
  }
  return ::testing::ValuesIn(newValues);
}

INSTANTIATE_TEST_SUITE_P(AddTests, OpModelBinaryEltwiseParam,
                         generateBinaryEltwiseParams(BinaryEltwiseOpType::Add,
                                                     binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MulTests, OpModelBinaryEltwiseParam,
                         generateBinaryEltwiseParams(BinaryEltwiseOpType::Mul,
                                                     binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    SubtractTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::Subtract,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    MaximumTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::Maximum,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    MinimumTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::Minimum,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    DivideTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::Divide,
                                binaryEltwiseParams, 2048));

INSTANTIATE_TEST_SUITE_P(EqualTests, OpModelBinaryEltwiseParam,
                         generateBinaryEltwiseParams(BinaryEltwiseOpType::Equal,
                                                     binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    NotEqualTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::NotEqual,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    GreaterEqualTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::GreaterEqual,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    GreaterThanTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::GreaterThan,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    LessEqualTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::LessEqual,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    LessThanTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::LessThan,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    LogicalAndTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::LogicalAnd,
                                binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    LogicalOrTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::LogicalOr,
                                binaryEltwiseParams, 4096));

INSTANTIATE_TEST_SUITE_P(
    LogicalXorTests, OpModelBinaryEltwiseParam,
    generateBinaryEltwiseParams(BinaryEltwiseOpType::LogicalXor,
                                binaryEltwiseParams, 4096));

// ==== Binary Eltwise Ops Ends ====

class OpModelLinearParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // inputA
                     detail::TestTensor,         // inputB
                     detail::TestTensor,         // bias
                     detail::TestTensor,         // output,
                     llvm::SmallVector<int64_t>, // physical grid
                     detail::ExpectedResult>> {};

TEST_P(OpModelLinearParam, LinearParam) {
  auto params = GetParam();
  const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
              inputVirtualGridA] = std::get<0>(params);
  const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
              inputVirtualGridB] = std::get<1>(params);
  const auto [biasShape, biasTensorLayout, biasBufferType, biasVirtualGrid] =
      std::get<2>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<3>(params);
  llvm::SmallVector<int64_t> physicalGrid = std::get<4>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<5>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const mlir::tt::ttnn::TTNNLayoutAttr biasLayout = CreateTiledLayout(
      biasShape, biasBufferType, biasTensorLayout, biasVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = LinearOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
      biasShape, biasLayout, outputShape, outputLayout, false, false);

  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = LinearOpInterface::getOpRuntime(
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, biasShape,
      biasLayout, outputShape, outputLayout, false, false);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
  mlir::tt::op_model::ttnn::SingletonDeviceContext::resetInstance();
}

INSTANTIATE_TEST_SUITE_P(
    LinearInterleavedTests, OpModelLinearParam,
    ::testing::Values(
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 655360, 0, 0}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 262144, 131072}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 262144, 0, 0}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 262144, 262144, 131072}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 262144, 0, 0}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 262144, 262144, 131072}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 262144, 131072})));

INSTANTIATE_TEST_SUITE_P(
    LinearShardedTests, OpModelLinearParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 430144, 229376, 114688}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 544832, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{
                llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 8256, 4096, 2048}),
        std::make_tuple(
            detail::TestTensor{
                {56 * 32, 1 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{
                llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 114688, 229376, 114688})));

class OpModelMatmulParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // inputA
                     detail::TestTensor,         // inputB
                     detail::TestTensor,         // output,
                     llvm::SmallVector<int64_t>, // physical grid
                     detail::ExpectedResult>> {};

TEST_P(OpModelMatmulParam, MatmulParam) {
  auto params = GetParam();
  const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
              inputVirtualGridA] = std::get<0>(params);
  const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
              inputVirtualGridB] = std::get<1>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<2>(params);
  llvm::SmallVector<int64_t> physicalGrid = std::get<3>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<4>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = MatmulOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
      outputShape, outputLayout, false, false);

  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = MatmulOpInterface::getOpRuntime(
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, outputShape,
      outputLayout, false, false);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterleavedTests, OpModelMatmulParam,
    ::testing::Values(
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 655360, 0, 0}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::interleaved2048X2048Dram,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        detail::inerleaved2048X2048L1,
                        llvm::SmallVector<int64_t>{8, 8},
                        detail::ExpectedResult{true, 786432, 131072, 131072})));

INSTANTIATE_TEST_SUITE_P(
    MatmulShardedTests, OpModelMatmulParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 430144, 114688, 114688}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 262144, 401408,
                                   401408}), // matmul bug shards to less cores
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 544832, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{
                llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 8256, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{
                {56 * 32, 1 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{
                llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{56, 1}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 114688, 114688, 114688})));

class OpModelConv2dParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // input
                     detail::TestTensor,         // weight
                     detail::TestTensor,         // output
                     uint32_t,                   // in_channels
                     uint32_t,                   // out_channels
                     uint32_t,                   // batch_size
                     uint32_t,                   // input_height
                     uint32_t,                   // input_width
                     llvm::SmallVector<int32_t>, // kernel_size
                     llvm::SmallVector<int32_t>, // stride
                     llvm::SmallVector<int32_t>, // padding
                     llvm::SmallVector<int32_t>, // dilation
                     uint32_t,                   // groups
                     detail::ExpectedResult>> {};

TEST_P(OpModelConv2dParam, Conv2d) {
  // Skipped due to hang. See https://github.com/tenstorrent/tt-mlir/issues/3901
  GTEST_SKIP();
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [weightShape, weightTensorLayout, weightBufferType,
              weightVirtualGrid] = std::get<1>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<2>(params);
  const auto in_channels = std::get<3>(params);
  const auto out_channels = std::get<4>(params);
  const auto batch_size = std::get<5>(params);
  const auto input_height = std::get<6>(params);
  const auto input_width = std::get<7>(params);
  const auto kernel_size = std::get<8>(params);
  const auto stride = std::get<9>(params);
  const auto padding = std::get<10>(params);
  const auto dilation = std::get<11>(params);
  const auto groups = std::get<12>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<13>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const mlir::tt::ttnn::TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  // This is not configurable, as the backend doesn't support it for now.
  // But this test shows that this information is parsed and passes to the
  // backend correctly.
  ::mlir::tt::ttnn::DeviceComputeKernelConfigAttr deviceConfig =
      ::mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(
          &context, /*mathFidelity=*/::mlir::tt::ttnn::MathFidelity::LoFi,
          /*mathApproxMode=*/::mlir::BoolAttr::get(&context, true),
          /*fp32DestAccEn=*/::mlir::BoolAttr::get(&context, true),
          /*packerL1Acc=*/::mlir::BoolAttr::get(&context, true),
          /*dstFullSyncEn=*/::mlir::BoolAttr::get(&context, true));

  auto constraintsExp = Conv2dOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, dilation, groups,
      std::nullopt, deviceConfig, outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  auto runtimeExp = Conv2dOpInterface::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, dilation, groups, std::nullopt,
      deviceConfig, outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    const auto runtime = runtimeExp.get();
    EXPECT_GT(runtime, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    Conv2dTests, OpModelConv2dParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{1, 1, 50176, 3},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{64, 3, 7, 7},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::SystemMemory},
            detail::TestTensor{{1, 1, 12544, 64},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
            llvm::SmallVector<int32_t>{2, 2}, llvm::SmallVector<int32_t>{3, 3},
            llvm::SmallVector<int32_t>{1, 1}, 1,
            detail::ExpectedResult{true, 229440, 190568, 0}),
        std::make_tuple(
            detail::TestTensor{{1, 1, 50176, 3},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{64, 3, 9, 7},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::SystemMemory},
            detail::TestTensor{{1, 1, 12544, 64},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
            llvm::SmallVector<int32_t>{2, 2}, llvm::SmallVector<int32_t>{3, 3},
            llvm::SmallVector<int32_t>{1, 1}, 1,
            detail::ExpectedResult{false, 0, 0, 0})));

class OpModelConvTranspose2dParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // input
                     detail::TestTensor,         // weight
                     detail::TestTensor,         // output
                     uint32_t,                   // in_channels
                     uint32_t,                   // out_channels
                     uint32_t,                   // batch_size
                     uint32_t,                   // input_height
                     uint32_t,                   // input_width
                     llvm::SmallVector<int32_t>, // kernel_size
                     llvm::SmallVector<int32_t>, // stride
                     llvm::SmallVector<int32_t>, // padding
                     llvm::SmallVector<int32_t>, // output_padding
                     llvm::SmallVector<int32_t>, // dilation
                     uint32_t,                   // groups
                     detail::ExpectedResult>> {};

TEST_P(OpModelConvTranspose2dParam, ConvTranspose2d) {
  // Skipped due to hang. See https://github.com/tenstorrent/tt-mlir/issues/3970
  GTEST_SKIP();

  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [weightShape, weightTensorLayout, weightBufferType,
              weightVirtualGrid] = std::get<1>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<2>(params);
  const auto in_channels = std::get<3>(params);
  const auto out_channels = std::get<4>(params);
  const auto batch_size = std::get<5>(params);
  const auto input_height = std::get<6>(params);
  const auto input_width = std::get<7>(params);
  const auto kernel_size = std::get<8>(params);
  const auto stride = std::get<9>(params);
  const auto padding = std::get<10>(params);
  const auto output_padding = std::get<11>(params);
  const auto dilation = std::get<12>(params);
  const auto groups = std::get<13>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<14>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout =
      CreateRowMajorLayout(inputShape, inputBufferType, inputTensorLayout,
                           inputVirtualGrid, GetPhysicalGridSize());
  const mlir::tt::ttnn::TTNNLayoutAttr weightLayout =
      CreateRowMajorLayout(weightShape, weightBufferType, weightTensorLayout,
                           weightVirtualGrid, GetPhysicalGridSize());
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  auto constraintsExp = ConvTranspose2dOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, output_padding,
      dilation, groups, std::nullopt, outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  auto runtimeExp = ConvTranspose2dOpInterface::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, output_padding, dilation,
      groups, std::nullopt, outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    const auto runtime = runtimeExp.get();
    EXPECT_GT(runtime, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ConvTranspose2dTests, OpModelConvTranspose2dParam,
    ::testing::Values(std::make_tuple(
        detail::TestTensor{{1, 1, 50176, 3},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::DRAM},
        detail::TestTensor{{3, 64, 7, 7},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::SystemMemory},
        detail::TestTensor{{1, 1, 12544, 64},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::DRAM},
        3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
        llvm::SmallVector<int32_t>{2, 2}, llvm::SmallVector<int32_t>{3, 3},
        llvm::SmallVector<int32_t>{0, 0}, llvm::SmallVector<int32_t>{1, 1}, 1,
        detail::ExpectedResult{true, 0, 0, 0})));

class OpModelMaxPool2DParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // input
                     detail::TestTensor,         // output
                     int32_t,                    // batch_size
                     int32_t,                    // input_height
                     int32_t,                    // input_width
                     int32_t,                    // input_channels
                     llvm::SmallVector<int32_t>, // kernel_size
                     llvm::SmallVector<int32_t>, // stride
                     llvm::SmallVector<int32_t>, // padding
                     llvm::SmallVector<int32_t>, // dilation
                     bool,                       // ceil_mode
                     bool                        // expected legal
                     >> {};

TEST_P(OpModelMaxPool2DParam, MaxPool2DParam) {
  // TODO(2976): Some of these test cases return L1 interleaved row major
  // tensors which triggers an assertion in TTNNLayoutAttr. Will be reenabled
  // when the linked issue is fixed
  GTEST_SKIP();
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto batchSize = std::get<2>(params);
  const auto inputHeight = std::get<3>(params);
  const auto inputWidth = std::get<4>(params);
  const auto inputChannels = std::get<5>(params);
  const auto kernelSize = std::get<6>(params);
  const auto stride = std::get<7>(params);
  const auto padding = std::get<8>(params);
  const auto dilation = std::get<9>(params);
  const auto ceilMode = std::get<10>(params);
  const auto expectedLegal = std::get<11>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = MaxPool2DOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, batchSize, inputHeight,
      inputWidth, inputChannels, kernelSize, stride, padding, dilation,
      ceilMode, outputShape, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp = MaxPool2DOpInterface::getOpRuntime(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MaxPool2DTests, OpModelMaxPool2DParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{1, 1, 128 * 128, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 1, 64 * 64, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            1, 128, 128, 32, llvm::SmallVector<int32_t>{2, 2},
            llvm::SmallVector<int32_t>{2, 2}, llvm::SmallVector<int32_t>{0, 0},
            llvm::SmallVector<int32_t>{1, 1}, false, true),
        std::make_tuple(
            detail::TestTensor{{1, 1, 256 * 256, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 1, 64 * 128, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            1, 256, 256, 32, llvm::SmallVector<int32_t>{3, 3},
            llvm::SmallVector<int32_t>{4, 2}, llvm::SmallVector<int32_t>{0, 0},
            llvm::SmallVector<int32_t>{1, 1}, false, true),
        std::make_tuple(
            detail::TestTensor{{1, 1, 17 * 21, 22},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 1, 5 * 11, 22},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            1, 256, 256, 22, llvm::SmallVector<int32_t>{3, 3},
            llvm::SmallVector<int32_t>{4, 2}, llvm::SmallVector<int32_t>{0, 0},
            llvm::SmallVector<int32_t>{1, 1}, false, false)));

class OpModelClampScalarParam : public OpModelTest,
                                public testing::WithParamInterface<
                                    std::tuple<detail::TestTensor, // input
                                               detail::TestTensor, // output
                                               float,              // min
                                               float,              // max
                                               bool // expected legal
                                               >> {};

TEST_P(OpModelClampScalarParam, ClampScalarParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto minVal = llvm::APFloat(std::get<2>(params));
  const auto maxVal = llvm::APFloat(std::get<3>(params));
  const auto expectedLegal = std::get<4>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = ClampScalarOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, minVal, maxVal, outputShape,
      outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp = ClampScalarOpInterface::getOpRuntime(
      inputShape, inputLayout, minVal, maxVal, outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ClampScalarTests, OpModelClampScalarParam,
    ::testing::Values(std::make_tuple(
        detail::TestTensor{{1, 1, 128 * 128, 32},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::DRAM},
        detail::TestTensor{{1, 1, 128 * 128, 32},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::L1},
        1.0, 5.0, true)));

class OpModelPermuteParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // input
                     detail::TestTensor,         // output
                     llvm::SmallVector<int64_t>, // permutation
                     float,                      // pad_value
                     bool                        // expected legal
                     >> {};

TEST_P(OpModelPermuteParam, PermuteParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto permutation = std::get<2>(params);
  const auto padValue = llvm::APFloat(std::get<3>(params));
  const auto expectedLegal = std::get<4>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = PermuteOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, permutation, padValue,
      outputShape, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp =
      PermuteOpInterface::getOpRuntime(inputShape, inputLayout, permutation,
                                       padValue, outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    PermuteTests, OpModelPermuteParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{1, 64, 128, 256},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 256, 64, 128},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            llvm::SmallVector<int64_t>{0, 3, 1, 2}, 0.0f, true),
        std::make_tuple(
            detail::TestTensor{{2, 1280, 8, 8},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{8, 8, 2, 1280},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            llvm::SmallVector<int64_t>{2, 3, 0, 1}, 0.0f, true),
        std::make_tuple(
            detail::TestTensor{{1, 2, 32, 64},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 2, 64, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            llvm::SmallVector<int64_t>{0, -3, -1, -2}, 0.0f, true)));

class OpModelUpsampleParam : public OpModelTest,
                             public testing::WithParamInterface<
                                 std::tuple<detail::TestTensor, // input
                                            detail::TestTensor, // output
                                            int,                // scale factor
                                                 // note: could also be a tuple
                                            std::string, // mode
                                            bool         // expected legal
                                            >> {};

TEST_P(OpModelUpsampleParam, UpsampleParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto scaleFactor = builder.getSI32IntegerAttr(std::get<2>(params));
  const auto mode = std::get<3>(params);
  const auto expectedLegal = std::get<4>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);

  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateRowMajorLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = UpsampleOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, scaleFactor, mode,
      outputShape, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
        constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp = UpsampleOpInterface::getOpRuntime(
      inputShape, inputLayout, scaleFactor, mode, outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    UpsampleTests, OpModelUpsampleParam,
    ::testing::Values(std::make_tuple(
        detail::TestTensor{{2, 128, 8, 8},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::DRAM},
        detail::TestTensor{{2, 256, 16, 8},
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                           mlir::tt::ttnn::BufferType::DRAM},
        2, "nearest", true)));

// ==== EmbeddingOp Tests ====
class OpModelEmbeddingParam : public OpModelTest,
                              public ::testing::WithParamInterface<
                                  std::tuple<detail::TestTensor, // input
                                             detail::TestTensor, // weight
                                             detail::ExpectedResult>> {
protected:
  void RunTest() {
    const auto [inputTensor, weightTensor, expected] = GetParam();
    const auto [inputShape, inputLayout, inputBufferType, inputVirtualGrid] =
        inputTensor;
    const auto [weightShape, weightLayout, weightBufferType,
                weightVirtualGrid] = weightTensor;
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = expected;
    // output shape: [batch, seq_len, hidden_size]
    llvm::SmallVector<int64_t> outputShape = {inputShape[0], inputShape[1],
                                              weightShape[1]};

    const mlir::tt::ttnn::TTNNLayoutAttr inputTiledLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputLayout, inputVirtualGrid);
    const mlir::tt::ttnn::TTNNLayoutAttr weightTiledLayout = CreateTiledLayout(
        weightShape, weightBufferType, weightLayout, weightVirtualGrid);
    const mlir::tt::ttnn::TTNNLayoutAttr outputTiledLayout = CreateTiledLayout(
        outputShape, mlir::tt::ttnn::BufferType::L1,
        mlir::tt::ttnn::TensorMemoryLayout::Interleaved, std::nullopt);

    auto constraintsExp =
        op_model::ttnn::EmbeddingOpInterface::getOpConstraints(
            CreateWorkerGrid(), inputShape, inputTiledLayout, weightShape,
            weightTiledLayout, outputShape, outputTiledLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();
      EXPECT_EQ(cbSize, expectedCbSize);
      EXPECT_EQ(peakSize, expectedPeakSize);
      EXPECT_EQ(outputSize, expectedOutputSize);
      ExpectLayoutsEQ(outputTiledLayout, outputLayoutReadBack);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    // Test runtime using the interface directly
    auto runtimeExp = op_model::ttnn::EmbeddingOpInterface::getOpRuntime(
        inputShape, inputTiledLayout, weightShape, weightTiledLayout,
        outputShape, outputTiledLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_GT(runtimeExp.get(), 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  } // namespace mlir::tt::op_model::ttnn
};

TEST_P(OpModelEmbeddingParam, EmbeddingParam) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    EmbeddingTests, OpModelEmbeddingParam,
    ::testing::Values(
        std::make_tuple(
            // Input: [batch=1, seq_len=1024]
            detail::TestTensor{{1, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            // Weight: [vocab_size=256, hidden_size=128]
            detail::TestTensor{{256, 128},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 16384, 8192, 4096}),
        std::make_tuple(
            // Input: [batch=2, seq_len=512] (sharded)
            detail::TestTensor{{2, 512},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{2, 1}},
            // Weight: [vocab_size=512, hidden_size=256]
            detail::TestTensor{{512, 256},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 32768, 16384, 8192})));

TEST_F(OpModelTest, Where) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  auto constraintsExp = WhereOpInterface::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, inputLayout, inputTensorShape,
      inputLayout, inputTensorShape, inputLayout, inputTensorShape,
      outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 12288);
  EXPECT_EQ(peakSize, 10240);
  EXPECT_EQ(outputSize, 2048);

  auto runtimeExp = WhereOpInterface::getOpRuntime(
      inputTensorShape, inputLayout, inputTensorShape, inputLayout,
      inputTensorShape, inputLayout, inputTensorShape, outputLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

} // namespace mlir::tt::op_model::ttnn
