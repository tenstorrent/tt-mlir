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
#include <iostream>
#include <optional>
#include <tuple>

namespace mlir::tt::ttnn::op_model {

template <typename T1, typename T2>
void EXPECT_EQ_OR_GE(const T1 &actual, const T2 &expected,
                     bool useGreaterThan = false) {
  if (useGreaterThan) {
    EXPECT_GE(actual, expected);
  } else {
    EXPECT_EQ(actual, expected);
  }
}

class OpModelTest : public OpModelFixture {};

namespace detail {
namespace {
struct TestTensor {
  llvm::SmallVector<int64_t> shape;
  TensorMemoryLayout layout;
  BufferType bufferType;
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
    TensorMemoryLayout::Interleaved,
    BufferType::DRAM};
const TestTensor interleavedN300X1024L1 = {
    {OpModelFixture::workerCoresN300, 1024},
    TensorMemoryLayout::Interleaved,
    BufferType::L1};

const TestTensor interleaved2048X2048Dram = {{2048, 2048},
                                             TensorMemoryLayout::Interleaved,
                                             BufferType::DRAM,
                                             llvm::SmallVector<int64_t>{8, 8}};

const TestTensor inerleaved2048X2048L1 = {{2048, 2048},
                                          TensorMemoryLayout::Interleaved,
                                          BufferType::L1,
                                          llvm::SmallVector<int64_t>{8, 8}};
} // namespace detail

// ==== Unary Eltwise Ops Starts ====

template <typename OpTy>
class OpModelUnaryEltwiseParam : public OpModelTest,
                                 public testing::WithParamInterface<
                                     std::tuple<detail::TestTensor, // input
                                                detail::TestTensor, // output
                                                detail::ExpectedResult>> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<1>(params);
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = std::get<2>(params);

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();

      bool useGreaterThan = std::is_same_v<OpTy, CbrtOp>;
      EXPECT_EQ_OR_GE(cbSize, expectedCbSize, useGreaterThan);
      EXPECT_EQ_OR_GE(peakSize, expectedPeakSize, useGreaterThan);
      EXPECT_EQ_OR_GE(outputSize, expectedOutputSize, useGreaterThan);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBack);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp =
        OpModel<OpTy>::getOpRuntime(inputShape, inputLayout, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }

  void RunTestInt32() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<1>(params);
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = std::get<2>(params);

    const TTNNLayoutAttr inputLayout = CreateTiledLayoutInt32(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayoutInt32(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();

      bool useGreaterThan = std::is_same_v<OpTy, BitwiseNotOp>;
      EXPECT_EQ_OR_GE(cbSize, expectedCbSize, useGreaterThan);
      EXPECT_EQ_OR_GE(peakSize, expectedPeakSize, useGreaterThan);
      EXPECT_EQ_OR_GE(outputSize, expectedOutputSize, useGreaterThan);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBack);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp =
        OpModel<OpTy>::getOpRuntime(inputShape, inputLayout, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

// Type aliases for unary operations
using OpModelReluParam = OpModelUnaryEltwiseParam<ReluOp>;
using OpModelSqrtParam = OpModelUnaryEltwiseParam<SqrtOp>;
using OpModelSigmoidParam = OpModelUnaryEltwiseParam<SigmoidOp>;
using OpModelSinParam = OpModelUnaryEltwiseParam<SinOp>;
using OpModelCosParam = OpModelUnaryEltwiseParam<CosOp>;
using OpModelExpParam = OpModelUnaryEltwiseParam<ExpOp>;
using OpModelTanhParam = OpModelUnaryEltwiseParam<TanhOp>;
using OpModelLogParam = OpModelUnaryEltwiseParam<LogOp>;
using OpModelAbsParam = OpModelUnaryEltwiseParam<AbsOp>;
using OpModelCeilParam = OpModelUnaryEltwiseParam<CeilOp>;
using OpModelSignParam = OpModelUnaryEltwiseParam<SignOp>;
using OpModelErfParam = OpModelUnaryEltwiseParam<ErfOp>;
using OpModelErfcParam = OpModelUnaryEltwiseParam<ErfcOp>;
using OpModelFloorParam = OpModelUnaryEltwiseParam<FloorOp>;
using OpModelGeluParam = OpModelUnaryEltwiseParam<GeluOp>;
using OpModelIsFiniteParam = OpModelUnaryEltwiseParam<IsFiniteOp>;
using OpModelLogicalNotParam = OpModelUnaryEltwiseParam<LogicalNotOp>;
using OpModelNegParam = OpModelUnaryEltwiseParam<NegOp>;
using OpModelTanParam = OpModelUnaryEltwiseParam<TanOp>;
using OpModelAtanParam = OpModelUnaryEltwiseParam<AtanOp>;
using OpModelRsqrtParam = OpModelUnaryEltwiseParam<RsqrtOp>;
using OpModelLog1pParam = OpModelUnaryEltwiseParam<Log1pOp>;
using OpModelExpm1Param = OpModelUnaryEltwiseParam<Expm1Op>;
using OpModelReciprocalParam = OpModelUnaryEltwiseParam<ReciprocalOp>;
using OpModelCbrtParam = OpModelUnaryEltwiseParam<CbrtOp>;
using OpModelBitwiseNotParam = OpModelUnaryEltwiseParam<BitwiseNotOp>;

TEST_P(OpModelReluParam, ReluOp) { RunTest(); }
TEST_P(OpModelSqrtParam, SqrtOp) { RunTest(); }
TEST_P(OpModelSigmoidParam, SigmoidOp) { RunTest(); }
TEST_P(OpModelSinParam, SinOp) { RunTest(); }
TEST_P(OpModelCosParam, CosOp) { RunTest(); }
TEST_P(OpModelExpParam, ExpOp) { RunTest(); }
TEST_P(OpModelTanhParam, TanhOp) { RunTest(); }
TEST_P(OpModelLogParam, LogOp) { RunTest(); }
TEST_P(OpModelAbsParam, AbsOp) { RunTest(); }
TEST_P(OpModelCeilParam, CeilOp) { RunTest(); }
TEST_P(OpModelSignParam, SignOp) { RunTest(); }
TEST_P(OpModelErfParam, ErfOp) { RunTest(); }
TEST_P(OpModelErfcParam, ErfcOp) { RunTest(); }
TEST_P(OpModelFloorParam, FloorOp) { RunTest(); }
TEST_P(OpModelReciprocalParam, ReciprocalOp) { RunTest(); }
TEST_P(OpModelGeluParam, GeluOp) { RunTest(); }
TEST_P(OpModelIsFiniteParam, IsFiniteOp) { RunTest(); }
TEST_P(OpModelLogicalNotParam, LogicalNotOp) { RunTest(); }
TEST_P(OpModelNegParam, NegOp) { RunTest(); }
TEST_P(OpModelTanParam, TanOp) { RunTest(); }
TEST_P(OpModelAtanParam, AtanOp) { RunTest(); }
TEST_P(OpModelRsqrtParam, RsqrtOp) { RunTest(); }
TEST_P(OpModelLog1pParam, Log1pOp) { RunTest(); }
TEST_P(OpModelExpm1Param, Expm1Op) { RunTest(); }
TEST_P(OpModelCbrtParam, CbrtOp) { RunTest(); }
TEST_P(OpModelBitwiseNotParam, BitwiseNotOp) { RunTestInt32(); }

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
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::ExpectedResult{true, 0, 14 * 32 * 32 * 2,
                                   14 * 32 * 32 * 2}),
        std::make_tuple(
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::Interleaved,
                               BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::Interleaved,
                               BufferType::L1},
            detail::ExpectedResult{false})};

INSTANTIATE_TEST_SUITE_P(ReluTests, OpModelReluParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SqrtTests, OpModelSqrtParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SigmoidTests, OpModelSigmoidParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SinTests, OpModelSinParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(CosTests, OpModelCosParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ExpTests, OpModelExpParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(TanhTests, OpModelTanhParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogTests, OpModelLogParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(AbsTests, OpModelAbsParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(CeilTests, OpModelCeilParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SignTests, OpModelSignParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ErfTests, OpModelErfParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ErfcTests, OpModelErfcParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(FloorTests, OpModelFloorParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(GeluTests, OpModelGeluParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(IsFiniteTests, OpModelIsFiniteParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalNotTests, OpModelLogicalNotParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(NegTests, OpModelNegParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(TanTests, OpModelTanParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(AtanTests, OpModelAtanParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(RsqrtTests, OpModelRsqrtParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(Log1pTests, OpModelLog1pParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(Expm1Tests, OpModelExpm1Param,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ReciprocalTests, OpModelReciprocalParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(CbrtTests, OpModelCbrtParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(BitwiseNotTests, OpModelBitwiseNotParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

// ==== Unary Eltwise Ops Ends ====

template <typename OpTy>
class OpModelReductionParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,                        // input
                     detail::TestTensor,                        // output
                     std::optional<llvm::SmallVector<int64_t>>, // dim arg
                     bool,                                      // keep dim
                     detail::ExpectedResult>> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);

    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<1>(params);
    const auto dimArg = std::get<2>(params);
    const auto keepDim = std::get<3>(params);
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = std::get<4>(params);

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
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

    auto runtimeExp = OpModel<OpTy>::getOpRuntime(
        inputShape, inputLayout, dimArg, keepDim, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

// Type aliases for reduction operations
using OpModelSumParam = OpModelReductionParam<SumOp>;
using OpModelMeanParam = OpModelReductionParam<MeanOp>;

TEST_P(OpModelSumParam, SumOp) { RunTest(); }
TEST_P(OpModelMeanParam, MeanOp) { RunTest(); }

// Test parameters for reduction operations
static const auto reductionParams = ::testing::Values(
    std::make_tuple(detail::interleavedN300X1024Dram,
                    detail::interleavedN300X1024Dram,
                    std::optional<llvm::SmallVector<int64_t>>{
                        llvm::SmallVector<int64_t>{1}},
                    true, detail::ExpectedResult{true, 12288, 0, 0}),
    std::make_tuple(detail::interleavedN300X1024Dram,
                    detail::interleavedN300X1024Dram,
                    std::optional<llvm::SmallVector<int64_t>>{
                        llvm::SmallVector<int64_t>{1, 2}},
                    false, detail::ExpectedResult{false, 0, 0, 0}),
    std::make_tuple(detail::interleavedN300X1024Dram,
                    detail::interleavedN300X1024Dram,
                    std::optional<llvm::SmallVector<int64_t>>{
                        llvm::SmallVector<int64_t>{1, 0}},
                    false, detail::ExpectedResult{true, 12288, 0, 0}),
    std::make_tuple(detail::interleavedN300X1024L1,
                    detail::interleavedN300X1024Dram,
                    std::optional<llvm::SmallVector<int64_t>>{
                        llvm::SmallVector<int64_t>{1}},
                    false, detail::ExpectedResult{true, 12288, 0, 0}));

INSTANTIATE_TEST_SUITE_P(SumTests, OpModelSumParam, reductionParams);

INSTANTIATE_TEST_SUITE_P(MeanTests, OpModelMeanParam, reductionParams);

TEST_F(OpModelTest, SoftmaxInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayout_dram = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayout_l1 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, false,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, false,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, false,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, false,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, false,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 137216);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  std::vector<std::tuple<TTNNLayoutAttr, TTNNLayoutAttr>> layout_combinations =
      {{inputLayout_dram, inputLayout_dram},
       {inputLayout_l1, inputLayout_dram},
       {inputLayout_dram, inputLayout_l1},
       {inputLayout_l1, inputLayout_l1}};
  for (const auto &[input_layout, output_layout] : layout_combinations) {
    auto runtimeExp = OpModel<SoftmaxOp>::getOpRuntime(
        tensorShape, input_layout, -1, false, output_layout);
    EXPECT_TRUE(static_cast<bool>(runtimeExp));
    EXPECT_TRUE(runtimeExp.get() > 0);
  }
}

TEST_F(OpModelTest, SoftmaxNumericStable) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const TTNNLayoutAttr inputLayout_dram = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, true,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  auto runtimeExp = OpModel<SoftmaxOp>::getOpRuntime(
      tensorShape, inputLayout_dram, -1, true, inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Reshape) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<ReshapeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, {workerCoresN300 * 4, 256},
      layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 5120);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  auto runtimeExp = OpModel<ReshapeOp>::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  constraintsExp = OpModel<ReshapeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, {workerCoresN300 * 4, 256},
      layoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 5120);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);
  // Need to reset device other wise hangs. See tt-metal issue #25772
  SingletonDeviceContext::resetInstance();

  runtimeExp = OpModel<ReshapeOp>::getOpRuntime(
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
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      inputTensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  llvm::SmallVector<int64_t> begins = {0, 0, 0, 0};
  llvm::SmallVector<int64_t> ends = {1, 56, 56, 95};
  llvm::SmallVector<int64_t> step = {1, 2, 1, 1};

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<SliceOp>::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, layoutDRAM, begins, ends, step,
      layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<SliceOp>::getOpRuntime(
      inputTensorShape, layoutDRAM, begins, ends, step, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, ToLayout) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAMTiled = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1RowMajorHS = CreateRowMajorLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<ToLayoutOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 131072);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayout);

  auto runtimeExp = OpModel<ToLayoutOp>::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = OpModel<ToLayoutOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutL1RowMajorHS);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = OpModel<ToLayoutOp>::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutL1RowMajorHS);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());

  constraintsExp = OpModel<ToLayoutOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMTiled, std::nullopt,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 131072);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayout);

  runtimeExp = OpModel<ToLayoutOp>::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, ToMemoryConfig) {
  const llvm::SmallVector<int64_t> tensorShape = {1, 8, 64, 128};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  const TTNNLayoutAttr inputLayoutL1Tiled = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayoutDRAMTiled = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  MemoryConfigAttr memoryConfig = MemoryConfigAttr::get(
      &context, outputLayoutDRAMTiled.getMemLayout(),
      BufferTypeAttr::get(&context, outputLayoutDRAMTiled.getBufferType()),
      std::nullopt /*shardSpec*/);
  auto constraintsExp = OpModel<ToMemoryConfigOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutL1Tiled, memoryConfig,
      outputLayoutDRAMTiled);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<ToMemoryConfigOp>::getOpRuntime(
      tensorShape, inputLayoutL1Tiled, memoryConfig, outputLayoutDRAMTiled);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  auto coreRangeSetAttr = CoreRangeSetAttr::get(
      &context, llvm::ArrayRef<CoreRangeAttr>{CoreRangeAttr::get(
                    &context, CoreCoordAttr::get(&context, 0, 0),
                    CoreCoordAttr::get(&context, 7, 0))});
  ShardSpecAttr shardSpec = ShardSpecAttr::get(
      &context, coreRangeSetAttr, ShapeAttr::get(&context, {64, 128}),
      ShardOrientationAttr::get(&context, ShardOrientation::RowMajor),
      ShardModeAttr::get(&context, ShardMode::Physical),
      /*physical_shard_shape=*/nullptr);
  const TTNNLayoutAttr outputLayoutL1Tiled = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  memoryConfig = MemoryConfigAttr::get(
      &context, outputLayoutL1Tiled.getMemLayout(),
      BufferTypeAttr::get(&context, outputLayoutL1Tiled.getBufferType()),
      shardSpec);
  constraintsExp = OpModel<ToMemoryConfigOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutL1Tiled, memoryConfig,
      outputLayoutL1Tiled);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 16384);
  EXPECT_EQ(opCstr.outputL1BufferSize, 16384);

  runtimeExp = OpModel<ToMemoryConfigOp>::getOpRuntime(
      tensorShape, inputLayoutL1Tiled, memoryConfig, outputLayoutL1Tiled);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Concat) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      inputTensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      inputTensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<ConcatOp>::getOpConstraints(
      CreateWorkerGrid(), {inputTensorShape, inputTensorShape},
      {layoutL1Interleaved, layoutL1Interleaved}, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 4096);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<ConcatOp>::getOpRuntime(
      {inputTensorShape, inputTensorShape},
      {layoutL1Interleaved, layoutL1Interleaved}, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Transpose) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1WSharded = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::WidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<TransposeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<TransposeOp>::getOpRuntime(tensorShape, layoutDRAM,
                                                       0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = OpModel<TransposeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, 1, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 8192);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 2048);
  EXPECT_EQ(opCstr.outputL1BufferSize, 2048);

  runtimeExp = OpModel<TransposeOp>::getOpRuntime(tensorShape, layoutDRAM, 0, 1,
                                                  layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = OpModel<TransposeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 0, 1,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = OpModel<TransposeOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, 0, 1, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

TEST_F(OpModelTest, SoftmaxSharded) {
  const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayout_l1_hs = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  const TTNNLayoutAttr inputLayout_l1_i = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_hs, -2, false,
      inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_hs, -2, false,
      inputLayout_l1_i);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_i, -2, false,
      inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 24576);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 32768);
  EXPECT_EQ(opCstr.outputL1BufferSize, 32768);

  auto runtimeExp = OpModel<SoftmaxOp>::getOpRuntime(
      tensorShape, inputLayout_l1_i, -2, false, inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Typecast) {
  const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayoutDRAMIBF16 = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutL1HSBF16 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  const TTNNLayoutAttr inputLayoutDRAMIF32 = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      std::nullopt, GetPhysicalGridSize(), builder.getF32Type());
  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<TypecastOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 12288);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<TypecastOp>::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = OpModel<TypecastOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());
  runtimeExp = OpModel<TypecastOp>::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32),
      inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

struct BinaryEltwiseParam {
  detail::TestTensor inputA;
  detail::TestTensor inputB;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};

template <typename OpTy>
class OpModelBinaryEltwiseParam
    : public OpModelTest,
      public testing::WithParamInterface<BinaryEltwiseParam> {
protected:
  // clang-format on
  void RunTest() {
    const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
                inputVirtualGridA] = GetParam().inputA;
    const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
                inputVirtualGridB] = GetParam().inputB;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = GetParam().expectedResult;

    const TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
        inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
    const TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
        inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB,
        inputLayoutB, outputLayout);
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

    llvm::Expected<size_t> runtimeExp = OpModel<OpTy>::getOpRuntime(
        inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

// Type aliases for binary operations
using OpModelAddParam = OpModelBinaryEltwiseParam<AddOp>;
using OpModelMultiplyParam = OpModelBinaryEltwiseParam<MultiplyOp>;
using OpModelSubtractParam = OpModelBinaryEltwiseParam<SubtractOp>;
using OpModelMaximumParam = OpModelBinaryEltwiseParam<MaximumOp>;
using OpModelMinimumParam = OpModelBinaryEltwiseParam<MinimumOp>;
using OpModelDivideParam = OpModelBinaryEltwiseParam<DivideOp>;
using OpModelEqualParam = OpModelBinaryEltwiseParam<EqualOp>;
using OpModelNotEqualParam = OpModelBinaryEltwiseParam<NotEqualOp>;
using OpModelGreaterEqualParam = OpModelBinaryEltwiseParam<GreaterEqualOp>;
using OpModelGreaterThanParam = OpModelBinaryEltwiseParam<GreaterThanOp>;
using OpModelLessEqualParam = OpModelBinaryEltwiseParam<LessEqualOp>;
using OpModelLessThanParam = OpModelBinaryEltwiseParam<LessThanOp>;
using OpModelLogicalAndParam = OpModelBinaryEltwiseParam<LogicalAndOp>;
using OpModelLogicalOrParam = OpModelBinaryEltwiseParam<LogicalOrOp>;
using OpModelLogicalXorParam = OpModelBinaryEltwiseParam<LogicalXorOp>;

TEST_P(OpModelAddParam, AddOp) { RunTest(); }
TEST_P(OpModelMultiplyParam, MultiplyOp) { RunTest(); }
TEST_P(OpModelSubtractParam, SubtractOp) { RunTest(); }
TEST_P(OpModelMaximumParam, MaximumOp) { RunTest(); }
TEST_P(OpModelMinimumParam, MinimumOp) { RunTest(); }
TEST_P(OpModelDivideParam, DivideOp) { RunTest(); }
TEST_P(OpModelEqualParam, EqualOp) { RunTest(); }
TEST_P(OpModelNotEqualParam, NotEqualOp) { RunTest(); }
TEST_P(OpModelGreaterEqualParam, GreaterEqualOp) { RunTest(); }
TEST_P(OpModelGreaterThanParam, GreaterThanOp) { RunTest(); }
TEST_P(OpModelLessEqualParam, LessEqualOp) { RunTest(); }
TEST_P(OpModelLessThanParam, LessThanOp) { RunTest(); }
TEST_P(OpModelLogicalAndParam, LogicalAndOp) { RunTest(); }
TEST_P(OpModelLogicalOrParam, LogicalOrOp) { RunTest(); }
TEST_P(OpModelLogicalXorParam, LogicalXorOp) { RunTest(); }

const std::initializer_list<BinaryEltwiseParam> binaryEltwiseParams = {
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {detail::interleavedN300X1024Dram, detail::interleaved2048X2048Dram,
     detail::interleaved2048X2048Dram,
     detail::ExpectedResult{false, 0, 0, 0}}, // incompatible dimensions at
                                              // the input
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram,
     detail::ExpectedResult{true, 12288, 0, 0}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1,
     detail::ExpectedResult{true, 12288, 2048, 2048}},
    {detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::HeightSharded,
                        BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::Interleaved,
                        BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::HeightSharded,
                        BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::ExpectedResult{true, 4096, 262144, 262144}},
    {detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::HeightSharded,
                        BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::Interleaved,
                        BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::Interleaved,
                        BufferType::DRAM},
     detail::ExpectedResult{true, 8192, 0, 0}},
    {detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::Interleaved,
                        BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::Interleaved,
                        BufferType::DRAM},
     detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                        TensorMemoryLayout::HeightSharded,
                        BufferType::L1,
                        llvm::SmallVector<int64_t>{8, 1}},
     detail::ExpectedResult{true, 8192, 262144, 262144}}};

::testing::internal::ParamGenerator<BinaryEltwiseParam>
generateBinaryEltwiseParams(std::initializer_list<BinaryEltwiseParam> values,
                            std::size_t extraCbRequirement = 0) {
  // The expected size of the circular buffer is the same for most binary ops,
  // but some of them (such as Divide, LogicalOr and LogicalXor) extra memory is
  // required due to the op's implementation.
  std::vector<BinaryEltwiseParam> newValues;
  for (const auto &v : values) {
    newValues.emplace_back(v);
    newValues.back().expectedResult.expectedCbSize += extraCbRequirement;
  }
  return ::testing::ValuesIn(newValues);
}

INSTANTIATE_TEST_SUITE_P(AddTests, OpModelAddParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MulTests, OpModelMultiplyParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SubtractTests, OpModelSubtractParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MaximumTests, OpModelMaximumParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MinimumTests, OpModelMinimumParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(DivideTests, OpModelDivideParam,
                         generateBinaryEltwiseParams(
                             binaryEltwiseParams, /*extraCbRequirement=*/2048));

INSTANTIATE_TEST_SUITE_P(EqualTests, OpModelEqualParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(NotEqualTests, OpModelNotEqualParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(GreaterEqualTests, OpModelGreaterEqualParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(GreaterThanTests, OpModelGreaterThanParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LessEqualTests, OpModelLessEqualParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LessThanTests, OpModelLessThanParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalAndTests, OpModelLogicalAndParam,
                         generateBinaryEltwiseParams(
                             binaryEltwiseParams, /*extraCbRequirement=*/4096));

INSTANTIATE_TEST_SUITE_P(LogicalOrTests, OpModelLogicalOrParam,
                         generateBinaryEltwiseParams(
                             binaryEltwiseParams, /*extraCbRequirement=*/4096));

INSTANTIATE_TEST_SUITE_P(LogicalXorTests, OpModelLogicalXorParam,
                         generateBinaryEltwiseParams(
                             binaryEltwiseParams, /*extraCbRequirement=*/4096));

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

  const TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const TTNNLayoutAttr biasLayout = CreateTiledLayout(
      biasShape, biasBufferType, biasTensorLayout, biasVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = OpModel<LinearOp>::getOpConstraints(
      CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
      biasShape, biasLayout, outputLayout, false, false);

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

  auto runtimeExp = OpModel<LinearOp>::getOpRuntime(
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, biasShape,
      biasLayout, outputLayout, false, false);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
  SingletonDeviceContext::resetInstance();
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
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{true, 430144, 229376, 114688}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{false}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{false}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{true, 544832, 0, 0}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                            TensorMemoryLayout::HeightSharded, BufferType::L1,
                            llvm::SmallVector<int64_t>{56, 1}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               TensorMemoryLayout::WidthSharded, BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               TensorMemoryLayout::Interleaved,
                               BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{{1 * 32, 56 * 32},
                               TensorMemoryLayout::Interleaved,
                               BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               TensorMemoryLayout::WidthSharded, BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 8256, 4096, 2048}),
        std::make_tuple(detail::TestTensor{{56 * 32, 1 * 32},
                                           TensorMemoryLayout::HeightSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{56, 1}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                            TensorMemoryLayout::Interleaved, BufferType::DRAM,
                            llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                            TensorMemoryLayout::HeightSharded, BufferType::L1,
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

  const TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = OpModel<MatmulOp>::getOpConstraints(
      CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
      outputLayout, false, false);

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
      OpModel<MatmulOp>::getOpRuntime(inputShapeA, inputLayoutA, inputShapeB,
                                      inputLayoutB, outputLayout, false, false);
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
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{true, 430144, 114688, 114688}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{false}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{
                            true, 262144, 401408,
                            401408}), // matmul bug shards to less cores
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{true, 544832, 0, 0}),
        std::make_tuple(detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::BlockSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                            TensorMemoryLayout::HeightSharded, BufferType::L1,
                            llvm::SmallVector<int64_t>{56, 1}},
                        detail::TestTensor{{56 * 32, 56 * 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM,
                                           llvm::SmallVector<int64_t>{7, 8}},
                        llvm::SmallVector<int64_t>{7, 8},
                        detail::ExpectedResult{false}),
        std::make_tuple(
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               TensorMemoryLayout::WidthSharded, BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            detail::TestTensor{{56 * 32, 56 * 32},
                               TensorMemoryLayout::Interleaved,
                               BufferType::DRAM,
                               llvm::SmallVector<int64_t>{7, 8}},
            detail::TestTensor{llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                               TensorMemoryLayout::WidthSharded, BufferType::L1,
                               llvm::SmallVector<int64_t>{1, 56}},
            llvm::SmallVector<int64_t>{7, 8},
            detail::ExpectedResult{true, 8256, 2048, 2048}),
        std::make_tuple(detail::TestTensor{{56 * 32, 1 * 32},
                                           TensorMemoryLayout::HeightSharded,
                                           BufferType::L1,
                                           llvm::SmallVector<int64_t>{56, 1}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
                            TensorMemoryLayout::Interleaved, BufferType::DRAM,
                            llvm::SmallVector<int64_t>{7, 8}},
                        detail::TestTensor{
                            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
                            TensorMemoryLayout::HeightSharded, BufferType::L1,
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

  const TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  // This is not configurable, as the backend doesn't support it for now.
  // But this test shows that this information is parsed and passes to the
  // backend correctly.
  DeviceComputeKernelConfigAttr deviceConfig =
      DeviceComputeKernelConfigAttr::get(
          &context, /*mathFidelity=*/MathFidelity::LoFi,
          /*mathApproxMode=*/::mlir::BoolAttr::get(&context, true),
          /*fp32DestAccEn=*/::mlir::BoolAttr::get(&context, true),
          /*packerL1Acc=*/::mlir::BoolAttr::get(&context, true),
          /*dstFullSyncEn=*/::mlir::BoolAttr::get(&context, true));

  auto constraintsExp = OpModel<Conv2dOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, dilation, groups,
      std::nullopt, deviceConfig, outputLayout);
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

  auto runtimeExp = OpModel<Conv2dOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, dilation, groups, std::nullopt,
      deviceConfig, outputLayout);
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
        std::make_tuple(detail::TestTensor{{1, 1, 50176, 3},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{64, 3, 7, 7},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::SystemMemory},
                        detail::TestTensor{{1, 1, 12544, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{1, 1}, 1,
                        detail::ExpectedResult{true, 229440, 190568, 0}),
        std::make_tuple(detail::TestTensor{{1, 1, 50176, 3},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{64, 3, 9, 7},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::SystemMemory},
                        detail::TestTensor{{1, 1, 12544, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{1, 1}, 1,
                        detail::ExpectedResult{true, 0, 0, 0})));

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

  const TTNNLayoutAttr inputLayout =
      CreateRowMajorLayout(inputShape, inputBufferType, inputTensorLayout,
                           inputVirtualGrid, GetPhysicalGridSize());
  const TTNNLayoutAttr weightLayout =
      CreateRowMajorLayout(weightShape, weightBufferType, weightTensorLayout,
                           weightVirtualGrid, GetPhysicalGridSize());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  auto constraintsExp = OpModel<ConvTranspose2dOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, output_padding,
      dilation, groups, std::nullopt, outputLayout);
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

  auto runtimeExp = OpModel<ConvTranspose2dOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, output_padding, dilation,
      groups, std::nullopt, outputLayout);
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
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{3, 64, 7, 7},
                           TensorMemoryLayout::Interleaved,
                           BufferType::SystemMemory},
        detail::TestTensor{{1, 1, 12544, 64},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
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

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = OpModel<MaxPool2dOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, batchSize, inputHeight,
      inputWidth, inputChannels, kernelSize, stride, padding, dilation,
      ceilMode, outputLayout);
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

  auto runtimeExp = OpModel<MaxPool2dOp>::getOpRuntime(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      outputLayout);
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
        std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 1, 64 * 64, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        1, 128, 128, 32, llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{0, 0},
                        llvm::SmallVector<int32_t>{1, 1}, false, true),
        std::make_tuple(detail::TestTensor{{1, 1, 256 * 256, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 1, 64 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        1, 256, 256, 32, llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{4, 2},
                        llvm::SmallVector<int32_t>{0, 0},
                        llvm::SmallVector<int32_t>{1, 1}, false, true),
        std::make_tuple(detail::TestTensor{{1, 1, 17 * 21, 22},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 1, 5 * 11, 22},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        1, 256, 256, 22, llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{4, 2},
                        llvm::SmallVector<int32_t>{0, 0},
                        llvm::SmallVector<int32_t>{1, 1}, false, false)));

class OpModelLeakyReluParam : public OpModelTest,
                              public testing::WithParamInterface<
                                  std::tuple<detail::TestTensor, // input
                                             detail::TestTensor, // output
                                             float,              // slope
                                             bool // expected legal
                                             >> {};

TEST_P(OpModelLeakyReluParam, LeakyReluParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto slope = llvm::APFloat(std::get<2>(params));
  const auto expectedLegal = std::get<3>(params);

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = op_model::OpModel<LeakyReluOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, slope, outputLayout);
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

  auto runtimeExp = op_model::OpModel<LeakyReluOp>::getOpRuntime(
      inputShape, inputLayout, slope, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(LeakyReluTests, OpModelLeakyReluParam,
                         ::testing::Values(std::make_tuple(
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::DRAM},
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::L1},
                             1.0, true)));

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

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = OpModel<ClampScalarOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, minVal, maxVal,
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

  auto runtimeExp = OpModel<ClampScalarOp>::getOpRuntime(
      inputShape, inputLayout, minVal, maxVal, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(ClampScalarTests, OpModelClampScalarParam,
                         ::testing::Values(std::make_tuple(
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::DRAM},
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::L1},
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

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = OpModel<PermuteOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, permutation, padValue,
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

  auto runtimeExp = OpModel<PermuteOp>::getOpRuntime(
      inputShape, inputLayout, permutation, padValue, outputLayout);
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
        std::make_tuple(detail::TestTensor{{1, 64, 128, 256},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 256, 64, 128},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        llvm::SmallVector<int64_t>{0, 3, 1, 2}, 0.0f, true),
        std::make_tuple(detail::TestTensor{{2, 1280, 8, 8},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{8, 8, 2, 1280},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        llvm::SmallVector<int64_t>{2, 3, 0, 1}, 0.0f, true),
        std::make_tuple(detail::TestTensor{{1, 2, 32, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 2, 64, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        llvm::SmallVector<int64_t>{0, -3, -1, -2}, 0.0f,
                        true)));

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

  const TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);

  const TTNNLayoutAttr outputLayout = CreateRowMajorLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  SingletonDeviceContext::resetInstance();

  auto constraintsExp = OpModel<UpsampleOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, scaleFactor, mode,
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
    EXPECT_EQ(peakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp = OpModel<UpsampleOp>::getOpRuntime(
      inputShape, inputLayout, scaleFactor, mode, outputLayout);
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
        detail::TestTensor{
            {2, 128, 8, 8}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {2, 256, 16, 8}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
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

    const TTNNLayoutAttr inputTiledLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputLayout, inputVirtualGrid);
    const TTNNLayoutAttr weightTiledLayout = CreateTiledLayout(
        weightShape, weightBufferType, weightLayout, weightVirtualGrid);
    const TTNNLayoutAttr outputTiledLayout =
        CreateTiledLayout(outputShape, BufferType::L1,
                          TensorMemoryLayout::Interleaved, std::nullopt);

    auto constraintsExp = OpModel<EmbeddingOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputTiledLayout, weightShape,
        weightTiledLayout, outputTiledLayout);

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
    auto runtimeExp = OpModel<EmbeddingOp>::getOpRuntime(
        inputShape, inputTiledLayout, weightShape, weightTiledLayout,
        outputTiledLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_GT(runtimeExp.get(), 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelEmbeddingParam, EmbeddingParam) { RunTest(); }

INSTANTIATE_TEST_SUITE_P(
    EmbeddingTests, OpModelEmbeddingParam,
    ::testing::Values(
        std::make_tuple(
            // Input: [batch=1, seq_len=1024]
            detail::TestTensor{
                {1, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
            // Weight: [vocab_size=256, hidden_size=128]
            detail::TestTensor{
                {256, 128}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
            detail::ExpectedResult{true, 16384, 8192, 4096}),
        std::make_tuple(
            // Input: [batch=2, seq_len=512] (sharded)
            detail::TestTensor{{2, 512},
                               TensorMemoryLayout::Interleaved,
                               BufferType::L1,
                               llvm::SmallVector<int64_t>{2, 1}},
            // Weight: [vocab_size=512, hidden_size=256]
            detail::TestTensor{
                {512, 256}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
            detail::ExpectedResult{true, 32768, 16384, 8192})));

TEST_F(OpModelTest, EmbeddingBackwardOp) {
  llvm::SmallVector<int64_t> inputShape = {2, 1024};
  llvm::SmallVector<int64_t> weightShape = {3200, 4096};
  llvm::SmallVector<int64_t> inGradientShape = {1, 1, 2048, 4096};

  auto inputLayout = CreateRowMajorLayout(inputShape, BufferType::DRAM,
                                          TensorMemoryLayout::Interleaved);
  auto weightLayout = CreateRowMajorLayout(weightShape, BufferType::DRAM,
                                           TensorMemoryLayout::Interleaved);
  auto inGradientLayout = CreateTiledLayout(inGradientShape, BufferType::L1,
                                            TensorMemoryLayout::Interleaved);
  auto outputLayout = CreateTiledLayout(inGradientShape, BufferType::L1,
                                        TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<EmbeddingBackwardOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      inGradientShape, inGradientLayout, outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 12400);
  EXPECT_EQ(peakSize, 409600);
  EXPECT_EQ(outputSize, 409600);

  auto runtimeExp = OpModel<EmbeddingBackwardOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, inGradientShape,
      inGradientLayout, outputLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

TEST_F(OpModelTest, Where) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputTensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      inputTensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<WhereOp>::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, inputLayout, inputTensorShape,
      inputLayout, inputTensorShape, inputLayout, outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 16384);
  EXPECT_EQ(peakSize, 2048);
  EXPECT_EQ(outputSize, 2048);

  auto runtimeExp = OpModel<WhereOp>::getOpRuntime(
      inputTensorShape, inputLayout, inputTensorShape, inputLayout,
      inputTensorShape, inputLayout, outputLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GT(runtimeExp.get(), 0);
}

TEST_F(OpModelTest, EmptyOp) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutL1Tiled =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::MemoryConfigAttr memoryConfig =
      mlir::tt::ttnn::MemoryConfigAttr::get(
          &context, inputLayoutL1Tiled.getMemLayout(),
          mlir::tt::ttnn::BufferTypeAttr::get(
              &context, inputLayoutL1Tiled.getBufferType()),
          std::nullopt /*shardSpec*/);
  mlir::tt::ttcore::DataTypeAttr dtype =
      mlir::tt::ttcore::DataTypeAttr::get(&context, ttcore::DataType::Float32);
  mlir::tt::ttnn::Layout layout = mlir::tt::ttnn::Layout::Tile;
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  auto constraintsExp =
      ttnn::op_model::OpModel<mlir::tt::ttnn::EmptyOp>::getOpConstraints(
          CreateWorkerGrid(), inputTensorShape, dtype, layout, memoryConfig,
          outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 0);
  EXPECT_EQ(peakSize, 4096);
  EXPECT_EQ(outputSize, 4096);
}

TEST_F(OpModelTest, ArangeOp) {
  const llvm::SmallVector<int64_t> inputTensorShape = {workerCoresN300, 1024};
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout =
      CreateTiledLayout(inputTensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  // Create IntegerAttr parameters
  ::mlir::IntegerAttr startAttr = builder.getI32IntegerAttr(0);
  ::mlir::IntegerAttr endAttr = builder.getI32IntegerAttr(10);
  ::mlir::IntegerAttr stepAttr = builder.getI32IntegerAttr(1);

  // Create optional dtype
  std::optional<mlir::tt::ttcore::DataType> dtype =
      mlir::tt::ttcore::DataType::Float32;

  // Create optional memory config
  std::optional<mlir::tt::ttnn::MemoryConfigAttr> memConfig =
      mlir::tt::ttnn::MemoryConfigAttr::get(
          &context, inputLayout.getMemLayout(),
          mlir::tt::ttnn::BufferTypeAttr::get(&context,
                                              inputLayout.getBufferType()),
          std::nullopt /*shardSpec*/);

  auto constraintsExp =
      ttnn::op_model::OpModel<mlir::tt::ttnn::ArangeOp>::getOpConstraints(
          CreateWorkerGrid(), startAttr, endAttr, stepAttr, dtype, memConfig,
          nullptr);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  // Basic assertions to verify the op constraints are computed
  EXPECT_EQ(cbSize, 0);
  EXPECT_EQ(peakSize, 0);
  EXPECT_EQ(outputSize, 0);
}

// ==== Creation Ops ====

template <typename OpTy>
class OpModelCreationParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<llvm::SmallVector<int64_t>, detail::ExpectedResult>> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [tensorShape, expectedResult] = params;
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = expectedResult;

    const mlir::tt::ttnn::TTNNLayoutAttr outputLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    auto shapeAttr = mlir::tt::ttnn::ShapeAttr::get(&context, tensorShape);

    auto constraintsExp = ttnn::op_model::OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), shapeAttr, std::nullopt, std::nullopt, std::nullopt,
        outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
          constraintsExp.get();
      EXPECT_EQ(cbSize, expectedCbSize);
      EXPECT_EQ(peakSize, expectedPeakSize);
      EXPECT_EQ(outputSize, expectedOutputSize);
    }
  }
};

using OpModelZerosParam = OpModelCreationParam<mlir::tt::ttnn::ZerosOp>;
using OpModelOnesParam = OpModelCreationParam<mlir::tt::ttnn::OnesOp>;

TEST_P(OpModelZerosParam, ZerosOpParameterized) { RunTest(); }
TEST_P(OpModelOnesParam, OnesOpParameterized) { RunTest(); }

// Test data for creation operations
const auto creationOpTestData = testing::Values(
    std::make_tuple(llvm::SmallVector<int64_t>{1024, 256},
                    detail::ExpectedResult{true, 0, 8192, 8192}));

INSTANTIATE_TEST_SUITE_P(CreationOps, OpModelZerosParam, creationOpTestData);
INSTANTIATE_TEST_SUITE_P(CreationOps, OpModelOnesParam, creationOpTestData);

// ==== FullOp Tests ====
TEST_F(OpModelTest, FullOp) {
  const llvm::SmallVector<int64_t> shape = {workerCoresN300, 1024};
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout =
      CreateTiledLayout(shape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  auto shapeAttr = mlir::tt::ttnn::ShapeAttr::get(&context, shape);
  auto constraintsExp =
      ttnn::op_model::OpModel<mlir::tt::ttnn::FullOp>::getOpConstraints(
          CreateWorkerGrid(), shapeAttr, builder.getI32IntegerAttr(0),
          std::nullopt, std::nullopt, std::nullopt, outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cbSize, peakSize, outputSize, outputLayoutReadBack] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 0);
  EXPECT_EQ(peakSize, 0);
  EXPECT_EQ(outputSize, 0);
}

} // namespace mlir::tt::ttnn::op_model
