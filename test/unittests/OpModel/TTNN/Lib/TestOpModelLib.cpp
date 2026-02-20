// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/OpModel/TTNN/TTNNOpConstraints.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "llvm/ADT/APFloat.h"
#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>

namespace mlir::tt::ttnn::op_model {

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
    const auto expectedLegal = std::get<2>(params).expectedLegal;

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
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();

      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
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
    const auto expectedLegal = std::get<2>(params).expectedLegal;

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
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();

      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
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
using OpModelRelu6Param = OpModelUnaryEltwiseParam<Relu6Op>;
using OpModelSiluParam = OpModelUnaryEltwiseParam<SiluOp>;
using OpModelSqrtParam = OpModelUnaryEltwiseParam<SqrtOp>;
using OpModelSigmoidParam = OpModelUnaryEltwiseParam<SigmoidOp>;
using OpModelHardsigmoidParam = OpModelUnaryEltwiseParam<HardsigmoidOp>;
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
TEST_P(OpModelRelu6Param, Relu6Op) { RunTest(); }
TEST_P(OpModelSiluParam, SiluOp) { RunTest(); }
TEST_P(OpModelSqrtParam, SqrtOp) { RunTest(); }
TEST_P(OpModelSigmoidParam, SigmoidOp) { RunTest(); }
TEST_P(OpModelHardsigmoidParam, HardsigmoidOp) { RunTest(); }
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
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true}),
        std::make_tuple(
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::ExpectedResult{true}),
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

const std::initializer_list<
    std::tuple<detail::TestTensor, detail::TestTensor, detail::ExpectedResult>>
    tanhParams = {
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true}),
        std::make_tuple(
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::TestTensor{{14 * OpModelFixture::workerCoresN300 * 32, 32},
                               TensorMemoryLayout::HeightSharded,
                               BufferType::L1},
            detail::ExpectedResult{true}),
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

INSTANTIATE_TEST_SUITE_P(Relu6Tests, OpModelRelu6Param,
                         ::testing::ValuesIn(unaryEltwiseParams));
INSTANTIATE_TEST_SUITE_P(SiluTests, OpModelSiluParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SqrtTests, OpModelSqrtParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SigmoidTests, OpModelSigmoidParam,
                         ::testing::ValuesIn(unaryEltwiseParams));
INSTANTIATE_TEST_SUITE_P(HardsigmoidTests, OpModelHardsigmoidParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SinTests, OpModelSinParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(CosTests, OpModelCosParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(ExpTests, OpModelExpParam,
                         ::testing::ValuesIn(unaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(TanhTests, OpModelTanhParam,
                         ::testing::ValuesIn(tanhParams));

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
    const auto expectedLegal = std::get<4>(params).expectedLegal;

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, dimArg, keepDim,
        outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
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
using OpModelMaxParam = OpModelReductionParam<MaxOp>;
using OpModelMinParam = OpModelReductionParam<MinOp>;

TEST_P(OpModelSumParam, SumOp) { RunTest(); }
TEST_P(OpModelMeanParam, MeanOp) { RunTest(); }
TEST_P(OpModelMaxParam, MaxOp) { RunTest(); }
TEST_P(OpModelMinParam, MinOp) { RunTest(); }

// Test parameters for reduction operations
static const auto reductionParams =
    ::testing::Values(std::make_tuple(detail::interleavedN300X1024Dram,
                                      detail::interleavedN300X1024Dram,
                                      std::optional<llvm::SmallVector<int64_t>>{
                                          llvm::SmallVector<int64_t>{1}},
                                      true, detail::ExpectedResult{true}),
                      std::make_tuple(detail::interleavedN300X1024Dram,
                                      detail::interleavedN300X1024Dram,
                                      std::optional<llvm::SmallVector<int64_t>>{
                                          llvm::SmallVector<int64_t>{1, 2}},
                                      false, detail::ExpectedResult{false}),
                      std::make_tuple(detail::interleavedN300X1024Dram,
                                      detail::interleavedN300X1024Dram,
                                      std::optional<llvm::SmallVector<int64_t>>{
                                          llvm::SmallVector<int64_t>{1, 0}},
                                      false, detail::ExpectedResult{true}),
                      std::make_tuple(detail::interleavedN300X1024L1,
                                      detail::interleavedN300X1024Dram,
                                      std::optional<llvm::SmallVector<int64_t>>{
                                          llvm::SmallVector<int64_t>{1}},
                                      false, detail::ExpectedResult{true}));

INSTANTIATE_TEST_SUITE_P(SumTests, OpModelSumParam, reductionParams);

INSTANTIATE_TEST_SUITE_P(MeanTests, OpModelMeanParam, reductionParams);

INSTANTIATE_TEST_SUITE_P(MaxTests, OpModelMaxParam, reductionParams);

INSTANTIATE_TEST_SUITE_P(MinTests, OpModelMinParam, reductionParams);

TEST_F(OpModelTest, ArgMax) {
  const llvm::SmallVector<int64_t> tensorShape = {64, 64};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  // ArgMax requires ROW_MAJOR layout for both input and output
  // Note: L1 + ROW_MAJOR doesn't work (see tt-mlir issue #2976), so we use DRAM
  const TTNNLayoutAttr layoutDRAMRowMajor = CreateRowMajorLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  // Test case 1: no keepDim
  auto constraintsExp = OpModel<ArgMaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMRowMajor, 1, false, false,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<ArgMaxOp>::getOpRuntime(
      tensorShape, layoutDRAMRowMajor, 1, false, false, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  // Test case 2: with keepDim
  constraintsExp = OpModel<ArgMaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAMRowMajor, 1, true, false,
      layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<ArgMaxOp>::getOpRuntime(
      tensorShape, layoutDRAMRowMajor, 1, true, false, layoutDRAMRowMajor);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  // Test case 3: Different tensor shape, no keepDim
  const llvm::SmallVector<int64_t> tensorShape2 = {32, 128};
  const TTNNLayoutAttr layoutDRAMRowMajor2 = CreateRowMajorLayout(
      tensorShape2, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  constraintsExp = OpModel<ArgMaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape2, layoutDRAMRowMajor2, 1, false, false,
      layoutDRAMRowMajor2);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<ArgMaxOp>::getOpRuntime(
      tensorShape2, layoutDRAMRowMajor2, 1, false, false, layoutDRAMRowMajor2);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Prod) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300,
                                                  workerCoresN300};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1WSharded = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::WidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = op_model::OpModel<ProdOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, false, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = op_model::OpModel<ProdOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, false,
      layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  constraintsExp = op_model::OpModel<ProdOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 0, false,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());
}

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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_EQ(outputSize, 0);
  EXPECT_EQ(l1PeakSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, false,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, false,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1, -1, false,
      inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_dram, -1, false,
      inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
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

TEST_F(OpModelTest, Scatter) {
  const llvm::SmallVector<int64_t> inputShape = {256, 1024};
  const llvm::SmallVector<int64_t> indexSourceShape = {128, 1024};
  const int32_t dim = 0;
  const ttcore::ReduceTypeAttr reduceTypeAttr =
      ttcore::ReduceTypeAttr::get(&context, ttcore::ReduceType::Invalid);

  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayoutDRAM = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutL1 = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr indexLayoutDRAM = CreateTiledLayoutInt32(
      indexSourceShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr indexLayoutL1 = CreateTiledLayoutInt32(
      indexSourceShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr sourceLayoutDRAM = CreateTiledLayout(
      indexSourceShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr sourceLayoutL1 = CreateTiledLayout(
      indexSourceShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  // DRAM layouts
  auto constraintsExp = OpModel<ScatterOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayoutDRAM, indexSourceShape,
      indexLayoutDRAM, indexSourceShape, sourceLayoutDRAM, dim, reduceTypeAttr,
      inputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.peakL1MemorySize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<ScatterOp>::getOpRuntime(
      inputShape, inputLayoutDRAM, indexSourceShape, indexLayoutDRAM,
      indexSourceShape, sourceLayoutDRAM, dim, reduceTypeAttr, inputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  // L1 layouts
  constraintsExp = OpModel<ScatterOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayoutL1, indexSourceShape,
      indexLayoutL1, indexSourceShape, sourceLayoutL1, dim, reduceTypeAttr,
      inputLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.peakL1MemorySize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<ScatterOp>::getOpRuntime(
      inputShape, inputLayoutL1, indexSourceShape, indexLayoutL1,
      indexSourceShape, sourceLayoutL1, dim, reduceTypeAttr, inputLayoutL1);
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<ReshapeOp>::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = OpModel<ReshapeOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, {workerCoresN300 * 4, 256},
      layoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<ReshapeOp>::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutL1);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
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

  auto constraintsExp = OpModel<SliceStaticOp>::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, layoutDRAM, begins, ends, step,
      layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<SliceStaticOp>::getOpRuntime(
      inputTensorShape, layoutDRAM, begins, ends, step, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, SliceDynamic) {
  const llvm::SmallVector<int64_t> inputTensorShape = {4, 32, 32};
  const llvm::SmallVector<int64_t> beginsShape = {3};
  const llvm::SmallVector<int64_t> endsShape = {3};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayoutDRAM = CreateTiledLayout(
      inputTensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr beginsLayoutDRAM = CreateTiledLayout(
      beginsShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr endsLayoutDRAM = CreateTiledLayout(
      endsShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayoutDRAM = CreateTiledLayout(
      inputTensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  std::optional<llvm::SmallVector<int64_t>> step =
      llvm::SmallVector<int64_t>{1, 1, 1};

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<SliceDynamicOp>::getOpConstraints(
      CreateWorkerGrid(), inputTensorShape, inputLayoutDRAM, beginsShape,
      beginsLayoutDRAM, endsShape, endsLayoutDRAM, step, outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<SliceDynamicOp>::getOpRuntime(
      inputTensorShape, inputLayoutDRAM, beginsShape, beginsLayoutDRAM,
      endsShape, endsLayoutDRAM, step, outputLayoutDRAM);
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayouts[0]);

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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  ExpectLayoutsEQ(layoutDRAMRowMajor, opCstr.outputLayouts[0]);

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
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
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
      ShardOrientationAttr::get(&context, ShardOrientation::RowMajor));
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

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

TEST_F(OpModelTest, MorehCumSum) {
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

  auto constraintsExp = op_model::OpModel<MorehCumSumOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = op_model::OpModel<MorehCumSumOp>::getOpRuntime(
      tensorShape, layoutDRAM, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<MorehCumSumOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<MorehCumSumOp>::getOpRuntime(
      tensorShape, layoutDRAM, 0, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<MorehCumSumOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 0,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = op_model::OpModel<MorehCumSumOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, 0, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

// ==== ConcatenateHeadsOp Tests ====
class OpModelConcatenateHeadsParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,    // input tensor config
                     detail::TestTensor,    // output tensor config
                     detail::ExpectedResult // expected results
                     >> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<1>(params);
    const auto expectedLegal = std::get<2>(params).expectedLegal;

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<ConcatenateHeadsOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, outputLayout);

    // Check if the operation is expected to be legal
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

    if (expectedLegal) {
      auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
            outputLayoutResults] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      EXPECT_TRUE(outputLayoutResults.size());
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
  }
};

TEST_P(OpModelConcatenateHeadsParam, ConcatenateHeadsOp) { RunTest(); }

const std::initializer_list<
    std::tuple<detail::TestTensor, detail::TestTensor, detail::ExpectedResult>>
    concatenateHeadsParams = {
        // Test case 1: Small transformer L1 to L1
        std::make_tuple(detail::TestTensor{{1, 8, 512, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 512, 512},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::ExpectedResult{true}),

        // Test case 2: DRAM to DRAM configuration
        std::make_tuple(detail::TestTensor{{2, 12, 1024, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{2, 1024, 768},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::ExpectedResult{true}),

        // Test case 3: Mixed memory (DRAM input, L1 output)
        std::make_tuple(detail::TestTensor{{1, 16, 256, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 256, 512},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::ExpectedResult{true}),

        // Test case 4: Large transformer configuration
        std::make_tuple(detail::TestTensor{{4, 24, 2048, 128},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{4, 2048, 3072},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::ExpectedResult{true})};

INSTANTIATE_TEST_SUITE_P(ConcatenateHeadsTests, OpModelConcatenateHeadsParam,
                         ::testing::ValuesIn(concatenateHeadsParams));

TEST_F(OpModelTest, NLPConcatHeadsDecodeOp) {
  const int64_t batchSizeUnpadded = 2;
  const int64_t batchSizePadded = 32;
  const int64_t seqLen = 1;
  const int64_t numHeadsPadded = 32;
  const int64_t headDim = 128;
  const uint32_t numHeadsUnpadded = 8;

  const llvm::SmallVector<int64_t> inputShape = {seqLen, batchSizeUnpadded,
                                                 numHeadsPadded, headDim};
  const llvm::SmallVector<int64_t> outputShape = {seqLen, 1, batchSizePadded,
                                                  numHeadsUnpadded * headDim};
  const llvm::SmallVector<int64_t> virtualGridHeightSharded = {
      batchSizeUnpadded, 1};
  const llvm::SmallVector<int64_t> virtualGridWidthSharded = {
      1, batchSizeUnpadded};

  ttcore::GridAttr workerGrid = CreateWorkerGrid(gridShapeHwN300);
  TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::HeightSharded,
      virtualGridHeightSharded);
  TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, BufferType::L1, TensorMemoryLayout::WidthSharded,
      virtualGridWidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp =
      op_model::OpModel<NLPConcatHeadsDecodeOp>::getOpConstraints(
          CreateWorkerGrid(), inputShape, inputLayout, numHeadsUnpadded,
          outputLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_EQ(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = op_model::OpModel<NLPConcatHeadsDecodeOp>::getOpRuntime(
      inputShape, inputLayout, numHeadsUnpadded, outputLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

// ==== SplitQueryKeyValueAndSplitHeadsOp Tests ====
class OpModelSplitQueryKeyValueAndSplitHeadsParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,    // input tensor config
                     detail::TestTensor,    // output query tensor config
                     detail::TestTensor,    // output key tensor config
                     detail::TestTensor,    // output value tensor config
                     uint32_t,              // num_heads
                     bool,                  // transpose_key
                     detail::ExpectedResult // expected results
                     >> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);
    const auto [outputQueryShape, outputQueryTensorLayout,
                outputQueryBufferType, outputQueryVirtualGrid] =
        std::get<1>(params);
    const uint32_t numHeads = std::get<4>(params);
    const bool transposeKey = std::get<5>(params);
    const auto expectedLegal = std::get<6>(params).expectedLegal;

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputQueryLayout =
        CreateTiledLayout(outputQueryShape, outputQueryBufferType,
                          outputQueryTensorLayout, outputQueryVirtualGrid);

    auto constraintsExp =
        OpModel<SplitQueryKeyValueAndSplitHeadsOp>::getOpConstraints(
            CreateWorkerGrid(), inputShape, inputLayout,
            std::nullopt, // inputKVShape
            std::nullopt, // inputKVLayout
            numHeads,
            std::nullopt, // numKVHeads
            transposeKey, outputQueryLayout);

    // Check if the operation is expected to be legal
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

    if (expectedLegal) {
      auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
            outputLayoutResults] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      EXPECT_TRUE(outputLayoutResults.size());
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
  }
};

TEST_P(OpModelSplitQueryKeyValueAndSplitHeadsParam,
       SplitQueryKeyValueAndSplitHeadsOp) {
  RunTest();
}
const std::initializer_list<
    std::tuple<detail::TestTensor,    // input tensor config
               detail::TestTensor,    // output query tensor config
               detail::TestTensor,    // output key tensor config
               detail::TestTensor,    // output value tensor config
               uint32_t,              // num_heads
               bool,                  // transpose_key
               detail::ExpectedResult // expected results
               >>
    splitQueryKeyValueAndSplitHeadsParams = {
        // Test case 1: Small transformer L1 to L1 (based on interface test
        // config)
        std::make_tuple(detail::TestTensor{{8, 16, 3072},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{8, 32, 16, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{8, 32, 16, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{8, 32, 16, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        32, false, detail::ExpectedResult{true}),

        // Test case 2: DRAM to DRAM configuration
        std::make_tuple(detail::TestTensor{{2, 1024, 2304},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{2, 12, 1024, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{2, 12, 1024, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{2, 12, 1024, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        12, false, detail::ExpectedResult{true}),

        // Test case 3: Mixed memory (DRAM input, L1 output)
        std::make_tuple(detail::TestTensor{{1, 256, 1536},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 16, 256, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 16, 256, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 16, 256, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        16, false, detail::ExpectedResult{true}),

        // Test case 4: Large transformer configuration
        std::make_tuple(detail::TestTensor{{4, 2048, 9216},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{4, 24, 2048, 128},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{4, 24, 2048, 128},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{4, 24, 2048, 128},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        24, false, detail::ExpectedResult{true}),

        // Test case 5: With transpose_key enabled
        std::make_tuple(detail::TestTensor{{1, 512, 1536},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 16, 512, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 16, 512, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        detail::TestTensor{{1, 16, 512, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        16, true, detail::ExpectedResult{true})};

INSTANTIATE_TEST_SUITE_P(
    SplitQueryKeyValueAndSplitHeadsTests,
    OpModelSplitQueryKeyValueAndSplitHeadsParam,
    ::testing::ValuesIn(splitQueryKeyValueAndSplitHeadsParams));

TEST_F(OpModelTest, RepeatInterleave) {
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

  auto constraintsExp = op_model::OpModel<RepeatInterleaveOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 2, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = op_model::OpModel<RepeatInterleaveOp>::getOpRuntime(
      tensorShape, layoutDRAM, 2, 0, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<RepeatInterleaveOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 2, 0, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<RepeatInterleaveOp>::getOpRuntime(
      tensorShape, layoutDRAM, 2, 0, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<RepeatInterleaveOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 2, 0,
      layoutL1WSharded);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<RepeatInterleaveOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, 2, 0, layoutL1WSharded);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Repeat) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  std::vector<int64_t> repeatDimsVec = {2, 1};
  llvm::ArrayRef<int64_t> repeatDims(repeatDimsVec);

  // Output shape after repeat: first dim is doubled
  const llvm::SmallVector<int64_t> outputShape = {2 * workerCoresN300, 1024};
  const TTNNLayoutAttr outputLayoutL1WSharded = CreateTiledLayout(
      outputShape, BufferType::L1, TensorMemoryLayout::WidthSharded);

  auto constraintsExp = op_model::OpModel<RepeatOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, repeatDims, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = op_model::OpModel<RepeatOp>::getOpRuntime(
      tensorShape, layoutDRAM, repeatDims, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<RepeatOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, repeatDims,
      layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  // Output is L1 Interleaved, so expect L1 buffer usage
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GE(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<RepeatOp>::getOpRuntime(
      tensorShape, layoutDRAM, repeatDims, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<RepeatOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, repeatDims,
      outputLayoutL1WSharded);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<RepeatOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, repeatDims, outputLayoutL1WSharded);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}

TEST_F(OpModelTest, Pad) {
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

  std::vector<int32_t> paddingVec = {0, 2, 0, 2};
  llvm::ArrayRef<int32_t> padding(paddingVec);
  llvm::APFloat padValue = llvm::APFloat(1.0f);

  auto constraintsExp = op_model::OpModel<PadOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, padding, padValue, false,
      layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = op_model::OpModel<PadOp>::getOpRuntime(
      tensorShape, layoutDRAM, padding, padValue, false, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<PadOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, padding, padValue, false,
      layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = op_model::OpModel<PadOp>::getOpRuntime(
      tensorShape, layoutDRAM, padding, padValue, false, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<PadOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, padding, padValue,
      false, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = op_model::OpModel<PadOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, padding, padValue, false,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

TEST_F(OpModelTest, Sort) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300,
                                                  workerCoresN300};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1WSharded = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::WidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = op_model::OpModel<SortOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, false, false, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  // Sort has two outputs: values and indices.
  ASSERT_EQ(opCstr.outputLayouts.size(), 2);

  auto runtimeExp = op_model::OpModel<SortOp>::getOpRuntime(
      tensorShape, layoutDRAM, 0, false, false, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<SortOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutDRAM, 0, false, false,
      layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
  // Sort has two outputs: values and indices.
  ASSERT_EQ(opCstr.outputLayouts.size(), 2);

  runtimeExp = op_model::OpModel<SortOp>::getOpRuntime(
      tensorShape, layoutDRAM, 0, false, false, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = op_model::OpModel<SortOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, layoutL1Interleaved, 0, false, false,
      layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = op_model::OpModel<SortOp>::getOpRuntime(
      tensorShape, layoutL1Interleaved, 0, false, false, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

TEST_F(OpModelTest, MaxPool2dWithIndices) {
  const llvm::SmallVector<int64_t> inputShape = {1, 1, 128 * 128, 32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr layoutDRAM = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1Interleaved = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr layoutL1WSharded = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::WidthSharded);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  // MaxPool2dWithIndices parameters
  int32_t batchSize = 1;
  int32_t inputHeight = 128;
  int32_t inputWidth = 128;
  int32_t inputChannels = 32;
  llvm::SmallVector<int32_t> kernelSize = {3, 3};
  llvm::SmallVector<int32_t> stride = {2, 2};
  llvm::SmallVector<int32_t> padding = {0, 0};
  llvm::SmallVector<int32_t> dilation = {1, 1};
  bool ceilMode = false;
  bool reallocateHaloOutput = true;
  bool deallocateInput = false;
  bool returnIndices = true;

  // Right now, OpRuntime tests aren't supported for MaxPool2dWithIndicesOp
  // TODO(umales): Add OpRuntime tests for MaxPool2dWithIndicesOp, once
  // https://github.com/tenstorrent/tt-metal/issues/31646 is resolved.

  auto constraintsExp =
      op_model::OpModel<MaxPool2dWithIndicesOp>::getOpConstraints(
          CreateWorkerGrid(), inputShape, layoutDRAM, batchSize, inputHeight,
          inputWidth, inputChannels, kernelSize, stride, padding, dilation,
          ceilMode, reallocateHaloOutput, deallocateInput, returnIndices,
          std::nullopt, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.peakL1MemorySize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = op_model::OpModel<MaxPool2dWithIndicesOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutDRAM, batchSize, inputHeight,
      inputWidth, inputChannels, kernelSize, stride, padding, dilation,
      ceilMode, reallocateHaloOutput, deallocateInput, returnIndices,
      std::nullopt, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.peakL1MemorySize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  constraintsExp = op_model::OpModel<MaxPool2dWithIndicesOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, layoutL1Interleaved, batchSize,
      inputHeight, inputWidth, inputChannels, kernelSize, stride, padding,
      dilation, ceilMode, reallocateHaloOutput, deallocateInput, returnIndices,
      std::nullopt, layoutL1WSharded);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());
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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_hs, -2, false,
      inputLayout_l1_i);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  constraintsExp = OpModel<SoftmaxOp>::getOpConstraints(
      CreateWorkerGrid(), tensorShape, inputLayout_l1_i, -2, false,
      inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

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
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
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
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

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
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();

      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
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

  void RunTestInt32() {
    const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
                inputVirtualGridA] = GetParam().inputA;
    const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
                inputVirtualGridB] = GetParam().inputB;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    const TTNNLayoutAttr inputLayoutA = CreateTiledLayoutInt32(
        inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
    const TTNNLayoutAttr inputLayoutB = CreateTiledLayoutInt32(
        inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
    const TTNNLayoutAttr outputLayout = CreateTiledLayoutInt32(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB,
        inputLayoutB, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();

      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
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

struct GeluBackwardParam {
  detail::TestTensor input;
  detail::TestTensor grad;
  std::string approximate;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};

class OpModelGeluBackwardParam
    : public OpModelTest,
      public testing::WithParamInterface<GeluBackwardParam> {
protected:
  void RunTest() {
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = GetParam().input;
    const auto [gradShape, gradTensorLayout, gradBufferType, gradVirtualGrid] =
        GetParam().grad;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto approximate = GetParam().approximate;
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;
    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr gradLayout = CreateTiledLayout(
        gradShape, gradBufferType, gradTensorLayout, gradVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<GeluBackwardOp>::getOpConstraints(
        CreateWorkerGrid(), gradShape, gradLayout, inputShape, inputLayout,
        approximate, outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<GeluBackwardOp>::getOpRuntime(
        gradShape, gradLayout, inputShape, inputLayout, approximate,
        outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelGeluBackwardParam, GeluBackwardOp) { RunTest(); }

const auto geluBackwardOpTestValues = testing::Values(
    // === Mixed memory, "none" approximation ===
    GeluBackwardParam{
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        "none",
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::ExpectedResult{true}},

    // === Mixed memory, "tanh" approximation ===
    GeluBackwardParam{
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        "tanh",
        detail::TestTensor{
            {1, 1, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::ExpectedResult{true}},

    // === All L1, "none" approximation ===
    GeluBackwardParam{
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        "none",
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::ExpectedResult{true}},

    // === All L1, "tanh" approximation ===
    GeluBackwardParam{
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        "tanh",
        detail::TestTensor{
            {1, 1, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::ExpectedResult{true}});
INSTANTIATE_TEST_SUITE_P(GeluBackwardTests, OpModelGeluBackwardParam,
                         geluBackwardOpTestValues);

// Type aliases for binary operations
using OpModelAddParam = OpModelBinaryEltwiseParam<AddOp>;
using OpModelMultiplyParam = OpModelBinaryEltwiseParam<MultiplyOp>;
using OpModelLogicalRightShiftParam =
    OpModelBinaryEltwiseParam<LogicalRightShiftOp>;
using OpModelLogicalLeftShiftParam =
    OpModelBinaryEltwiseParam<LogicalLeftShiftOp>;
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
using OpModelPowParam = OpModelBinaryEltwiseParam<PowTensorOp>;
using OpModelBitwiseAndParam = OpModelBinaryEltwiseParam<BitwiseAndOp>;
using OpModelBitwiseOrParam = OpModelBinaryEltwiseParam<BitwiseOrOp>;
using OpModelBitwiseXorParam = OpModelBinaryEltwiseParam<BitwiseXorOp>;
using OpModelRemainderParam = OpModelBinaryEltwiseParam<RemainderOp>;
using OpModelAtan2Param = OpModelBinaryEltwiseParam<Atan2Op>;

TEST_P(OpModelAddParam, AddOp) { RunTest(); }
TEST_P(OpModelMultiplyParam, MultiplyOp) { RunTest(); }
TEST_P(OpModelLogicalRightShiftParam, LogicalRightShiftOp) { RunTestInt32(); }
TEST_P(OpModelLogicalLeftShiftParam, LogicalLeftShiftOp) { RunTestInt32(); }
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
TEST_P(OpModelBitwiseAndParam, BitwiseAndOp) { RunTestInt32(); }
TEST_P(OpModelBitwiseOrParam, BitwiseOrOp) { RunTestInt32(); }
TEST_P(OpModelBitwiseXorParam, BitwiseXorOp) { RunTestInt32(); }
TEST_P(OpModelPowParam, PowTensorOp) { RunTest(); }
TEST_P(OpModelRemainderParam, RemainderOp) { RunTest(); }
TEST_P(OpModelAtan2Param, Atan2Op) { RunTest(); }

const std::initializer_list<BinaryEltwiseParam> binaryEltwiseParams = {
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024Dram, detail::interleaved2048X2048Dram,
     detail::interleaved2048X2048Dram,
     detail::ExpectedResult{false}}, // incompatible dimensions at the input
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
     detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
    {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
     detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
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
     detail::ExpectedResult{true}},
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
     detail::ExpectedResult{true}},
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
     detail::ExpectedResult{true}}};

// Power and Remainder tests are mostly similar to other binary ops, but they
// have subtle differences in terms of their memory footprint after metal
// uplift. As a workaround, we generate separate test parameters for these two
// ops. However, this will be completely resolved when we change the way we test
// OpModelLib (See this issue:
// https://github.com/tenstorrent/tt-mlir/issues/4288).
const std::initializer_list<BinaryEltwiseParam>
    binaryEltwiseParamsForRemainderAndPow = {
        {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
         detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024Dram, detail::interleaved2048X2048Dram,
         detail::interleaved2048X2048Dram,
         detail::ExpectedResult{false}}, // incompatible dimensions at the input
        {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
         detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
         detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
         detail::interleavedN300X1024Dram, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024L1, detail::interleavedN300X1024L1,
         detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024Dram, detail::interleavedN300X1024L1,
         detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024L1, detail::interleavedN300X1024Dram,
         detail::interleavedN300X1024L1, detail::ExpectedResult{true}},
        {detail::interleavedN300X1024Dram, detail::interleavedN300X1024Dram,
         detail::interleavedN300X1024L1, detail::ExpectedResult{true}}};

::testing::internal::ParamGenerator<BinaryEltwiseParam>
generateBinaryEltwiseParams(std::initializer_list<BinaryEltwiseParam> values) {
  return ::testing::ValuesIn(values);
}

::testing::internal::ParamGenerator<BinaryEltwiseParam>
generateBinaryBitwiseParams(std::initializer_list<BinaryEltwiseParam> values) {
  return ::testing::ValuesIn(values);
}

::testing::internal::ParamGenerator<BinaryEltwiseParam>
generateBinaryEltwiseParamsSameLayout(
    std::initializer_list<BinaryEltwiseParam> values) {
  std::vector<BinaryEltwiseParam> newValues;
  for (const auto &v : values) {
    newValues.emplace_back(v);
    if ((newValues.back().inputA.layout != newValues.back().inputB.layout) ||
        (newValues.back().inputA.layout != newValues.back().output.layout)) {
      newValues.back().expectedResult.expectedLegal = false;
    }
  }
  return ::testing::ValuesIn(newValues);
}

INSTANTIATE_TEST_SUITE_P(AddTests, OpModelAddParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MulTests, OpModelMultiplyParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(SubtractTests, OpModelSubtractParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalRightShiftTests, OpModelLogicalRightShiftParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalLeftShiftTests, OpModelLogicalLeftShiftParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MaximumTests, OpModelMaximumParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(MinimumTests, OpModelMinimumParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(DivideTests, OpModelDivideParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

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
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalOrTests, OpModelLogicalOrParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(LogicalXorTests, OpModelLogicalXorParam,
                         generateBinaryEltwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(BitwiseAndTests, OpModelBitwiseAndParam,
                         generateBinaryBitwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(BitwiseOrTests, OpModelBitwiseOrParam,
                         generateBinaryBitwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(BitwiseXorTests, OpModelBitwiseXorParam,
                         generateBinaryBitwiseParams(binaryEltwiseParams));

INSTANTIATE_TEST_SUITE_P(
    PowTests, OpModelPowParam,
    generateBinaryEltwiseParams(binaryEltwiseParamsForRemainderAndPow));

INSTANTIATE_TEST_SUITE_P(
    RemainderTests, OpModelRemainderParam,
    generateBinaryEltwiseParams(binaryEltwiseParamsForRemainderAndPow));

INSTANTIATE_TEST_SUITE_P(
    Atan2Tests, OpModelAtan2Param,
    generateBinaryEltwiseParamsSameLayout(binaryEltwiseParams));

class OpModelPowScalarParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,            // input
                     detail::TestTensor,            // output
                     std::variant<float, uint32_t>, // exponent
                     detail::ExpectedResult         // expected legal
                     >> {};

TEST_P(OpModelPowScalarParam, PowScalarParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  mlir::Attribute exponent;
  if (std::holds_alternative<float>(std::get<2>(params))) {
    exponent = builder.getF32FloatAttr(std::get<float>(std::get<2>(params)));
  } else {
    exponent =
        builder.getI32IntegerAttr(std::get<uint32_t>(std::get<2>(params)));
  }
  const auto expectedLegal = std::get<3>(params).expectedLegal;

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = OpModel<PowScalarOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, exponent, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();

    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = OpModel<PowScalarOp>::getOpRuntime(inputShape, inputLayout,
                                                       exponent, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    PowScalarTests, OpModelPowScalarParam,
    ::testing::Values(
        std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 1, 128 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        float(2.0), detail::ExpectedResult{true}),
        std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        detail::TestTensor{{1, 1, 128 * 128, 32},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::L1},
                        uint32_t(3), detail::ExpectedResult{true})));

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
  const auto expectedLegal = std::get<5>(params).expectedLegal;

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
      biasShape, biasLayout, outputLayout, false, false,
      /*programConfig=*/std::nullopt);

  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
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
}

INSTANTIATE_TEST_SUITE_P(
    LinearInterleavedTests, OpModelLinearParam,
    ::testing::Values(
        std::make_tuple(
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            detail::interleaved2048X2048Dram, detail::inerleaved2048X2048L1,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::interleaved2048X2048Dram, detail::inerleaved2048X2048L1,
            detail::inerleaved2048X2048L1, detail::interleaved2048X2048Dram,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::interleaved2048X2048Dram, detail::inerleaved2048X2048L1,
            detail::inerleaved2048X2048L1, detail::inerleaved2048X2048L1,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::inerleaved2048X2048L1, detail::interleaved2048X2048Dram,
            detail::inerleaved2048X2048L1, detail::interleaved2048X2048Dram,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::inerleaved2048X2048L1, detail::interleaved2048X2048Dram,
            detail::inerleaved2048X2048L1, detail::inerleaved2048X2048L1,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::inerleaved2048X2048L1, detail::inerleaved2048X2048L1,
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true}),
        std::make_tuple(
            detail::inerleaved2048X2048L1, detail::inerleaved2048X2048L1,
            detail::interleaved2048X2048Dram, detail::inerleaved2048X2048L1,
            llvm::SmallVector<int64_t>{8, 8}, detail::ExpectedResult{true})));

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
                        detail::ExpectedResult{true}),
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
                        detail::ExpectedResult{true}),
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
                        detail::ExpectedResult{true}),
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
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{true}),
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
                        detail::ExpectedResult{true})));

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
  const auto expectedLegal = std::get<4>(params).expectedLegal;

  const TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = OpModel<MatmulOp>::getOpConstraints(
      CreateWorkerGrid(), inputShapeA, inputLayoutA, inputShapeB, inputLayoutB,
      outputLayout, false, false, /*programConfig=*/std::nullopt);

  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
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
    ::testing::Values(std::make_tuple(detail::interleaved2048X2048Dram,
                                      detail::interleaved2048X2048Dram,
                                      detail::interleaved2048X2048Dram,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::interleaved2048X2048Dram,
                                      detail::interleaved2048X2048Dram,
                                      detail::inerleaved2048X2048L1,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::interleaved2048X2048Dram,
                                      detail::inerleaved2048X2048L1,
                                      detail::interleaved2048X2048Dram,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::interleaved2048X2048Dram,
                                      detail::inerleaved2048X2048L1,
                                      detail::inerleaved2048X2048L1,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::inerleaved2048X2048L1,
                                      detail::interleaved2048X2048Dram,
                                      detail::interleaved2048X2048Dram,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::inerleaved2048X2048L1,
                                      detail::interleaved2048X2048Dram,
                                      detail::inerleaved2048X2048L1,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::inerleaved2048X2048L1,
                                      detail::inerleaved2048X2048L1,
                                      detail::interleaved2048X2048Dram,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true}),
                      std::make_tuple(detail::inerleaved2048X2048L1,
                                      detail::inerleaved2048X2048L1,
                                      detail::inerleaved2048X2048L1,
                                      llvm::SmallVector<int64_t>{8, 8},
                                      detail::ExpectedResult{true})));

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
                        detail::ExpectedResult{true}),
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
                            true}), // matmul bug shards to less cores
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
                        detail::ExpectedResult{true}),
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
            llvm::SmallVector<int64_t>{7, 8}, detail::ExpectedResult{true}),
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
                        detail::ExpectedResult{true})));

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
  const auto expectedLegal = std::get<13>(params).expectedLegal;

  const TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

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
      std::nullopt, deviceConfig, std::nullopt, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(totalPeakSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = OpModel<Conv2dOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, dilation, groups, std::nullopt,
      deviceConfig, std::nullopt, outputLayout);
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
                        detail::ExpectedResult{true}),
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
                        detail::ExpectedResult{true})));

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
  GTEST_SKIP()
      << "Skipping ConvTranspose2d test until metal fix lands."
      << "Tracked here: https://github.com/tenstorrent/tt-metal/issues/35028";
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
  const auto expectedLegal = std::get<14>(params).expectedLegal;

  const TTNNLayoutAttr inputLayout =
      CreateRowMajorLayout(inputShape, inputBufferType, inputTensorLayout,
                           inputVirtualGrid, GetPhysicalGridSize());
  const TTNNLayoutAttr weightLayout =
      CreateRowMajorLayout(weightShape, weightBufferType, weightTensorLayout,
                           weightVirtualGrid, GetPhysicalGridSize());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = OpModel<ConvTranspose2dOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, output_padding,
      dilation, groups, std::nullopt, std::nullopt, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(totalPeakSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = OpModel<ConvTranspose2dOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, output_padding, dilation,
      groups, std::nullopt, std::nullopt, outputLayout);
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
        detail::ExpectedResult{true})));

class OpModelConv3dParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // input
                     detail::TestTensor,         // weight
                     detail::TestTensor,         // output
                     uint32_t,                   // in_channels
                     uint32_t,                   // out_channels
                     uint32_t,                   // batch_size
                     uint32_t,                   // input_depth
                     uint32_t,                   // input_height
                     uint32_t,                   // input_width
                     llvm::SmallVector<int32_t>, // kernel_size
                     llvm::SmallVector<int32_t>, // stride
                     llvm::SmallVector<int32_t>, // padding
                     uint32_t,                   // groups
                     llvm::StringRef,            // padding_mode
                     detail::ExpectedResult>> {};

TEST_P(OpModelConv3dParam, Conv3d) {
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
  const auto input_depth = std::get<6>(params);
  const auto input_height = std::get<7>(params);
  const auto input_width = std::get<8>(params);
  const auto kernel_size = std::get<9>(params);
  const auto stride = std::get<10>(params);
  const auto padding = std::get<11>(params);
  const auto groups = std::get<12>(params);
  const auto padding_mode = std::get<13>(params);
  const auto expectedLegal = std::get<14>(params).expectedLegal;

  // Conv3d requires BF16 data type and specific layouts
  const TTNNLayoutAttr inputLayout = CreateRowMajorLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid,
      GetPhysicalGridSize(), builder.getBF16Type());
  const TTNNLayoutAttr weightLayout = CreateTiledLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid,
      GetPhysicalGridSize(), builder.getBF16Type());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  DeviceComputeKernelConfigAttr deviceConfig =
      DeviceComputeKernelConfigAttr::get(
          &context, /*mathFidelity=*/MathFidelity::LoFi,
          /*mathApproxMode=*/::mlir::BoolAttr::get(&context, true),
          /*fp32DestAccEn=*/::mlir::BoolAttr::get(&context, true),
          /*packerL1Acc=*/::mlir::BoolAttr::get(&context, true),
          /*dstFullSyncEn=*/::mlir::BoolAttr::get(&context, true));

  auto constraintsExp = OpModel<Conv3dOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      std::nullopt, std::nullopt, in_channels, out_channels, batch_size,
      input_depth, input_height, input_width, kernel_size, stride, padding,
      groups, padding_mode, std::nullopt, std::nullopt, deviceConfig,
      outputLayout);
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    // Conv3d (experimental) ignores the requested L1 output memory config and
    // forces output to DRAM (copies input's memory config). Therefore:
    // - outputSize (l1_output_buffer_per_core) = 0 (output is in DRAM, not L1)
    // - l1PeakSize (l1_buffers_peak_per_core) = 0 (no intermediate L1 tensors)
    // - cbSize (cb_peak_size_per_core) > 0 (circular buffers ARE allocated in
    // L1) Circular buffers are staging queues in L1 SRAM that bridge DRAM and
    // compute engines. They're mandatory for all compute operations and must be
    // in L1 because the unpacker/packer hardware can only access L1, not DRAM
    // directly. See: ttnn/api/ttnn/graph/graph_query_op_constraints.hpp:124
    //      (query returns 0 for l1_output_buffer_per_core when
    //      output.buffer()->is_dram())
    EXPECT_GT(cbSize, 0);
  } else {
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = OpModel<Conv3dOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_depth,
      input_height, input_width, kernel_size, stride, padding, groups,
      padding_mode, std::nullopt, std::nullopt, deviceConfig, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    const auto runtime = runtimeExp.get();
    EXPECT_GT(runtime, 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    Conv3dTests, OpModelConv3dParam,
    ::testing::Values(std::make_tuple(
        // tt-metal Conv3d expects input in [N, D, H, W, C] format (channels
        // LAST)
        detail::TestTensor{{1, 5, 10, 10, 3}, // [N, D, H, W, C]
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        // Weight must be 2D: [kD*kH*kW*C_in, C_out] where C_in is input
        // channels patch_size = 3*3*3*3 = 81, out_channels = 64 (multiple of
        // 32)
        detail::TestTensor{
            {81, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        // Output dims: D_out=(5-3)/1+1=3, H_out=(10-3)/1+1=8,
        // W_out=(10-3)/1+1=8
        detail::TestTensor{{1, 3, 8, 8, 64}, // [N, D_out, H_out, W_out, C_out]
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        3, 64, 1, 5, 10, 10, llvm::SmallVector<int32_t>{3, 3, 3},
        llvm::SmallVector<int32_t>{1, 1, 1},
        llvm::SmallVector<int32_t>{0, 0, 0}, 1, "zeros",
        detail::ExpectedResult{true})));

template <typename OpTy>
class OpModelPool2DParam
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
                     bool,                       // reallocate_halo_output
                     bool                        // expected legal
                     >> {
protected:
  void RunTest() {
    auto params = this->GetParam();
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
    const auto reallocateHaloOutput = std::get<11>(params);
    const auto expectedLegal = std::get<12>(params);

    const TTNNLayoutAttr inputLayout = this->CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = this->CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<OpTy>::getOpConstraints(
        this->CreateWorkerGrid(), inputShape, inputLayout, batchSize,
        inputHeight, inputWidth, inputChannels, kernelSize, stride, padding,
        dilation, ceilMode, reallocateHaloOutput, std::nullopt, outputLayout);
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

    if (constraintsExp) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GT(cbSize, 0);
      EXPECT_GT(l1PeakSize, 0);
      EXPECT_EQ(outputSize, 0);
    } else {
      // Must clean up the error
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<OpTy>::getOpRuntime(
        inputShape, inputLayout, batchSize, inputHeight, inputWidth,
        inputChannels, kernelSize, stride, padding, dilation, ceilMode,
        reallocateHaloOutput, std::nullopt, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (runtimeExp) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

// Shared test values for Pool2D operations (MaxPool2D and AvgPool2D)
const auto pool2DTestValues = ::testing::Values(
    std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 64 * 64, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    1, 128, 128, 32, llvm::SmallVector<int32_t>{2, 2},
                    llvm::SmallVector<int32_t>{2, 2},
                    llvm::SmallVector<int32_t>{0, 0},
                    llvm::SmallVector<int32_t>{1, 1}, false, false, true),
    std::make_tuple(detail::TestTensor{{1, 1, 256 * 256, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 64 * 128, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    1, 256, 256, 32, llvm::SmallVector<int32_t>{3, 3},
                    llvm::SmallVector<int32_t>{4, 2},
                    llvm::SmallVector<int32_t>{0, 0},
                    llvm::SmallVector<int32_t>{1, 1}, false, false, true),
    std::make_tuple(detail::TestTensor{{1, 1, 17 * 21, 22},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 4 * 10, 22},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    1, 17, 21, 22, llvm::SmallVector<int32_t>{3, 3},
                    llvm::SmallVector<int32_t>{4, 2},
                    llvm::SmallVector<int32_t>{0, 0},
                    llvm::SmallVector<int32_t>{1, 1}, false, false, true),
    std::make_tuple(detail::TestTensor{{1, 1, 17 * 21, 22},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 4 * 11, 22},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    1, 17, 21, 22, llvm::SmallVector<int32_t>{3, 3},
                    llvm::SmallVector<int32_t>{4, 2},
                    llvm::SmallVector<int32_t>{0, 0, 1, 1},
                    llvm::SmallVector<int32_t>{1, 1}, false, false, true));

// MaxPool2D tests
class OpModelMaxPool2DParam : public OpModelPool2DParam<MaxPool2dOp> {};
TEST_P(OpModelMaxPool2DParam, MaxPool2DParam) { RunTest(); }
INSTANTIATE_TEST_SUITE_P(MaxPool2DTests, OpModelMaxPool2DParam,
                         pool2DTestValues);

// AvgPool2D tests
class OpModelAvgPool2DParam : public OpModelPool2DParam<AvgPool2dOp> {};
TEST_P(OpModelAvgPool2DParam, AvgPool2DParam) { RunTest(); }
INSTANTIATE_TEST_SUITE_P(AvgPool2DTests, OpModelAvgPool2DParam,
                         pool2DTestValues);

class OpModelGlobalAvgPool2dParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,    // input
                     detail::TestTensor,    // output
                     detail::ExpectedResult // expected result
                     >> {
protected:
  void RunTest() {
    auto params = GetParam();
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = std::get<0>(params);
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<1>(params);
    const auto expectedLegal = std::get<2>(params).expectedLegal;

    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<GlobalAvgPool2dOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, std::nullopt,
        outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<GlobalAvgPool2dOp>::getOpRuntime(
        inputShape, inputLayout, std::nullopt, outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelGlobalAvgPool2dParam, GlobalAvgPool2dOp) { RunTest(); }

// Test parameters for GlobalAvgPool2d operation
const auto globalAvgPool2dTestValues = testing::Values(
    // Test case 1: Basic 128x128 input with 32 channels (DRAM to DRAM)
    std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 1, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::ExpectedResult{true}),

    // Test case 2: DRAM input to L1 output
    std::make_tuple(detail::TestTensor{{1, 1, 128 * 128, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 1, 32},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::L1},
                    detail::ExpectedResult{true}),

    // Test case 3: L1 input to DRAM output
    std::make_tuple(detail::TestTensor{{1, 1, 64 * 64, 64},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::L1},
                    detail::TestTensor{{1, 1, 1, 64},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::ExpectedResult{true}),

    // Test case 4: Both L1 (most memory intensive)
    std::make_tuple(detail::TestTensor{{1, 1, 64 * 64, 64},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::L1},
                    detail::TestTensor{{1, 1, 1, 64},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::L1},
                    detail::ExpectedResult{true}),

    // Test case 5: Larger input (256x256 with 128 channels)
    std::make_tuple(detail::TestTensor{{1, 1, 256 * 256, 128},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{1, 1, 1, 128},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::ExpectedResult{true}),

    // Test case 6: Batch size > 1
    std::make_tuple(detail::TestTensor{{2, 1, 32 * 32, 16},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::DRAM},
                    detail::TestTensor{{2, 1, 1, 16},
                                       TensorMemoryLayout::Interleaved,
                                       BufferType::L1},
                    detail::ExpectedResult{true}));

// Test fails: https://github.com/tenstorrent/tt-mlir/issues/5313
INSTANTIATE_TEST_SUITE_P(GlobalAvgPool2dTests, OpModelGlobalAvgPool2dParam,
                         globalAvgPool2dTestValues);

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

  auto constraintsExp = op_model::OpModel<LeakyReluOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, slope, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

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

  auto constraintsExp = OpModel<ClampScalarOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, minVal, maxVal,
      outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

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

class OpModelClampTensorParam : public OpModelTest,
                                public testing::WithParamInterface<
                                    std::tuple<detail::TestTensor, // input
                                               detail::TestTensor, // output
                                               detail::TestTensor, // min
                                               detail::TestTensor, // max
                                               bool // expected legal
                                               >> {};

TEST_P(OpModelClampTensorParam, ClampTensorParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto [minShape, minTensorLayout, minBufferType, minVirtualGrid] =
      std::get<2>(params);
  const auto [maxShape, maxTensorLayout, maxBufferType, maxVirtualGrid] =
      std::get<3>(params);
  const auto expectedLegal = std::get<4>(params);

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);
  const TTNNLayoutAttr minLayout = CreateTiledLayout(
      minShape, minBufferType, minTensorLayout, minVirtualGrid);
  const TTNNLayoutAttr maxLayout = CreateTiledLayout(
      maxShape, maxBufferType, maxTensorLayout, maxVirtualGrid);

  auto constraintsExp = OpModel<ClampTensorOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, minShape, minLayout,
      maxShape, maxLayout, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = OpModel<ClampTensorOp>::getOpRuntime(
      inputShape, inputLayout, minShape, minLayout, maxShape, maxLayout,
      outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(ClampTensorTests, OpModelClampTensorParam,
                         ::testing::Values(std::make_tuple(
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::DRAM},
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::L1},
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::DRAM},
                             detail::TestTensor{{1, 1, 128 * 128, 32},
                                                TensorMemoryLayout::Interleaved,
                                                BufferType::L1},
                             true)));

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

  auto constraintsExp = OpModel<PermuteOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, permutation, padValue,
      outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(l1PeakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

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

  auto constraintsExp = OpModel<UpsampleOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, scaleFactor, mode,
      outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_EQ(l1PeakSize, 0);
    EXPECT_EQ(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

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
    const auto expectedLegal = expected.expectedLegal;

    const TTNNLayoutAttr inputTiledLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputLayout, inputVirtualGrid);
    const TTNNLayoutAttr weightTiledLayout = CreateTiledLayout(
        weightShape, weightBufferType, weightLayout, weightVirtualGrid);
    const TTNNLayoutAttr outputTiledLayout = CreateTiledLayout(
        {inputShape[0], inputShape[1], weightShape[1]}, BufferType::L1,
        TensorMemoryLayout::Interleaved, std::nullopt);

    auto constraintsExp = OpModel<EmbeddingOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputTiledLayout, weightShape,
        weightTiledLayout, outputTiledLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
      ExpectLayoutsEQ(outputTiledLayout, outputLayoutReadBacks[0]);
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
            detail::ExpectedResult{true}),
        std::make_tuple(
            // Input: [batch=2, seq_len=512] (sharded)
            detail::TestTensor{{2, 512},
                               TensorMemoryLayout::Interleaved,
                               BufferType::L1,
                               llvm::SmallVector<int64_t>{2, 1}},
            // Weight: [vocab_size=512, hidden_size=256]
            detail::TestTensor{
                {512, 256}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
            detail::ExpectedResult{true})));

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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_GT(l1PeakSize, 0);
  EXPECT_GT(totalPeakSize, 0);
  EXPECT_GT(outputSize, 0);

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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  EXPECT_GT(cbSize, 0);
  EXPECT_GT(l1PeakSize, 0);
  EXPECT_GT(outputSize, 0);

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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 0);
  EXPECT_GT(l1PeakSize, 0);
  EXPECT_GT(outputSize, 0);
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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  // Basic assertions to verify the op constraints are computed
  EXPECT_EQ(cbSize, 0);
  EXPECT_EQ(l1PeakSize, 0);
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
    const auto expectedLegal = expectedResult.expectedLegal;

    const mlir::tt::ttnn::TTNNLayoutAttr outputLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    auto shapeAttr = mlir::tt::ttnn::ShapeAttr::get(&context, tensorShape);

    auto constraintsExp = ttnn::op_model::OpModel<OpTy>::getOpConstraints(
        CreateWorkerGrid(), shapeAttr, std::nullopt, std::nullopt, std::nullopt,
        outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
            outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
    }
  }
};

using OpModelZerosParam = OpModelCreationParam<mlir::tt::ttnn::ZerosOp>;
using OpModelOnesParam = OpModelCreationParam<mlir::tt::ttnn::OnesOp>;

TEST_P(OpModelZerosParam, ZerosOpParameterized) { RunTest(); }
TEST_P(OpModelOnesParam, OnesOpParameterized) { RunTest(); }

// Test data for creation operations
const auto creationOpTestData = testing::Values(std::make_tuple(
    llvm::SmallVector<int64_t>{1024, 256}, detail::ExpectedResult{true}));

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
  auto [cbSize, l1PeakSize, totalPeakSize, outputSize, outputLayoutReadBacks] =
      constraintsExp.get();
  EXPECT_EQ(cbSize, 0);
  EXPECT_EQ(l1PeakSize, 0);
  EXPECT_EQ(totalPeakSize, 0);
  EXPECT_EQ(outputSize, 0);
}

class OpModelPrepareConv2dWeightsParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // weight
                     detail::TestTensor,         // output
                     ::mlir::tt::ttnn::Layout,   // input_tensor_layout
                     std::string,                // weights_format
                     uint32_t,                   // in_channels
                     uint32_t,                   // out_channels
                     uint32_t,                   // batch_size
                     uint32_t,                   // input_height
                     uint32_t,                   // input_width
                     llvm::SmallVector<int32_t>, // kernel_size
                     llvm::SmallVector<int32_t>, // stride
                     llvm::SmallVector<int32_t>, // padding
                     llvm::SmallVector<int32_t>, // dilation
                     bool,                       // has_bias
                     uint32_t,                   // groups
                     detail::ExpectedResult>> {};

TEST_P(OpModelPrepareConv2dWeightsParam, PrepareConv2dWeights) {
  auto params = GetParam();
  const auto [weightShape, weightTensorLayout, weightBufferType,
              weightVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto inputTensorLayout = std::get<2>(params);
  const auto weightsFormat = std::get<3>(params);
  const auto in_channels = std::get<4>(params);
  const auto out_channels = std::get<5>(params);
  const auto batch_size = std::get<6>(params);
  const auto input_height = std::get<7>(params);
  const auto input_width = std::get<8>(params);
  const auto kernel_size = std::get<9>(params);
  const auto stride = std::get<10>(params);
  const auto padding = std::get<11>(params);
  const auto dilation = std::get<12>(params);
  const auto has_bias = std::get<13>(params);
  const auto groups = std::get<14>(params);
  const auto expectedLegal = std::get<15>(params).expectedLegal;

  const TTNNLayoutAttr weightLayout = CreateRowMajorLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Create input memory config
  MemoryConfigAttr inputMemConfig = MemoryConfigAttr::get(
      &context,
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved),
      BufferTypeAttr::get(&context, BufferType::DRAM),
      std::nullopt /*shardSpec*/);

  auto constraintsExp = OpModel<PrepareConv2dWeightsOp>::getOpConstraints(
      CreateWorkerGrid(), weightLayout, weightShape, inputMemConfig,
      inputTensorLayout, weightsFormat, in_channels, out_channels, batch_size,
      input_height, input_width, kernel_size, stride, padding, dilation,
      has_bias, groups, ttcore::DataType::Float32, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, outputLayout);

  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
              outputLayoutReadBacks] = constraintsExp.get();
  EXPECT_GE(cbSize, 0);
  EXPECT_EQ(l1PeakSize, 0);
  EXPECT_EQ(totalPeakSize, 0);
  EXPECT_EQ(outputSize, 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareConv2dWeightsTests, OpModelPrepareConv2dWeightsParam,
    ::testing::Values(
        // Test case 1: Standard 7x7 conv weights preparation
        std::make_tuple(detail::TestTensor{{64, 3, 7, 7},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::SystemMemory},
                        detail::TestTensor{{64, 3, 7, 7},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        ::mlir::tt::ttnn::Layout::RowMajor, "OIHW", 3, 64, 1,
                        224, 224, llvm::SmallVector<int32_t>{7, 7},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{1, 1}, false, 1,
                        detail::ExpectedResult{true})));

//===----------------------------------------------------------------------===//
// PrepareConv2dBiasOp Tests
//===----------------------------------------------------------------------===//

class OpModelPrepareConv2dBiasParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,         // bias
                     detail::TestTensor,         // output
                     ::mlir::tt::ttnn::Layout,   // input_tensor_layout
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

TEST_P(OpModelPrepareConv2dBiasParam, PrepareConv2dBias) {
  auto params = GetParam();
  const auto [biasShape, biasTensorLayout, biasBufferType, biasVirtualGrid] =
      std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto inputTensorLayout = std::get<2>(params);
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
  const auto expectedLegal = std::get<13>(params).expectedLegal;

  const TTNNLayoutAttr biasLayout = CreateRowMajorLayout(
      biasShape, biasBufferType, biasTensorLayout, biasVirtualGrid,
      GetPhysicalGridSize(), builder.getF32Type());
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Create input memory config
  MemoryConfigAttr inputMemConfig = MemoryConfigAttr::get(
      &context,
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved),
      BufferTypeAttr::get(&context, BufferType::DRAM),
      std::nullopt /*shardSpec*/);

  //  get_cb_info expects conv_config.weights_dtype to be set otherwise it
  //  issues an error. See conv2d_op_program_factory_common.cpp in tt-metal.
  Conv2dConfigAttr conv2dConfig = Conv2dConfigAttr::get(&context);
  conv2dConfig = conv2dConfig.withWeightsDtype(ttcore::DataType::Float32);

  auto constraintsExp = OpModel<PrepareConv2dBiasOp>::getOpConstraints(
      CreateWorkerGrid(), biasLayout, biasShape, inputMemConfig,
      inputTensorLayout, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, dilation, groups,
      ttcore::DataType::Float32, std::nullopt, conv2dConfig, std::nullopt,
      outputLayout);

  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
              outputLayoutReadBacks] = constraintsExp.get();
  EXPECT_GE(cbSize, 0);
  EXPECT_EQ(l1PeakSize, 0);
  EXPECT_EQ(totalPeakSize, 0);
  EXPECT_EQ(outputSize, 0);
}

INSTANTIATE_TEST_SUITE_P(
    PrepareConv2dBiasTests, OpModelPrepareConv2dBiasParam,
    ::testing::Values(
        // Test case 1: Standard conv bias preparation
        std::make_tuple(detail::TestTensor{{1, 1, 1, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::SystemMemory},
                        detail::TestTensor{{1, 1, 1, 64},
                                           TensorMemoryLayout::Interleaved,
                                           BufferType::DRAM},
                        ::mlir::tt::ttnn::Layout::RowMajor, 3, 64, 1, 224, 224,
                        llvm::SmallVector<int32_t>{7, 7},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{1, 1}, 1,
                        detail::ExpectedResult{true})));

//===----------------------------------------------------------------------===//
// BatchNormOp Tests
//===----------------------------------------------------------------------===//

class OpModelBatchNormParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,                // input
                     detail::TestTensor,                // output
                     std::optional<detail::TestTensor>, // running_mean
                     std::optional<detail::TestTensor>, // running_var
                     std::optional<detail::TestTensor>, // weight
                     std::optional<detail::TestTensor>, // bias
                     bool,                              // training
                     float,                             // epsilon
                     float,                             // momentum
                     detail::ExpectedResult             // expected result
                     >> {};

TEST_P(OpModelBatchNormParam, BatchNormParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto runningMeanOpt = std::get<2>(params);
  const auto runningVarOpt = std::get<3>(params);
  const auto weightOpt = std::get<4>(params);
  const auto biasOpt = std::get<5>(params);
  const auto training = std::get<6>(params);
  const auto epsilon = llvm::APFloat(std::get<7>(params));
  const auto momentum = llvm::APFloat(std::get<8>(params));
  const auto expectedResult = std::get<9>(params);
  const auto expectedLegal = expectedResult.expectedLegal;

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Create optional layouts for running_mean, running_var, weight, bias
  std::optional<llvm::ArrayRef<int64_t>> runningMeanShape = std::nullopt;
  std::optional<TTNNLayoutAttr> runningMeanLayout = std::nullopt;
  if (runningMeanOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] =
        runningMeanOpt.value();
    runningMeanShape = shape;
    runningMeanLayout =
        CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  std::optional<llvm::ArrayRef<int64_t>> runningVarShape = std::nullopt;
  std::optional<TTNNLayoutAttr> runningVarLayout = std::nullopt;
  if (runningVarOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] =
        runningVarOpt.value();
    runningVarShape = shape;
    runningVarLayout =
        CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  std::optional<llvm::ArrayRef<int64_t>> weightShape = std::nullopt;
  std::optional<TTNNLayoutAttr> weightLayout = std::nullopt;
  if (weightOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = weightOpt.value();
    weightShape = shape;
    weightLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  std::optional<llvm::ArrayRef<int64_t>> biasShape = std::nullopt;
  std::optional<TTNNLayoutAttr> biasLayout = std::nullopt;
  if (biasOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = biasOpt.value();
    biasShape = shape;
    biasLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  // Test getOpConstraints
  llvm::Expected<OpConstraints> constraintsExp =
      training ? op_model::OpModel<BatchNormTrainingOp>::getOpConstraints(
                     CreateWorkerGrid(), inputShape, inputLayout,
                     runningMeanShape, runningMeanLayout, runningVarShape,
                     runningVarLayout, weightShape, weightLayout, biasShape,
                     biasLayout, epsilon, momentum, outputLayout)
               : op_model::OpModel<BatchNormInferenceOp>::getOpConstraints(
                     CreateWorkerGrid(), inputShape, inputLayout,
                     runningMeanShape, runningMeanLayout, runningVarShape,
                     runningVarLayout, weightShape, weightLayout, biasShape,
                     biasLayout, epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
  } else {
    llvm::consumeError(constraintsExp.takeError());
  }

  // Test getOpRuntime
  llvm::Expected<size_t> runtimeExp =
      training
          ? op_model::OpModel<BatchNormTrainingOp>::getOpRuntime(
                inputShape, inputLayout, runningMeanShape, runningMeanLayout,
                runningVarShape, runningVarLayout, weightShape, weightLayout,
                biasShape, biasLayout, epsilon, momentum, outputLayout)
          : op_model::OpModel<BatchNormInferenceOp>::getOpRuntime(
                inputShape, inputLayout, runningMeanShape, runningMeanLayout,
                runningVarShape, runningVarLayout, weightShape, weightLayout,
                biasShape, biasLayout, epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

// Shared test values for BatchNormOp operations
const auto batchNormTestValues = ::testing::Values(
    // Test case 1: Basic BatchNormInference with all optional tensors (4D
    // input)
    std::make_tuple(
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        false, 1e-05f, 0.1f, detail::ExpectedResult{true}),

    // Test case 2: BatchNormTraining without optional tensors
    std::make_tuple(
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, true, 1e-05f,
        0.1f, detail::ExpectedResult{true}),

    // Test case 3: Failing case: BatchNormInference supports tensors of rank 4
    // only.
    std::make_tuple(
        detail::TestTensor{{1, 16, 256, 256},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 16, 256, 256},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {16}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {16}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {16}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {16}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        false, 1e-05f, 0.01f, detail::ExpectedResult{false}),

    // Test case 4: BatchNormInference with L1 memory buffers
    std::make_tuple(
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        std::make_optional(detail::TestTensor{
            {1, 32, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        false, 1e-05f, 0.1f, detail::ExpectedResult{true}),

    // Test case 5: Failing case: running_mean and running_var must be defined
    // in evaluation mode
    std::make_tuple(
        detail::TestTensor{{1, 64, 112, 112},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 64, 112, 112},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::nullopt, std::nullopt,
        std::make_optional(detail::TestTensor{
            {1, 64, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {1, 64, 1, 1}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        false, 1e-05f, 0.1f, detail::ExpectedResult{false}));

INSTANTIATE_TEST_SUITE_P(BatchNormTests, OpModelBatchNormParam,
                         batchNormTestValues);

//===----------------------------------------------------------------------===//
// RMSNormOp Tests
//===----------------------------------------------------------------------===//

class OpModelRMSNormParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,                // input
                     detail::TestTensor,                // output
                     std::optional<detail::TestTensor>, // weight
                     std::optional<detail::TestTensor>, // bias
                     float,                             // epsilon
                     detail::ExpectedResult             // expected result
                     >> {};

TEST_P(OpModelRMSNormParam, RMSNormParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto weightOpt = std::get<2>(params);
  const auto biasOpt = std::get<3>(params);
  const auto epsilon = llvm::APFloat(std::get<4>(params));
  const auto expectedResult = std::get<5>(params);
  const auto expectedLegal = expectedResult.expectedLegal;

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Create optional layouts for weight and bias
  std::optional<llvm::ArrayRef<int64_t>> weightShape = std::nullopt;
  std::optional<TTNNLayoutAttr> weightLayout = std::nullopt;
  if (weightOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = weightOpt.value();
    weightShape = shape;
    weightLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  std::optional<llvm::ArrayRef<int64_t>> biasShape = std::nullopt;
  std::optional<TTNNLayoutAttr> biasLayout = std::nullopt;
  if (biasOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = biasOpt.value();
    biasShape = shape;
    biasLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  // Test getOpConstraints
  auto constraintsExp = op_model::OpModel<RMSNormOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      biasShape, biasLayout, epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
  } else {
    llvm::consumeError(constraintsExp.takeError());
  }

  // Test getOpRuntime
  auto runtimeExp = op_model::OpModel<RMSNormOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, biasShape, biasLayout,
      epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

// Shared test values for RMSNormOp operations
const auto rmsNormTestValues = ::testing::Values(
    // Test case 1: Basic RMSNorm with all optional tensors (weight and bias)
    std::make_tuple(
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        1e-12f, detail::ExpectedResult{true}),

    // Test case 2: RMSNorm without optional tensors (only input)
    std::make_tuple(
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, 1e-12f, detail::ExpectedResult{true}),

    // Test case 3: RMSNorm with L1 memory buffers
    std::make_tuple(
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional(detail::TestTensor{
            {32}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        std::make_optional(detail::TestTensor{
            {32}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        1e-12f, detail::ExpectedResult{true}),

    // Test case 4: RMSNorm with only weight (no bias)
    std::make_tuple(
        detail::TestTensor{
            {2, 16, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {2, 16, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::nullopt, 1e-8f, detail::ExpectedResult{true}),

    // Test case 5: RMSNorm with different epsilon value
    std::make_tuple(
        detail::TestTensor{
            {1, 16, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 16, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, 1e-6f, detail::ExpectedResult{true}));

INSTANTIATE_TEST_SUITE_P(RMSNormTests, OpModelRMSNormParam, rmsNormTestValues);

//===----------------------------------------------------------------------===//
// LayerNormOp Tests
//===----------------------------------------------------------------------===//

class OpModelLayerNormParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,                // input
                     detail::TestTensor,                // output
                     std::optional<detail::TestTensor>, // weight
                     std::optional<detail::TestTensor>, // bias
                     float,                             // epsilon
                     detail::ExpectedResult             // expected result
                     >> {};

TEST_P(OpModelLayerNormParam, LayerNormParam) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto weightOpt = std::get<2>(params);
  const auto biasOpt = std::get<3>(params);
  const auto epsilon = llvm::APFloat(std::get<4>(params));
  const auto expectedResult = std::get<5>(params);
  const auto expectedLegal = expectedResult.expectedLegal;

  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Create optional layouts for weight and bias
  std::optional<llvm::ArrayRef<int64_t>> weightShape = std::nullopt;
  std::optional<TTNNLayoutAttr> weightLayout = std::nullopt;
  if (weightOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = weightOpt.value();
    weightShape = shape;
    weightLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  std::optional<llvm::ArrayRef<int64_t>> biasShape = std::nullopt;
  std::optional<TTNNLayoutAttr> biasLayout = std::nullopt;
  if (biasOpt.has_value()) {
    const auto &[shape, layout, bufferType, virtualGrid] = biasOpt.value();
    biasShape = shape;
    biasLayout = CreateTiledLayout(shape, bufferType, layout, virtualGrid);
  }

  // Test getOpConstraints
  auto constraintsExp = op_model::OpModel<LayerNormOp>::getOpConstraints(
      CreateWorkerGrid(), inputShape, inputLayout, weightShape, weightLayout,
      biasShape, biasLayout, epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (constraintsExp) {
    const auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
                outputLayoutReadBacks] = constraintsExp.get();
    EXPECT_GE(cbSize, 0);
    EXPECT_GE(l1PeakSize, 0);
    EXPECT_GE(totalPeakSize, 0);
    EXPECT_GE(outputSize, 0);
  } else {
    llvm::consumeError(constraintsExp.takeError());
  }

  // Test getOpRuntime
  auto runtimeExp = op_model::OpModel<LayerNormOp>::getOpRuntime(
      inputShape, inputLayout, weightShape, weightLayout, biasShape, biasLayout,
      epsilon, outputLayout);

  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (runtimeExp) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

// Shared test values for LayerNormOp operations
const auto layerNormTestValues = ::testing::Values(
    // Test case 1: Basic LayerNorm with all optional tensors (weight and bias)
    std::make_tuple(
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 32, 128, 128},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        1e-12f, detail::ExpectedResult{true}),

    // Test case 2: LayerNorm without optional tensors (only input)
    std::make_tuple(
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 64, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, 1e-12f, detail::ExpectedResult{true}),

    // Test case 3: LayerNorm with L1 memory buffers
    std::make_tuple(
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1, 32, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional(detail::TestTensor{
            {32}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        std::make_optional(detail::TestTensor{
            {32}, TensorMemoryLayout::Interleaved, BufferType::L1}),
        1e-12f, detail::ExpectedResult{true}),

    // Test case 4: LayerNorm with only weight (no bias)
    std::make_tuple(
        detail::TestTensor{
            {2, 16, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {2, 16, 64, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        std::nullopt, 1e-8f, detail::ExpectedResult{true}),

    // Test case 5: LayerNorm with different epsilon value
    std::make_tuple(
        detail::TestTensor{
            {1, 16, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 16, 32, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, 1e-6f, detail::ExpectedResult{true}));

INSTANTIATE_TEST_SUITE_P(LayerNormTests, OpModelLayerNormParam,
                         layerNormTestValues);

// ==== ConstantOp Tests ====

template <typename DataType, typename MLIRType>
class OpModelConstantParam
    : public OpModelTest,
      public testing::WithParamInterface<std::tuple<
          llvm::SmallVector<int64_t>,             // tensor shape
          std::vector<DataType>,                  // constant data
          std::function<MLIRType(OpModelTest *)>, // type creator function
          std::optional<TTNNLayoutAttr>,          // output layout (optional)
          detail::ExpectedResult                  // expected results
          >> {
protected:
  void RunTest() {
    auto params = this->GetParam();
    const auto [tensorShape, constData, typeCreator, outputLayoutOpt,
                expectedResult] = params;
    const auto expectedLegal = expectedResult.expectedLegal;

    // Create element type using the provided function
    MLIRType elementType = typeCreator(this);

    // Create output layout or use default
    TTNNLayoutAttr outputLayout;
    if (outputLayoutOpt.has_value()) {
      outputLayout = outputLayoutOpt.value();
    } else {
      // For supported integer types, we need to use specific layout creation
      // methods
      if constexpr (std::is_same_v<DataType, int32_t>) {
        outputLayout = CreateTiledLayoutInt32(tensorShape, BufferType::L1,
                                              TensorMemoryLayout::Interleaved);
      } else {
        outputLayout = CreateTiledLayout(tensorShape, BufferType::L1,
                                         TensorMemoryLayout::Interleaved);
      }
    }

    // Create tensor type and dense elements attribute
    mlir::RankedTensorType tensorType =
        mlir::RankedTensorType::get(tensorShape, elementType);
    llvm::ArrayRef<DataType> dataRef(constData);
    mlir::DenseElementsAttr attr =
        mlir::DenseElementsAttr::get(tensorType, dataRef);

    // Test getOpConstraints
    auto constraintsExp =
        ttnn::op_model::OpModel<mlir::tt::ttnn::ConstantOp>::getOpConstraints(
            CreateWorkerGrid(), attr, outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      auto [cbSize, l1PeakSize, totalPeakSize, outputSize,
            outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSize, 0);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
  }
};

// Type aliases for different constant data types
using OpModelConstantUInt32Param = OpModelConstantParam<uint32_t, mlir::Type>;
using OpModelConstantInt32Param = OpModelConstantParam<int32_t, mlir::Type>;
using OpModelConstantUInt16Param = OpModelConstantParam<uint16_t, mlir::Type>;
using OpModelConstantUInt8Param = OpModelConstantParam<uint8_t, mlir::Type>;

TEST_P(OpModelConstantUInt32Param, ConstantOpUInt32) { RunTest(); }
TEST_P(OpModelConstantInt32Param, ConstantOpInt32) { RunTest(); }
TEST_P(OpModelConstantUInt16Param, ConstantOpUInt16) { RunTest(); }
TEST_P(OpModelConstantUInt8Param, ConstantOpUInt8) { RunTest(); }

// Test data for ConstantOp with different supported types
const auto constantOpUInt32TestData = testing::Values(
    // Basic 2x2 u32 tensor with L1 interleaved layout
    std::make_tuple(
        llvm::SmallVector<int64_t>{2, 2}, std::vector<uint32_t>{1, 2, 3, 4},
        [](OpModelTest *test) {
          return test->builder.getIntegerType(32, false);
        },
        std::nullopt, detail::ExpectedResult{true}));

const auto constantOpInt32TestData = testing::Values(
    // Basic 2x2 i32 tensor with L1 interleaved layout
    std::make_tuple(
        llvm::SmallVector<int64_t>{2, 2}, std::vector<int32_t>{1, 2, 3, 4},
        [](OpModelTest *test) { return test->builder.getI32Type(); },
        std::nullopt, detail::ExpectedResult{true}),

    // Larger 32x32 i32 tensor
    std::make_tuple(
        llvm::SmallVector<int64_t>{32, 32},
        std::vector<int32_t>(32 * 32, 42), // Fill with value 42
        [](OpModelTest *test) { return test->builder.getI32Type(); },
        std::nullopt, detail::ExpectedResult{true}));

const auto constantOpUInt16TestData = testing::Values(
    // Basic 2x3x4 u16 tensor
    std::make_tuple(
        llvm::SmallVector<int64_t>{2, 3, 4},
        std::vector<uint16_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
        [](OpModelTest *test) {
          return test->builder.getIntegerType(16, false);
        },
        std::nullopt, detail::ExpectedResult{true}));

const auto constantOpUInt8TestData = testing::Values(
    // Basic u8 tensor
    std::make_tuple(
        llvm::SmallVector<int64_t>{4, 4},
        std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                             16},
        [](OpModelTest *test) {
          return test->builder.getIntegerType(8, false);
        },
        std::nullopt, detail::ExpectedResult{true}));

INSTANTIATE_TEST_SUITE_P(ConstantOpUInt32Tests, OpModelConstantUInt32Param,
                         constantOpUInt32TestData);
INSTANTIATE_TEST_SUITE_P(ConstantOpInt32Tests, OpModelConstantInt32Param,
                         constantOpInt32TestData);
INSTANTIATE_TEST_SUITE_P(ConstantOpUInt16Tests, OpModelConstantUInt16Param,
                         constantOpUInt16TestData);
INSTANTIATE_TEST_SUITE_P(ConstantOpUInt8Tests, OpModelConstantUInt8Param,
                         constantOpUInt8TestData);

TEST_F(OpModelTest, RandOp) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr outputLayoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayoutL1 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  // Test RandOp with DRAM output
  auto shapeAttr = ttnn::ShapeAttr::get(&context, tensorShape);

  auto constraintsExp = OpModel<RandOp>::getOpConstraints(
      workerGrid, shapeAttr, ttcore::DataType::BFloat16, nullptr,
      ttnn::Layout::Tile, llvm::APFloat(0.0f), llvm::APFloat(1.0f), 0,
      outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  // Test RandOp with L1 output
  constraintsExp = OpModel<RandOp>::getOpConstraints(
      workerGrid, shapeAttr, ttcore::DataType::BFloat16, nullptr,
      ttnn::Layout::Tile, llvm::APFloat(0.0f), llvm::APFloat(1.0f), 0,
      outputLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GT(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  // Test RandOp with custom range parameters
  constraintsExp = OpModel<RandOp>::getOpConstraints(
      workerGrid, shapeAttr, ttcore::DataType::BFloat16, nullptr,
      ttnn::Layout::Tile, llvm::APFloat(-2.5f), llvm::APFloat(5.0f), 42,
      outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  // Test RandOp with Float32 data type
  constraintsExp = OpModel<RandOp>::getOpConstraints(
      workerGrid, shapeAttr, ttcore::DataType::Float32, nullptr,
      ttnn::Layout::Tile, llvm::APFloat(0.0f), llvm::APFloat(1.0f), 0,
      outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);
}

TEST_F(OpModelTest, DropoutOp) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const TTNNLayoutAttr inputLayoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutL1 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayoutDRAM = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr outputLayoutL1 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  auto constraintsExp = OpModel<DropoutOp>::getOpConstraints(
      workerGrid, tensorShape, inputLayoutDRAM, llvm::APFloat(0.0f),
      llvm::APFloat(1.0f), 0, true, outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GE(opCstr.cbL1PeakSize, 0);

  constraintsExp = OpModel<DropoutOp>::getOpConstraints(
      workerGrid, tensorShape, inputLayoutL1, llvm::APFloat(0.2f),
      llvm::APFloat(1.25f), 21, true, outputLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GE(opCstr.cbL1PeakSize, 0);

  constraintsExp = OpModel<DropoutOp>::getOpConstraints(
      workerGrid, tensorShape, inputLayoutDRAM, llvm::APFloat(0.5f),
      llvm::APFloat(2.0f), 21, false, outputLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GE(opCstr.cbL1PeakSize, 0);

  auto runtimeExp = OpModel<DropoutOp>::getOpRuntime(
      tensorShape, inputLayoutDRAM, llvm::APFloat(0.2f), llvm::APFloat(1.25f),
      21, true, outputLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_GE(runtimeExp.get(), 0);
}

TEST_F(OpModelTest, FillCacheOp) {
  // Test basic FillCacheOp with DRAM cache and input tensors
  const llvm::SmallVector<int64_t> cacheShape = {1, 32, 64, 512};
  const llvm::SmallVector<int64_t> inputShape = {1, 32, 3, 512};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  const TTNNLayoutAttr cacheLayoutDRAM = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutDRAM = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr cacheLayoutL1 = CreateTiledLayout(
      cacheShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutL1 = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  // Test FillCacheOp constraints with batch_offset = 0
  uint32_t batchOffset = 0;
  auto constraintsExp = OpModel<FillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM,
      batchOffset, cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
    EXPECT_EQ(opCstr.outputL1BufferSize, 0);
    // Captures updated cache, even though it is in place update
    EXPECT_EQ(opCstr.outputLayouts.size(), 1);
  }

  // Test with L1 output layout
  constraintsExp = OpModel<FillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM,
      batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with L1 cache layout
  constraintsExp = OpModel<FillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutL1, inputShape, inputLayoutDRAM,
      batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with L1 input layout
  constraintsExp = OpModel<FillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutL1,
      batchOffset, cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with all L1 layouts
  constraintsExp = OpModel<FillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutL1, inputShape, inputLayoutL1,
      batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  // Test FillCacheOp runtime estimation
  auto runtimeExp = OpModel<FillCacheOp>::getOpRuntime(
      cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM, batchOffset,
      cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));

  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  }
}

TEST_F(OpModelTest, UpdateCacheOp) {
  // Test basic UpdateCacheOp with DRAM cache, input, and update_index tensors
  const llvm::SmallVector<int64_t> cacheShape = {1, 32, 64, 512};
  const llvm::SmallVector<int64_t> inputShape = {1, 32, 3, 512};
  const llvm::SmallVector<int64_t> updateIndexShape = {1};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  const TTNNLayoutAttr cacheLayoutDRAM = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutDRAM = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr updateIndexLayoutDRAM = CreateTiledLayoutUInt32(
      updateIndexShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr cacheLayoutL1 = CreateTiledLayout(
      cacheShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayoutL1 = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr updateIndexLayoutL1 = CreateTiledLayoutUInt32(
      updateIndexShape, BufferType::L1, TensorMemoryLayout::Interleaved);

  // Test UpdateCacheOp constraints with batch_offset = 0
  uint32_t batchOffset = 0;
  auto constraintsExp = OpModel<UpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM,
      updateIndexShape, updateIndexLayoutDRAM, batchOffset, cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
    EXPECT_EQ(opCstr.outputL1BufferSize, 0);
    // Captures updated cache, even though it is in place update
    EXPECT_EQ(opCstr.outputLayouts.size(), 1);
  }

  // Test with L1 output layout
  constraintsExp = OpModel<UpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM,
      updateIndexShape, updateIndexLayoutDRAM, batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with L1 cache layout
  constraintsExp = OpModel<UpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutL1, inputShape, inputLayoutDRAM,
      updateIndexShape, updateIndexLayoutDRAM, batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with L1 input layout
  constraintsExp = OpModel<UpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutDRAM, inputShape, inputLayoutL1,
      updateIndexShape, updateIndexLayoutDRAM, batchOffset, cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  // Test with all L1 layouts
  constraintsExp = OpModel<UpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayoutL1, inputShape, inputLayoutL1,
      updateIndexShape, updateIndexLayoutL1, batchOffset, cacheLayoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  // Test UpdateCacheOp runtime estimation
  auto runtimeExp = OpModel<UpdateCacheOp>::getOpRuntime(
      cacheShape, cacheLayoutDRAM, inputShape, inputLayoutDRAM,
      updateIndexShape, updateIndexLayoutDRAM, batchOffset, cacheLayoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));

  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  }
}

TEST_F(OpModelTest, PagedUpdateCacheOp) {
  // Test basic PagedUpdateCacheOp with DRAM cache, input, update_index, and
  // page_table tensors
  const llvm::SmallVector<int64_t> cacheShape = {128, 4, 32, 256};
  const llvm::SmallVector<int64_t> inputShape = {1, 8, 12, 256};
  const llvm::SmallVector<int64_t> updateIndexShape = {8};
  const llvm::SmallVector<int64_t> pageTableShape = {8, 16};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  const TTNNLayoutAttr cacheLayout = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  const TTNNLayoutAttr updateIndexLayout = CreateRowMajorLayoutInt32(
      updateIndexShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr pageTableLayout = CreateRowMajorLayoutInt32(
      pageTableShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<PagedUpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayout, inputShape, inputLayout,
      updateIndexShape, updateIndexLayout, pageTableShape, pageTableLayout,
      false, cacheLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_GE(opCstr.tensorL1PeakSize, 0);
    EXPECT_EQ(opCstr.outputL1BufferSize, 0);
    EXPECT_GE(opCstr.peakL1MemorySize, 0);
    // Captures updated cache, even though it is in place update
    EXPECT_EQ(opCstr.outputLayouts.size(), 1);
  }
}

TEST_F(OpModelTest, PagedUpdateCacheOpWithoutPageTable) {
  // Test PagedUpdateCacheOp without optional page_table tensor
  // Cache shape: [batch, num_heads, block_size, head_dim]
  // Input shape: [1, batch, num_heads (padded to 32), head_dim]
  // The batch dimension must match: input[1] == cache[0]
  const llvm::SmallVector<int64_t> cacheShape = {8, 4, 32, 256};
  const llvm::SmallVector<int64_t> inputShape = {1, 8, 12, 256};
  const llvm::SmallVector<int64_t> updateIndexShape = {8};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  const TTNNLayoutAttr cacheLayout = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
  const TTNNLayoutAttr updateIndexLayout = CreateRowMajorLayoutInt32(
      updateIndexShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<PagedUpdateCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayout, inputShape, inputLayout,
      updateIndexShape, updateIndexLayout, std::nullopt, std::nullopt, false,
      cacheLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_GE(opCstr.tensorL1PeakSize, 0);
    EXPECT_EQ(opCstr.outputL1BufferSize, 0);
    EXPECT_GE(opCstr.peakL1MemorySize, 0);
    // Captures updated cache, even though it is in place update
    EXPECT_EQ(opCstr.outputLayouts.size(), 1);
  }

  auto runtimeExp = OpModel<PagedUpdateCacheOp>::getOpRuntime(
      cacheShape, cacheLayout, inputShape, inputLayout, updateIndexShape,
      updateIndexLayout, std::nullopt, std::nullopt, false, cacheLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  }
}

TEST_F(OpModelTest, PagedFillCacheOp) {
  // Test basic PagedUpdateCacheOp with DRAM cache, input, update_index, and
  // page_table tensors
  const llvm::SmallVector<int64_t> cacheShape = {128, 4, 32, 256};
  const llvm::SmallVector<int64_t> inputShape = {1, 12, 65, 256};
  const llvm::SmallVector<int64_t> batchOffsetShape = {1};
  const llvm::SmallVector<int64_t> pageTableShape = {8, 16};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  const TTNNLayoutAttr cacheLayout = CreateTiledLayout(
      cacheShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr batchOffsetLayout = CreateRowMajorLayoutInt32(
      batchOffsetShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  const TTNNLayoutAttr pageTableLayout = CreateRowMajorLayoutInt32(
      pageTableShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  auto constraintsExp = OpModel<PagedFillCacheOp>::getOpConstraints(
      workerGrid, cacheShape, cacheLayout, inputShape, inputLayout,
      pageTableShape, pageTableLayout, batchOffsetShape, batchOffsetLayout,
      cacheLayout);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));

  if (constraintsExp) {
    OpConstraints &opCstr = constraintsExp.get();
    EXPECT_GT(opCstr.cbL1PeakSize, 0);
    EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
    EXPECT_EQ(opCstr.outputL1BufferSize, 0);
    EXPECT_GT(opCstr.peakL1MemorySize, 0);
    // Captures updated cache, even though it is in place update
    EXPECT_EQ(opCstr.outputLayouts.size(), 1);
  }

  auto runtimeExp = OpModel<PagedFillCacheOp>::getOpRuntime(
      cacheShape, cacheLayout, inputShape, inputLayout, pageTableShape,
      pageTableLayout, batchOffsetShape, batchOffsetLayout, cacheLayout);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  if (runtimeExp) {
    EXPECT_GT(runtimeExp.get(), 0);
  }
}

//===----------------------------------------------------------------------===//
// QuantizeOp Tests
//===----------------------------------------------------------------------===//

struct QuantizeOpParam {
  detail::TestTensor input;     // BF16
  detail::TestTensor scale;     // BF16
  detail::TestTensor zeroPoint; // Int32
  detail::TestTensor output;    // Int32
  std::optional<int32_t> axis;
  detail::ExpectedResult expectedResult;
};

class OpModelQuantizeParam
    : public OpModelTest,
      public testing::WithParamInterface<QuantizeOpParam> {
protected:
  void RunTest() {
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = GetParam().input;
    const auto [scaleShape, scaleTensorLayout, scaleBufferType,
                scaleVirtualGrid] = GetParam().scale;
    const auto [zeroPointShape, zeroPointTensorLayout, zeroPointBufferType,
                zeroPointVirtualGrid] = GetParam().zeroPoint;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto axis = GetParam().axis;
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    // Create layouts - input and scale use BF16, zeroPoint and output use Int32
    const TTNNLayoutAttr inputLayout = CreateTiledLayout(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr scaleLayout = CreateTiledLayout(
        scaleShape, scaleBufferType, scaleTensorLayout, scaleVirtualGrid);
    const TTNNLayoutAttr zeroPointLayout =
        CreateTiledLayoutInt32(zeroPointShape, zeroPointBufferType,
                               zeroPointTensorLayout, zeroPointVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayoutInt32(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<QuantizeOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, scaleShape, scaleLayout,
        zeroPointShape, zeroPointLayout, axis,
        std::make_optional(ttcore::DataType::Int32), outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<QuantizeOp>::getOpRuntime(
        inputShape, inputLayout, scaleShape, scaleLayout, zeroPointShape,
        zeroPointLayout, axis, std::make_optional(ttcore::DataType::Int32),
        outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelQuantizeParam, QuantizeOp) { RunTest(); }

// Test data for QuantizeOp
const auto quantizeOpTestValues = testing::Values(
    // === L1 Memory Tests ===
    QuantizeOpParam{
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::nullopt,
        detail::ExpectedResult{true}}, // note: combined peak < cb+l1 peak
    QuantizeOpParam{
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1),
        detail::ExpectedResult{true}}, // note: combined peak < cb+l1 peak
    QuantizeOpParam{
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1),
        detail::ExpectedResult{true}}, // note: combined peak < cb+l1 peak

    // === DRAM Memory Tests ===
    QuantizeOpParam{
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},

    // === Mixed Memory Configuration Tests ===
    QuantizeOpParam{
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1),
        detail::ExpectedResult{true}}); // note: combined peak < cb+l1 peak

INSTANTIATE_TEST_SUITE_P(QuantizeTests, OpModelQuantizeParam,
                         quantizeOpTestValues);

//===----------------------------------------------------------------------===//
// RequantizeOp Tests
//===----------------------------------------------------------------------===//

struct RequantizeOpParam {
  detail::TestTensor input;        // Int32
  detail::TestTensor inScale;      // BF16
  detail::TestTensor inZeroPoint;  // Int32
  detail::TestTensor outScale;     // BF16
  detail::TestTensor outZeroPoint; // Int32
  detail::TestTensor output;       // Int32
  std::optional<int32_t> axis;
  detail::ExpectedResult expectedResult;
};

class OpModelRequantizeParam
    : public OpModelTest,
      public testing::WithParamInterface<RequantizeOpParam> {
protected:
  void RunTest() {
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = GetParam().input;
    const auto [inScaleShape, inScaleTensorLayout, inScaleBufferType,
                inScaleVirtualGrid] = GetParam().inScale;
    const auto [inZeroPointShape, inZeroPointTensorLayout,
                inZeroPointBufferType, inZeroPointVirtualGrid] =
        GetParam().inZeroPoint;
    const auto [outScaleShape, outScaleTensorLayout, outScaleBufferType,
                outScaleVirtualGrid] = GetParam().outScale;
    const auto [outZeroPointShape, outZeroPointTensorLayout,
                outZeroPointBufferType, outZeroPointVirtualGrid] =
        GetParam().outZeroPoint;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto axis = GetParam().axis;
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    // Create layouts - input and zero points use Int32, scales use BF16
    const TTNNLayoutAttr inputLayout = CreateTiledLayoutInt32(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr inScaleLayout =
        CreateTiledLayout(inScaleShape, inScaleBufferType, inScaleTensorLayout,
                          inScaleVirtualGrid);
    const TTNNLayoutAttr inZeroPointLayout =
        CreateTiledLayoutInt32(inZeroPointShape, inZeroPointBufferType,
                               inZeroPointTensorLayout, inZeroPointVirtualGrid);
    const TTNNLayoutAttr outScaleLayout =
        CreateTiledLayout(outScaleShape, outScaleBufferType,
                          outScaleTensorLayout, outScaleVirtualGrid);
    const TTNNLayoutAttr outZeroPointLayout = CreateTiledLayoutInt32(
        outZeroPointShape, outZeroPointBufferType, outZeroPointTensorLayout,
        outZeroPointVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayoutInt32(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<RequantizeOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, inScaleShape,
        inScaleLayout, inZeroPointShape, inZeroPointLayout, outScaleShape,
        outScaleLayout, outZeroPointShape, outZeroPointLayout, axis,
        std::make_optional(ttcore::DataType::Int32), outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<RequantizeOp>::getOpRuntime(
        inputShape, inputLayout, inScaleShape, inScaleLayout, inZeroPointShape,
        inZeroPointLayout, outScaleShape, outScaleLayout, outZeroPointShape,
        outZeroPointLayout, axis, std::make_optional(ttcore::DataType::Int32),
        outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelRequantizeParam, RequantizeOp) { RunTest(); }

// Test data for RequantizeOp
const auto requantizeOpTestValues = testing::Values(
    // === L1 Memory Tests ===
    RequantizeOpParam{
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::nullopt,
        detail::ExpectedResult{true}}, // note: combined peak < cb+l1 peak
    RequantizeOpParam{
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},

    // === DRAM Memory Tests ===
    RequantizeOpParam{
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},
    RequantizeOpParam{
        detail::TestTensor{
            {256, 512}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {256, 512}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, detail::ExpectedResult{true}},

    // === Mixed Memory Configuration Tests ===
    RequantizeOpParam{
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}});

INSTANTIATE_TEST_SUITE_P(RequantizeTests, OpModelRequantizeParam,
                         requantizeOpTestValues);

//===----------------------------------------------------------------------===//
// DequantizeOp Tests
//===----------------------------------------------------------------------===//

struct DequantizeOpParam {
  detail::TestTensor input;     // Int32
  detail::TestTensor scale;     // BF16
  detail::TestTensor zeroPoint; // Int32
  detail::TestTensor output;    // BF16
  std::optional<int32_t> axis;
  detail::ExpectedResult expectedResult;
};

class OpModelDequantizeParam
    : public OpModelTest,
      public testing::WithParamInterface<DequantizeOpParam> {
protected:
  void RunTest() {
    const auto [inputShape, inputTensorLayout, inputBufferType,
                inputVirtualGrid] = GetParam().input;
    const auto [scaleShape, scaleTensorLayout, scaleBufferType,
                scaleVirtualGrid] = GetParam().scale;
    const auto [zeroPointShape, zeroPointTensorLayout, zeroPointBufferType,
                zeroPointVirtualGrid] = GetParam().zeroPoint;
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;
    const auto axis = GetParam().axis;
    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    // Create layouts - input and zeroPoint use Int32, scale and output use BF16
    const TTNNLayoutAttr inputLayout = CreateTiledLayoutInt32(
        inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
    const TTNNLayoutAttr scaleLayout = CreateTiledLayout(
        scaleShape, scaleBufferType, scaleTensorLayout, scaleVirtualGrid);
    const TTNNLayoutAttr zeroPointLayout =
        CreateTiledLayoutInt32(zeroPointShape, zeroPointBufferType,
                               zeroPointTensorLayout, zeroPointVirtualGrid);
    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp = OpModel<DequantizeOp>::getOpConstraints(
        CreateWorkerGrid(), inputShape, inputLayout, scaleShape, scaleLayout,
        zeroPointShape, zeroPointLayout, axis,
        std::make_optional(ttcore::DataType::BFloat16), outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }

    auto runtimeExp = OpModel<DequantizeOp>::getOpRuntime(
        inputShape, inputLayout, scaleShape, scaleLayout, zeroPointShape,
        zeroPointLayout, axis, std::make_optional(ttcore::DataType::BFloat16),
        outputLayout);
    EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
    if (expectedLegal) {
      EXPECT_TRUE(runtimeExp.get() > 0);
    } else {
      llvm::consumeError(runtimeExp.takeError());
    }
  }
};

TEST_P(OpModelDequantizeParam, DequantizeOp) { RunTest(); }

// Test data for DequantizeOp
const auto dequantizeOpTestValues = testing::Values(
    // === L1 Memory Tests ===
    DequantizeOpParam{
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::nullopt, detail::ExpectedResult{true}},
    DequantizeOpParam{
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {32, 64}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},
    DequantizeOpParam{
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128, 256}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},

    // === DRAM Memory Tests ===
    DequantizeOpParam{
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {512, 1024}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},
    DequantizeOpParam{
        detail::TestTensor{
            {256, 512}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {256, 512}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, detail::ExpectedResult{true}},

    // === Mixed Memory Configuration Tests ===
    DequantizeOpParam{
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {64, 128}, TensorMemoryLayout::Interleaved, BufferType::L1},
        std::make_optional<int32_t>(1), detail::ExpectedResult{true}},
    DequantizeOpParam{
        detail::TestTensor{
            {128, 192}, TensorMemoryLayout::Interleaved, BufferType::L1},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {128, 192}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, detail::ExpectedResult{true}});

INSTANTIATE_TEST_SUITE_P(DequantizeTests, OpModelDequantizeParam,
                         dequantizeOpTestValues);

// === ScaledDotProductAttentionDecodeOp Tests ===

struct ScaledDotProductAttentionDecodeOpParam {
  detail::TestTensor query;
  detail::TestTensor key;
  detail::TestTensor value;
  std::optional<detail::TestTensor> curPosTensor;
  std::optional<detail::TestTensor> attentionMask;
  std::optional<detail::TestTensor> attentionSink;
  bool isCausal;
  bool withScale;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};

class OpModelScaledDotProductAttentionDecodeParam
    : public OpModelTest,
      public testing::WithParamInterface<
          ScaledDotProductAttentionDecodeOpParam> {
protected:
  void RunTest() {
    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
    const auto [queryShape, queryTensorLayout, queryBufferType,
                queryVirtualGrid] = GetParam().query;
    const auto [keyShape, keyTensorLayout, keyBufferType, keyVirtualGrid] =
        GetParam().key;
    const auto [valueShape, valueTensorLayout, valueBufferType,
                valueVirtualGrid] = GetParam().value;

    std::optional<SmallVector<int64_t>> curPosTensorShape = std::nullopt;
    std::optional<TensorMemoryLayout> curPosTensorTensorLayout = std::nullopt;
    std::optional<BufferType> curPosTensorBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> curPosTensorVirtualGrid = std::nullopt;

    std::optional<SmallVector<int64_t>> attentionMaskShape = std::nullopt;
    std::optional<TensorMemoryLayout> attentionMaskTensorLayout = std::nullopt;
    std::optional<BufferType> attentionMaskBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> attentionMaskVirtualGrid = std::nullopt;

    std::optional<SmallVector<int64_t>> attentionSinkShape = std::nullopt;
    std::optional<TensorMemoryLayout> attentionSinkTensorLayout = std::nullopt;
    std::optional<BufferType> attentionSinkBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> attentionSinkVirtualGrid = std::nullopt;

    if (auto curPosTensorDetail = GetParam().curPosTensor) {
      curPosTensorShape = curPosTensorDetail->shape;
      curPosTensorTensorLayout = curPosTensorDetail->layout;
      curPosTensorBufferType = curPosTensorDetail->bufferType;
      curPosTensorVirtualGrid = curPosTensorDetail->virtualGrid;
    }

    if (auto attentionMaskDetail = GetParam().attentionMask) {
      attentionMaskShape = attentionMaskDetail->shape;
      attentionMaskTensorLayout = attentionMaskDetail->layout;
      attentionMaskBufferType = attentionMaskDetail->bufferType;
      attentionMaskVirtualGrid = attentionMaskDetail->virtualGrid;
    }

    if (auto attentionSinkDetail = GetParam().attentionSink) {
      attentionSinkShape = attentionSinkDetail->shape;
      attentionSinkTensorLayout = attentionSinkDetail->layout;
      attentionSinkBufferType = attentionSinkDetail->bufferType;
      attentionSinkVirtualGrid = attentionSinkDetail->virtualGrid;
    }

    const auto isCausal = GetParam().isCausal;

    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;

    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    const TTNNLayoutAttr queryLayout = CreateTiledLayout(
        queryShape, queryBufferType, queryTensorLayout, queryVirtualGrid);
    const TTNNLayoutAttr keyLayout = CreateTiledLayout(
        keyShape, keyBufferType, keyTensorLayout, keyVirtualGrid);
    const TTNNLayoutAttr valueLayout = CreateTiledLayout(
        valueShape, valueBufferType, valueTensorLayout, valueVirtualGrid);

    std::optional<TTNNLayoutAttr> curPosTensorLayout = std::nullopt;
    std::optional<TTNNLayoutAttr> attentionMaskLayout = std::nullopt;
    std::optional<TTNNLayoutAttr> attentionSinkLayout = std::nullopt;

    if (curPosTensorShape) {
      curPosTensorLayout =
          CreateTiledLayout(*curPosTensorShape, *curPosTensorBufferType,
                            *curPosTensorTensorLayout, curPosTensorVirtualGrid);
    }
    if (attentionMaskShape) {
      attentionMaskLayout = CreateTiledLayout(
          *attentionMaskShape, *attentionMaskBufferType,
          *attentionMaskTensorLayout, attentionMaskVirtualGrid);
    }
    if (attentionSinkShape) {
      attentionSinkLayout = CreateTiledLayout(
          *attentionSinkShape, *attentionSinkBufferType,
          *attentionSinkTensorLayout, attentionSinkVirtualGrid);
    }

    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    const llvm::APFloat scaleAPFloat(1.0f);
    std::optional<llvm::APFloat> scale = std::nullopt;
    if (GetParam().withScale) {
      scale.emplace(scaleAPFloat);
    }

    auto constraintsExp =
        OpModel<ScaledDotProductAttentionDecodeOp>::getOpConstraints(
            CreateWorkerGrid(), queryShape, queryLayout, keyShape, keyLayout,
            valueShape, valueLayout, isCausal, attentionMaskShape,
            attentionMaskLayout, curPosTensorShape, curPosTensorLayout,
            attentionSinkShape, attentionSinkLayout, scale,
            /*programConfig=*/std::nullopt, outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
    // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
  }
};

TEST_P(OpModelScaledDotProductAttentionDecodeParam,
       ScaledDotProductAttentionDecodeOp) {
  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
  RunTest();
  // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
}

const auto scaledDotProductAttentionDecodeOpTestValues = testing::Values(
    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, true, false,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, std::nullopt, true, true,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 12, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        std::nullopt, false, false,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 12, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        std::nullopt, false, true,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt,
        std::make_optional(detail::TestTensor{
            {12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        true, false,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt,
        std::make_optional(detail::TestTensor{
            {12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        true, true,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 12, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        false, false,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 12, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        std::make_optional(detail::TestTensor{
            {12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM}),
        false, true,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{1, 12, 128, 32},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        std::nullopt, // curPosTensor is null
        std::make_optional(detail::TestTensor{{1, 1, 12, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        std::nullopt, // attentionSink
        false,        // is_causal
        false,        // withScale
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}});

INSTANTIATE_TEST_SUITE_P(ScaledDotProductAttentionDecodeTests,
                         OpModelScaledDotProductAttentionDecodeParam,
                         scaledDotProductAttentionDecodeOpTestValues);

// === PagedScaledDotProductAttentionDecodeOp Tests ===
struct PagedScaledDotProductAttentionDecodeOpParam {
  detail::TestTensor query;
  detail::TestTensor key;
  detail::TestTensor value;
  detail::TestTensor pageTable;
  std::optional<detail::TestTensor> attentionMask;
  std::optional<detail::TestTensor> curPosTensor;
  std::optional<detail::TestTensor> attentionSink;
  bool isCausal;
  float scale;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};

class OpModelPagedScaledDotProductAttentionDecodeParam
    : public OpModelTest,
      public testing::WithParamInterface<
          PagedScaledDotProductAttentionDecodeOpParam> {
protected:
  void RunTest() {
    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
    const auto [queryShape, queryTensorLayout, queryBufferType,
                queryVirtualGrid] = GetParam().query;
    const auto [keyShape, keyTensorLayout, keyBufferType, keyVirtualGrid] =
        GetParam().key;
    const auto [valueShape, valueTensorLayout, valueBufferType,
                valueVirtualGrid] = GetParam().value;

    const auto [pageTableShape, pageTableTensorLayout, pageTableBufferType,
                pageTableVirtualGrid] = GetParam().pageTable;

    std::optional<SmallVector<int64_t>> attentionMaskShape = std::nullopt;
    std::optional<TensorMemoryLayout> attentionMaskTensorLayout = std::nullopt;
    std::optional<BufferType> attentionMaskBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> attentionMaskVirtualGrid = std::nullopt;

    std::optional<SmallVector<int64_t>> curPosTensorShape = std::nullopt;
    std::optional<TensorMemoryLayout> curPosTensorTensorLayout = std::nullopt;
    std::optional<BufferType> curPosTensorBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> curPosTensorVirtualGrid = std::nullopt;

    std::optional<SmallVector<int64_t>> attentionSinkShape = std::nullopt;
    std::optional<TensorMemoryLayout> attentionSinkTensorLayout = std::nullopt;
    std::optional<BufferType> attentionSinkBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> attentionSinkVirtualGrid = std::nullopt;

    if (auto attentionMaskDetail = GetParam().attentionMask) {
      attentionMaskShape = attentionMaskDetail->shape;
      attentionMaskTensorLayout = attentionMaskDetail->layout;
      attentionMaskBufferType = attentionMaskDetail->bufferType;
      attentionMaskVirtualGrid = attentionMaskDetail->virtualGrid;
    }

    if (auto curPosTensorDetail = GetParam().curPosTensor) {
      curPosTensorShape = curPosTensorDetail->shape;
      curPosTensorTensorLayout = curPosTensorDetail->layout;
      curPosTensorBufferType = curPosTensorDetail->bufferType;
      curPosTensorVirtualGrid = curPosTensorDetail->virtualGrid;
    }

    if (auto attentionSinkDetail = GetParam().attentionSink) {
      attentionSinkShape = attentionSinkDetail->shape;
      attentionSinkTensorLayout = attentionSinkDetail->layout;
      attentionSinkBufferType = attentionSinkDetail->bufferType;
      attentionSinkVirtualGrid = attentionSinkDetail->virtualGrid;
    }

    const auto isCausal = GetParam().isCausal;

    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;

    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    const TTNNLayoutAttr queryLayout = CreateTiledLayout(
        queryShape, queryBufferType, queryTensorLayout, queryVirtualGrid);
    const TTNNLayoutAttr keyLayout = CreateTiledLayout(
        keyShape, keyBufferType, keyTensorLayout, keyVirtualGrid);
    const TTNNLayoutAttr valueLayout = CreateTiledLayout(
        valueShape, valueBufferType, valueTensorLayout, valueVirtualGrid);
    const TTNNLayoutAttr pageTableLayout =
        CreateRowMajorLayoutInt32(pageTableShape, pageTableBufferType,
                                  pageTableTensorLayout, pageTableVirtualGrid);

    std::optional<TTNNLayoutAttr> attentionMaskLayout = std::nullopt;
    std::optional<TTNNLayoutAttr> curPosTensorLayout = std::nullopt;
    std::optional<TTNNLayoutAttr> attentionSinkLayout = std::nullopt;

    if (attentionMaskShape) {
      attentionMaskLayout = CreateTiledLayout(
          *attentionMaskShape, *attentionMaskBufferType,
          *attentionMaskTensorLayout, attentionMaskVirtualGrid);
    }
    if (curPosTensorShape) {
      curPosTensorLayout = CreateRowMajorLayoutInt32(
          *curPosTensorShape, *curPosTensorBufferType,
          *curPosTensorTensorLayout, curPosTensorVirtualGrid);
    }
    if (attentionSinkShape) {
      attentionSinkLayout = CreateTiledLayout(
          *attentionSinkShape, *attentionSinkBufferType,
          *attentionSinkTensorLayout, attentionSinkVirtualGrid);
    }

    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    const llvm::APFloat scaleAPFloat(GetParam().scale);
    std::optional<llvm::APFloat> scale = scaleAPFloat;

    auto constraintsExp =
        OpModel<PagedScaledDotProductAttentionDecodeOp>::getOpConstraints(
            CreateWorkerGrid(), queryShape, queryLayout, keyShape, keyLayout,
            valueShape, valueLayout, pageTableShape, pageTableLayout, isCausal,
            attentionMaskShape, attentionMaskLayout, curPosTensorShape,
            curPosTensorLayout, attentionSinkShape, attentionSinkLayout, scale,
            outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
  }
  // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
};

TEST_P(OpModelPagedScaledDotProductAttentionDecodeParam,
       PagedScaledDotProductAttentionDecodeOp) {

  RunTest();
}

const auto pagedScaledDotProductAttentionDecodeOpTestValues =
    testing::Values(PagedScaledDotProductAttentionDecodeOpParam{
        detail::TestTensor{
            {1, 1, 12, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{{128, 12, 32, 64},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{{128, 12, 32, 64},
                           TensorMemoryLayout::Interleaved,
                           BufferType::DRAM},
        detail::TestTensor{
            {1, 128}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt,
        detail::TestTensor{
            {1}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, true, 0.125f,
        detail::TestTensor{
            {1, 1, 12, 32}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}});

INSTANTIATE_TEST_SUITE_P(PagedScaledDotProductAttentionDecodeTests,
                         OpModelPagedScaledDotProductAttentionDecodeParam,
                         pagedScaledDotProductAttentionDecodeOpTestValues);

// === ScaledDotProductAttentionOp Tests ===

struct ScaledDotProductAttentionOpParam {
  detail::TestTensor query;
  detail::TestTensor key;
  detail::TestTensor value;
  std::optional<detail::TestTensor> attentionMask;
  bool isCausal;
  bool withScale;
  detail::TestTensor output;
  detail::ExpectedResult expectedResult;
};
class OpModelScaledDotProductAttentionParam
    : public OpModelTest,
      public testing::WithParamInterface<ScaledDotProductAttentionOpParam> {
protected:
  void RunTest() {
    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
    const auto [queryShape, queryTensorLayout, queryBufferType,
                queryVirtualGrid] = GetParam().query;
    const auto [keyShape, keyTensorLayout, keyBufferType, keyVirtualGrid] =
        GetParam().key;
    const auto [valueShape, valueTensorLayout, valueBufferType,
                valueVirtualGrid] = GetParam().value;

    std::optional<SmallVector<int64_t>> attentionMaskShape = std::nullopt;
    std::optional<TensorMemoryLayout> attentionMaskTensorLayout = std::nullopt;
    std::optional<BufferType> attentionMaskBufferType = std::nullopt;
    std::optional<SmallVector<int64_t>> attentionMaskVirtualGrid = std::nullopt;

    if (auto attentionMaskDetail = GetParam().attentionMask) {
      attentionMaskShape = attentionMaskDetail->shape;
      attentionMaskTensorLayout = attentionMaskDetail->layout;
      attentionMaskBufferType = attentionMaskDetail->bufferType;
      attentionMaskVirtualGrid = attentionMaskDetail->virtualGrid;
    }

    const auto isCausal = GetParam().isCausal;

    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = GetParam().output;

    const auto expectedLegal = GetParam().expectedResult.expectedLegal;

    const TTNNLayoutAttr queryLayout = CreateTiledLayout(
        queryShape, queryBufferType, queryTensorLayout, queryVirtualGrid);
    const TTNNLayoutAttr keyLayout = CreateTiledLayout(
        keyShape, keyBufferType, keyTensorLayout, keyVirtualGrid);
    const TTNNLayoutAttr valueLayout = CreateTiledLayout(
        valueShape, valueBufferType, valueTensorLayout, valueVirtualGrid);

    std::optional<TTNNLayoutAttr> attentionMaskLayout = std::nullopt;

    if (attentionMaskShape) {
      attentionMaskLayout = CreateTiledLayout(
          *attentionMaskShape, *attentionMaskBufferType,
          *attentionMaskTensorLayout, attentionMaskVirtualGrid);
    }

    const TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    const llvm::APFloat scaleAPFloat(1.0f);
    std::optional<llvm::APFloat> scale = std::nullopt;
    if (GetParam().withScale) {
      scale.emplace(scaleAPFloat);
    }

    std::optional<uint32_t> slidingWindowSize = std::nullopt;

    auto constraintsExp =
        OpModel<ScaledDotProductAttentionOp>::getOpConstraints(
            CreateWorkerGrid(), queryShape, queryLayout, keyShape, keyLayout,
            valueShape, valueLayout, attentionMaskShape, attentionMaskLayout,
            isCausal, scale, slidingWindowSize, outputLayout);

    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, l1PeakSize, totalPeakSize, outputSizeResult,
                  outputLayoutReadBacks] = constraintsExp.get();
      EXPECT_GE(cbSize, 0);
      EXPECT_GE(l1PeakSize, 0);
      EXPECT_GE(totalPeakSize, 0);
      EXPECT_GE(outputSizeResult, 0);
      ExpectLayoutsEQ(outputLayout, outputLayoutReadBacks[0]);
    } else {
      llvm::consumeError(constraintsExp.takeError());
    }
    // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
  }
};

TEST_P(OpModelScaledDotProductAttentionParam, ScaledDotProductAttentionOp) {
  // NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)
  RunTest();
  // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
}

const auto scaledDotProductAttentionOpTestValues = testing::Values(
    ScaledDotProductAttentionOpParam{
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, true, false,
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionOpParam{
        detail::TestTensor{
            {1, 1, 32, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 32, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        false, false,
        detail::TestTensor{
            {1, 1, 32, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},
    ScaledDotProductAttentionOpParam{
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::nullopt, true, true,
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}},

    ScaledDotProductAttentionOpParam{
        detail::TestTensor{
            {1, 1, 32, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::TestTensor{
            {1, 1, 128, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        std::make_optional(detail::TestTensor{{1, 1, 32, 128},
                                              TensorMemoryLayout::Interleaved,
                                              BufferType::DRAM}),
        false, true,
        detail::TestTensor{
            {1, 1, 32, 64}, TensorMemoryLayout::Interleaved, BufferType::DRAM},
        detail::ExpectedResult{true}}

);

INSTANTIATE_TEST_SUITE_P(ScaledDotProductAttentionTests,
                         OpModelScaledDotProductAttentionParam,
                         scaledDotProductAttentionOpTestValues);

TEST_F(OpModelTest, AssignOp) {
  const llvm::SmallVector<int64_t> tensorShape = {64, 64};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);

  auto legalExp = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(static_cast<bool>(legalExp));

  const TTNNLayoutAttr tensorLayoutDRAM_F32 = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved,
      std::nullopt, GetPhysicalGridSize(), builder.getF32Type());
  MemoryConfigAttr memoryConfigDRAM_F32 = MemoryConfigAttr::get(
      &context, tensorLayoutDRAM_F32.getMemLayout(),
      BufferTypeAttr::get(&context, tensorLayoutDRAM_F32.getBufferType()),
      std::nullopt);

  // Test AssignOp with DRAM output
  auto constraintsExp = OpModel<AssignOp>::getOpConstraints(
      workerGrid, tensorShape, tensorLayoutDRAM_F32, memoryConfigDRAM_F32,
      std::nullopt);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  OpConstraints &opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_EQ(opCstr.tensorL1PeakSize, 0);
  EXPECT_EQ(opCstr.outputL1BufferSize, 0);

  auto runtimeExp = OpModel<AssignOp>::getOpRuntime(
      tensorShape, tensorLayoutDRAM_F32, memoryConfigDRAM_F32, std::nullopt);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  const TTNNLayoutAttr tensorLayoutL1_F32 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved,
      std::nullopt, GetPhysicalGridSize(), builder.getF32Type());
  MemoryConfigAttr memoryConfigL1_F32 = MemoryConfigAttr::get(
      &context, tensorLayoutL1_F32.getMemLayout(),
      BufferTypeAttr::get(&context, tensorLayoutL1_F32.getBufferType()),
      std::nullopt);
  // Test AssignOp with L1 output
  constraintsExp = OpModel<AssignOp>::getOpConstraints(
      workerGrid, tensorShape, tensorLayoutL1_F32, memoryConfigL1_F32,
      std::nullopt);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<AssignOp>::getOpRuntime(
      tensorShape, tensorLayoutL1_F32, memoryConfigL1_F32, std::nullopt);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  const TTNNLayoutAttr tensorLayoutL1_BF16 = CreateTiledLayout(
      tensorShape, BufferType::L1, TensorMemoryLayout::Interleaved,
      std::nullopt, GetPhysicalGridSize(), builder.getBF16Type());
  MemoryConfigAttr memoryConfigL1_BF16 = MemoryConfigAttr::get(
      &context, tensorLayoutL1_BF16.getMemLayout(),
      BufferTypeAttr::get(&context, tensorLayoutL1_BF16.getBufferType()),
      std::nullopt);
  std::optional<mlir::tt::ttcore::DataType> outputDtype =
      mlir::tt::ttcore::DataType::BFloat16;

  // Test AssignOp with output dtype
  constraintsExp = OpModel<AssignOp>::getOpConstraints(
      workerGrid, tensorShape, tensorLayoutL1_F32, memoryConfigL1_BF16,
      outputDtype);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  opCstr = constraintsExp.get();
  EXPECT_GT(opCstr.cbL1PeakSize, 0);
  EXPECT_GE(opCstr.tensorL1PeakSize, 0);
  EXPECT_GT(opCstr.outputL1BufferSize, 0);

  runtimeExp = OpModel<AssignOp>::getOpRuntime(
      tensorShape, tensorLayoutL1_F32, memoryConfigL1_BF16, outputDtype);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);
}
} // namespace mlir::tt::ttnn::op_model
