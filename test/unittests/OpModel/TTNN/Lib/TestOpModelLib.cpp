// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "SingletonDeviceContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

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

class OpModelUnaryEltwiseParam : public OpModelTest,
                                 public testing::WithParamInterface<
                                     std::tuple<detail::TestTensor, // input
                                                detail::TestTensor, // output
                                                detail::ExpectedResult>> {};

TEST_P(OpModelUnaryEltwiseParam, Relu) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);

  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<2>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = ReluOpInterface::getOpConstraints(
      inputShape, inputLayout, outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = ReluOpInterface::getOpRuntime(inputShape, inputLayout,
                                                  outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReluTests, OpModelUnaryEltwiseParam,
    ::testing::Values(
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
            detail::ExpectedResult{false})));

class OpModelReductionParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<detail::TestTensor,                        // input
                     detail::TestTensor,                        // output
                     std::optional<llvm::SmallVector<int64_t>>, // dim arg
                     bool,                                      // keep dim
                     detail::ExpectedResult>> {};

TEST_P(OpModelReductionParam, Reduction) {
  auto params = GetParam();
  const auto [inputShape, inputTensorLayout, inputBufferType,
              inputVirtualGrid] = std::get<0>(params);

  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<1>(params);
  const auto dimArg = std::get<2>(params);
  const auto keepDim = std::get<3>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<4>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp = MeanOpInterface::getOpConstraints(
      inputShape, inputLayout, dimArg, keepDim, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  auto runtimeExp = MeanOpInterface::getOpRuntime(
      inputShape, inputLayout, dimArg, keepDim, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MeanTests, OpModelReductionParam,
    ::testing::Values(
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        llvm::SmallVector<int64_t>{1}, true,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        llvm::SmallVector<int64_t>{1, 2}, false,
                        detail::ExpectedResult{false, 0, 0, 0}),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        llvm::SmallVector<int64_t>{1, 0}, false,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        llvm::SmallVector<int64_t>{1}, false,
                        detail::ExpectedResult{true, 12288, 0, 0})));

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
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 2048);
  EXPECT_EQ(peak_size, 2048);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_l1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 2048);
  EXPECT_EQ(peak_size, 2048);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

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
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 262144);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  auto runtimeExp = ReshapeOpInterface::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = ReshapeOpInterface::getOpConstraints(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutL1);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 262144);
  EXPECT_EQ(output_size, 2048);
  EXPECT_EQ(peak_size, 4096);

  runtimeExp = ReshapeOpInterface::getOpRuntime(
      tensorShape, layoutDRAM, {workerCoresN300 * 4, 256}, layoutL1);
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
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor, true);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 262144);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  auto runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor, true);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = ToLayoutOpInterface::getOpConstraints(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutL1RowMajorHS, true);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());

  runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutL1RowMajorHS, true);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());

  constraintsExp = ToLayoutOpInterface::getOpConstraints(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor, false);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 262144);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  runtimeExp = ToLayoutOpInterface::getOpRuntime(
      tensorShape, layoutDRAMTiled, std::nullopt, layoutDRAMRowMajor, false);
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
      tensorShape, layoutDRAM, 0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  auto runtimeExp = TransposeOpInterface::getOpRuntime(tensorShape, layoutDRAM,
                                                       0, 1, layoutDRAM);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TransposeOpInterface::getOpConstraints(
      tensorShape, layoutDRAM, 0, 1, layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 2048);
  EXPECT_EQ(peak_size, 2048);

  runtimeExp = TransposeOpInterface::getOpRuntime(tensorShape, layoutDRAM, 0, 1,
                                                  layoutL1Interleaved);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TransposeOpInterface::getOpConstraints(
      tensorShape, layoutL1Interleaved, 0, 1, layoutL1WSharded);
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
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_i);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  constraintsExp = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  std::tie(cb_size, peak_size, output_size) = constraintsExp.get();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

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
      tensorShape, inputLayoutDRAMIBF16,
      DataTypeAttr::get(&context, DataType::Float32), tensorShape,
      inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(constraintsExp));
  auto [cb_size, peak_size, output_size] = constraintsExp.get();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  auto runtimeExp = TypecastOpInterface::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      DataTypeAttr::get(&context, DataType::Float32), tensorShape,
      inputLayoutDRAMIF32);
  EXPECT_TRUE(static_cast<bool>(runtimeExp));
  EXPECT_TRUE(runtimeExp.get() > 0);

  constraintsExp = TypecastOpInterface::getOpConstraints(
      tensorShape, inputLayoutDRAMIBF16,
      DataTypeAttr::get(&context, DataType::Float32), tensorShape,
      inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(constraintsExp));
  llvm::consumeError(constraintsExp.takeError());
  runtimeExp = TypecastOpInterface::getOpRuntime(
      tensorShape, inputLayoutDRAMIBF16,
      DataTypeAttr::get(&context, DataType::Float32), tensorShape,
      inputLayoutL1HSBF16);
  EXPECT_FALSE(static_cast<bool>(runtimeExp));
  llvm::consumeError(runtimeExp.takeError());
}

enum class OpType { Add, Mul };
class OpModelBinaryEltwiseParam : public OpModelTest,
                                  public testing::WithParamInterface<
                                      std::tuple<OpType,
                                                 detail::TestTensor, // inputA
                                                 detail::TestTensor, // inputB
                                                 detail::TestTensor, // output
                                                 detail::ExpectedResult>> {

protected:
  std::map<OpType,
           std::function<llvm::Expected<size_t>(
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>>
      runtimeMap = {
          {OpType::Add, AddOpInterface::getOpRuntime},
          {OpType::Mul, MultiplyOpInterface::getOpRuntime},
      };

  std::map<OpType,
           std::function<llvm::Expected<std::tuple<size_t, size_t, size_t>>(
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr,
               llvm::ArrayRef<int64_t>, mlir::tt::ttnn::TTNNLayoutAttr)>>
      constraintsMap = {
          {OpType::Add, AddOpInterface::getOpConstraints},
          {OpType::Mul, MultiplyOpInterface::getOpConstraints},
      };

  void RunTest() {
    const auto opType = get<0>(GetParam());
    const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
                inputVirtualGridA] = std::get<1>(GetParam());
    const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
                inputVirtualGridB] = std::get<2>(GetParam());
    const auto [outputShape, outputTensorLayout, outputBufferType,
                outputVirtualGrid] = std::get<3>(GetParam());
    const auto [expectedLegal, expectedCbSize, expectedPeakSize,
                expectedOutputSize] = std::get<4>(GetParam());

    const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
        inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
    const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
        inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
    const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
        outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

    auto constraintsExp =
        constraintsMap[opType](inputShapeA, inputLayoutA, inputShapeB,
                               inputLayoutB, outputShape, outputLayout);
    // Manually cast to bool because EXPECT_TRUE requires a const bool operator
    // which llvm::Expected<T> does not have
    EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
    if (expectedLegal) {
      const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
      EXPECT_EQ(cbSize, expectedCbSize);
      EXPECT_EQ(peakSize, expectedPeakSize);
      EXPECT_EQ(outputSize, expectedOutputSize);
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

INSTANTIATE_TEST_SUITE_P(
    AddTests, OpModelBinaryEltwiseParam,
    ::testing::Values(
        std::make_tuple(OpType::Add, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            OpType::Add, detail::interleavedN300X1024Dram,
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            detail::ExpectedResult{false, 0, 0,
                                   0}), // incompatible dimensions at the input
        std::make_tuple(OpType::Add, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Add, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Add,
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                            mlir::tt::ttnn::BufferType::L1,
                            llvm::SmallVector<int64_t>{8, 1}},
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                            mlir::tt::ttnn::BufferType::DRAM},
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                            mlir::tt::ttnn::BufferType::L1,
                            llvm::SmallVector<int64_t>{8, 1}},
                        detail::ExpectedResult{true, 32768, 262144, 262144}),
        std::make_tuple(
            OpType::Add,
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 65536, 0, 0}),
        std::make_tuple(
            OpType::Add,
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::ExpectedResult{true, 65536, 262144, 262144})));

INSTANTIATE_TEST_SUITE_P(
    MulTests, OpModelBinaryEltwiseParam,
    ::testing::Values(
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            OpType::Mul, detail::interleavedN300X1024Dram,
            detail::interleaved2048X2048Dram, detail::interleaved2048X2048Dram,
            detail::ExpectedResult{false, 0, 0,
                                   0}), // incompatible dimensions at the input
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024L1,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Mul, detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1,
                        detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(OpType::Mul,
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                            mlir::tt::ttnn::BufferType::L1,
                            llvm::SmallVector<int64_t>{8, 1}},
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                            mlir::tt::ttnn::BufferType::DRAM},
                        detail::TestTensor{
                            {16 * OpModelFixture::workerCoresN300 * 32, 32},
                            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                            mlir::tt::ttnn::BufferType::L1,
                            llvm::SmallVector<int64_t>{8, 1}},
                        detail::ExpectedResult{true, 32768, 262144, 262144}),
        std::make_tuple(
            OpType::Mul,
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 65536, 0, 0}),
        std::make_tuple(
            OpType::Mul,
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::ExpectedResult{true, 65536, 262144, 262144})));

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
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, outputShape,
      outputLayout, false, false);

  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);
  if (expectedLegal) {
    const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
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
                     bool, bool>> {};

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
  const auto constraintsLegal = std::get<13>(params);
  const auto runtimeLegal = std::get<14>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout = CreateTiledLayout(
      inputShape, inputBufferType, inputTensorLayout, inputVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr weightLayout = CreateTiledLayout(
      weightShape, weightBufferType, weightTensorLayout, weightVirtualGrid);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  // Device hangs otherwise.
  SingletonDeviceContext::resetInstance();

  auto constraintsExp = Conv2dOpInterface::getOpConstraints(
      inputShape, inputLayout, weightShape, weightLayout, std::nullopt,
      std::nullopt, in_channels, out_channels, batch_size, input_height,
      input_width, kernel_size, stride, padding, dilation, groups, std::nullopt,
      outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(constraintsExp), constraintsLegal);
  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
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
      outputShape, outputLayout);
  // Manually cast to bool because EXPECT_TRUE requires a const bool operator
  // which llvm::Expected<T> does not have
  EXPECT_EQ(static_cast<bool>(runtimeExp), runtimeLegal);
  if (runtimeExp) {
    const auto runtime = runtimeExp.get();
    EXPECT_GT(runtime, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(runtimeExp.takeError());
  }
}

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

INSTANTIATE_TEST_SUITE_P(
    Conv2dTests, OpModelConv2dParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{1, 1, 50176, 3},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 1, 1568, 64},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{1, 1, 12544, 64},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            3, 64, 1, 224, 224, llvm::SmallVector<int32_t>{7, 7},
            llvm::SmallVector<int32_t>{2, 2}, llvm::SmallVector<int32_t>{3, 3},
            llvm::SmallVector<int32_t>{1, 1}, 1, false, true),
        std::make_tuple(detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024Dram,
                        detail::interleavedN300X1024L1, 3, 64, 32, 224, 224,
                        llvm::SmallVector<int32_t>{7, 7},
                        llvm::SmallVector<int32_t>{2, 2},
                        llvm::SmallVector<int32_t>{3, 3},
                        llvm::SmallVector<int32_t>{1, 1}, 1, false, false)));

TEST_P(OpModelMaxPool2DParam, MaxPool2DParam) {
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

  auto constraintsExp = MaxPool2DInterface::getOpConstraints(
      inputShape, inputLayout, batchSize, inputHeight, inputWidth,
      inputChannels, kernelSize, stride, padding, dilation, ceilMode,
      outputShape, outputLayout);
  if (!constraintsExp) {
    std::cout << "Error: " << llvm::toString(constraintsExp.takeError())
              << std::endl;
  }
  EXPECT_EQ(static_cast<bool>(constraintsExp), expectedLegal);

  if (constraintsExp) {
    const auto [cbSize, peakSize, outputSize] = constraintsExp.get();
    EXPECT_GT(cbSize, 0);
    EXPECT_GT(peakSize, 0);
    EXPECT_GT(outputSize, 0);
  } else {
    // Must clean up the error
    llvm::consumeError(constraintsExp.takeError());
  }

  SingletonDeviceContext::resetInstance();

  auto runtimeExp = MaxPool2DInterface::getOpRuntime(
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

} // namespace mlir::tt::op_model::ttnn
