// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"
#include <llvm/Support/Error.h>
#include <optional>

namespace mlir::tt::op_model::ttnn {

class OpModelTest : public OpModelFixture {};

namespace detail {
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
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 8192, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 8192, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 8192, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
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

// TEST_F(OpModelTest, SoftmaxInterleaved) {
//   const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
//   const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
//   const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
//       CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
//                         mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
//   const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
//       CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
//                         mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

//   bool legal = false;
//   std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
//   std::optional<std::string> errorMsg = "";
//   size_t cb_size = 0;
//   size_t peak_size = 0;
//   size_t output_size = 0;

//   std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
//   EXPECT_TRUE(legal);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
//   EXPECT_EQ(legal, true);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 137216);
//   EXPECT_EQ(output_size, 0);
//   EXPECT_EQ(peak_size, 0);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_l1);
//   EXPECT_EQ(legal, true);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 137216);
//   EXPECT_EQ(output_size, 2048);
//   EXPECT_EQ(peak_size, 2048);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_dram);
//   EXPECT_EQ(legal, true);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 137216);
//   EXPECT_EQ(output_size, 0);
//   EXPECT_EQ(peak_size, 0);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_l1);
//   EXPECT_EQ(legal, true);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 137216);
//   EXPECT_EQ(output_size, 2048);
//   EXPECT_EQ(peak_size, 2048);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
//   EXPECT_TRUE(legal);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 137216);
//   EXPECT_EQ(output_size, 0);
//   EXPECT_EQ(peak_size, 0);

//   std::vector<std::tuple<mlir::tt::ttnn::TTNNLayoutAttr,
//                          mlir::tt::ttnn::TTNNLayoutAttr>>
//       layout_combinations = {{inputLayout_dram, inputLayout_dram},
//                              {inputLayout_l1, inputLayout_dram},
//                              {inputLayout_dram, inputLayout_l1},
//                              {inputLayout_l1, inputLayout_l1}};
//   for (const auto &[input_layout, output_layout] : layout_combinations) {
//     auto runtimeExp = SoftmaxOpInterface::getOpRuntime(
//         tensorShape, input_layout, -1, tensorShape, output_layout);
//     EXPECT_TRUE(static_cast<bool>(runtimeExp));
//     EXPECT_TRUE(runtimeExp.get() > 0);
//   }
// }

// TEST_F(OpModelTest, SoftmaxSharded) {
//   const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
//                                                   32};
//   const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
//   const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
//       CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
//                         mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
//   const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
//       CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
//                         mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

//   bool legal = false;
//   std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
//   std::optional<std::string> errorMsg = "";
//   size_t cb_size = 0;
//   size_t peak_size = 0;
//   size_t output_size = 0;

//   std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
//   EXPECT_TRUE(legal);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_hs);
//   EXPECT_TRUE(legal);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 24576);
//   EXPECT_EQ(output_size, 32768);
//   EXPECT_EQ(peak_size, 32768);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_i);
//   EXPECT_TRUE(legal);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 24576);
//   EXPECT_EQ(output_size, 32768);
//   EXPECT_EQ(peak_size, 32768);

//   std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
//       tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs);
//   EXPECT_TRUE(legal);
//   EXPECT_TRUE(l1Usage.has_value());
//   EXPECT_FALSE(errorMsg.has_value());
//   std::tie(cb_size, peak_size, output_size) = l1Usage.value();
//   EXPECT_EQ(cb_size, 24576);
//   EXPECT_EQ(output_size, 32768);
//   EXPECT_EQ(peak_size, 32768);

//   auto runtimeExp = SoftmaxOpInterface::getOpRuntime(
//       tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs);
//   EXPECT_TRUE(static_cast<bool>(runtimeExp));
//   EXPECT_TRUE(runtimeExp.get() > 0);
// }

class OpModelBinaryEltwiseParam : public OpModelTest,
                                  public testing::WithParamInterface<
                                      std::tuple<detail::TestTensor, // inputA
                                                 detail::TestTensor, // inputB
                                                 detail::TestTensor, // output
                                                 detail::ExpectedResult>> {};

TEST_P(OpModelBinaryEltwiseParam, Add) {
  auto params = GetParam();
  const auto [inputShapeA, inputTensorLayoutA, inputBufferTypeA,
              inputVirtualGridA] = std::get<0>(params);
  const auto [inputShapeB, inputTensorLayoutB, inputBufferTypeB,
              inputVirtualGridB] = std::get<1>(params);
  const auto [outputShape, outputTensorLayout, outputBufferType,
              outputVirtualGrid] = std::get<2>(params);
  const auto [expectedLegal, expectedCbSize, expectedPeakSize,
              expectedOutputSize] = std::get<3>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  auto constraintsExp =
      AddOpInterface::getOpConstraints(inputShapeA, inputLayoutA, inputShapeB,
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

  auto runtimeExp =
      AddOpInterface::getOpRuntime(inputShapeA, inputLayoutA, inputShapeB,
                                   inputLayoutB, outputShape, outputLayout);
  EXPECT_EQ(static_cast<bool>(runtimeExp), expectedLegal);
  if (expectedLegal) {
    EXPECT_TRUE(runtimeExp.get() > 0);
  } else {
    llvm::consumeError(runtimeExp.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(
    AddTests, OpModelBinaryEltwiseParam,
    ::testing::Values(
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::ExpectedResult{true, 12288, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{{OpModelFixture::workerCoresN300, 1024},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1},
            detail::ExpectedResult{true, 12288, 2048, 2048}),
        std::make_tuple(
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::TestTensor{{16 * OpModelFixture::workerCoresN300 * 32, 32},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM},
            detail::TestTensor{
                {16 * OpModelFixture::workerCoresN300 * 32, 32},
                mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                mlir::tt::ttnn::BufferType::L1,
                llvm::SmallVector<int64_t>{8, 1}},
            detail::ExpectedResult{true, 32768, 262144, 262144}),
        std::make_tuple(
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
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 655360, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 131072, 131072}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::DRAM,
                               llvm::SmallVector<int64_t>{8, 8}},
            llvm::SmallVector<int64_t>{8, 8},
            detail::ExpectedResult{true, 786432, 0, 0}),
        std::make_tuple(
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
            detail::TestTensor{{2048, 2048},
                               mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                               mlir::tt::ttnn::BufferType::L1,
                               llvm::SmallVector<int64_t>{8, 8}},
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
} // namespace mlir::tt::op_model::ttnn
