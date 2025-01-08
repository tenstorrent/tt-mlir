// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

namespace mlir::tt::op_model::ttnn {

class OpModelTest : public OpModelFixture {};

TEST_F(OpModelTest, ReluInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  std::tie(legal, l1Usage, errorMsg) = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  std::tie(legal, l1Usage, errorMsg) = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);
}

TEST_F(OpModelTest, ReluSharded) {
  const llvm::SmallVector<int64_t> tensorShape = {14 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 0);
  EXPECT_EQ(output_size, tensorShape[0] * tensorShape[1] * 2 / workerCoresN300);
  EXPECT_EQ(peak_size, tensorShape[0] * tensorShape[1] * 2 / workerCoresN300);

  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_l1_i));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_i, tensorShape, inputLayout_l1_hs));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
}

TEST_F(OpModelTest, SoftmaxInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_l1);
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_dram);
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_l1);
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);
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

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_i);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  std::tie(legal, l1Usage, errorMsg) = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);
}

TEST_F(OpModelTest, AddInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {workerCoresN300, 1024};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);
}

TEST_F(OpModelTest, AddSharded) {
  const llvm::SmallVector<int64_t> tensorShape = {16 * workerCoresN300 * 32,
                                                  32};
  const auto workerGrid = CreateWorkerGrid(gridShapeHwN300);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{8, 1});
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  std::tie(legal, errorMsg) = Device::getDeviceConstraints(workerGrid);
  EXPECT_TRUE(legal);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_l1_hs);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 32768);
  EXPECT_EQ(peak_size, 229376);
  EXPECT_EQ(output_size, 229376);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_dram);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 65536);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  std::tie(legal, l1Usage, errorMsg) = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1_hs);
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 65536);
  EXPECT_EQ(peak_size, 229376);
  EXPECT_EQ(output_size, 229376);
}

class OpModelMatmulParam
    : public OpModelTest,
      public testing::WithParamInterface<
          std::tuple<llvm::SmallVector<int64_t>,         // input shape A
                     mlir::tt::ttnn::TensorMemoryLayout, // input layout A
                     mlir::tt::ttnn::BufferType,         // input buffer type A
                     llvm::SmallVector<int64_t>,         // input virtual grid A
                     llvm::SmallVector<int64_t>,         // input shape B
                     mlir::tt::ttnn::TensorMemoryLayout, // input layout B
                     mlir::tt::ttnn::BufferType,         // input buffer type B
                     llvm::SmallVector<int64_t>,         // input virtual grid B
                     llvm::SmallVector<int64_t>,         // output shape
                     mlir::tt::ttnn::TensorMemoryLayout, // output layout
                     mlir::tt::ttnn::BufferType,         // output buffer type
                     llvm::SmallVector<int64_t>,         // output virtual grid
                     llvm::SmallVector<int64_t>,         // physical grid
                     bool,                               // expected valid
                     size_t,                             // expected cb size
                     size_t,                             // expected peak size
                     size_t                              // expected output size
                     >> {};

TEST_P(OpModelMatmulParam, MatmulParam) {

  auto params = GetParam();
  llvm::SmallVector<int64_t> inputShapeA = std::get<0>(params);
  mlir::tt::ttnn::TensorMemoryLayout inputTensorLayoutA = std::get<1>(params);
  mlir::tt::ttnn::BufferType inputBufferTypeA = std::get<2>(params);
  llvm::SmallVector<int64_t> inputVirtualGridA = std::get<3>(params);
  llvm::SmallVector<int64_t> inputShapeB = std::get<4>(params);
  mlir::tt::ttnn::TensorMemoryLayout inputTensorLayoutB = std::get<5>(params);
  mlir::tt::ttnn::BufferType inputBufferTypeB = std::get<6>(params);
  llvm::SmallVector<int64_t> inputVirtualGridB = std::get<7>(params);
  llvm::SmallVector<int64_t> outputShape = std::get<8>(params);
  mlir::tt::ttnn::TensorMemoryLayout outputTensorLayout = std::get<9>(params);
  mlir::tt::ttnn::BufferType outputBufferType = std::get<10>(params);
  llvm::SmallVector<int64_t> outputVirtualGrid = std::get<11>(params);
  llvm::SmallVector<int64_t> physicalGrid = std::get<12>(params);
  bool expectedLegal = std::get<13>(params);
  size_t expectedCbSize = std::get<14>(params);
  size_t expectedPeakSize = std::get<15>(params);
  size_t expectedOutputSize = std::get<16>(params);

  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutA = CreateTiledLayout(
      inputShapeA, inputBufferTypeA, inputTensorLayoutA, inputVirtualGridA);
  const mlir::tt::ttnn::TTNNLayoutAttr inputLayoutB = CreateTiledLayout(
      inputShapeB, inputBufferTypeB, inputTensorLayoutB, inputVirtualGridB);
  const mlir::tt::ttnn::TTNNLayoutAttr outputLayout = CreateTiledLayout(
      outputShape, outputBufferType, outputTensorLayout, outputVirtualGrid);

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cbSize = 0;
  size_t peakSize = 0;
  size_t outputSize = 0;

  std::tie(legal, l1Usage, errorMsg) = MatmulOpInterface::getOpConstraints(
      inputShapeA, inputLayoutA, inputShapeB, inputLayoutB, outputShape,
      outputLayout, false, false);
  EXPECT_EQ(legal, expectedLegal);
  EXPECT_EQ(l1Usage.has_value(), expectedLegal);
  EXPECT_EQ(errorMsg.has_value(), !expectedLegal);

  if (l1Usage.has_value()) {
    std::tie(cbSize, peakSize, outputSize) = l1Usage.value();
    EXPECT_EQ(cbSize, expectedCbSize);
    EXPECT_EQ(peakSize, expectedPeakSize);
    EXPECT_EQ(outputSize, expectedOutputSize);
  }

  std::cout << errorMsg.value_or("No errors") << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterleavedTests, OpModelMatmulParam,
    ::testing::Values(
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 753664, 0, 0),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 151552, 151552),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 0, 0),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 151552, 151552),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 0, 0),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 151552, 151552),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 0, 0),
        std::make_tuple(
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{2048, 2048},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{8, 8},
            llvm::SmallVector<int64_t>{8, 8}, true, 786432, 151552, 151552)));

INSTANTIATE_TEST_SUITE_P(
    MatmulShardedTests, OpModelMatmulParam,
    ::testing::Values(
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{7, 8}, true, 430144, 114688, 114688),
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{7, 8}, false, -1, -1, -1),
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{7, 8}, true, 262144, 401408,
            401408), // matmul bug shards to less cores
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{7, 8}, true, 544832, 0, 0),
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{56, 1},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{7, 8}, false, -1, -1, -1),
        std::make_tuple(
            llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{1, 56},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{1, 56},
            llvm::SmallVector<int64_t>{7, 8}, true, 8256, 2048, 2048),
        std::make_tuple(
            llvm::SmallVector<int64_t>{56 * 32, 1 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{56, 1},
            llvm::SmallVector<int64_t>{1 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
            mlir::tt::ttnn::BufferType::DRAM, llvm::SmallVector<int64_t>{7, 8},
            llvm::SmallVector<int64_t>{56 * 32, 56 * 32},
            mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
            mlir::tt::ttnn::BufferType::L1, llvm::SmallVector<int64_t>{56, 1},
            llvm::SmallVector<int64_t>{7, 8}, true, 114688, 114688, 114688)));
} // namespace mlir::tt::op_model::ttnn
