// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"
#include "gtest/gtest.h"

#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"

namespace mlir::tt::op_model::ttnn {

class OpModelBase : public OpModelFixture {
public:
  mlir::tt::ttnn::TTNNLayoutAttr
  CreateLayout(llvm::ArrayRef<int64_t> tensorShape,
               mlir::tt::ttnn::BufferType bufferType,
               mlir::tt::ttnn::TensorMemoryLayout tensorMemoryLayout,
               ArrayRef<int64_t> virtualGrid = {8, 8}) {
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, TileType::get(&context, builder.getBF16Type()),
        bufferType, GridAttr::get(&context, virtualGrid),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }
};

TEST_F(OpModelBase, ReluInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  auto workerGrid =
      CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  opConstraint = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 8192);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);
}

TEST_F(OpModelBase, ReluSharded) {
  llvm::ArrayRef<int64_t> tensorShape = {16 * 64 * 32, 32};
  auto workerGrid = CreateWorkerGrid(
      mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 8});
  auto virtualGrid = GetVirtualGridShape(
      tensorShape, mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs = CreateLayout(
      tensorShape, mlir::tt::ttnn::BufferType::L1,
      mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, virtualGrid);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_l1_hs,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 0);
  EXPECT_EQ(output_size, tensorShape[0] * tensorShape[1] * 2 / 64);
  EXPECT_EQ(peak_size, tensorShape[0] * tensorShape[1] * 2 / 64);

  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_l1_i,
      workerGrid));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_i, tensorShape, inputLayout_l1_hs,
      workerGrid));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
}

TEST_F(OpModelBase, SoftmaxInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  auto workerGrid =
      CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_l1,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_dram,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_EQ(legal, true);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 4096);
  EXPECT_EQ(peak_size, 4096);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 137216);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);
}

TEST_F(OpModelBase, SoftmaxSharded) {
  llvm::ArrayRef<int64_t> tensorShape = {16 * 64 * 32, 32};
  auto workerGrid = CreateWorkerGrid(
      mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 8});

  auto virtualGrid = GetVirtualGridShape(
      tensorShape, mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs = CreateLayout(
      tensorShape, mlir::tt::ttnn::BufferType::L1,
      mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, virtualGrid);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_hs,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_i,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 38912);
  EXPECT_EQ(peak_size, 38912);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_i, -2, tensorShape, inputLayout_l1_hs,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);

  opConstraint = SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, -2, tensorShape, inputLayout_l1_hs,
      workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 24576);
  EXPECT_EQ(output_size, 32768);
  EXPECT_EQ(peak_size, 32768);
}

TEST_F(OpModelBase, AddInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};
  auto workerGrid =
      CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(peak_size, 4096);
  EXPECT_EQ(output_size, 4096);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 12288);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);
}

TEST_F(OpModelBase, AddSharded) {
  llvm::ArrayRef<int64_t> tensorShape = {16 * 64 * 32, 32};
  auto workerGrid = CreateWorkerGrid(
      mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::HeightSharded, {8, 1});
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_l1_hs, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 32768);
  EXPECT_EQ(peak_size, 262144);
  EXPECT_EQ(output_size, 262144);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_dram, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 65536);
  EXPECT_EQ(peak_size, 0);
  EXPECT_EQ(output_size, 0);

  opConstraint = AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1_hs, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 65536);
  EXPECT_EQ(peak_size, 262144);
  EXPECT_EQ(output_size, 262144);
}

TEST_F(OpModelBase, MatmulInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {2048, 2048};
  auto workerGrid =
      CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved, {8, 8});

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1 =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 151552);
  EXPECT_EQ(peak_size, 151552);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 151552);
  EXPECT_EQ(peak_size, 151552);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 151552);
  EXPECT_EQ(peak_size, 151552);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 0);
  EXPECT_EQ(peak_size, 0);

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 786432);
  EXPECT_EQ(output_size, 151552);
  EXPECT_EQ(peak_size, 151552);
}

TEST_F(OpModelBase, MatmulSharded) {
  const llvm::ArrayRef<int64_t> tensorShape = {1024, 1024};
  const SmallVector<int64_t> gridShape = {4, 4};
  auto workerGrid = CreateWorkerGrid(
      mlir::tt::ttnn::TensorMemoryLayout::BlockSharded, gridShape);

  mlir::tt::ttnn::TTNNLayoutAttr layout_l1_hs =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::BlockSharded, gridShape);
  mlir::tt::ttnn::TTNNLayoutAttr layout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved, gridShape);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;

  bool legal = false;
  std::optional<std::tuple<size_t, size_t, size_t>> l1Usage = std::nullopt;
  std::optional<std::string> errorMsg = "";
  size_t cb_size = 0;
  size_t peak_size = 0;
  size_t output_size = 0;

  opConstraint = MatmulOpInterface::getOpConstraints(
      tensorShape, layout_l1_hs, tensorShape, layout_dram, tensorShape,
      layout_l1_hs, false, false, workerGrid);
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 524352);
  EXPECT_EQ(output_size, 131072);
  EXPECT_EQ(peak_size, 131072);

  opConstraint = (MatmulOpInterface::getOpConstraints(
      tensorShape, layout_l1_hs, tensorShape, layout_dram, tensorShape,
      layout_dram, false, false, workerGrid));
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 655424);
  EXPECT_EQ(output_size, 0); // dram
  EXPECT_EQ(peak_size, 0);

  opConstraint = (MatmulOpInterface::getOpConstraints(
      tensorShape, layout_dram, tensorShape, layout_dram, tensorShape,
      layout_l1_hs, false, false, workerGrid));
  std::tie(legal, l1Usage, errorMsg) = opConstraint;
  EXPECT_TRUE(legal);
  EXPECT_TRUE(l1Usage.has_value());
  EXPECT_FALSE(errorMsg.has_value());
  std::tie(cb_size, peak_size, output_size) = l1Usage.value();
  EXPECT_EQ(cb_size, 262144);
  EXPECT_EQ(output_size, 524288);
  EXPECT_EQ(peak_size, 524288);
}

} // namespace mlir::tt::op_model::ttnn