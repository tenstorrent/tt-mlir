// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/TTNNOpModel.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::tt::op_model::ttnn {

class OpModelBase : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override { context.loadDialect<mlir::tt::ttnn::TTNNDialect>(); }
  void TearDown() override {}

  mlir::tt::ttnn::TTNNLayoutAttr
  CreateLayout(llvm::ArrayRef<int64_t> tensorShape,
               mlir::tt::ttnn::BufferType bufferType,
               mlir::tt::ttnn::TensorMemoryLayout tensorMemoryLayout,
               ArrayRef<int64_t> gridShape = {8, 8}) {
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, TileType::get(&context, builder.getBF16Type()),
        bufferType, GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }
};

TEST_F(OpModelBase, ReluInterleaved) {

  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};

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
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1));
  EXPECT_EQ(legal, true);
}

TEST_F(OpModelBase, ReluSharded) {
  llvm::ArrayRef<int64_t> tensorShape_a = {16 * 64 * 32, 32};

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateLayout(tensorShape_a, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateLayout(tensorShape_a, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_hs, tensorShape_a, inputLayout_l1_hs));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_hs, tensorShape_a, inputLayout_l1_i));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
  legal = std::get<0>(ReluOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_i, tensorShape_a, inputLayout_l1_hs));
  // Unary operation requires Input and Output memory layout to match.
  EXPECT_EQ(legal, false);
}

TEST_F(OpModelBase, SoftmaxInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};

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
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, -1, tensorShape, inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, -1, tensorShape, inputLayout_l1));
  EXPECT_EQ(legal, true);
}

TEST_F(OpModelBase, SoftmaxSharded) {
  llvm::ArrayRef<int64_t> tensorShape_a = {16 * 64 * 32, 32};

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateLayout(tensorShape_a, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_i =
      CreateLayout(tensorShape_a, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_hs, -2, tensorShape_a, inputLayout_l1_hs));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_hs, -2, tensorShape_a, inputLayout_l1_i));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(SoftmaxOpInterface::getOpConstraints(
      tensorShape_a, inputLayout_l1_i, -2, tensorShape_a, inputLayout_l1_hs));
  EXPECT_EQ(legal, true);
}

TEST_F(OpModelBase, AddInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {64, 1024};

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
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);

  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
}

TEST_F(OpModelBase, AddSharded) {
  llvm::ArrayRef<int64_t> tensorShape = {16 * 64 * 32, 32};

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  conversion::debug(tensorShape, inputLayout_l1_hs);

  conversion::debug(tensorShape, inputLayout_dram);

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_l1_hs));
  // FAILED AddOpInterface: TT_FATAL @
  //
  // / proj_sw / user_dev / mbezulj / work / tt - mlir / third_party / tt -
  //     metal / src / tt -
  //     metal / tt_metal / impl / kernels /
  //         kernel.cpp : 269 :
  // set_rt_args.size() == runtime_args.size() info: Illegal Runtime Args on
  // (x=0,y=7): Number of runtime args cannot be modified from 5793382363 to
  // 7!
  EXPECT_EQ(legal, false);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_dram));
  // FAILED AddOpInterface: TT_FATAL @
  //
  // / proj_sw / user_dev / mbezulj / work / tt - mlir / third_party / tt -
  //     metal / src / tt -
  //     metal / tt_metal / impl / kernels /
  //         kernel.cpp : 269 :
  // set_rt_args.size() == runtime_args.size() info: Illegal Runtime Args on
  // (x=0,y=7): Number of runtime args cannot be modified from 5793382388 to
  // 7!
  EXPECT_EQ(legal, false);
  legal = std::get<0>(AddOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1_hs));
  // FAILED AddOpInterface: TT_FATAL @
  //
  // / proj_sw / user_dev / mbezulj / work / tt - mlir / third_party / tt -
  //     metal / src / tt -
  //     metal / tt_metal / impl / kernels /
  //         kernel.cpp : 269 :
  // set_rt_args.size() == runtime_args.size() info: Illegal Runtime Args on
  // (x=0,y=7): Number of runtime args cannot be modified from 5793382372 to
  // 7!
  EXPECT_EQ(legal, false);
}

TEST_F(OpModelBase, MatmulInterleaved) {
  llvm::ArrayRef<int64_t> tensorShape = {2048, 2048};

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
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);

  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1, tensorShape, inputLayout_l1, tensorShape,
      inputLayout_l1));
  EXPECT_EQ(legal, true);
}

TEST_F(OpModelBase, MatmulSharded) {
  llvm::ArrayRef<int64_t> tensorShape = {1024, 1024};

  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_l1_hs =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                   mlir::tt::ttnn::TensorMemoryLayout::BlockSharded, {4, 4});
  mlir::tt::ttnn::TTNNLayoutAttr inputLayout_dram =
      CreateLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                   mlir::tt::ttnn::TensorMemoryLayout::Interleaved, {4, 4});

  std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>,
             std::optional<std::string>>
      opConstraint;
  bool legal = false;
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_l1_hs));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_l1_hs, tensorShape, inputLayout_dram,
      tensorShape, inputLayout_dram));
  EXPECT_EQ(legal, true);
  legal = std::get<0>(MatmulOpInterface::getOpConstraints(
      tensorShape, inputLayout_dram, tensorShape, inputLayout_dram, tensorShape,
      inputLayout_l1_hs));
  EXPECT_EQ(legal, true);
}

} // namespace mlir::tt::op_model::ttnn