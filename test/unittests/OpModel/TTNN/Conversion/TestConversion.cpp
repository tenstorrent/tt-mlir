// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <llvm-gtest/gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>
#include <tuple>

#include "Conversion.hpp"
#include "MetalHeaders.h"
#include "common/core_coord.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttnn/tensor/types.hpp"

class MlirToTtnnConversion : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override { context.loadDialect<mlir::tt::ttnn::TTNNDialect>(); }

  mlir::tt::ttnn::TTNNLayoutAttr CreateTiledLayout(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridShape = {8, 8}) {
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape,
        mlir::tt::TileType::get(&context, builder.getBF16Type()), bufferType,
        mlir::tt::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }
  mlir::tt::ttnn::TTNNLayoutAttr CreateRowMajorLayout(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridShape = {8, 8}) {
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, builder.getBF16Type(), bufferType,
        mlir::tt::GridAttr::get(&context, gridShape),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }
};

//================================================================================
// getDataType
//================================================================================
class MlirToTtnnConversionSimpleShape
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<mlir::SmallVector<int64_t>> {};

TEST_P(MlirToTtnnConversionSimpleShape, SimpleShape) {
  const auto &tensorShape = GetParam();
  const auto &shape = mlir::tt::op_model::ttnn::conversion::getSimpleShape(
      llvm::ArrayRef<int64_t>(tensorShape));

  EXPECT_EQ(shape.size(), tensorShape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    EXPECT_EQ(shape[i], tensorShape[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ToSimpleShape, MlirToTtnnConversionSimpleShape,
    ::testing::Values(mlir::SmallVector<int64_t>{64, 32},
                      mlir::SmallVector<int64_t>{64, 32, 128},
                      mlir::SmallVector<int64_t>{64, 32, 128, 256}));

//================================================================================
// getSimpleShape
//================================================================================
class MlirToTtnnConversionDataType
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::DataType, ::tt::tt_metal::DataType>> {};

TEST_P(MlirToTtnnConversionDataType, DataType) {
  const auto &dataType = std::get<0>(GetParam());
  const auto &expectedDataType = std::get<1>(GetParam());

  llvm::ArrayRef<int64_t> tensorShape = {32, 32};
  auto layout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, tensorShape,
      mlir::tt::TileType::get(&context, {32, 32}, dataType),
      mlir::tt::ttnn::BufferType::L1, mlir::tt::GridAttr::get(&context, {8, 8}),
      mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
          &context, mlir::tt::ttnn::TensorMemoryLayout::Interleaved));

  auto convertedDataType =
      mlir::tt::op_model::ttnn::conversion::getDataType(layout);
  EXPECT_EQ(convertedDataType, expectedDataType);
}

INSTANTIATE_TEST_SUITE_P(
    ToDataType, MlirToTtnnConversionDataType,
    ::testing::Values(std::make_tuple(mlir::tt::DataType::Float32,
                                      ::tt::tt_metal::DataType::FLOAT32),
                      std::make_tuple(mlir::tt::DataType::BFloat16,
                                      ::tt::tt_metal::DataType::BFLOAT16),
                      std::make_tuple(mlir::tt::DataType::BFP_BFloat8,
                                      ::tt::tt_metal::DataType::BFLOAT8_B),
                      std::make_tuple(mlir::tt::DataType::BFP_BFloat4,
                                      ::tt::tt_metal::DataType::BFLOAT4_B),
                      std::make_tuple(mlir::tt::DataType::UInt32,
                                      ::tt::tt_metal::DataType::UINT32),
                      std::make_tuple(mlir::tt::DataType::UInt16,
                                      ::tt::tt_metal::DataType::UINT16),
                      std::make_tuple(mlir::tt::DataType::UInt8,
                                      ::tt::tt_metal::DataType::UINT8)));

//================================================================================
// getShardShape
//================================================================================
TEST_F(MlirToTtnnConversion, ShardShape) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};

  // block sharded to 8x8 grid
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    const auto shardShape =
        mlir::tt::op_model::ttnn::conversion::getShardShape(layout);
    layout.dump(); // TODO(mbezulj) remove
    EXPECT_EQ(shardShape[0], tensorShape[0] / 8);
    EXPECT_EQ(shardShape[1], tensorShape[1] / 8);
  }

  // height sharded to 8x8 grid
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
    const auto shardShape =
        mlir::tt::op_model::ttnn::conversion::getShardShape(layout);
    layout.dump(); // TODO(mbezulj) remove

    // There is a known issue with the height sharded layout, when it's fixed
    // this test should be updated.
    EXPECT_NE(shardShape[0], tensorShape[0] / 64);
    EXPECT_NE(shardShape[1], tensorShape[1]);
  }

  // width sharded to 8x8 grid
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::WidthSharded);
    const auto shardShape =
        mlir::tt::op_model::ttnn::conversion::getShardShape(layout);
    layout.dump(); // TODO(mbezulj) remove

    // There is a known issue with the width sharded layout, when it's fixed
    // this test should be updated.
    EXPECT_NE(shardShape[0], tensorShape[0]);
    EXPECT_NE(shardShape[1], tensorShape[1] / 64);
  }
}

//================================================================================
// getPageLayout
//================================================================================
TEST_F(MlirToTtnnConversion, PageLayout) {
  llvm::ArrayRef<int64_t> tensorShape = {16 * 64 * 32, 32};

  mlir::tt::ttnn::TTNNLayoutAttr tiledLayout =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  EXPECT_EQ(mlir::tt::op_model::ttnn::conversion::getPageLayout(tiledLayout),
            ::tt::tt_metal::Layout::TILE);

  mlir::tt::ttnn::TTNNLayoutAttr rowLayout =
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
  EXPECT_EQ(mlir::tt::op_model::ttnn::conversion::getPageLayout(rowLayout),
            ::tt::tt_metal::Layout::ROW_MAJOR);
}

//================================================================================
// getCoreRangeSet
//================================================================================
class MlirToTtnnConversionCoreRangeSet
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<llvm::SmallVector<int64_t>> {};

TEST_P(MlirToTtnnConversionCoreRangeSet, CoreRangeSet) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};
  const auto &grid = GetParam();

  const auto layout =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded, grid);
  const auto coreRangeSet =
      mlir::tt::op_model::ttnn::conversion::getCoreRangeSet(layout);

  EXPECT_EQ(coreRangeSet.size(), 1);
  EXPECT_EQ(coreRangeSet.ranges()[0].start_coord, CoreCoord(0, 0));
  EXPECT_EQ(coreRangeSet.ranges()[0].end_coord,
            CoreCoord(grid[0] - 1, grid[1] - 1));
}

INSTANTIATE_TEST_SUITE_P(ToCoreRangeSet, MlirToTtnnConversionCoreRangeSet,
                         ::testing::Values(llvm::SmallVector<int64_t>{8, 8},
                                           llvm::SmallVector<int64_t>{3, 6},
                                           llvm::SmallVector<int64_t>{7, 2}));

//================================================================================
// getShardSpec
//================================================================================
TEST_F(MlirToTtnnConversion, ShardSpec) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};

  // dram interleaved
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                          mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
    const auto shardSpec =
        mlir::tt::op_model::ttnn::conversion::getShardSpec(layout);
    EXPECT_EQ(shardSpec.has_value(), false);
  }

  // l1 interleaved
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
    const auto shardSpec =
        mlir::tt::op_model::ttnn::conversion::getShardSpec(layout);
    EXPECT_EQ(shardSpec.has_value(), false);
  }

  // block sharded to 8x8 grid
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    const auto shardSpec =
        mlir::tt::op_model::ttnn::conversion::getShardSpec(layout);

    EXPECT_EQ(shardSpec.has_value(), true);
    EXPECT_EQ(shardSpec->shape[0], tensorShape[0] / 8);
    EXPECT_EQ(shardSpec->shape[1], tensorShape[1] / 8);
    EXPECT_EQ(shardSpec->grid.size(), 1);
    EXPECT_EQ(shardSpec->grid.ranges()[0].start_coord, CoreCoord(0, 0));
    EXPECT_EQ(shardSpec->grid.ranges()[0].end_coord, CoreCoord(7, 7));
    // These fields are not utilized on the compiler side, we are setting them
    // to default values. Purpose of testing them is to update the test if the
    // compiler side changes.
    EXPECT_EQ(shardSpec->orientation, ShardOrientation::ROW_MAJOR);
    EXPECT_EQ(shardSpec->halo, false);
    EXPECT_EQ(shardSpec->mode, ShardMode::PHYSICAL);
    EXPECT_EQ(shardSpec->physical_shard_shape.has_value(), false);
  }
}

//================================================================================
// getBufferType
//================================================================================
class MlirToTtnnConversionBufferType
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::BufferType, tt::tt_metal::BufferType>> {};

TEST_P(MlirToTtnnConversionBufferType, BufferType) {
  const auto &mlirBufferType = std::get<0>(GetParam());
  const auto &expectedBufferType = std::get<1>(GetParam());

  auto memRefType =
      buildMemRef<mlir::tt::ttnn::BufferType, mlir::tt::ttnn::BufferTypeAttr>(
          &context, {32, 32},
          mlir::tt::TileType::get(&context, builder.getBF16Type()),
          mlirBufferType);

  const auto bufferType =
      mlir::tt::op_model::ttnn::conversion::getBufferType(memRefType);
  EXPECT_EQ(bufferType, expectedBufferType);
}

INSTANTIATE_TEST_SUITE_P(
    ToBufferType, MlirToTtnnConversionBufferType,
    ::testing::Values(std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                                      tt::tt_metal::BufferType::L1),
                      std::make_tuple(mlir::tt::ttnn::BufferType::DRAM,
                                      tt::tt_metal::BufferType::DRAM),
                      std::make_tuple(mlir::tt::ttnn::BufferType::SystemMemory,
                                      tt::tt_metal::BufferType::SYSTEM_MEMORY),
                      std::make_tuple(mlir::tt::ttnn::BufferType::L1Small,
                                      tt::tt_metal::BufferType::L1_SMALL),
                      std::make_tuple(mlir::tt::ttnn::BufferType::Trace,
                                      tt::tt_metal::BufferType::TRACE)));

//================================================================================
// getTensorMemoryLayout
//================================================================================
class MlirToTnnConversionTensorMemoryLayout
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::TensorMemoryLayout,
                     tt::tt_metal::TensorMemoryLayout>> {};

TEST_P(MlirToTnnConversionTensorMemoryLayout, MemoryConfig) {
  const auto &mlirTensorMemoryLayout =
      std::get<mlir::tt::ttnn::TensorMemoryLayout>(GetParam());
  const auto &expectedTensorMemoryLayout =
      std::get<tt::tt_metal::TensorMemoryLayout>(GetParam());

  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};

  auto layout = CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                                  mlirTensorMemoryLayout);

  const auto tensorMemoryLayout =
      mlir::tt::op_model::ttnn::conversion::getTensorMemoryLayout(layout);
  EXPECT_EQ(tensorMemoryLayout, expectedTensorMemoryLayout);
}

INSTANTIATE_TEST_SUITE_P(
    ToTensorMemoryLayout, MlirToTnnConversionTensorMemoryLayout,
    ::testing::Values(
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                        tt::tt_metal::TensorMemoryLayout::INTERLEAVED),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::SingleBank,
                        tt::tt_metal::TensorMemoryLayout::SINGLE_BANK),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED)));

//================================================================================
// getMemoryConfig
//================================================================================
class MlirToTtnnConversionMemoryConfig
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<std::tuple<
          mlir::tt::ttnn::BufferType, mlir::tt::ttnn::TensorMemoryLayout>> {};

TEST_P(MlirToTtnnConversionMemoryConfig, MemoryConfig) {
  const auto &mlirBufferType = std::get<0>(GetParam());
  const auto &mlirTensorMemoryLayout = std::get<1>(GetParam());
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};

  auto layout =
      CreateTiledLayout(tensorShape, mlirBufferType, mlirTensorMemoryLayout);

  const auto memoryConfig =
      mlir::tt::op_model::ttnn::conversion::getMemoryConfig(layout);

  EXPECT_EQ(memoryConfig.is_l1(),
            mlirBufferType == mlir::tt::ttnn::BufferType::L1);
  EXPECT_EQ(memoryConfig.is_dram(),
            mlirBufferType == mlir::tt::ttnn::BufferType::DRAM);
  EXPECT_EQ(memoryConfig.is_sharded(),
            mlirTensorMemoryLayout !=
                mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
}

INSTANTIATE_TEST_SUITE_P(
    ToMemoryConfig, MlirToTtnnConversionMemoryConfig,
    ::testing::Combine(
        ::testing::Values(mlir::tt::ttnn::BufferType::DRAM,
                          mlir::tt::ttnn::BufferType::L1),
        ::testing::Values(mlir::tt::ttnn::TensorMemoryLayout::Interleaved,
                          mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                          mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded)));

//================================================================================
// getTensorLayout
//================================================================================
TEST_F(MlirToTtnnConversion, TensorLayout) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};
  // test tilized layout
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

    const auto tensorLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(layout);

    EXPECT_EQ(tensorLayout.get_data_type(), tt::tt_metal::DataType::BFLOAT16);
    EXPECT_EQ(tensorLayout.get_layout(), tt::tt_metal::Layout::TILE);
    EXPECT_EQ(tensorLayout.get_memory_config().is_sharded(), true);
  }
  // test row-major layout
  {
    const auto layout =
        CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                             mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

    const auto tensorLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(layout);

    EXPECT_EQ(tensorLayout.get_data_type(), tt::tt_metal::DataType::BFLOAT16);
    EXPECT_EQ(tensorLayout.get_layout(), tt::tt_metal::Layout::ROW_MAJOR);
    EXPECT_EQ(tensorLayout.get_memory_config().is_sharded(), true);
  }
}

//================================================================================
// getTensorSpec
//================================================================================
TEST_F(MlirToTtnnConversion, TensorSpec) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};
  // test tilized layout
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

    const auto ttnnSimpleShape =
        mlir::tt::op_model::ttnn::conversion::getSimpleShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(layout);

    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, layout);

    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnSimpleShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::TILE);
  }
  // test row-major layout
  {
    const auto layout =
        CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                             mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);

    const auto ttnnSimpleShape =
        mlir::tt::op_model::ttnn::conversion::getSimpleShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(layout);

    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, layout);

    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnSimpleShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::ROW_MAJOR);
  }
}