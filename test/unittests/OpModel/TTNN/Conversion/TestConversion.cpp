// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Conversion.hpp"
#include "OpModelFixture.h"

#include "gtest/gtest.h"

class MlirToTtnnConversion : public OpModelFixture {};

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
class MlirToTtnnConversionShardShape
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<std::tuple<
          llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>,
          mlir::tt::ttnn::BufferType, mlir::tt::ttnn::TensorMemoryLayout>> {};

TEST_P(MlirToTtnnConversionShardShape, ShardShape) {
  const auto &virtualGrid = std::get<0>(GetParam());
  const auto &tensorShape = std::get<1>(GetParam());
  const auto &bufferType = std::get<2>(GetParam());
  const auto &tensorMemoryLayout = std::get<3>(GetParam());

  if (tensorMemoryLayout == mlir::tt::ttnn::TensorMemoryLayout::WidthSharded) {
    EXPECT_EQ(virtualGrid[0], 1);
  } else if (tensorMemoryLayout ==
             mlir::tt::ttnn::TensorMemoryLayout::HeightSharded) {
    EXPECT_EQ(virtualGrid[1], 1);
  }

  const auto layout = CreateTiledLayout(tensorShape, bufferType,
                                        tensorMemoryLayout, virtualGrid);
  const auto shardShape =
      mlir::tt::op_model::ttnn::conversion::getShardShape(layout);

  EXPECT_EQ(shardShape[0],
            ttmlir::utils::alignUp(tensorShape[0] / virtualGrid[0], 32L));
  EXPECT_EQ(shardShape[1],
            ttmlir::utils::alignUp(tensorShape[1] / virtualGrid[1], 32L));
}

INSTANTIATE_TEST_SUITE_P(
    ToShardShape, MlirToTtnnConversionShardShape,
    ::testing::Values(
        std::make_tuple(llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{6, 6},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{64, 1},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{36, 1},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{1, 64},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{1, 36},
                        llvm::SmallVector<int64_t>{4096, 2048},
                        mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded)));

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

class ShardedCoreRangeSet
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::TensorMemoryLayout, // tensor memory layout
                     llvm::SmallVector<int64_t>,         // shard shape
                     llvm::SmallVector<int64_t>,         // virtual grid shape
                     ::tt::tt_metal::CoreRangeSet>> // expected core range set
{};

TEST_P(ShardedCoreRangeSet, ShardedCoreRangeSet) {
  const auto &tensorMemoryLayout = std::get<0>(GetParam());
  const auto &tensorShape = std::get<1>(GetParam());
  const auto &grid = std::get<2>(GetParam());
  const auto &expectedCoreRangeSet = std::get<3>(GetParam());

  const auto layout = CreateTiledLayout(
      tensorShape, mlir::tt::ttnn::BufferType::L1, tensorMemoryLayout, grid);

  const auto coreRangeSet =
      mlir::tt::op_model::ttnn::conversion::getCoreRangeSet(
          layout, CreateWorkerGrid(tensorMemoryLayout, {8, 8}));

  std::cout << coreRangeSet.str() << std::endl;

  EXPECT_EQ(coreRangeSet.size(), expectedCoreRangeSet.size());
  for (const auto &[v, r] :
       llvm::zip(coreRangeSet.ranges(), expectedCoreRangeSet.ranges())) {
    EXPECT_EQ(v.start_coord, r.start_coord);
    EXPECT_EQ(v.end_coord, r.end_coord);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ToCoreRangeSet, ShardedCoreRangeSet,
    ::testing::Values(
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{32, 64 * 32},
                        llvm::SmallVector<int64_t>{1, 64},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{32, 13 * 32},
                        llvm::SmallVector<int64_t>{1, 13},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{64 * 32, 32},
                        llvm::SmallVector<int64_t>{64, 1},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{13 * 32, 32},
                        llvm::SmallVector<int64_t>{13, 1},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{8 * 32, 8 * 32},
                        llvm::SmallVector<int64_t>{8, 8},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4 * 11 * 32, 8 * 13 * 32},
                        llvm::SmallVector<int64_t>{4, 8},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 3))})));

//================================================================================
// getShardSpec
//================================================================================

TEST_F(MlirToTtnnConversion, ShardWithInterleaved) {
  const llvm::ArrayRef<int64_t> tensorShape = {4096, 2048};

  // dram interleaved
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                          mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
    const auto shardSpec = mlir::tt::op_model::ttnn::conversion::getShardSpec(
        layout,
        CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved));
    EXPECT_EQ(shardSpec.has_value(), false);
  }

  // l1 interleaved
  {
    const auto layout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
    const auto shardSpec = mlir::tt::op_model::ttnn::conversion::getShardSpec(
        layout,
        CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::Interleaved));
    EXPECT_EQ(shardSpec.has_value(), false);
  }
}

class ShardSpecFixture
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::BufferType,         // buffer type
                     mlir::tt::ttnn::TensorMemoryLayout, // tensor memory layout
                     llvm::SmallVector<int64_t>,         // tensor shape
                     llvm::SmallVector<int64_t>,         // phy grid shape
                     llvm::SmallVector<int64_t>>>        // expected shard shape
{};

TEST_P(ShardSpecFixture, ShardSpec) {
  const auto bufferType = std::get<0>(GetParam());
  const auto tensorMemoryLayout = std::get<1>(GetParam());
  const auto tensorShape = std::get<2>(GetParam());
  const auto phyGridShape = std::get<3>(GetParam());
  const auto expected_shard_shape = std::get<4>(GetParam());

  auto virtualGrid =
      GetVirtualGridShape(tensorShape, tensorMemoryLayout, phyGridShape);

  const auto layout = CreateTiledLayout(tensorShape, bufferType,
                                        tensorMemoryLayout, virtualGrid);
  const auto shardSpec = mlir::tt::op_model::ttnn::conversion::getShardSpec(
      layout, CreateWorkerGrid(tensorMemoryLayout, phyGridShape));

  EXPECT_EQ(shardSpec.has_value(), true);

  EXPECT_EQ(shardSpec->shape[0], expected_shard_shape[0]);
  EXPECT_EQ(shardSpec->shape[1], expected_shard_shape[1]);
  EXPECT_EQ(shardSpec->grid.size(), 1);
  std::cout << shardSpec->grid.ranges()[0].start_coord.str() << std::endl;
  std::cout << shardSpec->grid.ranges()[0].end_coord.str() << std::endl;
  EXPECT_EQ(shardSpec->grid.ranges()[0].start_coord, CoreCoord(0, 0));
  EXPECT_EQ(shardSpec->grid.ranges()[0].end_coord,
            CoreCoord(phyGridShape[0] - 1, phyGridShape[1] - 1));
  // These fields are not utilized on the compiler
  // side, we are setting them to default values.
  // Purpose of testing them is to update the test
  // if the compiler side changes.
  EXPECT_EQ(shardSpec->orientation, ShardOrientation::ROW_MAJOR);
  EXPECT_EQ(shardSpec->halo, false);
  EXPECT_EQ(shardSpec->mode, ShardMode::PHYSICAL);
  EXPECT_EQ(shardSpec->physical_shard_shape.has_value(), false);
}

INSTANTIATE_TEST_SUITE_P(
    ToShardSpec, ShardSpecFixture,
    ::testing::Values(
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{512, 256}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{1024, 512}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{64, 2048}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{256, 2048}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{4096, 32}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{4096, 128})));

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
      mlir::tt::op_model::ttnn::conversion::getMemoryConfig(
          layout, CreateWorkerGrid(mlirTensorMemoryLayout));

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
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(
            layout,
            CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

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
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(
            layout,
            CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

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
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(
            layout,
            CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, layout,
        CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

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
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(
            layout,
            CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, layout,
        CreateWorkerGrid(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded));

    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnSimpleShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::ROW_MAJOR);
  }
}