// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "llvm/ADT/SmallVector.h"

class MlirToTtnnConversion : public OpModelFixture {};
class Conversion : public OpModelFixture {};
//================================================================================
// getShape
//================================================================================
class ConversionShape
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<mlir::SmallVector<int64_t>> {};

TEST_P(ConversionShape, Shape) {
  const auto &originalShape = GetParam();
  const ttnn::Shape &convertedShape =
      mlir::tt::op_model::ttnn::conversion::getShape(originalShape);

  // MLIR -> TTNN conversion
  EXPECT_EQ(convertedShape.size(), originalShape.size());
  for (size_t i = 0; i < convertedShape.size(); ++i) {
    EXPECT_EQ(convertedShape[i], originalShape[i]);
  }

  // TTNN -> MLIR conversion
  const mlir::SmallVector<int64_t> &reconvertedShape =
      mlir::tt::op_model::ttnn::conversion::getShape(convertedShape);
  EXPECT_EQ(reconvertedShape.size(), originalShape.size());
  for (size_t i = 0; i < reconvertedShape.size(); ++i) {
    EXPECT_EQ(reconvertedShape[i], originalShape[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ToShape, ConversionShape,
    ::testing::Values(mlir::SmallVector<int64_t>{64, 32},
                      mlir::SmallVector<int64_t>{64, 32, 128},
                      mlir::SmallVector<int64_t>{64, 32, 128, 256}));

//================================================================================
// getDataType
//================================================================================
class ConversionDataType
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::DataType, ::tt::tt_metal::DataType>> {};

TEST_P(ConversionDataType, DataType) {
  const mlir::tt::DataType &dataType = std::get<0>(GetParam());
  const ::tt::tt_metal::DataType &expectedDataType = std::get<1>(GetParam());

  llvm::SmallVector<int64_t> tensorShape = {32, 32};
  auto layout = mlir::tt::ttnn::TTNNLayoutAttr::get(
      &context, tensorShape,
      mlir::tt::TileType::get(&context, {32, 32}, dataType),
      mlir::tt::ttnn::BufferType::L1, mlir::tt::GridAttr::get(&context, {8, 8}),
      mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
          &context, mlir::tt::ttnn::TensorMemoryLayout::Interleaved));

  // MLIR -> TTNN conversion
  auto convertedDataType =
      mlir::tt::op_model::ttnn::conversion::getDataType(layout.getDataType());
  EXPECT_EQ(convertedDataType, expectedDataType);

  // TTNN -> MLIR conversion
  EXPECT_EQ(dataType, mlir::tt::op_model::ttnn::conversion::getDataType(
                          convertedDataType));
}

INSTANTIATE_TEST_SUITE_P(
    ToDataType, ConversionDataType,
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
                                      ::tt::tt_metal::DataType::UINT8),
                      std::make_tuple(mlir::tt::DataType::Int32,
                                      ::tt::tt_metal::DataType::INT32)));

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
  llvm::SmallVector<int64_t> tensorShape = {16 * 64 * 32, 32};

  mlir::tt::ttnn::TTNNLayoutAttr tiledLayout =
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  EXPECT_EQ(mlir::tt::op_model::ttnn::conversion::getPageLayout(tiledLayout),
            ::tt::tt_metal::Layout::TILE);

  mlir::tt::ttnn::TTNNLayoutAttr rowLayout =
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
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
      mlir::tt::op_model::ttnn::conversion::getCoreRangeSet(layout);

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
                        llvm::SmallVector<int64_t>{32, 56 * 32},
                        llvm::SmallVector<int64_t>{1, 56},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{32, 13 * 32},
                        llvm::SmallVector<int64_t>{1, 13},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{56 * 32, 32},
                        llvm::SmallVector<int64_t>{56, 1},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{13 * 32, 32},
                        llvm::SmallVector<int64_t>{13, 1},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{7 * 32, 8 * 32},
                        llvm::SmallVector<int64_t>{7, 8},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(mlir::tt::ttnn::TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4 * 11 * 32, 8 * 13 * 32},
                        llvm::SmallVector<int64_t>{4, 8},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 3))})));

//================================================================================
// getShardSpec
//================================================================================

TEST_F(MlirToTtnnConversion, ShardWithInterleaved) {
  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};

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

  const auto layout = CreateTiledLayout(
      tensorShape, bufferType, tensorMemoryLayout, virtualGrid, phyGridShape);

  const auto shardSpec =
      mlir::tt::op_model::ttnn::conversion::getShardSpec(layout);

  EXPECT_EQ(shardSpec.has_value(), true);

  EXPECT_EQ(shardSpec->shape[0], expected_shard_shape[0]);
  EXPECT_EQ(shardSpec->shape[1], expected_shard_shape[1]);
  EXPECT_EQ(shardSpec->grid.size(), 1);

  EXPECT_EQ(shardSpec->grid.ranges()[0].start_coord, CoreCoord(0, 0));
  EXPECT_EQ(shardSpec->grid.ranges()[0].end_coord,
            CoreCoord(phyGridShape[1] - 1, phyGridShape[0] - 1));
  // These fields are not utilized on the compiler
  // side, we are setting them to default values.
  // Purpose of testing them is to update the test
  // if the compiler side changes.
  EXPECT_EQ(shardSpec->orientation,
            ::tt::tt_metal::ShardOrientation::ROW_MAJOR);
  EXPECT_EQ(shardSpec->mode, ::tt::tt_metal::ShardMode::PHYSICAL);
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
                        llvm::SmallVector<int64_t>{7 * 512, 8 * 256},
                        llvm::SmallVector<int64_t>{7, 8},
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
                        llvm::SmallVector<int64_t>{56 * 64, 2048},
                        llvm::SmallVector<int64_t>{7, 8},
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
                        llvm::SmallVector<int64_t>{4096, 56 * 32},
                        llvm::SmallVector<int64_t>{7, 8},
                        llvm::SmallVector<int64_t>{4096, 32}),
        std::make_tuple(mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{4096, 128})));

//================================================================================
// getBufferType
//================================================================================
class ConversionBufferType
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::BufferType, tt::tt_metal::BufferType>> {};

TEST_P(ConversionBufferType, BufferType) {
  const mlir::tt::ttnn::BufferType &mlirBufferType = std::get<0>(GetParam());
  const tt::tt_metal::BufferType &expectedBufferType = std::get<1>(GetParam());

  auto layout =
      CreateTiledLayout({32, 32}, mlirBufferType,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved);

  // MLIR -> TTNN
  const tt::tt_metal::BufferType bufferType =
      mlir::tt::op_model::ttnn::conversion::getBufferType(layout);
  EXPECT_EQ(bufferType, expectedBufferType);

  // TTNN -> MLIR
  EXPECT_EQ(mlirBufferType,
            mlir::tt::op_model::ttnn::conversion::getBufferType(bufferType));
}

INSTANTIATE_TEST_SUITE_P(
    ToBufferType, ConversionBufferType,
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
class ConversionTensorMemoryLayout
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<mlir::tt::ttnn::TensorMemoryLayout,
                     tt::tt_metal::TensorMemoryLayout>> {};

TEST_P(ConversionTensorMemoryLayout, MemoryConfig) {
  const auto &mlirTensorMemoryLayout =
      std::get<mlir::tt::ttnn::TensorMemoryLayout>(GetParam());
  const auto &expectedTensorMemoryLayout =
      std::get<tt::tt_metal::TensorMemoryLayout>(GetParam());

  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};

  auto layout = CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                                  mlirTensorMemoryLayout);

  // MLIR -> TTNN
  const auto tensorMemoryLayout =
      mlir::tt::op_model::ttnn::conversion::getTensorMemoryLayout(
          layout.getMemLayout());
  EXPECT_EQ(tensorMemoryLayout, expectedTensorMemoryLayout);

  // TTNN -> MLIR
  EXPECT_EQ(mlirTensorMemoryLayout,
            mlir::tt::op_model::ttnn::conversion::getTensorMemoryLayout(
                tensorMemoryLayout));
}

INSTANTIATE_TEST_SUITE_P(
    ToTensorMemoryLayout, ConversionTensorMemoryLayout,
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
  const llvm::SmallVector<int64_t> tensorShape = {4096, 2048};

  auto layout =
      CreateTiledLayout(tensorShape, mlirBufferType, mlirTensorMemoryLayout);

  const auto memoryConfig =
      mlir::tt::op_model::ttnn::conversion::getMemoryConfig(layout);

  EXPECT_EQ(memoryConfig.is_l1(),
            mlirBufferType == mlir::tt::ttnn::BufferType::L1);
  EXPECT_EQ(memoryConfig.is_dram(),
            mlirBufferType == mlir::tt::ttnn::BufferType::DRAM);
  EXPECT_FALSE(layout.getIgnorePhysicalLayout());

  if (mlirTensorMemoryLayout !=
      mlir::tt::ttnn::TensorMemoryLayout::Interleaved) {
    EXPECT_TRUE(memoryConfig.is_sharded());
    EXPECT_TRUE(memoryConfig.shard_spec().has_value());

    auto partialLayout = layout.withIgnorePhysicalLayout(true);
    EXPECT_TRUE(partialLayout.getIgnorePhysicalLayout());
    EXPECT_TRUE(partialLayout.hasShardedTensorMemoryLayout());

    const auto partialConfig =
        mlir::tt::op_model::ttnn::conversion::getMemoryConfig(partialLayout);
    EXPECT_TRUE(partialConfig.is_sharded());
    EXPECT_FALSE(partialConfig.shard_spec().has_value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ToMemoryConfig, MlirToTtnnConversionMemoryConfig, ::testing::ValuesIn([] {
      using mlir::tt::ttnn::BufferType;
      using mlir::tt::ttnn::TensorMemoryLayout;

      std::vector<std::tuple<BufferType, TensorMemoryLayout>>
          validCombinations = {
              {BufferType::DRAM,
               TensorMemoryLayout::Interleaved}, // DRAM only with Interleaved
              {BufferType::L1, TensorMemoryLayout::Interleaved},
              {BufferType::L1, TensorMemoryLayout::HeightSharded},
              {BufferType::L1, TensorMemoryLayout::WidthSharded},
              {BufferType::L1, TensorMemoryLayout::BlockSharded}};
      return validCombinations;
    }()));

//================================================================================
// getTensorLayout
//================================================================================
TEST_F(MlirToTtnnConversion, TensorLayout) {
  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};
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
// getTensorSpec / getLayoutAttrFromTensorSpec
//================================================================================
TEST_F(Conversion, LayoutToTensorSpec) {
  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};
  // test tilized layout
  {
    const auto originalLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    const auto ttnnShape =
        mlir::tt::op_model::ttnn::conversion::getShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(originalLayout);
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::TILE);
  }

  {
    const auto originalLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::HeightSharded);
    const auto ttnnShape =
        mlir::tt::op_model::ttnn::conversion::getShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(originalLayout);
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::TILE);
  }

  {
    const auto originalLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                          mlir::tt::ttnn::TensorMemoryLayout::WidthSharded);
    const auto ttnnShape =
        mlir::tt::op_model::ttnn::conversion::getShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(originalLayout);
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::TILE);
  }

  // test DRAM Interleaved layout
  {
    const auto originalLayout =
        CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                          mlir::tt::ttnn::TensorMemoryLayout::Interleaved);
    const auto ttnnShape =
        mlir::tt::op_model::ttnn::conversion::getShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(originalLayout);
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::TILE);
  }

  // test row-major layout
  {
    const auto originalLayout =
        CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                             mlir::tt::ttnn::TensorMemoryLayout::BlockSharded);
    const auto ttnnShape =
        mlir::tt::op_model::ttnn::conversion::getShape(tensorShape);
    const auto ttnnLayout =
        mlir::tt::op_model::ttnn::conversion::getTensorLayout(originalLayout);
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              tt::tt_metal::Layout::ROW_MAJOR);
  }
}

TEST_F(Conversion, TensorSpecToLayout) {
  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};
  const std::vector<mlir::tt::ttnn::TTNNLayoutAttr> layouts = {
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::BlockSharded),
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::HeightSharded),
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::WidthSharded),
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::DRAM,
                           mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
      CreateTiledLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                        mlir::tt::ttnn::TensorMemoryLayout::Interleaved),
      CreateRowMajorLayout(tensorShape, mlir::tt::ttnn::BufferType::L1,
                           mlir::tt::ttnn::TensorMemoryLayout::BlockSharded)};
  for (mlir::tt::ttnn::TTNNLayoutAttr originalLayout : layouts) {
    const auto tensorSpec = mlir::tt::op_model::ttnn::conversion::getTensorSpec(
        tensorShape, originalLayout);
    EXPECT_TRUE(mlir::tt::op_model::ttnn::conversion::validateTensorSpec(
        tensorSpec, {8, 8}));

    const auto reconvertedLayout =
        mlir::tt::op_model::ttnn::conversion::getLayoutAttrFromTensorSpec(
            &context, tensorSpec, /*deviceGrid=*/{8, 8});
    ExpectLayoutsEQ(originalLayout, reconvertedLayout);
  }
}

TEST_F(Conversion, TensorSpecToLayoutReversed) {
  const ttnn::Shape tensorShape{56 * 32, 56 * 32};
  const std::vector<tt::tt_metal::TensorSpec> tensorSpecs = {
      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::TILE},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  tt::tt_metal::BufferType::DRAM}}},
      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::ROW_MAJOR},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  tt::tt_metal::BufferType::DRAM}}},
      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::TILE},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  tt::tt_metal::BufferType::L1}}},
      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::TILE},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                  tt::tt_metal::BufferType::L1,
                  tt::tt_metal::ShardSpec{
                      tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0},
                                              tt::tt_metal::CoreCoord{3, 3}},
                      {14 * 32, 14 * 32}}}}},

      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::TILE},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                  tt::tt_metal::BufferType::L1,
                  tt::tt_metal::ShardSpec{
                      tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0},
                                              tt::tt_metal::CoreCoord{3, 0}},
                      {14 * 32, 56 * 32}}}}},
      tt::tt_metal::TensorSpec{
          tensorShape,
          tt::tt_metal::TensorLayout{
              tt::tt_metal::DataType::BFLOAT16,
              tt::tt_metal::PageConfig{tt::tt_metal::Layout::TILE},
              tt::tt_metal::MemoryConfig{
                  tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                  tt::tt_metal::BufferType::L1,
                  tt::tt_metal::ShardSpec{
                      tt::tt_metal::CoreRange{tt::tt_metal::CoreCoord{0, 0},
                                              tt::tt_metal::CoreCoord{3, 0}},
                      {56 * 32, 14 * 32}}}}}};

  for (tt::tt_metal::TensorSpec originalTensorSpec : tensorSpecs) {
    const auto layout =
        mlir::tt::op_model::ttnn::conversion::getLayoutAttrFromTensorSpec(
            &context, originalTensorSpec, /*deviceGrid=*/{8, 8});
    const auto reconvertedTensorSpec =
        mlir::tt::op_model::ttnn::conversion::getTensorSpec(
            mlir::tt::op_model::ttnn::conversion::getShape(tensorShape),
            layout);
    EXPECT_EQ(reconvertedTensorSpec, originalTensorSpec);
    EXPECT_TRUE(mlir::tt::op_model::ttnn::conversion::validateTensorSpec(
        reconvertedTensorSpec, {8, 8}));
  }
}
