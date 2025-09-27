// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "OpModelFixture.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/OpModel/TTNN/Conversion.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::op_model {

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
  const ::ttnn::Shape &convertedShape = conversion::getShape(originalShape);

  // MLIR -> TTNN conversion
  EXPECT_EQ(convertedShape.size(), originalShape.size());
  for (size_t i = 0; i < convertedShape.size(); ++i) {
    EXPECT_EQ(convertedShape[i], originalShape[i]);
  }

  // TTNN -> MLIR conversion
  const mlir::SmallVector<int64_t> &reconvertedShape =
      conversion::getShape(convertedShape);
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
          std::tuple<ttcore::DataType, ::tt::tt_metal::DataType>> {};

TEST_P(ConversionDataType, DataType) {
  const ttcore::DataType &dataType = std::get<0>(GetParam());
  const ::tt::tt_metal::DataType &expectedDataType = std::get<1>(GetParam());

  llvm::SmallVector<int64_t> tensorShape = {32, 32};
  auto layout = TTNNLayoutAttr::get(
      &context, tensorShape,
      ttcore::TileType::get(&context, {32, 32}, dataType), BufferType::L1,
      ttcore::GridAttr::get(&context, {8, 8}),
      TensorMemoryLayoutAttr::get(&context, TensorMemoryLayout::Interleaved));

  // MLIR -> TTNN conversion
  auto convertedDataType = conversion::getDataType(layout.getDataType());
  EXPECT_EQ(convertedDataType, expectedDataType);

  // TTNN -> MLIR conversion
  EXPECT_EQ(dataType, conversion::getDataType(convertedDataType));
}

INSTANTIATE_TEST_SUITE_P(
    ToDataType, ConversionDataType,
    ::testing::Values(std::make_tuple(ttcore::DataType::Float32,
                                      ::tt::tt_metal::DataType::FLOAT32),
                      std::make_tuple(ttcore::DataType::BFloat16,
                                      ::tt::tt_metal::DataType::BFLOAT16),
                      std::make_tuple(ttcore::DataType::BFP_BFloat8,
                                      ::tt::tt_metal::DataType::BFLOAT8_B),
                      std::make_tuple(ttcore::DataType::BFP_BFloat4,
                                      ::tt::tt_metal::DataType::BFLOAT4_B),
                      std::make_tuple(ttcore::DataType::UInt32,
                                      ::tt::tt_metal::DataType::UINT32),
                      std::make_tuple(ttcore::DataType::UInt16,
                                      ::tt::tt_metal::DataType::UINT16),
                      std::make_tuple(ttcore::DataType::UInt8,
                                      ::tt::tt_metal::DataType::UINT8),
                      std::make_tuple(ttcore::DataType::Int32,
                                      ::tt::tt_metal::DataType::INT32)));

//================================================================================
// getShardShape
//================================================================================
class MlirToTtnnConversionShardShape
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>,
                     BufferType, TensorMemoryLayout>> {};

TEST_P(MlirToTtnnConversionShardShape, ShardShape) {
  const auto &virtualGrid = std::get<0>(GetParam());
  const auto &tensorShape = std::get<1>(GetParam());
  const auto &bufferType = std::get<2>(GetParam());
  const auto &tensorMemoryLayout = std::get<3>(GetParam());

  if (tensorMemoryLayout == TensorMemoryLayout::WidthSharded) {
    EXPECT_EQ(virtualGrid[0], 1);
  } else if (tensorMemoryLayout == TensorMemoryLayout::HeightSharded) {
    EXPECT_EQ(virtualGrid[1], 1);
  }

  const auto layout = CreateTiledLayout(tensorShape, bufferType,
                                        tensorMemoryLayout, virtualGrid);
  const auto shardShape = conversion::getShardShape(layout);

  EXPECT_EQ(shardShape[0],
            ttmlir::utils::alignUp(tensorShape[0] / virtualGrid[0], 32L));
  EXPECT_EQ(shardShape[1],
            ttmlir::utils::alignUp(tensorShape[1] / virtualGrid[1], 32L));
}

INSTANTIATE_TEST_SUITE_P(
    ToShardShape, MlirToTtnnConversionShardShape,
    ::testing::Values(
        std::make_tuple(llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::BlockSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{6, 6},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::BlockSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{64, 1},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::HeightSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{36, 1},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::HeightSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{1, 64},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::WidthSharded),
        std::make_tuple(llvm::SmallVector<int64_t>{1, 36},
                        llvm::SmallVector<int64_t>{4096, 2048}, BufferType::L1,
                        TensorMemoryLayout::WidthSharded)));

//================================================================================
// getPageLayout
//================================================================================
TEST_F(MlirToTtnnConversion, PageLayout) {
  llvm::SmallVector<int64_t> tensorShape = {16 * 64 * 32, 32};

  TTNNLayoutAttr tiledLayout = CreateTiledLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);

  EXPECT_EQ(conversion::getPageLayout(tiledLayout),
            ::tt::tt_metal::Layout::TILE);

  TTNNLayoutAttr rowLayout = CreateRowMajorLayout(
      tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
  EXPECT_EQ(conversion::getPageLayout(rowLayout),
            ::tt::tt_metal::Layout::ROW_MAJOR);
}

//================================================================================
// getCoreRangeSet
//================================================================================

class ShardedCoreRangeSet
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<TensorMemoryLayout,            // tensor memory layout
                     llvm::SmallVector<int64_t>,    // shard shape
                     llvm::SmallVector<int64_t>,    // virtual grid shape
                     ::tt::tt_metal::CoreRangeSet>> // expected core range set
{};

TEST_P(ShardedCoreRangeSet, ShardedCoreRangeSet) {
  const auto &tensorMemoryLayout = std::get<0>(GetParam());
  const auto &tensorShape = std::get<1>(GetParam());
  const auto &grid = std::get<2>(GetParam());
  const auto &expectedCoreRangeSet = std::get<3>(GetParam());

  const auto layout =
      CreateTiledLayout(tensorShape, BufferType::L1, tensorMemoryLayout, grid);

  const auto coreRangeSet = conversion::getCoreRangeSet(layout);

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
        std::make_tuple(TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{32, 56 * 32},
                        llvm::SmallVector<int64_t>{1, 56},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{32, 13 * 32},
                        llvm::SmallVector<int64_t>{1, 13},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{56 * 32, 32},
                        llvm::SmallVector<int64_t>{56, 1},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{13 * 32, 32},
                        llvm::SmallVector<int64_t>{13, 1},
                        CoreRangeSet{std::set<CoreRange>{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 0)),
                            CoreRange(CoreCoord(0, 1), CoreCoord(4, 1))}}),
        std::make_tuple(TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{7 * 32, 8 * 32},
                        llvm::SmallVector<int64_t>{7, 8},
                        CoreRangeSet{
                            CoreRange(CoreCoord(0, 0), CoreCoord(7, 6))}),
        std::make_tuple(TensorMemoryLayout::BlockSharded,
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
    const auto layout = CreateTiledLayout(tensorShape, BufferType::DRAM,
                                          TensorMemoryLayout::Interleaved);
    const auto shardSpec = conversion::getShardSpec(layout);
    EXPECT_EQ(shardSpec.has_value(), false);
  }

  // l1 interleaved
  {
    const auto layout = CreateTiledLayout(tensorShape, BufferType::L1,
                                          TensorMemoryLayout::Interleaved);
    const auto shardSpec = conversion::getShardSpec(layout);
    EXPECT_EQ(shardSpec.has_value(), false);
  }
}

class ShardSpecFixture
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<BufferType,                  // buffer type
                     TensorMemoryLayout,          // tensor memory layout
                     llvm::SmallVector<int64_t>,  // tensor shape
                     llvm::SmallVector<int64_t>,  // phy grid shape
                     llvm::SmallVector<int64_t>>> // expected shard shape
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

  const auto shardSpec = conversion::getShardSpec(layout);

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
}

INSTANTIATE_TEST_SUITE_P(
    ToShardSpec, ShardSpecFixture,
    ::testing::Values(
        std::make_tuple(BufferType::L1, TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{512, 256}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{7 * 512, 8 * 256},
                        llvm::SmallVector<int64_t>{7, 8},
                        llvm::SmallVector<int64_t>{512, 256}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::BlockSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{1024, 512}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{64, 2048}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{56 * 64, 2048},
                        llvm::SmallVector<int64_t>{7, 8},
                        llvm::SmallVector<int64_t>{64, 2048}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::HeightSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{256, 2048}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{8, 8},
                        llvm::SmallVector<int64_t>{4096, 32}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 56 * 32},
                        llvm::SmallVector<int64_t>{7, 8},
                        llvm::SmallVector<int64_t>{4096, 32}),
        std::make_tuple(BufferType::L1, TensorMemoryLayout::WidthSharded,
                        llvm::SmallVector<int64_t>{4096, 2048},
                        llvm::SmallVector<int64_t>{4, 4},
                        llvm::SmallVector<int64_t>{4096, 128})));

//================================================================================
// getBufferType
//================================================================================
class ConversionBufferType
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<BufferType, ::tt::tt_metal::BufferType>> {};

TEST_P(ConversionBufferType, BufferType) {
  const BufferType &mlirBufferType = std::get<0>(GetParam());
  const ::tt::tt_metal::BufferType &expectedBufferType =
      std::get<1>(GetParam());

  auto layout = CreateTiledLayout({32, 32}, mlirBufferType,
                                  TensorMemoryLayout::Interleaved);

  // MLIR -> TTNN
  const ::tt::tt_metal::BufferType bufferType =
      conversion::getBufferType(layout);
  EXPECT_EQ(bufferType, expectedBufferType);

  // TTNN -> MLIR
  EXPECT_EQ(mlirBufferType, conversion::getBufferType(bufferType));
}

INSTANTIATE_TEST_SUITE_P(
    ToBufferType, ConversionBufferType,
    ::testing::Values(
        std::make_tuple(BufferType::L1, ::tt::tt_metal::BufferType::L1),
        std::make_tuple(BufferType::DRAM, ::tt::tt_metal::BufferType::DRAM),
        std::make_tuple(BufferType::SystemMemory,
                        ::tt::tt_metal::BufferType::SYSTEM_MEMORY),
        std::make_tuple(BufferType::L1Small,
                        ::tt::tt_metal::BufferType::L1_SMALL),
        std::make_tuple(BufferType::Trace, ::tt::tt_metal::BufferType::TRACE)));

//================================================================================
// getTensorMemoryLayout
//================================================================================
class ConversionTensorMemoryLayout
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<TensorMemoryLayout, ::tt::tt_metal::TensorMemoryLayout>> {
};

TEST_P(ConversionTensorMemoryLayout, MemoryConfig) {
  const auto &mlirTensorMemoryLayout = std::get<TensorMemoryLayout>(GetParam());
  const auto &expectedTensorMemoryLayout =
      std::get<::tt::tt_metal::TensorMemoryLayout>(GetParam());

  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};

  auto layout =
      CreateTiledLayout(tensorShape, BufferType::L1, mlirTensorMemoryLayout);

  // MLIR -> TTNN
  const auto tensorMemoryLayout =
      conversion::getTensorMemoryLayout(layout.getMemLayout());
  EXPECT_EQ(tensorMemoryLayout, expectedTensorMemoryLayout);

  // TTNN -> MLIR
  EXPECT_EQ(mlirTensorMemoryLayout,
            conversion::getTensorMemoryLayout(tensorMemoryLayout));
}

INSTANTIATE_TEST_SUITE_P(
    ToTensorMemoryLayout, ConversionTensorMemoryLayout,
    ::testing::Values(
        std::make_tuple(TensorMemoryLayout::Interleaved,
                        ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED),
        std::make_tuple(TensorMemoryLayout::HeightSharded,
                        ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
        std::make_tuple(TensorMemoryLayout::WidthSharded,
                        ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED),
        std::make_tuple(TensorMemoryLayout::BlockSharded,
                        ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED)));

//================================================================================
// getMemoryConfig
//================================================================================
class MlirToTtnnConversionMemoryConfig
    : public MlirToTtnnConversion,
      public testing::WithParamInterface<
          std::tuple<BufferType, TensorMemoryLayout>> {};

TEST_P(MlirToTtnnConversionMemoryConfig, MemoryConfig) {
  const auto &mlirBufferType = std::get<0>(GetParam());
  const auto &mlirTensorMemoryLayout = std::get<1>(GetParam());
  const llvm::SmallVector<int64_t> tensorShape = {4096, 2048};

  auto layout =
      CreateTiledLayout(tensorShape, mlirBufferType, mlirTensorMemoryLayout);

  const auto memoryConfig = conversion::getMemoryConfig(layout);

  EXPECT_EQ(memoryConfig.is_l1(), mlirBufferType == BufferType::L1);
  EXPECT_EQ(memoryConfig.is_dram(), mlirBufferType == BufferType::DRAM);
  EXPECT_FALSE(layout.getIgnorePhysicalLayout());

  if (mlirTensorMemoryLayout != TensorMemoryLayout::Interleaved) {
    EXPECT_TRUE(memoryConfig.is_sharded());
    EXPECT_TRUE(memoryConfig.shard_spec().has_value());

    auto partialLayout = layout.withIgnorePhysicalLayout(true);
    EXPECT_TRUE(partialLayout.getIgnorePhysicalLayout());
    EXPECT_TRUE(partialLayout.hasShardedTensorMemoryLayout());

    const auto partialConfig = conversion::getMemoryConfig(partialLayout);
    EXPECT_TRUE(partialConfig.is_sharded());
    EXPECT_FALSE(partialConfig.shard_spec().has_value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ToMemoryConfig, MlirToTtnnConversionMemoryConfig, ::testing::ValuesIn([] {
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
    const auto layout = CreateTiledLayout(tensorShape, BufferType::L1,
                                          TensorMemoryLayout::BlockSharded);

    const auto tensorLayout = conversion::getTensorLayout(layout);

    EXPECT_EQ(tensorLayout.get_data_type(), ::tt::tt_metal::DataType::BFLOAT16);
    EXPECT_EQ(tensorLayout.get_layout(), ::tt::tt_metal::Layout::TILE);
    EXPECT_EQ(tensorLayout.get_memory_config().is_sharded(), true);
  }
  // test row-major layout
  {
    const auto layout = CreateRowMajorLayout(tensorShape, BufferType::L1,
                                             TensorMemoryLayout::BlockSharded);

    const auto tensorLayout = conversion::getTensorLayout(layout);

    EXPECT_EQ(tensorLayout.get_data_type(), ::tt::tt_metal::DataType::BFLOAT16);
    EXPECT_EQ(tensorLayout.get_layout(), ::tt::tt_metal::Layout::ROW_MAJOR);
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
    const auto originalLayout = CreateTiledLayout(
        tensorShape, BufferType::L1, TensorMemoryLayout::BlockSharded);
    const auto ttnnShape = conversion::getShape(tensorShape);
    const auto ttnnLayout = conversion::getTensorLayout(originalLayout);
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              ::tt::tt_metal::Layout::TILE);
  }

  {
    const auto originalLayout = CreateTiledLayout(
        tensorShape, BufferType::L1, TensorMemoryLayout::HeightSharded);
    const auto ttnnShape = conversion::getShape(tensorShape);
    const auto ttnnLayout = conversion::getTensorLayout(originalLayout);
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              ::tt::tt_metal::Layout::TILE);
  }

  {
    const auto originalLayout = CreateTiledLayout(
        tensorShape, BufferType::L1, TensorMemoryLayout::WidthSharded);
    const auto ttnnShape = conversion::getShape(tensorShape);
    const auto ttnnLayout = conversion::getTensorLayout(originalLayout);
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              ::tt::tt_metal::Layout::TILE);
  }

  // test DRAM Interleaved layout
  {
    const auto originalLayout = CreateTiledLayout(
        tensorShape, BufferType::DRAM, TensorMemoryLayout::Interleaved);
    const auto ttnnShape = conversion::getShape(tensorShape);
    const auto ttnnLayout = conversion::getTensorLayout(originalLayout);
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              ::tt::tt_metal::Layout::TILE);
  }

  // test row-major layout
  {
    const auto originalLayout = CreateRowMajorLayout(
        tensorShape, BufferType::L1, TensorMemoryLayout::BlockSharded);
    const auto ttnnShape = conversion::getShape(tensorShape);
    const auto ttnnLayout = conversion::getTensorLayout(originalLayout);
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_EQ(tensorSpec.logical_shape().volume(), ttnnShape.volume());
    EXPECT_EQ(tensorSpec.page_config().get_layout(),
              ::tt::tt_metal::Layout::ROW_MAJOR);
  }
}

TEST_F(Conversion, TensorSpecToLayout) {
  const llvm::SmallVector<int64_t> tensorShape = {56 * 32, 56 * 32};
  const std::vector<TTNNLayoutAttr> layouts = {
      CreateTiledLayout(tensorShape, BufferType::L1,
                        TensorMemoryLayout::BlockSharded),
      CreateTiledLayout(tensorShape, BufferType::L1,
                        TensorMemoryLayout::HeightSharded),
      CreateTiledLayout(tensorShape, BufferType::L1,
                        TensorMemoryLayout::WidthSharded),
      CreateTiledLayout(tensorShape, BufferType::DRAM,
                        TensorMemoryLayout::Interleaved),
      CreateRowMajorLayout(tensorShape, BufferType::DRAM,
                           TensorMemoryLayout::Interleaved),
      CreateTiledLayout(tensorShape, BufferType::L1,
                        TensorMemoryLayout::Interleaved),
      CreateRowMajorLayout(tensorShape, BufferType::L1,
                           TensorMemoryLayout::BlockSharded)};
  for (TTNNLayoutAttr originalLayout : layouts) {
    const auto tensorSpec =
        conversion::getTensorSpec(tensorShape, originalLayout);
    EXPECT_TRUE(conversion::validateTensorSpec(tensorSpec, {8, 8}));

    const auto reconvertedLayout = conversion::getLayoutAttrFromTensorSpec(
        &context, tensorSpec, /*deviceGrid=*/{8, 8});
    ExpectLayoutsEQ(originalLayout, reconvertedLayout);
  }
}

TEST_F(Conversion, TensorSpecToLayoutReversed) {
  const ::ttnn::Shape tensorShape{56 * 32, 56 * 32};
  const std::vector<::tt::tt_metal::TensorSpec> tensorSpecs = {
      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::TILE},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  ::tt::tt_metal::BufferType::DRAM}}},
      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::ROW_MAJOR},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  ::tt::tt_metal::BufferType::DRAM}}},
      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::TILE},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                  ::tt::tt_metal::BufferType::L1}}},
      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::TILE},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                  ::tt::tt_metal::BufferType::L1,
                  ::tt::tt_metal::ShardSpec{
                      ::tt::tt_metal::CoreRange{
                          ::tt::tt_metal::CoreCoord{0, 0},
                          ::tt::tt_metal::CoreCoord{3, 3}},
                      {14 * 32, 14 * 32}}}}},

      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::TILE},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
                  ::tt::tt_metal::BufferType::L1,
                  ::tt::tt_metal::ShardSpec{
                      ::tt::tt_metal::CoreRange{
                          ::tt::tt_metal::CoreCoord{0, 0},
                          ::tt::tt_metal::CoreCoord{3, 0}},
                      {14 * 32, 56 * 32}}}}},
      ::tt::tt_metal::TensorSpec{
          tensorShape,
          ::tt::tt_metal::TensorLayout{
              ::tt::tt_metal::DataType::BFLOAT16,
              ::tt::tt_metal::PageConfig{::tt::tt_metal::Layout::TILE},
              ::tt::tt_metal::MemoryConfig{
                  ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                  ::tt::tt_metal::BufferType::L1,
                  ::tt::tt_metal::ShardSpec{
                      ::tt::tt_metal::CoreRange{
                          ::tt::tt_metal::CoreCoord{0, 0},
                          ::tt::tt_metal::CoreCoord{3, 0}},
                      {56 * 32, 14 * 32}}}}}};

  for (::tt::tt_metal::TensorSpec originalTensorSpec : tensorSpecs) {
    const auto layout = conversion::getLayoutAttrFromTensorSpec(
        &context, originalTensorSpec, /*deviceGrid=*/{8, 8});
    const auto reconvertedTensorSpec =
        conversion::getTensorSpec(conversion::getShape(tensorShape), layout);
    EXPECT_EQ(reconvertedTensorSpec, originalTensorSpec);
    EXPECT_TRUE(conversion::validateTensorSpec(reconvertedTensorSpec, {8, 8}));
  }
}

} // namespace mlir::tt::ttnn::op_model
