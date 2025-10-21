// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
#define UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <optional>

class OpModelFixture : public ::testing::Test {
public:
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override {
    // Lifetime of the device in `SingletonDeviceContext` is managed by the
    // user. Hence we need to open and close the device explicitly in the test.
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .openDevice();

    // Initialize context and module
    context.loadDialect<mlir::tt::ttcore::TTCoreDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::ttcore::registerDevice(module.get());
  }

  void TearDown() override {
    mlir::tt::ttnn::op_model::SingletonDeviceContext::getInstance()
        .closeInstance();
  }

  static llvm::SmallVector<int64_t> GetPhysicalGridSize() {
    llvm::SmallVector<int64_t> grid = {gridShapeHwN300[0], gridShapeHwN300[1]};
    return grid;
  }

  // helper function to get virtual grid shape based on tensor shape and
  // memory
  llvm::SmallVector<int64_t> GetVirtualGridShape(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridPhyCores = GetPhysicalGridSize()) {

    auto tilePaddedShape =
        mlir::tt::ttnn::utils::getTilePaddedShape(tensorShape);
    size_t rank = tilePaddedShape.size();
    auto physicalShape =
        llvm::SmallVector<int64_t, 2>({1, tilePaddedShape[rank - 1]});
    for (size_t i = 0; i < rank - 1; ++i) {
      physicalShape[0] *= tilePaddedShape[i];
    }
    auto shapeInTiles = llvm::SmallVector<int64_t, 2>(
        {physicalShape[0] / TILE_HEIGHT, physicalShape[1] / TILE_WIDTH});

    int64_t tensorTiles = 1;
    for (const auto &dim : physicalShape) {
      tensorTiles *= dim;
    }

    switch (tensorMemoryLayout) {
    case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
      return {1, std::min(tensorTiles, gridPhyCores[0] * gridPhyCores[1])};
    case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
      return {std::min(tensorTiles, gridPhyCores[0] * gridPhyCores[1]), 1};
    case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
      assert(shapeInTiles.size() == 2);
      return {std::min(gridPhyCores[0], shapeInTiles[0]),
              std::min(gridPhyCores[1], shapeInTiles[1])};
    default:
      return {gridPhyCores[0], gridPhyCores[1]};
    }
  }

  mlir::tt::ttnn::TTNNLayoutAttr CreateTiledLayout(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const std::optional<llvm::SmallVector<int64_t>> &virtualGrid =
          std::nullopt,
      const llvm::SmallVector<int64_t> physicalGrid = GetPhysicalGridSize(),
      const std::optional<mlir::FloatType> dtype = std::nullopt) {
    const auto &virtualGridSelected =
        virtualGrid.has_value()
            ? virtualGrid.value()
            : GetVirtualGridShape(tensorShape, tensorMemoryLayout);

    auto memLayoutAttr = bufferType == mlir::tt::ttnn::BufferType::SystemMemory
                             ? mlir::tt::ttnn::TensorMemoryLayoutAttr{}
                             : mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                                   &context, tensorMemoryLayout);
    const auto dtypeSelected =
        dtype.has_value() ? dtype.value() : builder.getBF16Type();
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, mlir::tt::ttcore::TileType::get(dtypeSelected),
        bufferType,
        CreateGrid(&context, bufferType, tensorMemoryLayout,
                   virtualGridSelected, physicalGrid),
        memLayoutAttr);
  }

  mlir::tt::ttnn::TTNNLayoutAttr CreateTiledLayoutInt32(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const std::optional<llvm::SmallVector<int64_t>> &virtualGrid =
          std::nullopt,
      const llvm::SmallVector<int64_t> physicalGrid = GetPhysicalGridSize()) {
    const auto &virtualGridSelected =
        virtualGrid.has_value()
            ? virtualGrid.value()
            : GetVirtualGridShape(tensorShape, tensorMemoryLayout);

    auto memLayoutAttr = bufferType == mlir::tt::ttnn::BufferType::SystemMemory
                             ? mlir::tt::ttnn::TensorMemoryLayoutAttr{}
                             : mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                                   &context, tensorMemoryLayout);
    auto int32DataType = mlir::tt::ttcore::DataType::Int32;
    auto tileType =
        mlir::tt::ttcore::TileType::get(&context, {32, 32}, int32DataType);

    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, tileType, bufferType,
        CreateGrid(&context, bufferType, tensorMemoryLayout,
                   virtualGridSelected, physicalGrid),
        memLayoutAttr);
  }

  mlir::tt::ttnn::TTNNLayoutAttr CreateRowMajorLayout(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const std::optional<llvm::SmallVector<int64_t>> &virtualGrid =
          std::nullopt,
      const llvm::SmallVector<int64_t> physicalGrid = GetPhysicalGridSize(),
      const std::optional<mlir::FloatType> dtype = std::nullopt) {
    const auto &virtualGridSelected =
        virtualGrid.has_value()
            ? virtualGrid.value()
            : GetVirtualGridShape(tensorShape, tensorMemoryLayout);

    auto memLayoutAttr = bufferType == mlir::tt::ttnn::BufferType::SystemMemory
                             ? mlir::tt::ttnn::TensorMemoryLayoutAttr{}
                             : mlir::tt::ttnn::TensorMemoryLayoutAttr::get(
                                   &context, tensorMemoryLayout);
    const auto dtypeSelected =
        dtype.has_value() ? dtype.value() : builder.getBF16Type();

    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, dtypeSelected, bufferType,
        CreateGrid(&context, bufferType, tensorMemoryLayout,
                   virtualGridSelected, physicalGrid),
        memLayoutAttr);
  }

  mlir::tt::ttcore::GridAttr
  CreateGrid(::mlir::MLIRContext *context,
             const mlir::tt::ttnn::BufferType bufferType,
             const mlir::tt::ttnn::TensorMemoryLayout tensorMemoryLayout,
             const llvm::ArrayRef<int64_t> virtualGridSize,
             const llvm::ArrayRef<int64_t> physicalGridSize) {

    auto affineMap = mlir::tt::ttnn::optimizer_utils::
        createSingleDeviceVirtualToPhysicalAffineMap(
            context, tensorMemoryLayout, physicalGridSize);

    llvm::SmallVector<int64_t> gridShape(virtualGridSize);
    if (!mlir::tt::ttnn::isL1BufferType(bufferType)) {
      gridShape = {1, 1};
    }
    return mlir::tt::ttcore::GridAttr::get(context, gridShape, affineMap);
  }

  mlir::tt::ttcore::GridAttr CreateWorkerGrid(
      const llvm::ArrayRef<int64_t> physicalGridSize = GetPhysicalGridSize()) {
    return mlir::tt::ttcore::GridAttr::get(&context, physicalGridSize);
  }

  void ExpectLayoutsEQ(mlir::tt::ttnn::TTNNLayoutAttr layoutA,
                       mlir::tt::ttnn::TTNNLayoutAttr layoutB) {
    EXPECT_EQ(layoutA.getLayout(), layoutB.getLayout());
    EXPECT_EQ(layoutA.getBufferType(), layoutB.getBufferType());
    EXPECT_EQ(layoutA.getElementType(), layoutB.getElementType());
    EXPECT_EQ(layoutA.getDataType(), layoutB.getDataType());
    EXPECT_EQ(layoutA.getMemLayout().getValue(),
              layoutB.getMemLayout().getValue());
    EXPECT_EQ(layoutA.getGrid().getGridVolume(),
              layoutB.getGrid().getGridVolume());
    ASSERT_EQ(layoutA.getGrid().getShape().size(),
              layoutB.getGrid().getShape().size());
    for (size_t i = 0; i < layoutA.getGrid().getShape().size(); ++i) {
      EXPECT_EQ(layoutA.getGrid().getShape()[i],
                layoutB.getGrid().getShape()[i]);
    }
    ASSERT_EQ(layoutA.getShardShape().size(), layoutB.getShardShape().size());
    for (size_t i = 0; i < layoutA.getShardShape().size(); ++i) {
      EXPECT_EQ(layoutA.getShardShape()[i], layoutB.getShardShape()[i]);
    }
  }

  static constexpr std::array<int64_t, 2> gridShapeHwN300 = {8, 8};
  static constexpr size_t workerCoresN300 = 64;
  static constexpr int64_t TILE_HEIGHT = 32;
  static constexpr int64_t TILE_WIDTH = 32;
};

#endif // UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
