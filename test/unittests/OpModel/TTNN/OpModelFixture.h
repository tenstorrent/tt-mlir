// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
#define UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TT/Transforms/Transforms.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
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
    // Initialize context and module
    context.loadDialect<mlir::tt::TTDialect>();
    context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(&module->getBodyRegion().front());
    mlir::tt::registerDevice(module.get());
  }

  // helper function
  llvm::SmallVector<int64_t>
  GetTensorShapeInTiles(const llvm::ArrayRef<int64_t> &tensorShape) {
    llvm::SmallVector<int64_t> shapeInTiles;
    for (const auto &dim : tensorShape) {
      shapeInTiles.push_back(ttmlir::utils::alignUp(dim, 32L));
    }
    return shapeInTiles;
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

    // Usually tensors are of rank 2, but in case of MaxPool2D or Conv2D ops, it
    // is 4. Anyway this tensor will be flattened to {1, 1, Y, X} shape.
    assert(tensorShape.size() >= 2);
    for (size_t i = 0; i < tensorShape.size() - 2; i++) {
      assert(tensorShape[i] == 1);
    }
    int32_t tensorRank = tensorShape.size();
    int64_t tensorSizeX = tensorShape[tensorRank - 1];
    int64_t tensorSizeY = tensorShape[tensorRank - 2];

    llvm::SmallVector<int64_t> tensorShapeTiles =
        GetTensorShapeInTiles({tensorSizeY, tensorSizeX});

    int64_t tensorTiles = 1;
    for (const auto &dim : tensorShapeTiles) {
      tensorTiles *= dim;
    }

    switch (tensorMemoryLayout) {
    case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
      return {1, std::min(tensorTiles, gridPhyCores[0] * gridPhyCores[1])};
    case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
      return {std::min(tensorTiles, gridPhyCores[0] * gridPhyCores[1]), 1};
    case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
      assert(tensorShapeTiles.size() == 2);
      return {std::min(gridPhyCores[0], tensorShapeTiles[0]),
              std::min(gridPhyCores[1], tensorShapeTiles[1])};
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
        &context, tensorShape, mlir::tt::TileType::get(&context, dtypeSelected),
        bufferType,
        CreateGrid(&context, tensorMemoryLayout, virtualGridSelected,
                   physicalGrid),
        memLayoutAttr);
  }

  mlir::tt::ttnn::TTNNLayoutAttr CreateRowMajorLayout(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::BufferType &bufferType,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridShape = GetPhysicalGridSize()) {
    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape, builder.getBF16Type(), bufferType,
        CreateGrid(&context, tensorMemoryLayout,
                   GetVirtualGridShape(tensorShape, tensorMemoryLayout),
                   GetPhysicalGridSize()),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
  }

  mlir::tt::GridAttr
  CreateGrid(::mlir::MLIRContext *context,
             const mlir::tt::ttnn::TensorMemoryLayout tensorMemoryLayout,
             const llvm::ArrayRef<int64_t> virtualGridSize,
             const llvm::ArrayRef<int64_t> physicalGridSize) {

    auto affineMap = mlir::tt::ttnn::optimizer_utils::
        createSingleDeviceVirtualToPhysicalAffineMap(
            context, tensorMemoryLayout, physicalGridSize);

    return mlir::tt::GridAttr::get(context, virtualGridSize, affineMap);
  }

  mlir::tt::GridAttr CreateWorkerGrid(
      const llvm::ArrayRef<int64_t> physicalGridSize = GetPhysicalGridSize()) {
    return mlir::tt::GridAttr::get(&context, physicalGridSize);
  }

  static constexpr std::array<int64_t, 2> gridShapeHwN300 = {8, 8};
  static constexpr size_t workerCoresN300 = 64;
};

#endif // UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
