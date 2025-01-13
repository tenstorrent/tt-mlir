// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
#define UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/VirtualToPhysicalAffineMap.h"
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
  mlir::OpBuilder builder = mlir::OpBuilder(&context);

  void SetUp() override { context.loadDialect<mlir::tt::ttnn::TTNNDialect>(); }

  // helper function
  llvm::SmallVector<int64_t>
  GetTensorShapeInTiles(const llvm::ArrayRef<int64_t> &tensorShape) {
    return {ttmlir::utils::alignUp(tensorShape[0], 32L),
            ttmlir::utils::alignUp(tensorShape[1], 32L)};
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

    llvm::SmallVector<int64_t> tensorShapeTiles =
        GetTensorShapeInTiles(tensorShape);

    assert(tensorShape.size() == 2);
    switch (tensorMemoryLayout) {
    case mlir::tt::ttnn::TensorMemoryLayout::WidthSharded:
      return {1, std::min(tensorShapeTiles[0] * tensorShapeTiles[1],
                          gridPhyCores[0] * gridPhyCores[1])};
    case mlir::tt::ttnn::TensorMemoryLayout::HeightSharded:
      return {std::min(tensorShapeTiles[0] * tensorShapeTiles[1],
                       gridPhyCores[0] * gridPhyCores[1]),
              1};
    case mlir::tt::ttnn::TensorMemoryLayout::BlockSharded:
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
      const llvm::SmallVector<int64_t> physicalGrid = GetPhysicalGridSize()) {
    const auto &virtualGridSelected =
        virtualGrid.has_value()
            ? virtualGrid.value()
            : GetVirtualGridShape(tensorShape, tensorMemoryLayout);

    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape,
        mlir::tt::TileType::get(&context, builder.getBF16Type()), bufferType,
        CreateGrid(&context, tensorMemoryLayout, virtualGridSelected,
                   physicalGrid),
        mlir::tt::ttnn::TensorMemoryLayoutAttr::get(&context,
                                                    tensorMemoryLayout));
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

    auto affineMap =
        mlir::tt::ttnn::utils::CreateSingleDeviceVirtualToPhysicalAffineMap(
            context, tensorMemoryLayout, physicalGridSize);

    return mlir::tt::GridAttr::get(context, virtualGridSize, affineMap);
  }

  mlir::tt::GridAttr CreateWorkerGrid(
      const llvm::ArrayRef<int64_t> physicalGridSize = GetPhysicalGridSize()) {
    return mlir::tt::GridAttr::get(&context, physicalGridSize);
  }

  static constexpr std::array<int64_t, 2> gridShapeHwN300 = {7, 8};
  static constexpr size_t workerCoresN300 = 56;
};

#endif // UNITTESTS_OPMODEL_TTNN_OPMODELFIXTURE_H
