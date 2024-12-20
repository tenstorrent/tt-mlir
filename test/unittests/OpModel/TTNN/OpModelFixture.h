// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/VirtualToPhysicalGrid.h"
#include "ttmlir/Utils.h"
#include <llvm-gtest/gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinTypes.h>

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

  // helper function to get virtual grid shape based on tensor shape and
  // memory
  llvm::SmallVector<int64_t> GetVirtualGridShape(
      const llvm::ArrayRef<int64_t> &tensorShape,
      const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
      const llvm::ArrayRef<int64_t> &gridPhyCores = {8, 8}) {

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
          std::nullopt) {

    const auto &virtualGridSelected =
        virtualGrid.has_value()
            ? virtualGrid.value()
            : GetVirtualGridShape(tensorShape, tensorMemoryLayout);

    return mlir::tt::ttnn::TTNNLayoutAttr::get(
        &context, tensorShape,
        mlir::tt::TileType::get(&context, builder.getBF16Type()), bufferType,
        mlir::tt::GridAttr::get(&context, virtualGridSelected),
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

  mlir::tt::GridAttr
  CreateWorkerGrid(const mlir::tt::ttnn::TensorMemoryLayout &tensorMemoryLayout,
                   const llvm::ArrayRef<int64_t> gridShapeHw = {8, 8}) {
    auto affineMap =
        mlir::tt::ttnn::utils::SingleDeviceCreateVirtualToPhysicalLayoutMap(
            &context, tensorMemoryLayout, gridShapeHw);
    return mlir::tt::GridAttr::get(&context, gridShapeHw, affineMap);
  }
};
