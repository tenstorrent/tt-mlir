// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tt;

#include "ttmlir/Dialect/TT/IR/TTOpsEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"

mlir::tt::SystemDescAttr
mlir::tt::SystemDescAttr::getDefault(MLIRContext *context) {
  return tt::SystemDescAttr::get(
      context,
      // Chip Descriptors
      {
          tt::ChipDescAttr::get(
              context, tt::ArchAttr::get(context, tt::Arch::WormholeB0),
              tt::GridAttr::get(context, {8, 8}), (1 << 20), 12, (1 << 20), 16,
              32, 32),
      },
      // Chip Descriptor Indices
      {
          0,
      },
      // Chip capabilities
      {
          tt::ChipCapabilityAttr::get(context,
                                      // NOLINTNEXTLINE
                                      tt::ChipCapability::PCIE |
                                          tt::ChipCapability::HostMMIO),
      },
      // Chip Mesh Coordinates
      {
          tt::ChipCoordAttr::get(context, 0, 0, 0, 0),
      },
      // Chip Channel Connections
      {});
}

static mlir::MemRefType buildMemRef(::mlir::MLIRContext *context,
                                    ::llvm::ArrayRef<int64_t> shardShape,
                                    ::mlir::Type elementType,
                                    MemorySpace memorySpace) {
  if (elementType.isa<TileType>()) {
    shardShape = elementType.cast<TileType>().getTiledShape(
        ::llvm::SmallVector<int64_t>(shardShape));
  }
  return mlir::MemRefType::get(
      shardShape, elementType,
      mlir::AffineMap::getMultiDimIdentityMap(shardShape.size(), context),
      MemorySpaceAttr::get(context, memorySpace));
}

//
// This function creates an affine map that represents collapsing the tensor
// dims onto an n-dimensional grid. E.g. (Where <> is some join operator)
//
//   - 3D tensor onto a 2D grid:
//       (d0, d1, d2) -> (d0 <> d1, d2)
//
//   - 4D tensor onto a 2D grid:
//       (d0, d1, d2, d3) -> (d0 <> d1 <> d2, d3)
//
// Note there are many ways we could collapse the above dims, by default we
// just collapse the interval [0, -1), which collapses dim0 up to but not
// including the last dim.  By using collapseIntervals we can achieve flexible
// collapsing of any set of consecutive dimension ranges.
//
//   - 4D tensor onto a 3D grid collapseIntervals=[(1, -1)]:
//       (d0, d1, d2, d3) -> (d0, d1 <> d2, d3)
//
//   - 4D tensor onto a 3D grid collapseIntervals=[(0, 2)]:
//       (d0, d1, d2, d3) -> (d0 <> d1, d2, d3)
//
//   - 7D tensor onto a 4D grid collapseIntervals=[(0, 3), (-3, -1)]:
//       (d0, d1, d2, d3, d4, d5, d6) -> (d0 <> d1 <> d2, d3, d4 <> d5, d6)
//
static mlir::AffineMap collapsedLinearAffineMap(
    ::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape,
    ::llvm::ArrayRef<int64_t> gridShape,
    ::llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(shape.size() >= gridShape.size() && "shape must be >= gridShape");

  auto map = mlir::AffineMap::getMinorIdentityMap(shape.size(),
                                                  gridShape.size(), context);

  std::int64_t minimumDim = static_cast<std::int64_t>(shape.size());
  for (auto [begin, end] : collapseIntervals) {
    if (begin < 0) {
      begin += shape.size();
    }
    if (end < 0) {
      end += shape.size();
    }
    assert(end > 0);
    minimumDim = std::min(minimumDim, begin);
    auto collapsed = getAffineConstantExpr(0, context);
    int multiplier = 1;
    for (std::int64_t d = end - 1; d >= begin; --d) {
      collapsed = getAffineDimExpr(d, context) * multiplier + collapsed;
      multiplier *= shape[d];
    }
    map = map.dropResult(begin);
    map = map.insertResult(collapsed, begin);
  }

  // Fill in implicit lower dims
  for (std::int64_t d = 0; d < minimumDim; ++d) {
    map = map.dropResult(d);
    map = map.insertResult(getAffineDimExpr(d, context), d);
  }

  // Assert that all dims are represented on the RHS of the AffineMap
  for (std::size_t d = 0; d < shape.size(); ++d) {
    bool found = false;
    for (auto result : map.getResults()) {
      found |= result.isFunctionOfDim(d);
    }
    assert(found && "Dim does not participate in AffineMap RHS");
  }
  return map;
}

LayoutAttr LayoutAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape,
    Type elementType, MemorySpace memorySpace, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal) {
  if (not grid) {
    grid = tensorShape.size() == 1 ? GridAttr::get(context, {1})
                                   : GridAttr::get(context, {1, 1});
  }

  SmallVector<AffineExpr> tensorShapeExprs(
      llvm::map_range(tensorShape, [context](std::int64_t e) {
        return getAffineConstantExpr(e - 1, context);
      }));
  auto linear = collapsedLinearAffineMap(context, tensorShape, grid.getShape(),
                                         collapseIntervals);
  SmallVector<std::int64_t> shardShape(linear.getNumResults());
  assert(linear.getNumResults() == grid.getShape().size());
  for (unsigned i = 0; i < linear.getNumResults(); ++i) {
    AffineExpr expr = linear.getResult(i);
    AffineExpr constantExpr = expr.replaceDims(tensorShapeExprs);
    std::int64_t constant =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue() + 1;
    shardShape[i] = (constant + grid.getShape()[i] - 1) / grid.getShape()[i];
  }
  auto memref = buildMemRef(context, shardShape, elementType, memorySpace);
  return get(context, linear, oobVal, grid, memref);
}

LayoutAttr LayoutAttr::get(
    ::mlir::MLIRContext *context, RankedTensorType ty, MemorySpace memorySpace,
    GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals,
    OOBVal oobVal) {
  assert(ty);
  return get(context, ty.getShape(), ty.getElementType(), memorySpace, grid,
             collapseIntervals, oobVal);
}

// From the logical shape of the tensor and the affine map of the layout,
// compute the physical shape of the tensor, i.e the shape of the tensor
// after the dimensions have been collapsed onto a grid.
llvm::SmallVector<int64_t>
LayoutAttr::getPhysicalShape(ArrayRef<int64_t> logicalShape) const {
  llvm::SmallVector<int64_t> physicalShape(getGrid().getShape().size());
  SmallVector<AffineExpr> logicalShapeExprs(
      llvm::map_range(logicalShape, [context = getContext()](std::int64_t e) {
        return getAffineConstantExpr(e - 1, context);
      }));

  for (size_t i = 0; i < physicalShape.size(); i++) {
    AffineExpr expr = getLinear().getResult(i);
    AffineExpr constantExpr = expr.replaceDims(logicalShapeExprs);
    std::int64_t constant =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue() + 1;
    physicalShape[i] = constant;
  }

  return physicalShape;
}

llvm::SmallVector<int64_t>
LayoutAttr::getStride(ArrayRef<int64_t> logicalShape) const {

  llvm::SmallVector<int64_t> stride(logicalShape.size());

  auto physicalShape = getPhysicalShape(logicalShape);

  // Origin point in the logical space (0, 0, ...)
  SmallVector<AffineExpr> originPoint(logicalShape.size(),
                                      getAffineConstantExpr(0, getContext()));

  auto linearMap = getLinear();
  size_t prevDimElems = 1;

  // Iterates through physical dimensions (starting from the inner one).
  for (int i = linearMap.getNumResults() - 1; i >= 0; i--) {
    AffineExpr expr = linearMap.getResult(i);

    // Get coordinate of the i-th dimension (in physical space) of the origin
    // (in logical space).
    AffineExpr constantExpr = expr.replaceDims(originPoint);
    std::int64_t valueAtZero =
        llvm::cast<AffineConstantExpr>(constantExpr).getValue();

    for (size_t j = 0; j < logicalShape.size(); j++) {
      if (!expr.isFunctionOfDim(j)) {
        continue;
      }

      // Move from the origin point by one in the j-th dimension,
      // and get the coordinate of the i-th dimension (in physical space).
      auto newPoint = originPoint;
      newPoint[j] = getAffineConstantExpr(1, getContext());
      constantExpr = expr.replaceDims(newPoint);
      std::int64_t valueAtOne =
          llvm::cast<AffineConstantExpr>(constantExpr).getValue();

      // One step in the j-th dimension, jumps delta * prevDimElems elements in
      // the physical space.
      int64_t delta = valueAtOne - valueAtZero;
      stride[j] = prevDimElems * delta;
    }

    prevDimElems *= physicalShape[i];
  }

  return stride;
}

llvm::SmallVector<int64_t> LayoutAttr::getShardShape() const {
  SmallVector<int64_t> shardShape(getMemref().getShape());
  auto elementType = getElementType();
  if (elementType.isa<TileType>()) {
    return elementType.cast<TileType>().getScalarShape(shardShape);
  }
  return shardShape;
}

mlir::Type LayoutAttr::getElementType() const {
  return getMemref().getElementType();
}

LayoutAttr LayoutAttr::withGrid(
    ::mlir::MLIRContext *context, ArrayRef<int64_t> tensorShape, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  return get(context, tensorShape, getElementType(), getMemorySpace(), grid,
             collapseIntervals, getOobVal());
}

LayoutAttr LayoutAttr::withGrid(
    ::mlir::MLIRContext *context, RankedTensorType ty, GridAttr grid,
    ArrayRef<std::pair<std::int64_t, std::int64_t>> collapseIntervals) {
  assert(ty);
  return LayoutAttr::withGrid(context, ty.getShape(), grid, collapseIntervals);
}

LayoutAttr LayoutAttr::withElementType(::mlir::MLIRContext *context,
                                       Type elementType) {
  return LayoutAttr::get(
      context, getLinear(), getOobVal(), getGrid(),
      buildMemRef(context, getShardShape(), elementType, getMemorySpace()));
}

MemorySpace LayoutAttr::getMemorySpace() const {
  return getMemref()
      .getMemorySpace()
      .template cast<mlir::tt::MemorySpaceAttr>()
      .getValue();
}

llvm::SmallVector<int64_t>
TileType::getScalarShape(SmallVector<int64_t> tiledShape) const {
  assert(tiledShape.size() >= 2 && "expected at least 2D shape");
  tiledShape[tiledShape.size() - 2] *= getHeight();
  tiledShape[tiledShape.size() - 1] *= getWidth();
  return tiledShape;
}

llvm::SmallVector<int64_t>
TileType::getTiledShape(SmallVector<int64_t> scalarShape) const {
  assert(scalarShape.size() >= 2 && "expected at least 2D shape");
  scalarShape[scalarShape.size() - 2] =
      (scalarShape[scalarShape.size() - 2] + getHeight() - 1) / getHeight();
  scalarShape[scalarShape.size() - 1] =
      (scalarShape[scalarShape.size() - 1] + getWidth() - 1) / getWidth();
  return scalarShape;
}

void TTDialect::registerTypes() {
  // NOLINTNEXTLINE
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.cpp.inc"
      >();
}
