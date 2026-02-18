// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AffineExpr.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace mlir::tt::d2m::utils {

llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape) {
  const int64_t minGridValue = *llvm::min_element(targetGridShape);

  llvm::SmallVector<int64_t, 2> squareGrid(targetGridShape.size(),
                                           minGridValue);
  return squareGrid;
}

// Helper to find the largest DST element type in a region.
// Returns nullptr if no DST-using ops are found.
static Type findLargestDstElemType(Region &region) {
  auto getTypeNumberOfBits = [](Type type) {
    return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(type));
  };

  Type largestType = nullptr;
  region.walk([&](OperandLoadStoreRegisterOpInterface op) {
    for (auto [operandIdx, v] :
         llvm::enumerate(op.getOperation()->getOperands())) {
      // Skip scalar operands.
      if (op.isScalarOperand(operandIdx)) {
        continue;
      }

      Type t = ttcore::getOperandInnerElementType(v);

      if (!largestType ||
          (getTypeNumberOfBits(t) > getTypeNumberOfBits(largestType))) {
        largestType = t;
      }

      if (largestType && getTypeNumberOfBits(largestType) >= 32u) {
        return WalkResult::interrupt();
      }
    }
    // Check output type for typecast operations that cast to a larger type.
    if (op.getOperation()->getNumResults() > 0) {
      Type outputType =
          ttcore::getOperandInnerElementType(op.getOperation()->getResult(0));
      if (!largestType || (getTypeNumberOfBits(outputType) >
                           getTypeNumberOfBits(largestType))) {
        largestType = outputType;
      }
      if (largestType && getTypeNumberOfBits(largestType) >= 32u) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return largestType;
}

Type getRegionLargestDstElemType(Region &region) {
  Type largestType = findLargestDstElemType(region);
  assert(largestType);
  auto getTypeNumberOfBits = [](Type type) {
    return ttcore::getNumberOfBits(ttcore::elementTypeToDataType(type));
  };
  TT_assert(getTypeNumberOfBits(largestType) <= 32u);
  return largestType;
}

Type getRegionLargestDstElemTypeOrNull(Region &region) {
  return findLargestDstElemType(region);
}

RankedTensorType reblockTensor(RankedTensorType oldTensor,
                               ArrayRef<int64_t> newGridShape) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());
  if (oldLayout.getGridShape(oldTensor) == newGridShape) {
    return oldTensor;
  }

  auto newShape = ttmlir::utils::calculateReblockShapeForGrid(
      oldTensor.getShape(), newGridShape);

  return RankedTensorType::get(newShape, oldTensor.getElementType(), oldLayout);
}

std::optional<SmallVector<int64_t>>
computeDimConstraints(mlir::ArrayRef<mlir::AffineMap> indexingMaps,
                      mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes) {
  TT_assert(!indexingMaps.empty());
  TT_assert(indexingMaps.size() == shapes.size());
  auto numDims = indexingMaps.front().getNumDims();
  SmallVector<int64_t> constrainedDims(numDims, 0);
  for (auto [shapeIdx, shape] : llvm::enumerate(shapes)) {
    auto dimProjectionMap =
        mlir::inverseAndBroadcastProjectedPermutation(indexingMaps[shapeIdx]);
    auto impliedDimConstraints = dimProjectionMap.compose(shape);

    for (auto [dimIdx, dimConstraint] :
         llvm::enumerate(impliedDimConstraints)) {
      if (dimConstraint == 0) {
        continue;
      }

      // Early exit if shapes are incompatible.
      if (constrainedDims[dimIdx] != 0 &&
          constrainedDims[dimIdx] != dimConstraint) {
        return std::nullopt;
      }
      constrainedDims[dimIdx] = dimConstraint;
    }
  }
  return constrainedDims;
}

SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                    AffineMap indexingMap) {
  // Create dimension values by creating BlockIndexOp for each dimension
  SmallVector<Value> dimValues;
  for (unsigned i = 0; i < indexingMap.getNumDims(); ++i) {
    dimValues.push_back(
        builder.create<BlockIndexOp>(loc, static_cast<int64_t>(i)));
  }

  // For each result expression, use expandAffineExpr to translate to arith ops
  SmallVector<Value> indices;
  for (unsigned i = 0; i < indexingMap.getNumResults(); ++i) {
    AffineExpr expr = indexingMap.getResult(i);
    Value result = mlir::affine::expandAffineExpr(builder, loc, expr, dimValues,
                                                  /*symbolValues=*/{});
    indices.push_back(result);
  }

  TT_assert(indices.size() == indexingMap.getNumResults());
  return indices;
}

static llvm::SmallVector<int64_t>
getPhysicalGridShapeFromShapeAndMap(ArrayRef<int64_t> overallDeviceShape,
                                    AffineMap map) {
  TT_assert(map.getNumResults() >= 2u);
  auto gridResultMap = ttmlir::utils::affineMapTakeFrontResults(map, 2);
  TT_assert(overallDeviceShape.size() == gridResultMap.getNumDims());
  return ttmlir::utils::evalShape(gridResultMap, overallDeviceShape);
}

SmallVector<int64_t> getPhysicalGridShape(Value tensorOrMemref) {
  // Handle view-like ops first.
  if (auto viewOp = tensorOrMemref.getDefiningOp<d2m::ViewOpInterface>()) {
    ttcore::DeviceAttr device = ttcore::lookupDevice(viewOp);
    auto deviceGridShape = device.getWorkerGrid().getShape();
    SmallVector<int64_t> outputGridShape;
    TT_assert(ttcore::hasDeviceLayout(tensorOrMemref));
    outputGridShape = llvm::to_vector(ttcore::getGridShape(tensorOrMemref));

    bool rankMismatch = outputGridShape.size() != deviceGridShape.size();
    bool outOfDeviceGridBounds = (outputGridShape[0] > deviceGridShape[0]) &&
                                 (outputGridShape[1] > deviceGridShape[1]);

    // For views, assume that if direct 1:1 mapping to device grid shape is
    // impossible, the physical grid shape is given by collapsing the ND grid
    // to a 2D physical grid that fits within the device.  This is checked
    // against actual gridAttr inverse map and output virtual grid shape in
    // GenericOp::verify().
    if (rankMismatch || outOfDeviceGridBounds) {
      return llvm::to_vector<2>(
          collapseToPhysicalGrid2D(outputGridShape, deviceGridShape));
    }
    // View virtual and physical grid shapes are equivalent if directly mappable
    // to device grid.
    return SmallVector<int64_t>(outputGridShape);
  }

  // After the virtual-grid refactor, virtualization maps live on EmptyOp
  // (virtualGridMapping attr) rather than on views or the layout attribute.
  // Check for an explicit virtualGridMapping first.
  if (auto vgm = utils::getVirtualGridMapping(tensorOrMemref)) {
    // The virtualGridMapping is the inverse map (physical → virtual).
    // Derive the physical grid shape from the virtual grid shape using
    // the shared getPhysicalGridExtent utility.
    ttcore::DeviceLayoutInterface layout =
        ttcore::getDeviceLayout(tensorOrMemref);
    TT_assert(layout);
    SmallVector<int64_t> gridShape = to_vector(
        layout.getGridShape(dyn_cast<ShapedType>(tensorOrMemref.getType())));
    if (auto *definingOp = tensorOrMemref.getDefiningOp()) {
      ttcore::DeviceAttr device = ttcore::lookupDevice(definingOp);
      auto workerGridShape = device.getWorkerGrid().getShape();
      return ttmlir::d2m::utils::grids::getPhysicalGridExtent(gridShape,
                                                              workerGridShape);
    }
  }

  // Check for a reblocking remapping on a view/stream op.
  auto shapeType = tensorOrMemref.getType();
  SmallVector<int64_t> deviceShape =
      llvm::to_vector(dyn_cast<ShapedType>(shapeType).getShape());

  if (auto remapping = utils::getAssociatedRemapping(tensorOrMemref)) {
    if (!remapping->isEmpty()) {
      return getPhysicalGridShapeFromShapeAndMap(deviceShape, *remapping);
    }
  }

  // No virtualGridMapping and no reblocking remapping.  Derive the grid shape
  // from the layout and, for grids that exceed the physical device bounds or
  // are ND (> 2D), collapse to a valid 2D physical grid.
  ttcore::DeviceLayoutInterface layout =
      ttcore::getDeviceLayout(tensorOrMemref);
  TT_assert(layout);
  SmallVector<int64_t> gridShape =
      to_vector(layout.getGridShape(dyn_cast<ShapedType>(shapeType)));

  // If we can look up the device (requires a defining op), check whether the
  // grid needs to be collapsed to fit the physical device bounds.
  if (auto *definingOp = tensorOrMemref.getDefiningOp()) {
    ttcore::DeviceAttr device = ttcore::lookupDevice(definingOp);
    auto deviceGridShape = device.getWorkerGrid().getShape();

    if (gridShape.size() > 2 || gridShape[0] > deviceGridShape[0] ||
        gridShape[1] > deviceGridShape[1]) {
      return llvm::to_vector<2>(
          collapseToPhysicalGrid2D(gridShape, deviceGridShape));
    }
  }
  return gridShape;
}

std::optional<AffineMap> getVirtualGridMapping(Value val) {
  // Direct check on the defining op.
  if (auto *defOp = val.getDefiningOp()) {
    // d2m.empty has a declared optional attribute.
    if (auto emptyOp = mlir::dyn_cast<EmptyOp>(defOp)) {
      if (auto vgm = emptyOp.getVirtualGridMappingAttr()) {
        return vgm.getValue();
      }
      return std::nullopt;
    }

    // Trace through d2m.to_layout to its output EmptyOp.
    if (auto toLayoutOp = mlir::dyn_cast<ToLayoutOp>(defOp)) {
      return getVirtualGridMapping(toLayoutOp.getOutput());
    }

    // Trace through d2m.view_layout to its input.
    if (auto viewOp = mlir::dyn_cast<ViewLayoutOp>(defOp)) {
      return getVirtualGridMapping(viewOp.getInput());
    }

    // Trace through d2m.stream_layout to its storage EmptyOp.
    if (auto streamOp = mlir::dyn_cast<StreamLayoutOp>(defOp)) {
      return getVirtualGridMapping(streamOp.getStorage());
    }

    // For ops from other dialects (memref::AllocOp, ttmetal::CreateBufferOp),
    // check for a discardable attribute.
    if (auto vgm = defOp->getAttrOfType<AffineMapAttr>("virtualGridMapping")) {
      return vgm.getValue();
    }
  }
  return std::nullopt;
}

std::optional<AffineMap> getAssociatedRemapping(Value val) {
  if (auto viewOp = val.getDefiningOp<ViewLayoutOp>()) {
    AffineMap map = viewOp.getRemapping();
    return map;
  }
  if (auto streamOp = val.getDefiningOp<StreamLayoutOp>()) {
    AffineMap map = streamOp.getRemapping();
    return map;
  }
  return std::nullopt;
}

AffineMap resolveEffectiveAffineMap(Value val, MemRefType memrefType) {
  if (auto layout =
          mlir::dyn_cast<MemRefLayoutAttrInterface>(memrefType.getLayout())) {
    if (mlir::isa<ttcore::ViewLayoutAttr>(layout)) {
      if (auto *definingOp = val.getDefiningOp()) {
        return applyViews(definingOp).second;
      }
      return AffineMap::getMultiDimIdentityMap(memrefType.getRank(),
                                               memrefType.getContext());
    }
    return layout.getAffineMap();
  }
  return AffineMap::getMultiDimIdentityMap(memrefType.getRank(),
                                           memrefType.getContext());
}

AffineMap getMemoryMap(ttcore::DeviceAttr device, MemRefType memrefType,
                       size_t pageSize, std::optional<AffineMap> view,
                       size_t baseOffset) {
  ttcore::MemorySpace memorySpace =
      mlir::cast<ttcore::MemorySpaceAttr>(memrefType.getMemorySpace())
          .getValue();
  AffineMap affineMap;
  if (auto layout =
          mlir::dyn_cast<MemRefLayoutAttrInterface>(memrefType.getLayout())) {
    affineMap = layout.getAffineMap();
  } else {
    affineMap = AffineMap::getMultiDimIdentityMap(memrefType.getRank(),
                                                  memrefType.getContext());
  }

  if (auto shardLayout =
          mlir::dyn_cast<ttcore::ShardLayoutAttr>(memrefType.getLayout())) {

    unsigned shardRank = shardLayout.getRank();
    unsigned gridRank = memrefType.getRank() - shardRank;

    auto gridShape = memrefType.getShape().take_front(gridRank);
    auto deviceGridShape = device.getWorkerGrid().getShape();

    bool needsCoreVirtualization =
        (gridRank != 2) || (gridShape[0] > deviceGridShape[0] ||
                            gridShape[1] > deviceGridShape[1]);

    if (needsCoreVirtualization) {
      auto physicalGrid = ttmlir::d2m::utils::grids::getPhysicalGridExtent(
          llvm::SmallVector<int64_t>(gridShape.begin(), gridShape.end()),
          llvm::SmallVector<int64_t>(deviceGridShape.begin(),
                                     deviceGridShape.end()));

      auto [forwardMap, inverseMap] =
          ttmlir::d2m::utils::grids::createCoreVirtMaps(
              memrefType.getContext(),
              llvm::SmallVector<int64_t>(gridShape.begin(), gridShape.end()),
              physicalGrid);

      AffineMap coreVirtMap = forwardMap;

      if (affineMap.getNumDims() > coreVirtMap.getNumResults()) {
        auto dimsToRemove =
            affineMap.getNumDims() - coreVirtMap.getNumResults();
        llvm::SmallBitVector projectedDims(affineMap.getNumDims());
        projectedDims.set(0, dimsToRemove);

        affineMap = getProjectedMap(affineMap, projectedDims);
        affineMap = affineMap.dropResults(projectedDims);
      }

      affineMap = affineMap.compose(coreVirtMap);
    }

    if (view) {
      affineMap = affineMap.compose(*view);
    }

    switch (memorySpace) {
    case ttcore::MemorySpace::DeviceL1: {
      SmallVector<int64_t> symbols = {static_cast<int64_t>(baseOffset)};
      auto resolvedL1Map =
          ttmlir::utils::replaceAffineMapSymbols(device.getL1Map(), symbols);
      return resolvedL1Map.compose(affineMap);
    }
    case ttcore::MemorySpace::DeviceDRAM: {
      pageSize = device.getMemrefSizeBytes(memrefType);
      assert(pageSize > 0 && "expected positive page size");
      SmallVector<int64_t> symbols(memrefType.getShape());
      symbols.push_back(static_cast<int64_t>(pageSize));
      symbols.push_back(static_cast<int64_t>(baseOffset));
      symbols.push_back(
          ttcore::getElementSizeBytes(memrefType.getElementType()));
      return ttmlir::utils::replaceAffineMapSymbols(device.getDramMap(),
                                                    symbols)
          .compose(affineMap);
    }
    default: {
      llvm_unreachable("Unsupported memory space");
    }
    }
  } else if (mlir::isa<ttcore::InterleavedLayoutAttr>(memrefType.getLayout())) {

    if (view) {
      affineMap = affineMap.compose(*view);
    }

    assert(memorySpace == ttcore::MemorySpace::DeviceDRAM &&
           "interleavedLayoutAttr only supported for deviceDRAM memory space");

    auto interleavedLayout =
        mlir::cast<ttcore::InterleavedLayoutAttr>(memrefType.getLayout());

    int64_t elementSizeBytes =
        ttcore::getElementSizeBytes(memrefType.getElementType());
    pageSize = mlir::isa<ttcore::TileType>(memrefType.getElementType())
                   ? elementSizeBytes
                   : interleavedLayout.getStride().front();

    assert(ttmlir::utils::volume(interleavedLayout.getGridShape(memrefType)) ==
               1 &&
           "All dims in grid shape for DRAM interleaved memref must be 1 (i.e. "
           "1x1x...x1) ");

    SmallVector<int64_t> symbols(memrefType.getShape());
    symbols.push_back(static_cast<int64_t>(pageSize));
    symbols.push_back(static_cast<int64_t>(baseOffset));
    symbols.push_back(elementSizeBytes);

    return ttmlir::utils::replaceAffineMapSymbols(device.getDramMap(), symbols)
        .compose(affineMap);
  } else {
    llvm_unreachable("Unsupported memory layout");
  }
}

AffineMap getMemoryMap(ttcore::DeviceAttr device,
                       std::pair<MemRefType, AffineMap> memrefAndView,
                       size_t pageSize, size_t baseOffset) {
  return getMemoryMap(device, memrefAndView.first, pageSize,
                      memrefAndView.second, baseOffset);
}

llvm::SmallVector<int64_t>
findLegalPhysicalGridForVolume(int64_t gridVolume,
                               ArrayRef<int64_t> targetGridShape) {
  assert(gridVolume > 0 && "Grid volume must be positive");
  assert(targetGridShape.size() >= 2u &&
         "Target grid shape must provide at least two dimensions");
  assert((targetGridShape[0] > 0 && targetGridShape[1] > 0) &&
         "Target grid dimensions must be positive");

  auto fitsTarget = [&](int64_t dimY, int64_t dimX) {
    return dimY <= targetGridShape[0] && dimX <= targetGridShape[1];
  };

  int64_t y = 1;
  // Find the largest factor of grid volume that is <= sqrt(gridVolume).
  for (int64_t i = static_cast<int64_t>(std::sqrt(gridVolume)); i > 0; --i) {
    if (gridVolume % i == 0) {
      int64_t candidateY = i;
      int64_t candidateX = gridVolume / i;
      if (fitsTarget(candidateY, candidateX)) {
        return {candidateY, candidateX};
      }
      if (fitsTarget(candidateX, candidateY)) {
        return {candidateX, candidateY};
      }
      if (y == 1) {
        y = candidateY;
      }
    }
  }
  return {};
}

llvm::SmallVector<int64_t, 2>
collapseToPhysicalGrid2D(ArrayRef<int64_t> gridShape,
                         ArrayRef<int64_t> deviceGridShape) {
  // Compute the volume of the virtual grid
  int64_t volume = 1;
  for (int64_t dim : gridShape) {
    volume *= dim;
  }

  // Try to find an optimal factorization (matches main's behavior).
  // This finds factors near sqrt to balance Y and X dimensions.
  auto result = findLegalPhysicalGridForVolume(volume, deviceGridShape);
  TT_assert(!result.empty());
  return result;
}

} // namespace mlir::tt::d2m::utils
