// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::d2m::utils {

llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetGrid) {
  llvm::SmallVector<int64_t> grid(physicalShape.size(), 1);

  TT_assert(targetGrid.size() == 2u);
  TT_assert(physicalShape.size() >= targetGrid.size());

  // For tensors with rank > 2, only shard across the last two dimensions
  // (which correspond to the 2D worker grid).
  const size_t dimOffset = physicalShape.size() - targetGrid.size();

  for (size_t i = 0; i < targetGrid.size(); ++i) {
    const int64_t dim = physicalShape[dimOffset + i];
    TT_assert(dim > 0);
    // Search downward from the target grid size to find the largest divisor.
    for (int64_t g = targetGrid[i]; g > 0; g--) {
      if (dim % g == 0) {
        grid[dimOffset + i] = g;
        break;
      }
    }
  }

  TT_assert(grid.size() == physicalShape.size());
  return grid;
}

// Find the dimension whose size-to-(product-of-others) ratio is largest.
// Returns 0 if no dim exceeds ratio 1.0 (i.e. balanced shape).
static unsigned findShardedDimIndex(ArrayRef<int64_t> physicalShape) {
  double bestRatio = 1.0;
  unsigned bestIndex = 0;
  for (size_t i = 0; i < physicalShape.size(); ++i) {
    double ratio = physicalShape[i];
    for (size_t j = 0; j < physicalShape.size(); ++j) {
      if (i == j) {
        continue;
      }
      ratio /= physicalShape[j];
    }
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestIndex = i;
    }
  }
  return bestIndex;
}

llvm::SmallVector<int64_t>
computeOptimalVirtualGrid(ArrayRef<int64_t> physicalShape,
                          ArrayRef<int64_t> targetGrid) {

  int64_t targetGridVolume = ttmlir::utils::volume(targetGrid);
  if (physicalShape.size() != 2) {

    // Compute factors for all dims.
    SmallVector<SmallVector<int64_t>> factors =
        llvm::to_vector(llvm::map_range(physicalShape, [](int64_t dim) {
          return ttmlir::utils::getFactors(dim);
        }));

    auto factorCombinations =
        ttmlir::utils::computeCartesianProduct<int64_t>(factors);

    // Find grid with the greatest volume that is less than or equal to the
    // target grid volume.
    SmallVector<int64_t> bestGrid = {0};
    int64_t bestGridVolume = 0;
    for (const auto &grid : factorCombinations) {
      int64_t gridVolume = ttmlir::utils::volume<int64_t>(grid);
      if (gridVolume <= targetGridVolume && gridVolume > bestGridVolume) {
        auto physGrid =
            utils::findLegalPhysicalGridForVolume(gridVolume, targetGrid);
        if (!physGrid.empty()) {

          bestGrid = grid;
          bestGridVolume = ttmlir::utils::volume<int64_t>(bestGrid);
        }
      }
    }
    return bestGrid;
  }

  // If not ND sharded, compute grid for 2D height or width sharding (Nx1, 1xN).
  unsigned shardedDimIndex = findShardedDimIndex(physicalShape);

  // Find the largest factor of the sharded dimension that fits within the
  // target grid volume.
  int64_t bestFactor = 0;
  const auto factors =
      ttmlir::utils::getFactors(physicalShape[shardedDimIndex]);
  for (int64_t factor : llvm::reverse(factors)) {
    if (factor <= targetGridVolume) {
      auto physGrid = utils::findLegalPhysicalGridForVolume(factor, targetGrid);
      if (!physGrid.empty()) {
        bestFactor = factor;
        break;
      }
    }
  }

  // If packing utilization is too low (<=25%), signal infeasibility by
  // returning an empty grid so the caller can fall back to block sharding.
  if (bestFactor == 0 ||
      bestFactor <= static_cast<int64_t>(0.25 * targetGridVolume)) {
    return {};
  }

  llvm::SmallVector<int64_t> grid;
  for (size_t i = 0; i < physicalShape.size(); ++i) {
    if (i == shardedDimIndex) {
      grid.push_back(bestFactor);
    } else {
      grid.push_back(1);
    }
  }
  return grid;
}

bool shouldImplementAsVirtualGrid(mlir::RankedTensorType tensorType,
                                  ArrayRef<int64_t> physicalShape,
                                  ArrayRef<int64_t> targetGrid) {

  ttcore::MetalLayoutAttr layout =
      mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());

  // For now, only non-collapsed 2D virtual grids on L1 are supported.
  if (layout.hasNonTrivialCollapsedDims(tensorType.getShape()) ||
      layout.getMemoryLayout() == ttcore::TensorMemoryLayout::Interleaved) {
    return false;
  }
  if (physicalShape.size() != 2) {
    return true;
  }
  auto blockShardedGrid =
      computeOptimalBlockShardedGrid(physicalShape, targetGrid);
  auto blockShardedGridVolume =
      ttmlir::utils::volume<int64_t>(blockShardedGrid);
  int64_t targetGridVolume = ttmlir::utils::volume<int64_t>(targetGrid);
  bool lowGridUtilization = blockShardedGridVolume < 0.5 * targetGridVolume;
  return lowGridUtilization;
}

llvm::SmallVector<int64_t> computeOptimalGrid(mlir::RankedTensorType tensorType,
                                              ArrayRef<int64_t> physicalShape,
                                              ArrayRef<int64_t> targetGrid) {
  if (shouldImplementAsVirtualGrid(tensorType, physicalShape, targetGrid)) {
    auto virtualGrid = computeOptimalVirtualGrid(physicalShape, targetGrid);
    if (!virtualGrid.empty()) {
      return virtualGrid;
    }
  }
  return computeOptimalBlockShardedGrid(physicalShape, targetGrid);
}

llvm::SmallVector<int64_t> computePhysicalShape(mlir::Value operand,
                                                ArrayRef<int64_t> targetGrid,
                                                bool ttnnMode,
                                                mlir::OpBuilder &builder) {

  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());

  // TTNN tensors do not require dim-alignment.
  if (ttnnMode) {
    return layout.getPhysicalShape(ttcore::TileType::getDefaultShape());
  }

  llvm::SmallVector<int64_t> tileShape;
  if (auto tileType =
          mlir::dyn_cast<ttcore::TileType>(tensorType.getElementType())) {
    tileShape = llvm::to_vector(tileType.getShape());
  } else {
    // Always tile-align when calculating the physical shape, even in the row
    // major case.
    tileShape = llvm::to_vector(ttcore::TileType::getDefaultShape());
  }

  llvm::SmallVector<int64_t> alignments =
      ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
          layout.getLogicalShape(), targetGrid,
          layout.getNormalizedIntervals());

  auto tempLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), layout.getLogicalShape(), layout.getOobVal(),
      layout.getMemorySpace(), layout.getMemoryLayout(),
      layout.getCollapsedIntervals(), alignments);

  return tempLayout.getPhysicalShape(
      llvm::ArrayRef(tileShape.data(), tileShape.size()));
}

ttcore::MetalLayoutAttr layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                                              ArrayRef<int64_t> targetGrid,
                                              bool ttnnMode,
                                              ArrayRef<int64_t> optimalGrid,
                                              mlir::OpBuilder &builder) {
  llvm::SmallVector<int64_t> newDimAlignments;
  if (ttnnMode) {
    // TTNN tensors use simple tile-aligned dim alignments without grid-aware
    // padding adjustments.
    newDimAlignments.assign(oldLayout.getLogicalShape().size(), 1);
    auto defaultTileShape = ttcore::TileType::getDefaultShape();
    newDimAlignments[newDimAlignments.size() - 1] = defaultTileShape[1];
    newDimAlignments[newDimAlignments.size() - 2] = defaultTileShape[0];
  } else {
    newDimAlignments = ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
        oldLayout.getLogicalShape(), targetGrid,
        oldLayout.getNormalizedIntervals());
  }

  return ttcore::MetalLayoutAttr::get(
      builder.getContext(), oldLayout.getLogicalShape(), oldLayout.getOobVal(),
      oldLayout.getMemorySpace(), oldLayout.getMemoryLayout(),
      oldLayout.getCollapsedIntervals(), newDimAlignments);
}

mlir::RankedTensorType tensorWithOptimalGrid(mlir::RankedTensorType oldTensor,
                                             ArrayRef<int64_t> targetGrid,
                                             bool ttnnMode,
                                             ArrayRef<int64_t> optimalGrid,
                                             mlir::OpBuilder &builder) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());

  llvm::SmallVector<int64_t> tileShape;
  Type elementType = oldTensor.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
  }

  ttcore::MetalLayoutAttr newLayout = layoutWithOptimalGrid(
      oldLayout, targetGrid, ttnnMode, optimalGrid, builder);

  llvm::SmallVector<int64_t> deviceShape = newLayout.getDeviceShape(
      optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

  Type newElementType =
      tileShape.empty()
          ? elementType
          : ttcore::TileType::get(elementType, llvm::ArrayRef(tileShape));
  return RankedTensorType::get(deviceShape, newElementType, newLayout);
}

} // namespace mlir::tt::d2m::utils
