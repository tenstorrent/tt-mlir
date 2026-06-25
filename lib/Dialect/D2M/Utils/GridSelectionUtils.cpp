// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <array>
#include <numeric>

namespace mlir::tt::d2m::utils {

d2m::ToLayoutOp getToLayoutProducerBehindViews(mlir::Value operand) {
  bool sawView = false;
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    sawView = true;
    operand = view.getInput();
  }
  return sawView ? operand.getDefiningOp<d2m::ToLayoutOp>() : d2m::ToLayoutOp();
}

llvm::SmallVector<int64_t>
findDownstreamTiledToLayoutTileShape(mlir::Value value) {
  llvm::SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Value, 8> visited{value};

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();

      if (auto toLayout = dyn_cast<d2m::ToLayoutOp>(user)) {
        auto outputType = mlir::dyn_cast<RankedTensorType>(toLayout.getType(0));
        if (!outputType) {
          continue;
        }
        if (auto tileType =
                mlir::dyn_cast<ttcore::TileType>(outputType.getElementType())) {
          return llvm::to_vector(tileType.getShape());
        }
        if (visited.insert(toLayout.getResult(0)).second) {
          worklist.push_back(toLayout.getResult(0));
        }
        continue;
      }

      if (auto view = dyn_cast<d2m::ViewLayoutOp>(user)) {
        if (visited.insert(view.getResult()).second) {
          worklist.push_back(view.getResult());
        }
        continue;
      }

      if (auto mask = dyn_cast<d2m::MaskOp>(user)) {
        if (visited.insert(mask.getResult()).second) {
          worklist.push_back(mask.getResult());
        }
        continue;
      }
    }
  }

  return {};
}

llvm::SmallVector<int64_t>
findUpstreamTiledLayoutBridgeTileShape(mlir::Value value) {
  while (auto valueType = mlir::dyn_cast<RankedTensorType>(value.getType())) {
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(valueType.getElementType())) {
      return llvm::to_vector(tileType.getShape());
    }
    if (auto view = value.getDefiningOp<d2m::ViewLayoutOp>()) {
      value = view.getInput();
      continue;
    }
    if (auto toLayout = value.getDefiningOp<d2m::ToLayoutOp>()) {
      value = toLayout.getInput();
      continue;
    }
    break;
  }
  return {};
}

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

    // Find grid with the greatest volume that is less than or equal to the
    // target grid volume.
    SmallVector<int64_t> bestGrid = {0};
    int64_t bestGridVolume = 0;

    llvm::DenseMap<int64_t, bool> legalVolumeCache;
    auto isLegalVolume = [&](int64_t gridVolume) {
      auto it = legalVolumeCache.find(gridVolume);
      if (it != legalVolumeCache.end()) {
        return it->second;
      }
      bool isLegal =
          !utils::findLegalPhysicalGridForVolume(gridVolume, targetGrid)
               .empty();
      legalVolumeCache[gridVolume] = isLegal;
      return isLegal;
    };

    SmallVector<int64_t> candidateGrid(physicalShape.size(), 1);
    auto enumerateGrids = [&](auto &&self, size_t dim,
                              int64_t currentVolume) -> void {
      if (currentVolume > targetGridVolume) {
        return;
      }
      if (dim == factors.size()) {
        if (currentVolume > bestGridVolume && isLegalVolume(currentVolume)) {
          bestGrid.assign(candidateGrid.begin(), candidateGrid.end());
          bestGridVolume = currentVolume;
        }
        return;
      }

      for (int64_t factor : factors[dim]) {
        candidateGrid[dim] = factor;
        self(self, dim + 1, currentVolume * factor);
      }
    };
    enumerateGrids(enumerateGrids, /*dim=*/0, /*currentVolume=*/1);
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

  if (layout.getMemoryLayout() == ttcore::TensorMemoryLayout::Interleaved) {
    return false;
  }

  // Non-2D physical shapes always go through virtual grid. Collapsed-dim
  // layouts that have already been normalized to a 2D physical grid stay
  // VGM-compatible — the VGM maps only the materialized grid dimensions while
  // the layout keeps the logical-to-physical collapse.
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
                                                bool ttnnMode) {

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
      operand.getContext(), layout.getLogicalShape(), layout.getMemorySpace(),
      layout.getMemoryLayout(), layout.getCollapsedIntervals(), alignments);

  return tempLayout.getPhysicalShape(
      llvm::ArrayRef(tileShape.data(), tileShape.size()));
}

int64_t computeCollapsedIntervalSize(ArrayRef<int64_t> logicalShape,
                                     ArrayRef<int64_t> alignments,
                                     int64_t intervalStart,
                                     int64_t intervalEnd) {
  TT_assert(intervalStart < intervalEnd);

  int64_t collapsedSize = ttmlir::utils::alignUp(logicalShape[intervalEnd - 1],
                                                 alignments[intervalEnd - 1]);
  for (int64_t dim = intervalEnd - 2; dim >= intervalStart; --dim) {
    collapsedSize = ttmlir::utils::alignUp(logicalShape[dim] * collapsedSize,
                                           alignments[dim]);
  }
  return collapsedSize;
}

static llvm::SmallVector<int64_t>
computeSelectedGridDimAlignments(ttcore::MetalLayoutAttr layout,
                                 ArrayRef<int64_t> selectedGrid,
                                 ArrayRef<int64_t> tileShape) {
  llvm::SmallVector<int64_t> normalizedIntervals =
      layout.getNormalizedIntervals();
  const int64_t tensorGridRank = normalizedIntervals.size() / 2;

  TT_assertv(selectedGrid.size() == static_cast<size_t>(tensorGridRank),
             "Selected grid rank {} must match tensor grid rank {}.",
             selectedGrid.size(), tensorGridRank);

  constexpr std::array<int64_t, 2> defaultTileShape =
      ttcore::TileType::getDefaultShape();

  llvm::SmallVector<int64_t> alignments =
      ttcore::MetalLayoutAttr::computeTileAlignments(layout.getLogicalShape(),
                                                     normalizedIntervals);

  for (int64_t intervalIdx = 0; intervalIdx < tensorGridRank; ++intervalIdx) {
    const int64_t gridDim = selectedGrid[intervalIdx];
    if (gridDim <= 1) {
      continue;
    }

    const bool isTileInterval = intervalIdx >= tensorGridRank - 2;
    const int64_t tileIdx = intervalIdx - (tensorGridRank - 2);
    const int64_t gridAwareThreshold =
        gridDim * (isTileInterval ? defaultTileShape[tileIdx] : 1);

    const int64_t intervalStart = normalizedIntervals[intervalIdx * 2];
    const int64_t intervalEnd = normalizedIntervals[intervalIdx * 2 + 1];
    const int64_t alignmentDim =
        (intervalStart + 1 == intervalEnd) ? intervalEnd - 1 : intervalStart;

    int64_t collapsedSize = computeCollapsedIntervalSize(
        layout.getLogicalShape(), alignments, intervalStart, intervalEnd);
    if (collapsedSize > gridAwareThreshold) {
      alignments[alignmentDim] =
          std::lcm(alignments[alignmentDim], gridAwareThreshold);
      collapsedSize = computeCollapsedIntervalSize(
          layout.getLogicalShape(), alignments, intervalStart, intervalEnd);
    }

    const bool dividesTiles = !tileShape.empty() && isTileInterval;
    const int64_t physicalDivisor =
        gridDim * (dividesTiles ? tileShape[tileIdx] : 1);
    if (collapsedSize % physicalDivisor != 0) {
      const int64_t baseAlignment = alignments[alignmentDim];
      const int64_t maxAlignment = std::lcm(baseAlignment, physicalDivisor);
      for (int64_t candidate = baseAlignment; candidate <= maxAlignment;
           candidate += baseAlignment) {
        alignments[alignmentDim] = candidate;
        collapsedSize = computeCollapsedIntervalSize(
            layout.getLogicalShape(), alignments, intervalStart, intervalEnd);
        if (collapsedSize % physicalDivisor == 0) {
          break;
        }
      }
      TT_assertv(collapsedSize % physicalDivisor == 0,
                 "Unable to make collapsed interval size {} divisible by {}",
                 collapsedSize, physicalDivisor);
    }
  }

  return alignments;
}

ttcore::MetalLayoutAttr layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                                              ArrayRef<int64_t> selectedGrid,
                                              bool ttnnMode,
                                              ArrayRef<int64_t> tileShape) {
  llvm::SmallVector<int64_t> newDimAlignments;
  if (ttnnMode) {
    // TTNN tensors use simple tile-aligned dim alignments without grid-aware
    // padding adjustments.
    newDimAlignments.assign(oldLayout.getLogicalShape().size(), 1);
    auto defaultTileShape = ttcore::TileType::getDefaultShape();
    if (newDimAlignments.size() == 1) {
      newDimAlignments[0] = defaultTileShape[1];
    } else {
      newDimAlignments[newDimAlignments.size() - 1] = defaultTileShape[1];
      newDimAlignments[newDimAlignments.size() - 2] = defaultTileShape[0];
    }
  } else {
    newDimAlignments =
        computeSelectedGridDimAlignments(oldLayout, selectedGrid, tileShape);
  }

  return ttcore::MetalLayoutAttr::get(
      oldLayout.getContext(), oldLayout.getLogicalShape(),
      oldLayout.getMemorySpace(), oldLayout.getMemoryLayout(),
      oldLayout.getCollapsedIntervals(), newDimAlignments);
}

mlir::RankedTensorType
tensorWithOptimalGrid(mlir::RankedTensorType oldTensor, bool ttnnMode,
                      ArrayRef<int64_t> optimalGrid,
                      ArrayRef<int64_t> paddingTileShape) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());

  llvm::SmallVector<int64_t> tileShape;
  Type elementType = oldTensor.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
  }

  ArrayRef<int64_t> layoutTileShape = paddingTileShape.empty()
                                          ? ArrayRef<int64_t>(tileShape)
                                          : paddingTileShape;
  ttcore::MetalLayoutAttr newLayout =
      layoutWithOptimalGrid(oldLayout, optimalGrid, ttnnMode, layoutTileShape);

  llvm::SmallVector<int64_t> deviceShape = newLayout.getDeviceShape(
      optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

  Type newElementType =
      tileShape.empty()
          ? elementType
          : ttcore::TileType::get(elementType, llvm::ArrayRef(tileShape));
  return RankedTensorType::get(deviceShape, newElementType, newLayout);
}

} // namespace mlir::tt::d2m::utils
