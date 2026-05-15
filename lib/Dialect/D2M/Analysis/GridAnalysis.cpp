// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinOps.h"

#include <functional>
#include <limits>

namespace mlir::tt::d2m {

bool GridAnalysis::isTTNNOperand(Value operand) {
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    operand = view.getInput();
  }
  return operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>() != nullptr;
}

static ToLayoutOp getSourceToLayoutThroughViews(Value operand) {
  bool sawView = false;
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    sawView = true;
    operand = view.getInput();
  }
  if (!sawView) {
    return {};
  }
  return operand.getDefiningOp<d2m::ToLayoutOp>();
}

static llvm::SmallVector<int64_t>
computeTrailingDimLayoutGrid(ArrayRef<int64_t> selectedGrid,
                             ArrayRef<int64_t> targetGrid) {
  llvm::SmallVector<int64_t> layoutGrid(targetGrid.size(), 1);
  unsigned dimOffset =
      selectedGrid.size() > targetGrid.size()
          ? static_cast<unsigned>(selectedGrid.size() - targetGrid.size())
          : 0;
  for (auto [targetIdx, targetDim] : llvm::enumerate(targetGrid)) {
    unsigned selectedIdx = dimOffset + targetIdx;
    if (selectedIdx >= selectedGrid.size()) {
      continue;
    }
    int64_t selectedDim = selectedGrid[selectedIdx];
    layoutGrid[targetIdx] = selectedDim > targetDim ? targetDim : selectedDim;
  }
  return layoutGrid;
}

// Find the largest value <= maxFactor that divides all the given physical
// dimensions. Returns 1 if no better common factor exists.
static int64_t findLargestCommonFactor(int64_t maxFactor,
                                       ArrayRef<int64_t> physicalDims) {
  for (int64_t f = maxFactor; f > 1; --f) {
    bool dividesAll = true;
    for (int64_t dim : physicalDims) {
      if (dim % f != 0) {
        dividesAll = false;
        break;
      }
    }
    if (dividesAll) {
      return f;
    }
  }
  return 1;
}

static bool isMatmulGeneric(GenericOp genericOp) {
  return genericOp
      ->walk([](Operation *op) {
        if (mlir::isa<d2m::TileMatmulOp, d2m::TileMatmulBlockOp>(op)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool isEmbeddingGeneric(GenericOp genericOp) {
  return genericOp
      ->walk([](Operation *op) {
        if (mlir::isa<d2m::EmbeddingOp, d2m::IndexedRowCopyOp>(op)) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool isTrailingDimOnly1DVirtualGrid(ArrayRef<int64_t> grid,
                                           ArrayRef<int64_t> targetGrid) {
  unsigned nonUnitDims = 0;
  unsigned trailingDimOffset =
      grid.size() > targetGrid.size()
          ? static_cast<unsigned>(grid.size() - targetGrid.size())
          : 0;
  for (auto [dimIdx, dim] : llvm::enumerate(grid)) {
    if (dim <= 1) {
      continue;
    }
    if (dimIdx < trailingDimOffset) {
      return false;
    }
    ++nonUnitDims;
  }
  return nonUnitDims <= 1;
}

static bool isBatchDimOnlyVirtualGrid(ArrayRef<int64_t> grid,
                                      ArrayRef<int64_t> targetGrid) {
  if (grid.size() <= targetGrid.size()) {
    return false;
  }

  unsigned trailingDimOffset =
      static_cast<unsigned>(grid.size() - targetGrid.size());
  for (auto [dimIdx, dim] : llvm::enumerate(grid)) {
    if (dim <= 1) {
      continue;
    }
    if (dimIdx >= trailingDimOffset) {
      return false;
    }
  }
  return true;
}

// Matmul kernels require a physical (non-virtual) grid or a 1D virtual grid
// along the logical height/width dimensions. Higher-rank matmuls may also
// shard only batch dimensions virtually since each batch tile accumulates
// independently.
static bool isGridSupportedByMatmulKernel(ArrayRef<int64_t> grid,
                                          ArrayRef<int64_t> targetGrid) {
  return !ttmlir::d2m::utils::grids::requiresVirtualGrid(grid, targetGrid) ||
         isTrailingDimOnly1DVirtualGrid(grid, targetGrid) ||
         isBatchDimOnlyVirtualGrid(grid, targetGrid);
}

static bool hasLegalPhysicalPlacement(ArrayRef<int64_t> grid,
                                      ArrayRef<int64_t> targetGrid) {
  return !utils::findLegalPhysicalGridForVolume(
              ttmlir::utils::volume<int64_t>(grid), targetGrid)
              .empty();
}

static bool canReblockCurrentTypeToGrid(Value operand, ArrayRef<int64_t> grid) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  ArrayRef<int64_t> currentGridShape = layout.getGridShape(tensorType);
  ArrayRef<int64_t> currentShardShape = layout.getShardShape(tensorType);

  if (currentGridShape.size() != grid.size() ||
      currentShardShape.size() != grid.size()) {
    return false;
  }

  for (auto [idx, gridDim] : llvm::enumerate(grid)) {
    if ((currentGridShape[idx] * currentShardShape[idx]) % gridDim != 0) {
      return false;
    }
  }
  return true;
}

static bool canMaterializeOperandGrid(Value operand, ArrayRef<int64_t> grid,
                                      ArrayRef<int64_t> targetGrid,
                                      bool ttnnMode) {
  if (!hasLegalPhysicalPlacement(grid, targetGrid)) {
    return false;
  }

  GridDecision decision = utils::makeGridDecision(grid, targetGrid);
  llvm::SmallVector<int64_t> materializedPhysicalShape =
      utils::computePhysicalShape(operand, decision.layoutGrid, ttnnMode);
  if (materializedPhysicalShape.size() != grid.size()) {
    return false;
  }

  for (auto [physicalDim, gridDim] :
       llvm::zip_equal(materializedPhysicalShape, grid)) {
    if (physicalDim % gridDim != 0) {
      return false;
    }
  }
  return true;
}

static llvm::SmallVector<int64_t>
computeMatmulTrailingDim1DGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetGrid) {
  int64_t targetGridVolume = ttmlir::utils::volume(targetGrid);
  llvm::SmallVector<int64_t> bestGrid(physicalShape.size(), 1);
  int64_t bestGridVolume = 1;
  unsigned trailingDimOffset =
      physicalShape.size() > targetGrid.size()
          ? static_cast<unsigned>(physicalShape.size() - targetGrid.size())
          : 0;

  for (auto [dimIdx, physicalDim] : llvm::enumerate(physicalShape)) {
    if (dimIdx < trailingDimOffset) {
      continue;
    }
    for (int64_t factor :
         llvm::reverse(ttmlir::utils::getFactors(physicalDim))) {
      if (factor > targetGridVolume || factor <= bestGridVolume) {
        continue;
      }
      if (utils::findLegalPhysicalGridForVolume(factor, targetGrid).empty()) {
        continue;
      }
      bestGrid.assign(physicalShape.size(), 1);
      bestGrid[dimIdx] = factor;
      bestGridVolume = factor;
      break;
    }
  }

  return bestGrid;
}

struct GridDecisionAndShape {
  GridDecision decision;
  llvm::SmallVector<int64_t> physicalShape;
};

static llvm::SmallVector<int64_t>
computeSelectedGrid(mlir::Value operand, ArrayRef<int64_t> physicalShape,
                    ArrayRef<int64_t> targetGrid, bool allowVirtualGrid) {
  auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  llvm::SmallVector<int64_t> grid =
      utils::computeOptimalGrid(operandType, physicalShape, targetGrid);
  if (allowVirtualGrid || isGridSupportedByMatmulKernel(grid, targetGrid)) {
    return grid;
  }
  // Matmul cannot use ND virtual grids; prefer block-sharded.
  grid = utils::computeOptimalBlockShardedGrid(physicalShape, targetGrid);
  if (isGridSupportedByMatmulKernel(grid, targetGrid)) {
    return grid;
  }
  // Last resort: pick a 1D virtual grid.
  return computeMatmulTrailingDim1DGrid(physicalShape, targetGrid);
}

static llvm::SmallVector<int64_t>
computeTypePhysicalShape(mlir::RankedTensorType tensorType) {
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  ArrayRef<int64_t> gridShape = layout.getGridShape(tensorType);
  ArrayRef<int64_t> shardShape = layout.getShardShape(tensorType);

  llvm::SmallVector<int64_t> physicalShape;
  physicalShape.reserve(gridShape.size());
  for (auto [gridDim, shardDim] : llvm::zip_equal(gridShape, shardShape)) {
    physicalShape.push_back(gridDim * shardDim);
  }
  return physicalShape;
}

static llvm::SmallVector<int64_t> computeCurrentPhysicalShape(Value operand) {
  return computeTypePhysicalShape(
      mlir::cast<mlir::RankedTensorType>(operand.getType()));
}

static bool canGridSelectionRewriteOperand(Value operand) {
  if (GridAnalysis::isTTNNOperand(operand)) {
    return true;
  }
  if (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    return !view.getReinterpretLayout();
  }
  return operand.getDefiningOp<d2m::CompositeViewOp>() ||
         operand.getDefiningOp<d2m::ToLayoutOp>() ||
         operand.getDefiningOp<d2m::EmptyOp>();
}

static mlir::RankedTensorType
computePlannedOperandType(Value operand, const GridDecision &decision,
                          bool ttnnMode) {
  auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  if (!canGridSelectionRewriteOperand(operand)) {
    return operandType;
  }
  return utils::tensorWithOptimalGrid(operandType, decision.layoutGrid,
                                      ttnnMode, decision.selectedGrid);
}

static bool shouldChooseMinimalMemoryGrid(GenericOp genericOp) {
  // d2m.spatial regions are explicit physical placement scopes. Do not shrink
  // their region-local grid just because a DMA-only body would fit on fewer
  // cores.
  if (mlir::isa<d2m::SpatialOp>(genericOp->getParentOp())) {
    return false;
  }

  for (Value input : genericOp.getInputs()) {
    if (input.getDefiningOp<d2m::CompositeViewOp>()) {
      return false;
    }
  }
  return genericOp.isDMAOnlyForm();
}

static uint64_t estimateOperandShardBytes(Type elementType,
                                          ArrayRef<int64_t> physicalShape,
                                          ArrayRef<int64_t> grid) {
  TT_assert(physicalShape.size() == grid.size());
  uint64_t elementSizeBytes = ttcore::getElementSizeBytes(elementType);
  uint64_t elementCount = 1;
  for (auto [physicalDim, gridDim] : llvm::zip_equal(physicalShape, grid)) {
    TT_assert(physicalDim % gridDim == 0);
    elementCount *= static_cast<uint64_t>(physicalDim / gridDim);
  }
  return elementCount * elementSizeBytes;
}

static uint64_t estimateDmaOnlyScratchBytes(
    GenericOp genericOp, ArrayRef<llvm::SmallVector<int64_t>> operandGrids,
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes) {
  if (!genericOp.isDMAOnlyForm()) {
    return 0;
  }

  uint64_t maxBlockBytes = 0;
  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    maxBlockBytes = std::max(
        maxBlockBytes, estimateOperandShardBytes(tensorType.getElementType(),
                                                 physicalShapes[operandIndex],
                                                 operandGrids[operandIndex]));
  }
  return maxBlockBytes;
}

static uint64_t
estimateGenericL1Bytes(GenericOp genericOp,
                       ArrayRef<llvm::SmallVector<int64_t>> operandGrids,
                       ArrayRef<llvm::SmallVector<int64_t>> physicalShapes) {
  uint64_t totalBytes = 0;
  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    totalBytes += estimateOperandShardBytes(tensorType.getElementType(),
                                            physicalShapes[operandIndex],
                                            operandGrids[operandIndex]);
  }
  totalBytes +=
      estimateDmaOnlyScratchBytes(genericOp, operandGrids, physicalShapes);
  return totalBytes;
}

static bool isBetterMemoryCandidate(uint64_t candidateEstimatedBytes,
                                    uint64_t candidateOutputGridVolume,
                                    bool candidateFitsL1,
                                    uint64_t bestEstimatedBytes,
                                    uint64_t bestOutputGridVolume,
                                    bool bestFitsL1) {
  if (candidateFitsL1 != bestFitsL1) {
    return candidateFitsL1;
  }
  if (candidateFitsL1) {
    return candidateOutputGridVolume < bestOutputGridVolume ||
           (candidateOutputGridVolume == bestOutputGridVolume &&
            candidateEstimatedBytes < bestEstimatedBytes);
  }
  return candidateEstimatedBytes < bestEstimatedBytes;
}

static GridDecisionAndShape
computeCurrentBlockShardedDecision(mlir::Value operand,
                                   ArrayRef<int64_t> targetGrid) {
  GridDecisionAndShape result;
  result.physicalShape = computeCurrentPhysicalShape(operand);
  auto selectedGrid =
      utils::computeOptimalBlockShardedGrid(result.physicalShape, targetGrid);
  result.decision = utils::makeGridDecision(selectedGrid, targetGrid);
  result.decision.layoutGrid =
      computeTrailingDimLayoutGrid(selectedGrid, targetGrid);
  return result;
}

static GridDecisionAndShape
computeMatmulGridDecision(mlir::Value operand, ArrayRef<int64_t> targetGrid,
                          bool ttnnMode) {
  GridDecisionAndShape result;
  result.physicalShape =
      utils::computePhysicalShape(operand, targetGrid, ttnnMode);
  result.decision = utils::makeGridDecision(
      computeSelectedGrid(operand, result.physicalShape, targetGrid,
                          /*allowVirtualGrid=*/false),
      targetGrid);

  // Preserve main's matmul padding contract: selectedGrid controls execution
  // parallelism, while the full target grid controls dim
  // alignments/materialized physical shapes. Otherwise GQA matmuls can rebuild
  // with mismatched local M extents between LHS and output.
  result.decision.layoutGrid = llvm::SmallVector<int64_t>(targetGrid);
  return result;
}

static GridDecisionAndShape
computeGridDecision(mlir::Value operand, ArrayRef<int64_t> initialLayoutGrid,
                    ArrayRef<int64_t> targetGrid, bool ttnnMode,
                    bool allowVirtualGrid) {
  constexpr unsigned kMaxGridDecisionIterations = 8;

  llvm::SmallVector<int64_t> layoutGrid(initialLayoutGrid);
  for (unsigned iteration = 0; iteration < kMaxGridDecisionIterations;
       ++iteration) {
    GridDecisionAndShape result;
    result.physicalShape =
        utils::computePhysicalShape(operand, layoutGrid, ttnnMode);
    result.decision = utils::makeGridDecision(
        computeSelectedGrid(operand, result.physicalShape, targetGrid,
                            allowVirtualGrid),
        targetGrid);
    if (!allowVirtualGrid) {
      result.decision.layoutGrid = computeTrailingDimLayoutGrid(
          result.decision.selectedGrid, targetGrid);
    }

    if (result.decision.layoutGrid == layoutGrid) {
      return result;
    }

    layoutGrid = result.decision.layoutGrid;
  }

  // Grid-aware padding can change factor availability, so a virtual decision
  // may theoretically oscillate. Prefer a physical block-sharded grid over
  // emitting an inconsistent virtual layout.
  return computeCurrentBlockShardedDecision(operand, targetGrid);
}

llvm::SmallVector<llvm::SmallVector<int64_t>>
GridAnalysis::normalizeOperandGridsForGeneric(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes,
    ArrayRef<int64_t> targetGrid, bool ttnnMode,
    bool requireCurrentTypeReblockable, uint64_t usableL1Bytes) {
  if (optimalOperandGrids.empty()) {
    return {};
  }

  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());
  TT_assert(physicalShapes.size() == optimalOperandGrids.size());

  llvm::SmallVector<Value> operands(genericOp.getInputsAndOutputs().begin(),
                                    genericOp.getInputsAndOutputs().end());
  llvm::SmallVector<llvm::SmallVector<int64_t>> currentReblockShapes;
  currentReblockShapes.reserve(operands.size());
  if (requireCurrentTypeReblockable) {
    for (Value operand : operands) {
      currentReblockShapes.push_back(computeCurrentPhysicalShape(operand));
    }
  }

  bool requireMatmulCompatibleGrid = isMatmulGeneric(genericOp);
  bool hasIndexedRowAccess = isEmbeddingGeneric(genericOp);
  auto indexingMaps = genericOp.getIndexingMapsValue();
  unsigned numLoopDims = indexingMaps.front().getNumDims();
  llvm::SmallVector<llvm::SmallVector<std::pair<uint64_t, uint64_t>>>
      loopDimOperandDims(numLoopDims);

  for (uint64_t operandIndex = 0; operandIndex < indexingMaps.size();
       ++operandIndex) {
    AffineMap operandIndexingMap = indexingMaps[operandIndex];
    auto results = operandIndexingMap.getResults();
    for (auto [operandDimIdx, expr] : llvm::enumerate(results)) {
      auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      unsigned loopDim = dimExpr.getPosition();
      loopDimOperandDims[loopDim].push_back(
          std::make_pair(operandIndex, static_cast<uint64_t>(operandDimIdx)));
    }
  }

  // Build a loop-space grid wish-list first. For each loop dimension, choose
  // the largest factor requested by any operand that also divides every
  // physical operand dimension that indexes the loop.
  llvm::SmallVector<int64_t> desiredLoopGrid(numLoopDims, 1);
  int64_t targetGridVolume = ttmlir::utils::volume<int64_t>(targetGrid);
  for (unsigned loopDim = 0; loopDim < numLoopDims; ++loopDim) {
    auto &entries = loopDimOperandDims[loopDim];
    if (entries.empty()) {
      continue;
    }

    int64_t maxFactor = 0;
    SmallVector<int64_t> physDimsForLoop;
    for (auto [operandIndex, operandDimIdx] : entries) {
      maxFactor =
          std::max(maxFactor, optimalOperandGrids[operandIndex][operandDimIdx]);
      physDimsForLoop.push_back(physicalShapes[operandIndex][operandDimIdx]);
      if (requireCurrentTypeReblockable) {
        physDimsForLoop.push_back(
            currentReblockShapes[operandIndex][operandDimIdx]);
      }
    }

    if (requireMatmulCompatibleGrid) {
      maxFactor = std::max(maxFactor, targetGridVolume);
    }
    desiredLoopGrid[loopDim] =
        findLargestCommonFactor(maxFactor, physDimsForLoop);
  }

  auto projectLoopGridToOperands = [&](ArrayRef<int64_t> loopGrid) {
    llvm::SmallVector<llvm::SmallVector<int64_t>> projectedOperandGrids;
    projectedOperandGrids.reserve(optimalOperandGrids.size());
    for (ArrayRef<int64_t> operandGrid : optimalOperandGrids) {
      projectedOperandGrids.emplace_back(operandGrid.size(), 1);
    }
    for (uint64_t operandIndex = 0; operandIndex < indexingMaps.size();
         ++operandIndex) {
      auto results = indexingMaps[operandIndex].getResults();
      for (auto [operandDimIdx, expr] : llvm::enumerate(results)) {
        auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        projectedOperandGrids[operandIndex][operandDimIdx] =
            loopGrid[dimExpr.getPosition()];
      }
    }
    return projectedOperandGrids;
  };

  auto allOperandGridsAreLegal =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
        return llvm::all_of(llvm::enumerate(operandGrids), [&](auto indexed) {
          auto [operandIndex, operandGrid] = indexed;
          if (requireCurrentTypeReblockable &&
              !canReblockCurrentTypeToGrid(operands[operandIndex],
                                           operandGrid)) {
            return false;
          }

          if (physicalShapes[operandIndex].size() != operandGrid.size()) {
            return false;
          }
          for (auto [physicalDim, gridDim] :
               llvm::zip_equal(physicalShapes[operandIndex], operandGrid)) {
            if (physicalDim % gridDim != 0) {
              return false;
            }
          }
          if (!canMaterializeOperandGrid(operands[operandIndex], operandGrid,
                                         targetGrid, ttnnMode)) {
            return false;
          }
          return !requireMatmulCompatibleGrid ||
                 isGridSupportedByMatmulKernel(operandGrid, targetGrid);
        });
      };

  llvm::SmallVector<int64_t> bestLoopGrid(numLoopDims, 1);
  uint64_t bestLoopGridVolume = 1;
  uint64_t bestOutputGridVolume = 1;
  uint64_t bestEstimatedBytes = std::numeric_limits<uint64_t>::max();
  bool bestFitsL1 = false;
  bool minimizeMemoryGrid = shouldChooseMinimalMemoryGrid(genericOp);
  const unsigned outputOperandIndex = genericOp.getInputs().size();
  llvm::SmallVector<int64_t> candidateLoopGrid(numLoopDims, 1);
  llvm::SmallVector<llvm::SmallVector<int64_t>> loopFactorChoices;
  loopFactorChoices.reserve(numLoopDims);
  for (int64_t desiredFactor : desiredLoopGrid) {
    loopFactorChoices.push_back(ttmlir::utils::getFactors(desiredFactor));
  }

  // Only projected operand/output grids consume physical cores. Loop factors
  // that do not project to the output, such as matmul K-splits, become block
  // factors on the selected cores and should not be capped by target volume.
  std::function<void(unsigned, uint64_t)> searchLoopGrids =
      [&](unsigned loopDim, uint64_t volumeSoFar) {
        if (loopDim == numLoopDims) {
          auto candidateOperandGrids =
              projectLoopGridToOperands(candidateLoopGrid);
          if (!allOperandGridsAreLegal(candidateOperandGrids)) {
            return;
          }

          if (minimizeMemoryGrid) {
            uint64_t candidateEstimatedBytes = estimateGenericL1Bytes(
                genericOp, candidateOperandGrids, physicalShapes);
            bool candidateFitsL1 =
                usableL1Bytes == 0 || candidateEstimatedBytes <= usableL1Bytes;
            uint64_t candidateOutputGridVolume =
                static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
                    candidateOperandGrids[outputOperandIndex]));

            if (isBetterMemoryCandidate(candidateEstimatedBytes,
                                        candidateOutputGridVolume,
                                        candidateFitsL1, bestEstimatedBytes,
                                        bestOutputGridVolume, bestFitsL1)) {
              bestLoopGrid = candidateLoopGrid;
              bestLoopGridVolume = volumeSoFar;
              bestOutputGridVolume = candidateOutputGridVolume;
              bestEstimatedBytes = candidateEstimatedBytes;
              bestFitsL1 = candidateFitsL1;
            }
            return;
          }

          if (volumeSoFar > bestLoopGridVolume) {
            bestLoopGrid = candidateLoopGrid;
            bestLoopGridVolume = volumeSoFar;
          }
          return;
        }

        for (int64_t factor : llvm::reverse(loopFactorChoices[loopDim])) {
          uint64_t candidateVolume =
              volumeSoFar * static_cast<uint64_t>(factor);
          candidateLoopGrid[loopDim] = factor;
          searchLoopGrids(loopDim + 1, candidateVolume);
        }
        candidateLoopGrid[loopDim] = 1;
      };
  searchLoopGrids(/*loopDim=*/0, /*volumeSoFar=*/1);

  auto normalizedOperandGrids = projectLoopGridToOperands(bestLoopGrid);
  if (hasIndexedRowAccess && outputOperandIndex < operands.size() &&
      targetGrid.size() >= 2 && operands.size() >= 3 &&
      !normalizedOperandGrids[0].empty() &&
      !normalizedOperandGrids[1].empty() &&
      !normalizedOperandGrids[outputOperandIndex].empty()) {
    llvm::SmallVector<llvm::SmallVector<int64_t>> bestEmbeddingGrids =
        normalizedOperandGrids;
    uint64_t bestEmbeddingEstimatedBytes =
        estimateGenericL1Bytes(genericOp, bestEmbeddingGrids, physicalShapes);
    uint64_t bestEmbeddingOutputGridVolume = static_cast<uint64_t>(
        ttmlir::utils::volume<int64_t>(bestEmbeddingGrids[outputOperandIndex]));
    bool bestEmbeddingFitsL1 =
        usableL1Bytes == 0 || bestEmbeddingEstimatedBytes <= usableL1Bytes;

    int64_t maxRowFactor = findLargestCommonFactor(
        targetGrid[0],
        {physicalShapes[1][0], physicalShapes[outputOperandIndex][0]});
    for (int64_t rowFactor : ttmlir::utils::getFactors(maxRowFactor)) {
      if (rowFactor <= normalizedOperandGrids[outputOperandIndex][0]) {
        continue;
      }
      llvm::SmallVector<llvm::SmallVector<int64_t>> candidateOperandGrids =
          normalizedOperandGrids;
      candidateOperandGrids[0][0] = 1;
      candidateOperandGrids[1][0] = rowFactor;
      candidateOperandGrids[outputOperandIndex][0] = rowFactor;
      if (!allOperandGridsAreLegal(candidateOperandGrids)) {
        continue;
      }

      uint64_t candidateEstimatedBytes = estimateGenericL1Bytes(
          genericOp, candidateOperandGrids, physicalShapes);
      uint64_t candidateOutputGridVolume =
          static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
              candidateOperandGrids[outputOperandIndex]));
      bool candidateFitsL1 =
          usableL1Bytes == 0 || candidateEstimatedBytes <= usableL1Bytes;
      if (isBetterMemoryCandidate(
              candidateEstimatedBytes, candidateOutputGridVolume,
              candidateFitsL1, bestEmbeddingEstimatedBytes,
              bestEmbeddingOutputGridVolume, bestEmbeddingFitsL1)) {
        bestEmbeddingGrids = candidateOperandGrids;
        bestEmbeddingEstimatedBytes = candidateEstimatedBytes;
        bestEmbeddingOutputGridVolume = candidateOutputGridVolume;
        bestEmbeddingFitsL1 = candidateFitsL1;
      }
    }

    normalizedOperandGrids = bestEmbeddingGrids;
  }
  if (allOperandGridsAreLegal(normalizedOperandGrids)) {
    return normalizedOperandGrids;
  }

  // Keep this as an invariant rather than a fallback: normalization is
  // responsible for constructing only legal execution grids.
  TT_assertv(allOperandGridsAreLegal(normalizedOperandGrids),
             "Grid normalization produced illegal operand grids within target "
             "grid {}",
             ttmlir::utils::formatIterable(targetGrid, "x"));

  return normalizedOperandGrids;
}

GenericGridAnalysisResult GridAnalysis::analyzeGenericOp(
    GenericOp genericOp,
    const EffectiveTargetGridRange &effectiveTargetGridRange) {
  GenericGridAnalysisResult result;
  result.effectiveTargetGridRange = effectiveTargetGridRange;
  ArrayRef<int64_t> targetGridShape = result.effectiveTargetGridRange.shape;

  // Build per-operand target grids. When a loop dimension maps to different
  // operand-dim positions across operands (e.g. matmul K is dim 1 of LHS but
  // dim 0 of RHS), those positions must use min(gridDims) so that
  // computeGridAwareDimAlignments produces consistent alignments.
  int64_t minGridDim = *llvm::min_element(targetGridShape);
  auto indexingMaps = genericOp.getIndexingMapsValue();
  unsigned numLoopDims = indexingMaps.front().getNumDims();

  // For each loop dim, find which operand-dim positions it maps to. If it
  // appears at different positions across operands, mark those positions as
  // needing the min constraint.
  // crossAxisPositions[operandIdx] = set of operand-dim positions that need
  // min(gridDims).
  SmallVector<llvm::DenseSet<unsigned>> crossAxisPositions(
      genericOp.getInputsAndOutputs().size());
  for (unsigned loopDim = 0; loopDim < numLoopDims; ++loopDim) {
    SmallVector<std::pair<unsigned, unsigned>> occurrences;
    for (auto [operandIdx, map] : llvm::enumerate(indexingMaps)) {
      for (auto [pos, expr] : llvm::enumerate(map.getResults())) {
        auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
        if (dimExpr && dimExpr.getPosition() == loopDim) {
          occurrences.push_back({operandIdx, pos});
        }
      }
    }
    bool isCrossAxis = false;
    if (occurrences.size() > 1) {
      unsigned firstPos = occurrences.front().second;
      for (auto [opIdx, pos] : occurrences) {
        if (pos != firstPos) {
          isCrossAxis = true;
          break;
        }
      }
    }
    if (isCrossAxis) {
      for (auto [opIdx, pos] : occurrences) {
        crossAxisPositions[opIdx].insert(pos);
      }
    }
  }

  // Build per-operand target grids.
  SmallVector<SmallVector<int64_t>> perOperandTargetGrids;
  for (auto [operandIdx, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    auto gridShape = operandLayout.getGridShape(operandType);
    unsigned rank = gridShape.size();

    // Start from the full device grid, then cap cross-axis positions.
    SmallVector<int64_t> opTarget(targetGridShape);
    // targetGridShape is 2D; cross-axis positions index into the last 2 dims
    // of the operand shape. Map them to the 2D target grid.
    for (unsigned pos : crossAxisPositions[operandIdx]) {
      if (pos >= rank - opTarget.size()) {
        unsigned targetIdx = pos - (rank - opTarget.size());
        opTarget[targetIdx] = minGridDim;
      }
    }
    perOperandTargetGrids.push_back(opTarget);
  }

  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  llvm::SmallVector<llvm::SmallVector<int64_t>> physicalShapes;
  bool allowVirtualGrid = !isMatmulGeneric(genericOp);

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    TT_assertv(operandLayout,
               "GridSelection expects GenericOp inputs/outputs to have "
               "MetalLayoutAttr");

    ArrayRef<int64_t> targetGrid = perOperandTargetGrids[operandIndex];
    GridDecisionAndShape decisionAndShape =
        allowVirtualGrid
            ? computeGridDecision(operand, targetGrid, targetGrid, ttnnMode,
                                  allowVirtualGrid)
            : computeMatmulGridDecision(operand, targetGrid, ttnnMode);
    optimalOperandGrids.push_back(decisionAndShape.decision.selectedGrid);
    physicalShapes.push_back(computeTypePhysicalShape(computePlannedOperandType(
        operand, decisionAndShape.decision, ttnnMode)));

    OperandGridInfo info;
    info.operand = operand;
    // Will be updated after normalization for view operands.
    info.grid = std::move(decisionAndShape.decision);

    result.operandInfos.push_back(std::move(info));
  }

  result.normalizedOperandGrids = normalizeOperandGridsForGeneric(
      genericOp, optimalOperandGrids, physicalShapes, targetGridShape, ttnnMode,
      /*requireCurrentTypeReblockable=*/false, usableL1Bytes);

  // Producer rewrites keep their own materialization decisions except when the
  // producer is part of the consuming GenericOp's view semantics. Composite
  // views are included here because DMA read expansion expects to see the
  // composite view directly rather than through an extra reblock view.
  for (unsigned idx = 0; idx < result.operandInfos.size(); ++idx) {
    OperandGridInfo &info = result.operandInfos[idx];
    if (isTTNNOperand(info.operand)) {
      continue;
    }
    auto view = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
    bool isCompositeView =
        info.operand.getDefiningOp<d2m::CompositeViewOp>() != nullptr;
    bool isToLayout = info.operand.getDefiningOp<d2m::ToLayoutOp>() != nullptr;
    bool isOutputOperand = idx >= genericOp.getInputs().size();
    GridDecision normalizedGrid = utils::makeGridDecision(
        result.normalizedOperandGrids[idx], perOperandTargetGrids[idx]);
    if (!allowVirtualGrid) {
      // Match main's matmul behavior: normalization adjusts execution grids,
      // not the full target-grid dim alignments used to materialize operands.
      normalizedGrid.layoutGrid =
          llvm::SmallVector<int64_t>(perOperandTargetGrids[idx]);
    }
    if (isToLayout && !isOutputOperand && !allowVirtualGrid) {
      if (ttmlir::utils::volume<int64_t>(normalizedGrid.selectedGrid) >
              ttmlir::utils::volume<int64_t>(info.grid.selectedGrid) &&
          canMaterializeOperandGrid(info.operand, normalizedGrid.selectedGrid,
                                    normalizedGrid.targetGrid, ttnnMode)) {
        info.grid = normalizedGrid;
      }
      continue;
    }
    if (!isOutputOperand && !isCompositeView &&
        (!view || view.getReinterpretLayout())) {
      continue;
    }
    info.grid = normalizedGrid;
  }

  // If an operand is a view chain over a ToLayoutOp, compute the producer's
  // grid from the final operand grid. This keeps producer-side materialization
  // aligned with the consumer plan, including loop-space normalization.
  for (OperandGridInfo &info : result.operandInfos) {
    if (auto toLayoutOp = getSourceToLayoutThroughViews(info.operand)) {
      if (toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      GridDecision sourceGrid =
          computeGridDecision(toLayoutOp.getResult(0), info.grid.targetGrid,
                              info.grid.targetGrid, ttnnMode,
                              /*allowVirtualGrid=*/true)
              .decision;
      if (allowVirtualGrid &&
          ttmlir::utils::volume<int64_t>(sourceGrid.selectedGrid) >
              ttmlir::utils::volume<int64_t>(info.grid.selectedGrid)) {
        info.viewSourceGrid = sourceGrid;
        continue;
      }
      if (canMaterializeOperandGrid(toLayoutOp.getResult(0),
                                    info.grid.selectedGrid,
                                    info.grid.targetGrid, ttnnMode)) {
        info.viewSourceGrid = info.grid;
        continue;
      }
      info.viewSourceGrid = sourceGrid;
    }
  }

  return result;
}

EffectiveTargetGridRange
GridAnalysis::getTargetGridRange(GenericOp genericOp) const {
  EffectiveTargetGridRange targetGridRange;
  mlir::Region *region = genericOp->getParentRegion();
  if (auto spatialOp = mlir::dyn_cast<d2m::SpatialOp>(region->getParentOp())) {
    mlir::ArrayAttr gridRangesAttr = spatialOp.getGridRanges();
    unsigned regionIndex = region->getRegionNumber();
    if (gridRangesAttr && regionIndex < gridRangesAttr.size()) {
      ttcore::CoreRangeAttr range =
          mlir::cast<ttcore::CoreRangeAttr>(gridRangesAttr[regionIndex]);
      targetGridRange.shape = {
          range.getEndCoord().getY() - range.getStartCoord().getY() + 1,
          range.getEndCoord().getX() - range.getStartCoord().getX() + 1};
      targetGridRange.offset = {range.getStartCoord().getY(),
                                range.getStartCoord().getX()};
      return targetGridRange;
    }
  }
  targetGridRange.shape = llvm::SmallVector<int64_t>(deviceGridShape);
  targetGridRange.offset = {0, 0};
  return targetGridRange;
}

GridAnalysis::GridAnalysis(Operation *moduleOp,
                           ArrayRef<int64_t> deviceGridShape, bool ttnnMode)
    : deviceGridShape(deviceGridShape),
      usableL1Bytes(ttcore::getCurrentScopeSystemDesc(moduleOp)
                        .getChipDesc(0)
                        .getUsableL1Size()),
      ttnnMode(ttnnMode) {
  moduleOp->walk([&](GenericOp genericOp) {
    // Skip explicit datamovement form — users manage grids manually.
    if (genericOp.isExplicitDatamovementForm()) {
      return;
    }
    if (genericOp->hasAttr("d2m.skip_grid_selection")) {
      return;
    }

    EffectiveTargetGridRange targetGridRange = getTargetGridRange(genericOp);
    GenericGridAnalysisResult result =
        analyzeGenericOp(genericOp, targetGridRange);
    results[genericOp.getOperation()] =
        std::make_unique<GenericGridAnalysisResult>(std::move(result));
  });
}

const GenericGridAnalysisResult *
GridAnalysis::lookup(GenericOp genericOp) const {
  auto it = results.find(genericOp.getOperation());
  if (it == results.end()) {
    return nullptr;
  }
  return it->second.get();
}

} // namespace mlir::tt::d2m
