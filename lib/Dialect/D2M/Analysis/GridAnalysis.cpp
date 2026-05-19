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
#include <optional>

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

static bool isBatchPlusTrailingDim1DVirtualGrid(ArrayRef<int64_t> grid,
                                                ArrayRef<int64_t> targetGrid) {
  unsigned trailingDimOffset =
      grid.size() > targetGrid.size()
          ? static_cast<unsigned>(grid.size() - targetGrid.size())
          : 0;
  unsigned nonUnitMatrixDims = 0;
  for (auto [dimIdx, dim] : llvm::enumerate(grid)) {
    if (dim <= 1 || dimIdx < trailingDimOffset) {
      continue;
    }
    ++nonUnitMatrixDims;
  }
  return nonUnitMatrixDims <= 1;
}

// Matmul kernels require a physical (non-virtual) grid or a 1D virtual grid
// along the logical height/width dimensions. Higher-rank matmuls may also
// shard batch dimensions virtually since each batch tile accumulates
// independently.
static bool isGridSupportedByMatmulKernel(ArrayRef<int64_t> grid,
                                          ArrayRef<int64_t> targetGrid) {
  return !ttmlir::d2m::utils::grids::requiresVirtualGrid(grid, targetGrid) ||
         isBatchPlusTrailingDim1DVirtualGrid(grid, targetGrid);
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

static GridDecision
makeGridDecisionForSelectedGrid(ArrayRef<int64_t> selectedGrid,
                                ArrayRef<int64_t> targetGrid,
                                bool useTargetGridForLayout = false) {
  GridDecision decision = utils::makeGridDecision(selectedGrid, targetGrid);
  if (useTargetGridForLayout) {
    decision.layoutGrid = llvm::SmallVector<int64_t>(targetGrid);
  }
  return decision;
}

struct GridDecisionAndShape {
  GridDecision decision;
  llvm::SmallVector<int64_t> physicalShape;
};

static uint64_t estimateOperandShardBytes(Type elementType,
                                          ArrayRef<int64_t> physicalShape,
                                          ArrayRef<int64_t> grid);

static llvm::SmallVector<int64_t>
computeSelectedGrid(mlir::Value operand, ArrayRef<int64_t> physicalShape,
                    ArrayRef<int64_t> targetGrid, bool allowVirtualGrid) {
  if (!allowVirtualGrid) {
    return utils::computeOptimalBlockShardedGrid(physicalShape, targetGrid);
  }

  auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  return utils::computeOptimalGrid(operandType, physicalShape, targetGrid);
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

static llvm::SmallVector<int64_t>
computeGridSearchPhysicalShape(Value operand, ArrayRef<int64_t> targetGrid,
                               bool ttnnMode, bool allowVirtualGrid) {
  if (!allowVirtualGrid || ttnnMode) {
    return utils::computePhysicalShape(operand, targetGrid, ttnnMode);
  }

  llvm::SmallVector<int64_t> tileOnlyGrid(targetGrid.size(), 1);
  return utils::computePhysicalShape(operand, tileOnlyGrid, ttnnMode);
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

static bool viewOperandHasRewritableToLayoutSource(Value operand) {
  auto toLayoutOp = getSourceToLayoutThroughViews(operand);
  return toLayoutOp &&
         !toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
}

static bool genericHasFixedNonReinterpretViewOperand(GenericOp genericOp) {
  return llvm::any_of(genericOp.getInputsAndOutputs(), [](Value operand) {
    auto view = operand.getDefiningOp<d2m::ViewLayoutOp>();
    if (!view || view.getReinterpretLayout() ||
        GridAnalysis::isTTNNOperand(operand)) {
      return false;
    }
    return !viewOperandHasRewritableToLayoutSource(operand);
  });
}

static llvm::SmallVector<int64_t>
computeMaterializedPhysicalShape(Value operand, const GridDecision &decision,
                                 bool ttnnMode) {
  auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  if (!canGridSelectionRewriteOperand(operand)) {
    return computeTypePhysicalShape(operandType);
  }

  auto oldLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
  llvm::SmallVector<int64_t> tileShape;
  if (auto tileType =
          mlir::dyn_cast<ttcore::TileType>(operandType.getElementType())) {
    tileShape = llvm::to_vector(tileType.getShape());
  } else {
    tileShape = llvm::to_vector(ttcore::TileType::getDefaultShape());
  }

  ttcore::MetalLayoutAttr newLayout =
      utils::layoutWithOptimalGrid(oldLayout, decision.layoutGrid, ttnnMode);
  return newLayout.getPhysicalShape(llvm::ArrayRef(tileShape));
}

static bool gridDividesPhysicalShape(ArrayRef<int64_t> physicalShape,
                                     ArrayRef<int64_t> grid) {
  if (physicalShape.size() != grid.size()) {
    return false;
  }
  for (auto [physicalDim, gridDim] : llvm::zip_equal(physicalShape, grid)) {
    if (gridDim == 0 || physicalDim % gridDim != 0) {
      return false;
    }
  }
  return true;
}

static std::optional<GridDecisionAndShape>
computeDecisionAndShapeForSelectedGrid(mlir::Value operand,
                                       ArrayRef<int64_t> selectedGrid,
                                       ArrayRef<int64_t> targetGrid,
                                       bool ttnnMode,
                                       bool useTargetGridForLayout = false) {
  if (!hasLegalPhysicalPlacement(selectedGrid, targetGrid)) {
    return std::nullopt;
  }

  GridDecision baseDecision = makeGridDecisionForSelectedGrid(
      selectedGrid, targetGrid, useTargetGridForLayout);

  auto buildCandidate =
      [&](GridDecision decision) -> std::optional<GridDecisionAndShape> {
    llvm::SmallVector<int64_t> physicalShape =
        computeMaterializedPhysicalShape(operand, decision, ttnnMode);
    if (!gridDividesPhysicalShape(physicalShape, selectedGrid)) {
      return std::nullopt;
    }

    GridDecisionAndShape candidate;
    candidate.decision = std::move(decision);
    candidate.physicalShape = std::move(physicalShape);
    return candidate;
  };

  std::optional<GridDecisionAndShape> baseCandidate =
      buildCandidate(baseDecision);
  if (baseCandidate || useTargetGridForLayout || !baseDecision.isVirtual() ||
      ttnnMode || targetGrid.size() != 2) {
    return baseCandidate;
  }

  auto isBetterFallbackPlacement = [](const GridDecisionAndShape &candidate,
                                      const GridDecisionAndShape &best) {
    uint64_t candidatePhysicalVolume = static_cast<uint64_t>(
        ttmlir::utils::volume<int64_t>(candidate.physicalShape));
    uint64_t bestPhysicalVolume = static_cast<uint64_t>(
        ttmlir::utils::volume<int64_t>(best.physicalShape));
    if (candidatePhysicalVolume != bestPhysicalVolume) {
      return candidatePhysicalVolume < bestPhysicalVolume;
    }

    for (auto [candidateDim, bestDim] : llvm::zip_equal(
             candidate.decision.physicalGrid, best.decision.physicalGrid)) {
      if (candidateDim != bestDim) {
        return candidateDim < bestDim;
      }
    }
    return false;
  };

  std::optional<GridDecisionAndShape> best;
  int64_t selectedGridVolume = ttmlir::utils::volume<int64_t>(selectedGrid);
  for (int64_t rowFactor = 1; rowFactor <= targetGrid[0]; ++rowFactor) {
    for (int64_t colFactor = 1; colFactor <= targetGrid[1]; ++colFactor) {
      if (rowFactor * colFactor != selectedGridVolume) {
        continue;
      }

      GridDecision candidateDecision = baseDecision;
      candidateDecision.physicalGrid = {rowFactor, colFactor};
      candidateDecision.layoutGrid = candidateDecision.physicalGrid;
      std::optional<GridDecisionAndShape> candidate =
          buildCandidate(std::move(candidateDecision));
      if (!candidate) {
        continue;
      }
      if (!best || isBetterFallbackPlacement(*candidate, *best)) {
        best = std::move(candidate);
      }
    }
  }

  return best;
}

static bool isBetterInitialGridCandidate(uint64_t candidateShardBytes,
                                         uint64_t candidateGridVolume,
                                         uint64_t candidatePhysicalVolume,
                                         uint64_t bestShardBytes,
                                         uint64_t bestGridVolume,
                                         uint64_t bestPhysicalVolume) {
  if (candidateGridVolume != bestGridVolume) {
    return candidateGridVolume > bestGridVolume;
  }
  if (candidateShardBytes != bestShardBytes) {
    return candidateShardBytes < bestShardBytes;
  }
  return candidatePhysicalVolume < bestPhysicalVolume;
}

static bool
isStrictlyBetterThanBaselineInitialPlan(Type elementType,
                                        const GridDecisionAndShape &candidate,
                                        const GridDecisionAndShape &baseline) {
  uint64_t candidateShardBytes = estimateOperandShardBytes(
      elementType, candidate.physicalShape, candidate.decision.selectedGrid);
  uint64_t baselineShardBytes = estimateOperandShardBytes(
      elementType, baseline.physicalShape, baseline.decision.selectedGrid);
  uint64_t candidateGridVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(candidate.decision.selectedGrid));
  uint64_t baselineGridVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(baseline.decision.selectedGrid));
  uint64_t candidatePhysicalVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(candidate.physicalShape));
  uint64_t baselinePhysicalVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(baseline.physicalShape));

  if (candidateShardBytes < baselineShardBytes &&
      candidateGridVolume >= baselineGridVolume) {
    return true;
  }
  return candidateGridVolume > baselineGridVolume &&
         candidateShardBytes <= baselineShardBytes &&
         candidatePhysicalVolume <= baselinePhysicalVolume;
}

static std::optional<GridDecisionAndShape>
computeBestInitialGridDecision(Value operand, ArrayRef<int64_t> targetGrid,
                               bool ttnnMode, bool allowVirtualGrid,
                               bool useTargetGridForLayout) {
  llvm::SmallVector<int64_t> targetPhysicalShape =
      utils::computePhysicalShape(operand, targetGrid, ttnnMode);
  // TTNN mode already has a legacy representation for height/width virtual
  // grids coming from TTNN layout casts. Preserve that main-style selection,
  // but do not run the broader D2M virtual-grid search below.
  llvm::SmallVector<int64_t> baselineGrid = computeSelectedGrid(
      operand, targetPhysicalShape, targetGrid, allowVirtualGrid || ttnnMode);
  std::optional<GridDecisionAndShape> baselinePlan =
      computeDecisionAndShapeForSelectedGrid(operand, baselineGrid, targetGrid,
                                             ttnnMode, useTargetGridForLayout);
  if (!allowVirtualGrid || ttnnMode) {
    return baselinePlan;
  }

  llvm::SmallVector<int64_t> searchPhysicalShape =
      computeGridSearchPhysicalShape(operand, targetGrid, ttnnMode,
                                     allowVirtualGrid);
  int64_t targetGridVolume = ttmlir::utils::volume<int64_t>(targetGrid);
  llvm::SmallVector<llvm::SmallVector<int64_t>> factorChoices(
      searchPhysicalShape.size());
  auto addFactors = [&](ArrayRef<int64_t> physicalShape) {
    for (auto [dim, physicalDim] : llvm::enumerate(physicalShape)) {
      factorChoices[dim].push_back(1);
      for (int64_t factor : ttmlir::utils::getFactors(physicalDim)) {
        if (factor <= targetGridVolume) {
          factorChoices[dim].push_back(factor);
        }
      }
    }
  };
  addFactors(searchPhysicalShape);
  for (llvm::SmallVector<int64_t> &factors : factorChoices) {
    llvm::sort(factors);
    factors.erase(std::unique(factors.begin(), factors.end()), factors.end());
  }

  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  std::optional<GridDecisionAndShape> best;
  uint64_t bestShardBytes = std::numeric_limits<uint64_t>::max();
  uint64_t bestGridVolume = 0;
  uint64_t bestPhysicalVolume = std::numeric_limits<uint64_t>::max();
  llvm::SmallVector<int64_t> candidateGrid(searchPhysicalShape.size(), 1);

  std::function<void(unsigned, uint64_t)> search = [&](unsigned dim,
                                                       uint64_t volumeSoFar) {
    if (dim == candidateGrid.size()) {
      std::optional<GridDecisionAndShape> candidate =
          computeDecisionAndShapeForSelectedGrid(operand, candidateGrid,
                                                 targetGrid, ttnnMode,
                                                 useTargetGridForLayout);
      if (!candidate) {
        return;
      }
      uint64_t candidateShardBytes = estimateOperandShardBytes(
          tensorType.getElementType(), candidate->physicalShape, candidateGrid);
      uint64_t candidatePhysicalVolume = static_cast<uint64_t>(
          ttmlir::utils::volume<int64_t>(candidate->physicalShape));
      if (!best ||
          isBetterInitialGridCandidate(candidateShardBytes, volumeSoFar,
                                       candidatePhysicalVolume, bestShardBytes,
                                       bestGridVolume, bestPhysicalVolume)) {
        best = std::move(candidate);
        bestShardBytes = candidateShardBytes;
        bestGridVolume = volumeSoFar;
        bestPhysicalVolume = candidatePhysicalVolume;
      }
      return;
    }

    for (int64_t factor : llvm::reverse(factorChoices[dim])) {
      uint64_t candidateVolume = volumeSoFar * static_cast<uint64_t>(factor);
      if (candidateVolume > static_cast<uint64_t>(targetGridVolume)) {
        continue;
      }
      candidateGrid[dim] = factor;
      search(dim + 1, candidateVolume);
    }
    candidateGrid[dim] = 1;
  };
  search(/*dim=*/0, /*volumeSoFar=*/1);
  if (!baselinePlan) {
    return best;
  }
  if (best && isStrictlyBetterThanBaselineInitialPlan(
                  tensorType.getElementType(), *best, *baselinePlan)) {
    return best;
  }
  return baselinePlan;
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

static bool isBetterSourceGridCandidate(uint64_t candidateShardBytes,
                                        uint64_t candidateGridVolume,
                                        uint64_t candidatePhysicalVolume,
                                        uint64_t bestShardBytes,
                                        uint64_t bestGridVolume,
                                        uint64_t bestPhysicalVolume) {
  if (candidateShardBytes != bestShardBytes) {
    return candidateShardBytes < bestShardBytes;
  }
  if (candidateGridVolume != bestGridVolume) {
    return candidateGridVolume > bestGridVolume;
  }
  return candidatePhysicalVolume < bestPhysicalVolume;
}

static std::optional<GridDecisionAndShape>
computeBestSourceGridDecision(Value operand, ArrayRef<int64_t> targetGrid,
                              bool ttnnMode, bool allowVirtualGrid,
                              bool useTargetGridForLayout) {
  llvm::SmallVector<int64_t> currentPhysicalShape =
      computeCurrentPhysicalShape(operand);
  llvm::SmallVector<int64_t> targetPhysicalShape =
      utils::computePhysicalShape(operand, targetGrid, ttnnMode);
  if (currentPhysicalShape.size() != targetPhysicalShape.size()) {
    return std::nullopt;
  }

  int64_t targetGridVolume = ttmlir::utils::volume<int64_t>(targetGrid);
  llvm::SmallVector<llvm::SmallVector<int64_t>> factorChoices;
  factorChoices.reserve(currentPhysicalShape.size());
  for (auto [currentDim, targetDim] :
       llvm::zip_equal(currentPhysicalShape, targetPhysicalShape)) {
    llvm::SmallVector<int64_t> factors{1};
    for (int64_t factor : ttmlir::utils::getFactors(currentDim)) {
      if (factor <= targetGridVolume) {
        factors.push_back(factor);
      }
    }
    for (int64_t factor : ttmlir::utils::getFactors(targetDim)) {
      if (factor <= targetGridVolume) {
        factors.push_back(factor);
      }
    }
    llvm::sort(factors);
    factors.erase(std::unique(factors.begin(), factors.end()), factors.end());
    factorChoices.push_back(std::move(factors));
  }

  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  std::optional<GridDecisionAndShape> best;
  uint64_t bestShardBytes = std::numeric_limits<uint64_t>::max();
  uint64_t bestGridVolume = 0;
  uint64_t bestPhysicalVolume = std::numeric_limits<uint64_t>::max();
  llvm::SmallVector<int64_t> candidateGrid(currentPhysicalShape.size(), 1);

  std::function<void(unsigned, uint64_t)> search = [&](unsigned dim,
                                                       uint64_t volumeSoFar) {
    if (dim == candidateGrid.size()) {
      if (!allowVirtualGrid && ttmlir::d2m::utils::grids::requiresVirtualGrid(
                                   candidateGrid, targetGrid)) {
        return;
      }
      auto candidate = computeDecisionAndShapeForSelectedGrid(
          operand, candidateGrid, targetGrid, ttnnMode, useTargetGridForLayout);
      if (!candidate) {
        return;
      }
      uint64_t candidateShardBytes = estimateOperandShardBytes(
          tensorType.getElementType(), candidate->physicalShape, candidateGrid);
      uint64_t candidatePhysicalVolume = static_cast<uint64_t>(
          ttmlir::utils::volume<int64_t>(candidate->physicalShape));
      if (!best ||
          isBetterSourceGridCandidate(candidateShardBytes, volumeSoFar,
                                      candidatePhysicalVolume, bestShardBytes,
                                      bestGridVolume, bestPhysicalVolume)) {
        best = std::move(candidate);
        bestShardBytes = candidateShardBytes;
        bestGridVolume = volumeSoFar;
        bestPhysicalVolume = candidatePhysicalVolume;
      }
      return;
    }

    for (int64_t factor : llvm::reverse(factorChoices[dim])) {
      uint64_t candidateVolume = volumeSoFar * static_cast<uint64_t>(factor);
      if (candidateVolume > static_cast<uint64_t>(targetGridVolume)) {
        continue;
      }
      candidateGrid[dim] = factor;
      search(dim + 1, candidateVolume);
    }
    candidateGrid[dim] = 1;
  };
  search(/*dim=*/0, /*volumeSoFar=*/1);
  return best;
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

static uint64_t
computeLoopGridVolume(GenericOp genericOp,
                      ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
  auto indexingMaps = genericOp.getIndexingMapsValue();
  unsigned numLoopDims = indexingMaps.front().getNumDims();
  llvm::SmallVector<int64_t> loopGrid(numLoopDims, 1);

  for (auto [operandIndex, indexingMap] : llvm::enumerate(indexingMaps)) {
    for (auto [operandDimIdx, expr] :
         llvm::enumerate(indexingMap.getResults())) {
      auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      loopGrid[dimExpr.getPosition()] =
          std::max(loopGrid[dimExpr.getPosition()],
                   operandGrids[operandIndex][operandDimIdx]);
    }
  }

  return static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(loopGrid));
}

struct GenericGridPlan {
  llvm::SmallVector<GridDecisionAndShape, 4> operandPlans;
  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids;
  llvm::SmallVector<llvm::SmallVector<int64_t>> physicalShapes;
  uint64_t loopGridVolume = 1;
  uint64_t outputGridVolume = 1;
};

static bool computePlanStats(GenericOp genericOp, GenericGridPlan &plan) {
  if (plan.normalizedOperandGrids.empty() || plan.physicalShapes.empty()) {
    return false;
  }

  const unsigned outputOperandIndex = genericOp.getInputs().size();
  if (outputOperandIndex >= plan.normalizedOperandGrids.size()) {
    return false;
  }

  plan.loopGridVolume =
      computeLoopGridVolume(genericOp, plan.normalizedOperandGrids);
  plan.outputGridVolume = static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
      plan.normalizedOperandGrids[outputOperandIndex]));
  return true;
}

static std::optional<GenericGridPlan> computeGenericGridPlanFromCurrentTypes(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool allowVirtualGrid,
    bool useTargetGridForLayout, bool requireCurrentTypeReblockable) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> physicalShapes;
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  physicalShapes.reserve(genericOp.getInputsAndOutputs().size());
  optimalOperandGrids.reserve(genericOp.getInputsAndOutputs().size());

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    std::optional<GridDecisionAndShape> initialPlan =
        computeBestInitialGridDecision(
            operand, perOperandTargetGrids[operandIndex], ttnnMode,
            allowVirtualGrid, useTargetGridForLayout);
    if (!initialPlan) {
      return std::nullopt;
    }
    physicalShapes.push_back(initialPlan->physicalShape);
    optimalOperandGrids.push_back(initialPlan->decision.selectedGrid);
  }

  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids =
      GridAnalysis::normalizeOperandGridsForGeneric(
          genericOp, optimalOperandGrids, physicalShapes, perOperandTargetGrids,
          targetGrid, ttnnMode, useTargetGridForLayout,
          requireCurrentTypeReblockable);
  if (normalizedOperandGrids.empty()) {
    return std::nullopt;
  }

  GenericGridPlan plan;
  plan.normalizedOperandGrids = std::move(normalizedOperandGrids);
  plan.operandPlans.reserve(plan.normalizedOperandGrids.size());
  plan.physicalShapes.reserve(plan.normalizedOperandGrids.size());

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    std::optional<GridDecisionAndShape> operandPlan =
        computeDecisionAndShapeForSelectedGrid(
            operand, plan.normalizedOperandGrids[operandIndex],
            perOperandTargetGrids[operandIndex], ttnnMode,
            useTargetGridForLayout);
    if (!operandPlan) {
      return std::nullopt;
    }
    plan.physicalShapes.push_back(operandPlan->physicalShape);
    plan.operandPlans.push_back(std::move(*operandPlan));
  }

  if (!computePlanStats(genericOp, plan)) {
    return std::nullopt;
  }
  return plan;
}

static llvm::SmallVector<llvm::SmallVector<int64_t>>
normalizeMatmulOperandGridsLikeMain(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids(
      optimalOperandGrids.begin(), optimalOperandGrids.end());

  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<uint64_t, uint64_t>>>
      dimToInputOperandDims;
  auto indexingMaps = genericOp.getIndexingMapsValue();
  for (uint64_t operandIndex = 0; operandIndex < genericOp.getInputs().size();
       ++operandIndex) {
    auto results = indexingMaps[operandIndex].getResults();
    for (auto [operandDimIdx, expr] : llvm::enumerate(results)) {
      auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      dimToInputOperandDims[dimExpr.getPosition()].push_back(
          {operandIndex, static_cast<uint64_t>(operandDimIdx)});
    }
  }

  for (auto &entry : dimToInputOperandDims) {
    auto &operandDims = entry.second;
    if (operandDims.size() < 2) {
      continue;
    }

    int64_t maxFactor = 0;
    llvm::SmallVector<int64_t> physicalDimsForLoop;
    for (auto [operandIndex, operandDimIdx] : operandDims) {
      maxFactor = std::max(maxFactor,
                           normalizedOperandGrids[operandIndex][operandDimIdx]);
      physicalDimsForLoop.push_back(
          physicalShapes[operandIndex][operandDimIdx]);
    }

    int64_t commonFactor =
        findLargestCommonFactor(maxFactor, physicalDimsForLoop);
    for (auto [operandIndex, operandDimIdx] : operandDims) {
      normalizedOperandGrids[operandIndex][operandDimIdx] = commonFactor;
    }
  }

  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  auto outputIndexingMap = indexingMaps[outputOperandIndex];
  auto outputShape = optimalOperandGrids[outputOperandIndex];
  std::optional<llvm::SmallVector<int64_t>> outputConstraints =
      d2m::utils::computeDimConstraints(
          llvm::ArrayRef<AffineMap>(outputIndexingMap),
          llvm::ArrayRef<llvm::SmallVector<int64_t>>(outputShape));

  if (!outputConstraints) {
    return normalizedOperandGrids;
  }

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputsMutable())) {
    if (genericOp.isDpsInit(&operand)) {
      continue;
    }

    auto results = indexingMaps[operandIndex].getResults();
    TT_assertv(results.size() == normalizedOperandGrids[operandIndex].size(),
               "indexing map results size does not match normalized operand "
               "grids size");

    for (auto [resultIdx, expr] : llvm::enumerate(results)) {
      auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      int64_t constraint = (*outputConstraints)[dimExpr.getPosition()];
      if (constraint == 0) {
        continue;
      }
      int64_t physicalDim = physicalShapes[operandIndex][resultIdx];
      normalizedOperandGrids[operandIndex][resultIdx] =
          physicalDim % constraint == 0
              ? constraint
              : findLargestCommonFactor(constraint, {physicalDim, constraint});
    }
  }

  return normalizedOperandGrids;
}

static std::optional<GenericGridPlan> computeMatmulGridPlan(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids, bool ttnnMode) {
  GenericGridPlan plan;
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  optimalOperandGrids.reserve(genericOp.getInputsAndOutputs().size());
  plan.physicalShapes.reserve(genericOp.getInputsAndOutputs().size());
  plan.operandPlans.reserve(genericOp.getInputsAndOutputs().size());

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    ArrayRef<int64_t> targetGrid = perOperandTargetGrids[operandIndex];
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    llvm::SmallVector<int64_t> physicalShape =
        utils::computePhysicalShape(operand, targetGrid, ttnnMode);
    llvm::SmallVector<int64_t> selectedGrid =
        utils::computeOptimalGrid(operandType, physicalShape, targetGrid);
    std::optional<GridDecisionAndShape> operandPlan =
        computeDecisionAndShapeForSelectedGrid(operand, selectedGrid,
                                               targetGrid, ttnnMode,
                                               /*useTargetGridForLayout=*/true);
    if (!operandPlan) {
      return std::nullopt;
    }

    plan.physicalShapes.push_back(operandPlan->physicalShape);
    plan.operandPlans.push_back(std::move(*operandPlan));
    optimalOperandGrids.push_back(std::move(selectedGrid));
  }

  // Preserve main's matmul contract: normalization may introduce execution
  // block factors (especially along K), but producer materialization remains
  // anchored to the per-operand selected grids computed above.
  plan.normalizedOperandGrids = normalizeMatmulOperandGridsLikeMain(
      genericOp, optimalOperandGrids, plan.physicalShapes);

  if (!computePlanStats(genericOp, plan)) {
    return std::nullopt;
  }
  return plan;
}

static std::optional<GenericGridPlan> computeGenericGridPlan(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool allowVirtualGrid) {
  bool useTargetGridForLayout = !allowVirtualGrid && !ttnnMode;
  bool requireCurrentTypeReblockable =
      genericHasFixedNonReinterpretViewOperand(genericOp);
  return computeGenericGridPlanFromCurrentTypes(
      genericOp, perOperandTargetGrids, targetGrid, ttnnMode, allowVirtualGrid,
      useTargetGridForLayout, requireCurrentTypeReblockable);
}

static std::optional<GenericGridPlan> computePreserveCurrentGridPlan(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids, bool ttnnMode) {
  GenericGridPlan plan;
  plan.normalizedOperandGrids.reserve(genericOp.getInputsAndOutputs().size());
  plan.physicalShapes.reserve(genericOp.getInputsAndOutputs().size());
  plan.operandPlans.reserve(genericOp.getInputsAndOutputs().size());

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    llvm::SmallVector<int64_t> currentGrid(
        operandLayout.getGridShape(operandType));
    std::optional<GridDecisionAndShape> operandPlan =
        computeDecisionAndShapeForSelectedGrid(
            operand, currentGrid, perOperandTargetGrids[operandIndex], ttnnMode,
            /*useTargetGridForLayout=*/false);
    if (!operandPlan) {
      return std::nullopt;
    }

    plan.normalizedOperandGrids.push_back(currentGrid);
    plan.physicalShapes.push_back(operandPlan->physicalShape);
    plan.operandPlans.push_back(std::move(*operandPlan));
  }

  if (!computePlanStats(genericOp, plan)) {
    return std::nullopt;
  }
  return plan;
}

llvm::SmallVector<llvm::SmallVector<int64_t>>
GridAnalysis::normalizeOperandGridsForGeneric(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool useTargetGridForLayout,
    bool requireCurrentTypeReblockable) {
  if (optimalOperandGrids.empty()) {
    return {};
  }

  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());
  TT_assert(physicalShapes.size() == optimalOperandGrids.size());
  TT_assert(perOperandTargetGrids.size() == optimalOperandGrids.size());

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
  auto iteratorTypes = genericOp.getIteratorTypesValue();
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
    if (!requireMatmulCompatibleGrid &&
        iteratorTypes[loopDim] == ttcore::IteratorType::Reduction) {
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

  auto computeMaterializedPhysicalShapesForGrids =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids)
      -> std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>> {
    llvm::SmallVector<llvm::SmallVector<int64_t>> materializedPhysicalShapes;
    materializedPhysicalShapes.reserve(operandGrids.size());
    for (auto [operandIndex, operandGrid] : llvm::enumerate(operandGrids)) {
      if (requireCurrentTypeReblockable &&
          !canReblockCurrentTypeToGrid(operands[operandIndex], operandGrid)) {
        return std::nullopt;
      }

      std::optional<GridDecisionAndShape> operandPlan =
          computeDecisionAndShapeForSelectedGrid(
              operands[operandIndex], operandGrid,
              perOperandTargetGrids[operandIndex], ttnnMode,
              useTargetGridForLayout);
      if (!operandPlan ||
          operandPlan->physicalShape.size() != operandGrid.size()) {
        return std::nullopt;
      }

      for (auto [physicalDim, gridDim] :
           llvm::zip_equal(operandPlan->physicalShape, operandGrid)) {
        if (gridDim == 0 || physicalDim % gridDim != 0) {
          return std::nullopt;
        }
      }

      if (requireMatmulCompatibleGrid &&
          !isGridSupportedByMatmulKernel(operandGrid,
                                         perOperandTargetGrids[operandIndex])) {
        return std::nullopt;
      }

      materializedPhysicalShapes.push_back(
          std::move(operandPlan->physicalShape));
    }
    return materializedPhysicalShapes;
  };

  auto localLoopShapesAreConsistent =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids,
          ArrayRef<llvm::SmallVector<int64_t>> materializedPhysicalShapes) {
        if (!genericOp.hasComputeOpsInRegion() || hasIndexedRowAccess) {
          return true;
        }
        for (unsigned loopDim = 0; loopDim < numLoopDims; ++loopDim) {
          std::optional<int64_t> expectedLocalDim;
          for (auto [operandIndex, operandDimIdx] :
               loopDimOperandDims[loopDim]) {
            int64_t gridDim = operandGrids[operandIndex][operandDimIdx];
            if (gridDim == 0 ||
                materializedPhysicalShapes[operandIndex][operandDimIdx] %
                        gridDim !=
                    0) {
              return false;
            }
            int64_t localDim =
                materializedPhysicalShapes[operandIndex][operandDimIdx] /
                gridDim;
            if (!expectedLocalDim) {
              expectedLocalDim = localDim;
              continue;
            }
            if (*expectedLocalDim != localDim) {
              return false;
            }
          }
        }
        return true;
      };

  auto computeLegalMaterializedPhysicalShapes =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids)
      -> std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>> {
    std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        materializedPhysicalShapes =
            computeMaterializedPhysicalShapesForGrids(operandGrids);
    if (!materializedPhysicalShapes ||
        !localLoopShapesAreConsistent(operandGrids,
                                      *materializedPhysicalShapes)) {
      return std::nullopt;
    }
    return materializedPhysicalShapes;
  };

  auto allOperandGridsAreLegal =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
        return computeLegalMaterializedPhysicalShapes(operandGrids).has_value();
      };

  llvm::SmallVector<int64_t> bestLoopGrid(numLoopDims, 1);
  uint64_t bestLoopGridVolume = 0;
  uint64_t bestOutputGridVolume = 0;
  uint64_t bestEstimatedBytes = std::numeric_limits<uint64_t>::max();
  bool foundLegalGrid = false;
  const unsigned outputOperandIndex = genericOp.getInputs().size();
  auto isBetterGridCandidate =
      [](uint64_t candidateEstimatedBytes, uint64_t candidateOutputGridVolume,
         uint64_t candidateLoopGridVolume, uint64_t bestEstimatedBytes,
         uint64_t bestOutputGridVolume, uint64_t bestLoopGridVolume) {
        if (candidateOutputGridVolume != bestOutputGridVolume) {
          return candidateOutputGridVolume > bestOutputGridVolume;
        }
        if (candidateLoopGridVolume != bestLoopGridVolume) {
          return candidateLoopGridVolume > bestLoopGridVolume;
        }
        return candidateEstimatedBytes < bestEstimatedBytes;
      };
  llvm::SmallVector<int64_t> candidateLoopGrid(numLoopDims, 1);
  llvm::SmallVector<llvm::SmallVector<int64_t>> loopFactorChoices;
  auto addLoopFactorChoices = [&](llvm::SmallVector<int64_t> &factors,
                                  int64_t value) {
    for (int64_t factor : ttmlir::utils::getFactors(value)) {
      if (factor <= targetGridVolume || requireMatmulCompatibleGrid) {
        factors.push_back(factor);
      }
    }
  };
  auto buildLoopFactorChoices = [&]() {
    llvm::SmallVector<llvm::SmallVector<int64_t>> choices;
    choices.reserve(numLoopDims);
    for (unsigned loopDim = 0; loopDim < numLoopDims; ++loopDim) {
      llvm::SmallVector<int64_t> factors;
      if (!requireMatmulCompatibleGrid &&
          iteratorTypes[loopDim] == ttcore::IteratorType::Reduction) {
        factors.push_back(1);
        choices.push_back(std::move(factors));
        continue;
      }
      addLoopFactorChoices(factors, desiredLoopGrid[loopDim]);
      for (auto [operandIndex, operandDimIdx] : loopDimOperandDims[loopDim]) {
        addLoopFactorChoices(factors,
                             optimalOperandGrids[operandIndex][operandDimIdx]);
        if (requireCurrentTypeReblockable) {
          addLoopFactorChoices(
              factors, currentReblockShapes[operandIndex][operandDimIdx]);
        }
      }
      if (factors.empty()) {
        factors.push_back(1);
      }
      llvm::sort(factors);
      factors.erase(std::unique(factors.begin(), factors.end()), factors.end());
      choices.push_back(std::move(factors));
    }
    return choices;
  };

  // Candidate materialization below is the final legality check for projected
  // operand grids. Search all known factor bases once; virtual-grid candidates
  // were already seeded from the unpadded physical shape, and materialization
  // recomputes padding from the final chosen physical placement.
  std::function<void(unsigned, uint64_t)> searchLoopGrids =
      [&](unsigned loopDim, uint64_t volumeSoFar) {
        if (loopDim == numLoopDims) {
          auto candidateOperandGrids =
              projectLoopGridToOperands(candidateLoopGrid);
          std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
              candidatePhysicalShapes =
                  computeLegalMaterializedPhysicalShapes(candidateOperandGrids);
          if (!candidatePhysicalShapes) {
            return;
          }

          uint64_t candidateEstimatedBytes = estimateGenericL1Bytes(
              genericOp, candidateOperandGrids, *candidatePhysicalShapes);
          uint64_t candidateOutputGridVolume =
              static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
                  candidateOperandGrids[outputOperandIndex]));

          if (!foundLegalGrid ||
              isBetterGridCandidate(candidateEstimatedBytes,
                                    candidateOutputGridVolume, volumeSoFar,
                                    bestEstimatedBytes, bestOutputGridVolume,
                                    bestLoopGridVolume)) {
            bestLoopGrid = candidateLoopGrid;
            bestLoopGridVolume = volumeSoFar;
            bestOutputGridVolume = candidateOutputGridVolume;
            bestEstimatedBytes = candidateEstimatedBytes;
          }
          foundLegalGrid = true;
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
  loopFactorChoices = buildLoopFactorChoices();

  auto desiredOperandGrids = projectLoopGridToOperands(desiredLoopGrid);
  if (std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
          desiredPhysicalShapes =
              computeLegalMaterializedPhysicalShapes(desiredOperandGrids)) {
    bestLoopGrid = desiredLoopGrid;
    bestLoopGridVolume =
        static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(desiredLoopGrid));
    bestOutputGridVolume = static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
        desiredOperandGrids[outputOperandIndex]));
    bestEstimatedBytes = estimateGenericL1Bytes(genericOp, desiredOperandGrids,
                                                *desiredPhysicalShapes);
    foundLegalGrid = true;
  }

  if (ttnnMode && foundLegalGrid) {
    return desiredOperandGrids;
  }

  searchLoopGrids(/*loopDim=*/0, /*volumeSoFar=*/1);

  if (!foundLegalGrid) {
    return {};
  }

  auto normalizedOperandGrids = projectLoopGridToOperands(bestLoopGrid);
  if (hasIndexedRowAccess && outputOperandIndex < operands.size() &&
      targetGrid.size() >= 2 && operands.size() >= 3 &&
      !normalizedOperandGrids[0].empty() &&
      !normalizedOperandGrids[1].empty() &&
      !normalizedOperandGrids[outputOperandIndex].empty()) {
    llvm::SmallVector<llvm::SmallVector<int64_t>> bestEmbeddingGrids =
        normalizedOperandGrids;
    std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
        bestEmbeddingPhysicalShapes =
            computeLegalMaterializedPhysicalShapes(bestEmbeddingGrids);
    TT_assert(bestEmbeddingPhysicalShapes);
    uint64_t bestEmbeddingEstimatedBytes = estimateGenericL1Bytes(
        genericOp, bestEmbeddingGrids, *bestEmbeddingPhysicalShapes);
    uint64_t bestEmbeddingOutputGridVolume = static_cast<uint64_t>(
        ttmlir::utils::volume<int64_t>(bestEmbeddingGrids[outputOperandIndex]));
    uint64_t bestEmbeddingLoopGridVolume =
        computeLoopGridVolume(genericOp, bestEmbeddingGrids);

    int64_t maxRowFactor = findLargestCommonFactor(
        targetGrid[0], {(*bestEmbeddingPhysicalShapes)[1][0],
                        (*bestEmbeddingPhysicalShapes)[outputOperandIndex][0]});
    for (int64_t rowFactor : ttmlir::utils::getFactors(maxRowFactor)) {
      if (rowFactor <= normalizedOperandGrids[outputOperandIndex][0]) {
        continue;
      }
      llvm::SmallVector<llvm::SmallVector<int64_t>> candidateOperandGrids =
          normalizedOperandGrids;
      candidateOperandGrids[0][0] = 1;
      candidateOperandGrids[1][0] = rowFactor;
      candidateOperandGrids[outputOperandIndex][0] = rowFactor;
      std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
          candidatePhysicalShapes =
              computeLegalMaterializedPhysicalShapes(candidateOperandGrids);
      if (!candidatePhysicalShapes) {
        continue;
      }

      uint64_t candidateEstimatedBytes = estimateGenericL1Bytes(
          genericOp, candidateOperandGrids, *candidatePhysicalShapes);
      uint64_t candidateOutputGridVolume =
          static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
              candidateOperandGrids[outputOperandIndex]));
      uint64_t candidateLoopGridVolume =
          computeLoopGridVolume(genericOp, candidateOperandGrids);
      if (isBetterGridCandidate(
              candidateEstimatedBytes, candidateOutputGridVolume,
              candidateLoopGridVolume, bestEmbeddingEstimatedBytes,
              bestEmbeddingOutputGridVolume, bestEmbeddingLoopGridVolume)) {
        bestEmbeddingGrids = candidateOperandGrids;
        bestEmbeddingEstimatedBytes = candidateEstimatedBytes;
        bestEmbeddingOutputGridVolume = candidateOutputGridVolume;
        bestEmbeddingLoopGridVolume = candidateLoopGridVolume;
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

  bool isMatmul = isMatmulGeneric(genericOp);
  bool inSpatialRegion =
      mlir::isa<d2m::SpatialOp>(genericOp->getParentRegion()->getParentOp());
  // TTNN conversion can round-trip legacy height/width sharding, but not
  // arbitrary D2M virtual grids. Spatial regions also carry explicit physical
  // ranges, so keep their grids directly placeable in that range.
  bool allowVirtualGrid = !isMatmul && !ttnnMode && !inSpatialRegion;
  for (Value operand : genericOp.getInputsAndOutputs()) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    TT_assertv(operandLayout,
               "GridSelection expects GenericOp inputs/outputs to have "
               "MetalLayoutAttr");
  }

  std::optional<GenericGridPlan> plan =
      isMatmul
          ? computeMatmulGridPlan(genericOp, perOperandTargetGrids, ttnnMode)
          : (inSpatialRegion
                 ? computePreserveCurrentGridPlan(
                       genericOp, perOperandTargetGrids, ttnnMode)
                 : computeGenericGridPlan(genericOp, perOperandTargetGrids,
                                          targetGridShape, ttnnMode,
                                          allowVirtualGrid));
  TT_assertv(plan, "GridSelection failed to produce a legal grid plan");

  result.normalizedOperandGrids = plan->normalizedOperandGrids;
  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    OperandGridInfo info;
    info.operand = operand;
    info.grid = plan->operandPlans[operandIndex].decision;
    result.operandInfos.push_back(std::move(info));
  }

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
    bool isOutputOperand = idx >= genericOp.getInputs().size();

    if (isMatmul) {
      if (view && !view.getReinterpretLayout()) {
        info.grid = plan->operandPlans[idx].decision;
      }
      continue;
    }

    if (!isOutputOperand && !isCompositeView &&
        (!view || view.getReinterpretLayout())) {
      continue;
    }
    info.grid = plan->operandPlans[idx].decision;
  }

  if (isMatmul) {
    for (OperandGridInfo &info : result.operandInfos) {
      auto viewLayout = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
      if (!viewLayout) {
        continue;
      }
      auto toLayoutOp = viewLayout.getInput().getDefiningOp<d2m::ToLayoutOp>();
      if (!toLayoutOp ||
          toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      auto inputType =
          mlir::cast<mlir::RankedTensorType>(viewLayout.getInput().getType());
      llvm::SmallVector<int64_t> inputPhysicalShape =
          utils::computePhysicalShape(viewLayout.getInput(),
                                      info.grid.targetGrid, ttnnMode);
      llvm::SmallVector<int64_t> inputOptimalGrid = utils::computeOptimalGrid(
          inputType, inputPhysicalShape, info.grid.targetGrid);
      std::optional<GridDecisionAndShape> sourcePlan =
          computeDecisionAndShapeForSelectedGrid(
              viewLayout.getInput(), inputOptimalGrid, info.grid.targetGrid,
              ttnnMode, /*useTargetGridForLayout=*/true);
      if (sourcePlan) {
        info.viewSourceGrid = sourcePlan->decision;
      }
    }
  } else {
    // If an operand is a view chain over a ToLayoutOp, compute the producer's
    // grid from the final operand grid. This keeps producer-side
    // materialization aligned with the consumer plan, including loop-space
    // normalization.
    for (OperandGridInfo &info : result.operandInfos) {
      if (auto toLayoutOp = getSourceToLayoutThroughViews(info.operand)) {
        if (toLayoutOp.getInput()
                .getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
          continue;
        }
        if (computeDecisionAndShapeForSelectedGrid(
                toLayoutOp.getResult(0), info.grid.selectedGrid,
                info.grid.targetGrid, ttnnMode,
                /*useTargetGridForLayout=*/!allowVirtualGrid && !ttnnMode)) {
          info.viewSourceGrid = info.grid;
          continue;
        }

        std::optional<GridDecisionAndShape> sourcePlan =
            computeBestSourceGridDecision(
                toLayoutOp.getResult(0), info.grid.targetGrid, ttnnMode,
                allowVirtualGrid, /*useTargetGridForLayout=*/false);
        if (sourcePlan) {
          info.viewSourceGrid = sourcePlan->decision;
        }
      }
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
    : deviceGridShape(deviceGridShape), ttnnMode(ttnnMode) {
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
