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

static bool
isBetterLayoutGridCandidate(ArrayRef<int64_t> candidatePhysicalShape,
                            ArrayRef<int64_t> candidateLayoutGrid,
                            const GridDecisionAndShape &best) {
  uint64_t candidatePhysicalVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(candidatePhysicalShape));
  uint64_t bestPhysicalVolume =
      static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(best.physicalShape));
  if (candidatePhysicalVolume != bestPhysicalVolume) {
    return candidatePhysicalVolume < bestPhysicalVolume;
  }

  uint64_t candidateLayoutVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(candidateLayoutGrid));
  uint64_t bestLayoutVolume = static_cast<uint64_t>(
      ttmlir::utils::volume<int64_t>(best.decision.layoutGrid));
  if (candidateLayoutVolume != bestLayoutVolume) {
    return candidateLayoutVolume < bestLayoutVolume;
  }

  for (auto [candidateDim, bestDim] :
       llvm::zip_equal(candidateLayoutGrid, best.decision.layoutGrid)) {
    if (candidateDim != bestDim) {
      return candidateDim < bestDim;
    }
  }
  return false;
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

  if (useTargetGridForLayout || !baseDecision.isVirtual() || ttnnMode ||
      targetGrid.size() != 2) {
    return buildCandidate(std::move(baseDecision));
  }

  std::optional<GridDecisionAndShape> best = buildCandidate(baseDecision);
  for (int64_t rowFactor = 1; rowFactor <= targetGrid[0]; ++rowFactor) {
    for (int64_t colFactor = 1; colFactor <= targetGrid[1]; ++colFactor) {
      GridDecision candidateDecision = baseDecision;
      candidateDecision.layoutGrid = {rowFactor, colFactor};
      std::optional<GridDecisionAndShape> candidate =
          buildCandidate(std::move(candidateDecision));
      if (!candidate) {
        continue;
      }
      if (!best ||
          isBetterLayoutGridCandidate(candidate->physicalShape,
                                      candidate->decision.layoutGrid, *best)) {
        best = std::move(candidate);
      }
    }
  }

  return best;
}

static bool canMaterializeOperandGrid(Value operand, ArrayRef<int64_t> grid,
                                      ArrayRef<int64_t> targetGrid,
                                      bool ttnnMode,
                                      bool useTargetGridForLayout = false) {
  return computeDecisionAndShapeForSelectedGrid(
             operand, grid, targetGrid, ttnnMode, useTargetGridForLayout)
      .has_value();
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
  uint64_t estimatedBytes = std::numeric_limits<uint64_t>::max();
  bool fitsL1 = false;
};

static bool computePlanStats(GenericOp genericOp, GenericGridPlan &plan,
                             uint64_t usableL1Bytes) {
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
  plan.estimatedBytes = estimateGenericL1Bytes(
      genericOp, plan.normalizedOperandGrids, plan.physicalShapes);
  plan.fitsL1 = usableL1Bytes == 0 || plan.estimatedBytes <= usableL1Bytes;
  return true;
}

static std::optional<GenericGridPlan> computeGenericGridPlanFromCurrentTypes(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool allowVirtualGrid,
    bool useTargetGridForLayout, bool requireCurrentTypeReblockable,
    uint64_t usableL1Bytes) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> physicalShapes;
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  physicalShapes.reserve(genericOp.getInputsAndOutputs().size());
  optimalOperandGrids.reserve(genericOp.getInputsAndOutputs().size());

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    llvm::SmallVector<int64_t> physicalShape = utils::computePhysicalShape(
        operand, perOperandTargetGrids[operandIndex], ttnnMode);
    llvm::SmallVector<int64_t> selectedGrid = computeSelectedGrid(
        operand, physicalShape, perOperandTargetGrids[operandIndex],
        allowVirtualGrid);
    if (!gridDividesPhysicalShape(physicalShape, selectedGrid)) {
      return std::nullopt;
    }
    physicalShapes.push_back(std::move(physicalShape));
    optimalOperandGrids.push_back(std::move(selectedGrid));
  }

  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids =
      GridAnalysis::normalizeOperandGridsForGeneric(
          genericOp, optimalOperandGrids, physicalShapes, perOperandTargetGrids,
          targetGrid, ttnnMode, useTargetGridForLayout,
          requireCurrentTypeReblockable, usableL1Bytes);
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

  if (!computePlanStats(genericOp, plan, usableL1Bytes)) {
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
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids, bool ttnnMode,
    uint64_t usableL1Bytes) {
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

  if (!computePlanStats(genericOp, plan, usableL1Bytes)) {
    return std::nullopt;
  }
  return plan;
}

static std::optional<GenericGridPlan> computeGenericGridPlan(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool allowVirtualGrid,
    uint64_t usableL1Bytes) {
  bool useTargetGridForLayout = !allowVirtualGrid;
  bool requireCurrentTypeReblockable =
      genericHasFixedNonReinterpretViewOperand(genericOp);
  return computeGenericGridPlanFromCurrentTypes(
      genericOp, perOperandTargetGrids, targetGrid, ttnnMode, allowVirtualGrid,
      useTargetGridForLayout, requireCurrentTypeReblockable, usableL1Bytes);
}

llvm::SmallVector<llvm::SmallVector<int64_t>>
GridAnalysis::normalizeOperandGridsForGeneric(
    GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes,
    ArrayRef<llvm::SmallVector<int64_t>> perOperandTargetGrids,
    ArrayRef<int64_t> targetGrid, bool ttnnMode, bool useTargetGridForLayout,
    bool requireCurrentTypeReblockable, uint64_t usableL1Bytes) {
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

  auto localLoopShapesAreConsistent =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
        if (!genericOp.hasComputeOpsInRegion() || hasIndexedRowAccess) {
          return true;
        }
        for (unsigned loopDim = 0; loopDim < numLoopDims; ++loopDim) {
          std::optional<int64_t> expectedLocalDim;
          for (auto [operandIndex, operandDimIdx] :
               loopDimOperandDims[loopDim]) {
            int64_t gridDim = operandGrids[operandIndex][operandDimIdx];
            if (gridDim == 0 ||
                physicalShapes[operandIndex][operandDimIdx] % gridDim != 0) {
              return false;
            }
            int64_t localDim =
                physicalShapes[operandIndex][operandDimIdx] / gridDim;
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

  auto allOperandGridsAreLegal =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
        if (!localLoopShapesAreConsistent(operandGrids)) {
          return false;
        }
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
                                         perOperandTargetGrids[operandIndex],
                                         ttnnMode, useTargetGridForLayout)) {
            return false;
          }
          return !requireMatmulCompatibleGrid ||
                 isGridSupportedByMatmulKernel(
                     operandGrid, perOperandTargetGrids[operandIndex]);
        });
      };

  llvm::SmallVector<int64_t> bestLoopGrid(numLoopDims, 1);
  uint64_t bestLoopGridVolume = 0;
  uint64_t bestOutputGridVolume = 0;
  uint64_t bestEstimatedBytes = std::numeric_limits<uint64_t>::max();
  bool bestFitsL1 = false;
  bool foundLegalGrid = false;
  const unsigned outputOperandIndex = genericOp.getInputs().size();
  auto isBetterGridCandidate =
      [](uint64_t candidateEstimatedBytes, uint64_t candidateOutputGridVolume,
         uint64_t candidateLoopGridVolume, bool candidateFitsL1,
         uint64_t bestEstimatedBytes, uint64_t bestOutputGridVolume,
         uint64_t bestLoopGridVolume, bool bestFitsL1) {
        if (candidateFitsL1 != bestFitsL1) {
          return candidateFitsL1;
        }
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

          uint64_t candidateEstimatedBytes = estimateGenericL1Bytes(
              genericOp, candidateOperandGrids, physicalShapes);
          bool candidateFitsL1 =
              usableL1Bytes == 0 || candidateEstimatedBytes <= usableL1Bytes;
          uint64_t candidateOutputGridVolume =
              static_cast<uint64_t>(ttmlir::utils::volume<int64_t>(
                  candidateOperandGrids[outputOperandIndex]));

          if (!foundLegalGrid ||
              isBetterGridCandidate(
                  candidateEstimatedBytes, candidateOutputGridVolume,
                  volumeSoFar, candidateFitsL1, bestEstimatedBytes,
                  bestOutputGridVolume, bestLoopGridVolume, bestFitsL1)) {
            bestLoopGrid = candidateLoopGrid;
            bestLoopGridVolume = volumeSoFar;
            bestOutputGridVolume = candidateOutputGridVolume;
            bestEstimatedBytes = candidateEstimatedBytes;
            bestFitsL1 = candidateFitsL1;
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
    uint64_t bestEmbeddingEstimatedBytes =
        estimateGenericL1Bytes(genericOp, bestEmbeddingGrids, physicalShapes);
    uint64_t bestEmbeddingOutputGridVolume = static_cast<uint64_t>(
        ttmlir::utils::volume<int64_t>(bestEmbeddingGrids[outputOperandIndex]));
    bool bestEmbeddingFitsL1 =
        usableL1Bytes == 0 || bestEmbeddingEstimatedBytes <= usableL1Bytes;
    uint64_t bestEmbeddingLoopGridVolume =
        computeLoopGridVolume(genericOp, bestEmbeddingGrids);

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
      uint64_t candidateLoopGridVolume =
          computeLoopGridVolume(genericOp, candidateOperandGrids);
      if (isBetterGridCandidate(
              candidateEstimatedBytes, candidateOutputGridVolume,
              candidateLoopGridVolume, candidateFitsL1,
              bestEmbeddingEstimatedBytes, bestEmbeddingOutputGridVolume,
              bestEmbeddingLoopGridVolume, bestEmbeddingFitsL1)) {
        bestEmbeddingGrids = candidateOperandGrids;
        bestEmbeddingEstimatedBytes = candidateEstimatedBytes;
        bestEmbeddingOutputGridVolume = candidateOutputGridVolume;
        bestEmbeddingLoopGridVolume = candidateLoopGridVolume;
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

  bool isMatmul = isMatmulGeneric(genericOp);
  // TTNN conversion can round-trip legacy height/width sharding, but not
  // arbitrary D2M virtual grids. Keep new generic virtual grids on the TTMetal
  // path until TTNN has a representation for them.
  bool allowVirtualGrid = !isMatmul && !ttnnMode;
  for (Value operand : genericOp.getInputsAndOutputs()) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    TT_assertv(operandLayout,
               "GridSelection expects GenericOp inputs/outputs to have "
               "MetalLayoutAttr");
  }

  std::optional<GenericGridPlan> plan =
      isMatmul ? computeMatmulGridPlan(genericOp, perOperandTargetGrids,
                                       ttnnMode, usableL1Bytes)
               : computeGenericGridPlan(genericOp, perOperandTargetGrids,
                                        targetGridShape, ttnnMode,
                                        allowVirtualGrid, usableL1Bytes);
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
        std::optional<GridDecisionAndShape> sourcePlan =
            computeBestSourceGridDecision(
                toLayoutOp.getResult(0), info.grid.targetGrid, ttnnMode,
                allowVirtualGrid, /*useTargetGridForLayout=*/false);
        if (sourcePlan && allowVirtualGrid &&
            ttmlir::utils::volume<int64_t>(sourcePlan->decision.selectedGrid) >
                ttmlir::utils::volume<int64_t>(info.grid.selectedGrid)) {
          info.viewSourceGrid = sourcePlan->decision;
          continue;
        }
        if (canMaterializeOperandGrid(
                toLayoutOp.getResult(0), info.grid.selectedGrid,
                info.grid.targetGrid, ttnnMode,
                /*useTargetGridForLayout=*/!allowVirtualGrid)) {
          info.viewSourceGrid = info.grid;
          continue;
        }
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
