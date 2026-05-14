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

// Matmul kernels require a physical (non-virtual) grid or a 1D virtual grid
// along the logical height/width dimensions.
static bool isGridSupportedByMatmulKernel(ArrayRef<int64_t> grid,
                                          ArrayRef<int64_t> targetGrid) {
  return !ttmlir::d2m::utils::grids::requiresVirtualGrid(grid, targetGrid) ||
         isTrailingDimOnly1DVirtualGrid(grid, targetGrid);
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

static llvm::SmallVector<int64_t> computeCurrentPhysicalShape(Value operand) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
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
    ArrayRef<int64_t> targetGrid, bool requireCurrentTypeReblockable) {
  if (optimalOperandGrids.empty()) {
    return {};
  }

  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());
  TT_assert(physicalShapes.size() == optimalOperandGrids.size());

  if (genericOp.isDMAOnlyForm()) {
    return llvm::SmallVector<llvm::SmallVector<int64_t>>(optimalOperandGrids);
  }

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

    desiredLoopGrid[loopDim] =
        findLargestCommonFactor(maxFactor, physDimsForLoop);
  }

  auto projectLoopGridToOperands = [&](ArrayRef<int64_t> loopGrid) {
    llvm::SmallVector<llvm::SmallVector<int64_t>> projectedOperandGrids(
        optimalOperandGrids.begin(), optimalOperandGrids.end());
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

  auto allOperandGridsArePlaceable =
      [&](ArrayRef<llvm::SmallVector<int64_t>> operandGrids) {
        return llvm::all_of(llvm::enumerate(operandGrids), [&](auto indexed) {
          auto [operandIndex, operandGrid] = indexed;
          if (requireCurrentTypeReblockable &&
              !canReblockCurrentTypeToGrid(operands[operandIndex],
                                           operandGrid)) {
            return false;
          }
          if (!hasLegalPhysicalPlacement(operandGrid, targetGrid)) {
            return false;
          }
          return !requireMatmulCompatibleGrid ||
                 isGridSupportedByMatmulKernel(operandGrid, targetGrid);
        });
      };

  llvm::SmallVector<int64_t> bestLoopGrid(numLoopDims, 1);
  int64_t bestLoopGridVolume = 1;
  llvm::SmallVector<int64_t> candidateLoopGrid(numLoopDims, 1);
  llvm::SmallVector<llvm::SmallVector<int64_t>> loopFactorChoices;
  loopFactorChoices.reserve(numLoopDims);
  for (int64_t desiredFactor : desiredLoopGrid) {
    loopFactorChoices.push_back(ttmlir::utils::getFactors(desiredFactor));
  }

  int64_t targetGridVolume = ttmlir::utils::volume(targetGrid);
  std::function<void(unsigned, int64_t)> searchLoopGrids =
      [&](unsigned loopDim, int64_t volumeSoFar) {
        if (loopDim == numLoopDims) {
          if (!hasLegalPhysicalPlacement(candidateLoopGrid, targetGrid)) {
            return;
          }
          auto candidateOperandGrids =
              projectLoopGridToOperands(candidateLoopGrid);
          if (!allOperandGridsArePlaceable(candidateOperandGrids)) {
            return;
          }
          if (volumeSoFar > bestLoopGridVolume) {
            bestLoopGrid = candidateLoopGrid;
            bestLoopGridVolume = volumeSoFar;
          }
          return;
        }

        for (int64_t factor : llvm::reverse(loopFactorChoices[loopDim])) {
          int64_t candidateVolume = volumeSoFar * factor;
          if (candidateVolume > targetGridVolume) {
            continue;
          }
          candidateLoopGrid[loopDim] = factor;
          searchLoopGrids(loopDim + 1, candidateVolume);
        }
        candidateLoopGrid[loopDim] = 1;
      };
  searchLoopGrids(/*loopDim=*/0, /*volumeSoFar=*/1);

  auto normalizedOperandGrids = projectLoopGridToOperands(bestLoopGrid);
  if (allOperandGridsArePlaceable(normalizedOperandGrids)) {
    return normalizedOperandGrids;
  }

  // Keep this as an invariant rather than a fallback: normalization is
  // responsible for constructing only placeable grids.
  for (auto [operandIndex, operandGrid] :
       llvm::enumerate(normalizedOperandGrids)) {
    TT_assertv(hasLegalPhysicalPlacement(operandGrid, targetGrid),
               "Grid normalization produced unplaceable operand grid {} for "
               "operand {} within target grid {}",
               ttmlir::utils::formatIterable(operandGrid, "x"), operandIndex,
               ttmlir::utils::formatIterable(targetGrid, "x"));
  }

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
    GridDecisionAndShape decisionAndShape = computeGridDecision(
        operand, targetGrid, targetGrid, ttnnMode, allowVirtualGrid);
    physicalShapes.push_back(decisionAndShape.physicalShape);
    optimalOperandGrids.push_back(decisionAndShape.decision.selectedGrid);

    OperandGridInfo info;
    info.operand = operand;
    // Will be updated after normalization for view operands.
    info.grid = std::move(decisionAndShape.decision);

    result.operandInfos.push_back(std::move(info));
  }

  result.normalizedOperandGrids = normalizeOperandGridsForGeneric(
      genericOp, optimalOperandGrids, physicalShapes, targetGridShape);

  // Propagate normalized grids back to operands so producer rewrites and the
  // recreated generic share one final grid decision.
  for (unsigned idx = 0; idx < result.operandInfos.size(); ++idx) {
    OperandGridInfo &info = result.operandInfos[idx];
    if (isTTNNOperand(info.operand)) {
      continue;
    }
    auto view = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
    if (view && view.getReinterpretLayout()) {
      continue;
    }
    info.grid = utils::makeGridDecision(result.normalizedOperandGrids[idx],
                                        perOperandTargetGrids[idx]);
    if (!allowVirtualGrid) {
      info.grid.layoutGrid = computeTrailingDimLayoutGrid(
          info.grid.selectedGrid, perOperandTargetGrids[idx]);
    }
  }

  // If an operand is a view chain over a ToLayoutOp, compute the producer's
  // grid decision from the consuming operand's layout grid. That keeps source
  // tilization aligned with the virtual placement selected for the generic.
  for (OperandGridInfo &info : result.operandInfos) {
    if (auto toLayoutOp = getSourceToLayoutThroughViews(info.operand)) {
      if (toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      info.viewSourceGrid =
          computeGridDecision(toLayoutOp.getResult(0), info.grid.layoutGrid,
                              info.grid.targetGrid, ttnnMode, allowVirtualGrid)
              .decision;
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
