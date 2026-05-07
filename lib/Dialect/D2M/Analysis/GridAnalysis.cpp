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

static GridDecision makeGridDecision(ArrayRef<int64_t> selectedGrid,
                                     ArrayRef<int64_t> targetGrid) {
  GridDecision decision;
  decision.selectedGrid = llvm::SmallVector<int64_t>(selectedGrid);
  decision.targetGrid = llvm::SmallVector<int64_t>(targetGrid);

  if (!ttmlir::d2m::utils::grids::requiresVirtualGrid(selectedGrid,
                                                      targetGrid)) {
    decision.physicalGrid = llvm::SmallVector<int64_t>(selectedGrid);
    decision.layoutGrid = llvm::SmallVector<int64_t>(targetGrid);
    return decision;
  }

  auto physicalGrid = utils::findLegalPhysicalGridForVolume(
      ttmlir::utils::volume<int64_t>(selectedGrid), targetGrid);
  TT_assertv(!physicalGrid.empty(),
             "Unable to find physical grid for virtual grid {} within {}",
             ttmlir::utils::formatIterable(selectedGrid, "x"),
             ttmlir::utils::formatIterable(targetGrid, "x"));
  decision.physicalGrid = physicalGrid;
  decision.layoutGrid = physicalGrid;
  return decision;
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

struct GridDecisionAndShape {
  GridDecision decision;
  llvm::SmallVector<int64_t> physicalShape;
};

static llvm::SmallVector<int64_t>
computeSelectedGrid(mlir::Value operand, ArrayRef<int64_t> physicalShape,
                    ArrayRef<int64_t> targetGrid, bool allowVirtualGrid) {
  if (!allowVirtualGrid) {
    return utils::computeOptimalBlockShardedGrid(physicalShape, targetGrid);
  }

  auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  return utils::computeOptimalGrid(operandType, physicalShape, targetGrid);
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
  result.decision = makeGridDecision(selectedGrid, targetGrid);
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
    result.decision =
        makeGridDecision(computeSelectedGrid(operand, result.physicalShape,
                                             targetGrid, allowVirtualGrid),
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
    ArrayRef<llvm::SmallVector<int64_t>> physicalShapes) {
  if (optimalOperandGrids.empty()) {
    return {};
  }

  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());
  TT_assert(physicalShapes.size() == optimalOperandGrids.size());

  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids(
      optimalOperandGrids.begin(), optimalOperandGrids.end());

  // Map: loopDim -> list of (operandIndex, operandDimIdx) pairs that reference
  // this loop dimension in their indexing maps.
  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<uint64_t, uint64_t>>>
      dimToInputOperandDims;

  auto indexingMaps = genericOp.getIndexingMapsValue();
  for (uint64_t operandIndex = 0; operandIndex < indexingMaps.size();
       ++operandIndex) {
    AffineMap operandIndexingMap = indexingMaps[operandIndex];
    auto results = operandIndexingMap.getResults();
    for (auto [operandDimIdx, expr] : llvm::enumerate(results)) {
      auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr) {
        continue;
      }
      int64_t loopDim = dimExpr.getPosition();
      dimToInputOperandDims[loopDim].push_back(
          std::make_pair(operandIndex, static_cast<uint64_t>(operandDimIdx)));
    }
  }

  // For each loop dimension shared by multiple operands, find the largest grid
  // factor that evenly divides every participating physical shape. Generic
  // recreation derives operand grids from loop factors, so outputs and inputs
  // must agree on any loop dimension they both reference.
  for (auto &it : dimToInputOperandDims) {
    auto &entries = it.second;
    if (entries.size() < 2) {
      continue;
    }

    int64_t maxFactor = 0;
    SmallVector<int64_t> physDimsForLoop;
    for (auto [operandIndex, operandDimIdx] : entries) {
      maxFactor = std::max(maxFactor,
                           normalizedOperandGrids[operandIndex][operandDimIdx]);
      physDimsForLoop.push_back(physicalShapes[operandIndex][operandDimIdx]);
    }

    int64_t commonFactor = findLargestCommonFactor(maxFactor, physDimsForLoop);

    for (auto [operandIndex, operandDimIdx] : entries) {
      normalizedOperandGrids[operandIndex][operandDimIdx] = commonFactor;
    }
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

  // Normalize the operand grids for the generic operation.
  result.normalizedOperandGrids = normalizeOperandGridsForGeneric(
      genericOp, optimalOperandGrids, physicalShapes);

  // Propagate normalized grids back to operands so producer rewrites and the
  // recreated generic agree on grid shape and virtual-grid metadata.
  for (unsigned idx = 0; idx < result.operandInfos.size(); ++idx) {
    OperandGridInfo &info = result.operandInfos[idx];
    if (isTTNNOperand(info.operand)) {
      continue;
    }
    auto view = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
    if (view && view.getReinterpretLayout()) {
      continue;
    }
    info.grid = makeGridDecision(result.normalizedOperandGrids[idx],
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
