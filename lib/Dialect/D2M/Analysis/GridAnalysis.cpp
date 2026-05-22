// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/GridAnalysis.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::d2m {

bool GridAnalysis::isTTNNOperand(Value operand) {
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    operand = view.getInput();
  }
  return operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>() != nullptr;
}

static d2m::ToLayoutOp getToLayoutProducerBehindViews(Value operand) {
  bool sawView = false;
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    sawView = true;
    operand = view.getInput();
  }
  return sawView ? operand.getDefiningOp<d2m::ToLayoutOp>() : d2m::ToLayoutOp();
}

static llvm::SmallVector<int64_t> getTiledConsumerTileShape(Value value) {
  llvm::SmallVector<Value> worklist{value};

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
        worklist.push_back(toLayout.getResult(0));
        continue;
      }

      if (auto view = dyn_cast<d2m::ViewLayoutOp>(user)) {
        worklist.push_back(view.getResult());
        continue;
      }

      if (auto mask = dyn_cast<d2m::MaskOp>(user)) {
        worklist.push_back(mask.getResult());
        continue;
      }
    }
  }

  return {};
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

static int64_t getVolume(ArrayRef<int64_t> shape) {
  int64_t volume = 1;
  for (int64_t dim : shape) {
    volume *= dim;
  }
  return volume;
}

static int64_t computeCollapsedIntervalSize(ArrayRef<int64_t> logicalShape,
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

static llvm::SmallVector<int64_t> capCompositeScalarInputGrid(
    ttcore::MetalLayoutAttr materializedLayout, ArrayRef<int64_t> alignments,
    ArrayRef<int64_t> selectedGrid, ArrayRef<int64_t> targetGrid,
    int32_t concatDim, ArrayRef<int64_t> paddingShape) {
  llvm::SmallVector<int64_t> grid = llvm::to_vector(selectedGrid);
  auto intervals = materializedLayout.getNormalizedIntervals();
  const int64_t tensorGridRank = intervals.size() / 2;
  TT_assert(static_cast<int64_t>(grid.size()) == tensorGridRank);

  // The concat dimension is intentionally only padded to a tile. If the
  // composite output grid splits that small input tile further, a scalar DMA
  // kernel can end up attaching a full tile CB to a sub-tile L1 shard. Keep the
  // already-selected grid on every other axis, but cap this input's concat-axis
  // grid by the number of full tiles it can actually provide.
  for (int64_t intervalIdx = 0; intervalIdx < tensorGridRank; ++intervalIdx) {
    const int64_t intervalStart = intervals[intervalIdx * 2];
    const int64_t intervalEnd = intervals[intervalIdx * 2 + 1];
    if (concatDim < intervalStart || concatDim >= intervalEnd) {
      continue;
    }

    const int64_t logicalRank = materializedLayout.getLogicalShape().size();
    const int64_t tileHWIdx = concatDim - (logicalRank - 2);
    const bool isTileInterval = intervalIdx >= tensorGridRank - 2;
    const int64_t minShard =
        (isTileInterval && tileHWIdx >= 0 &&
         static_cast<size_t>(tileHWIdx) < paddingShape.size())
            ? paddingShape[tileHWIdx]
            : 1;
    const int64_t intervalSize =
        computeCollapsedIntervalSize(materializedLayout.getLogicalShape(),
                                     alignments, intervalStart, intervalEnd);
    const int64_t maxGridDim = std::max<int64_t>(1, intervalSize / minShard);
    const int64_t upperBound = std::min(grid[intervalIdx], maxGridDim);

    llvm::SmallVector<int64_t> candidateGrid = grid;
    for (int64_t candidate = upperBound; candidate > 0; --candidate) {
      if (intervalSize % candidate != 0 ||
          intervalSize / candidate < minShard) {
        continue;
      }

      candidateGrid[intervalIdx] = candidate;
      if (!utils::findLegalPhysicalGridForVolume(getVolume(candidateGrid),
                                                 targetGrid)
               .empty()) {
        grid = candidateGrid;
        break;
      }
    }
    break;
  }

  return grid;
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

  uint64_t numInputs = genericOp.getInputs().size();
  // Map: loopDim -> list of (operandIndex, operandDimIdx) pairs that reference
  // this loop dimension in their indexing maps.
  llvm::DenseMap<int64_t, llvm::SmallVector<std::pair<uint64_t, uint64_t>>>
      dimToInputOperandDims;

  auto indexingMaps = genericOp.getIndexingMapsValue();
  for (uint64_t operandIndex = 0; operandIndex < numInputs; ++operandIndex) {
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

  // For each loop dimension shared by multiple inputs, find the largest grid
  // factor that evenly divides all operands' physical shapes for that dim.
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

  // Compute grid dim constraints implied by the generic's outputs.
  auto outputIndexingMap =
      genericOp.getIndexingMapsValue()[genericOp.getOutputs()
                                           .getBeginOperandIndex()];
  auto outputShape =
      optimalOperandGrids[genericOp.getOutputs().getBeginOperandIndex()];
  std::optional<SmallVector<int64_t>> outputConstraints =
      d2m::utils::computeDimConstraints(
          llvm::ArrayRef<AffineMap>(outputIndexingMap),
          llvm::ArrayRef<SmallVector<int64_t>>(outputShape));

  // Apply output constraints to input operands, but only if the constraint
  // divides the input's physical shape for that dimension.
  if (outputConstraints) {
    for (auto [operandIndex, operand] :
         llvm::enumerate(genericOp.getInputsAndOutputsMutable())) {
      if (genericOp.isDpsInit(&operand)) {
        continue;
      }

      AffineMap indexingMap = genericOp.getIndexingMap(operandIndex);
      auto results = indexingMap.getResults();
      TT_assertv(results.size() == normalizedOperandGrids[operandIndex].size(),
                 "indexing map results size does not match normalized operand "
                 "grids size");

      for (auto [resultIdx, expr] : llvm::enumerate(results)) {
        auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }
        int64_t dimPos = dimExpr.getPosition();
        int64_t constraint = (*outputConstraints)[dimPos];
        if (constraint != 0) {
          int64_t physDim = physicalShapes[operandIndex][resultIdx];
          if (physDim % constraint == 0) {
            normalizedOperandGrids[operandIndex][resultIdx] = constraint;
          } else {
            // Output constraint doesn't divide this operand's physical dim;
            // find the largest factor that divides both.
            normalizedOperandGrids[operandIndex][resultIdx] =
                findLargestCommonFactor(constraint, {physDim, constraint});
          }
        }
      }
    }
  }

  return normalizedOperandGrids;
}

static llvm::SmallVector<CompositeInputGridInfo>
computeCompositeInputGridInfos(d2m::CompositeViewOp compositeView,
                               ArrayRef<int64_t> targetGrid,
                               ArrayRef<int64_t> selectedGrid, bool ttnnMode,
                               ArrayRef<int64_t> paddingTileShape = {}) {
  llvm::SmallVector<CompositeInputGridInfo> inputInfos;
  inputInfos.reserve(compositeView.getInputs().size());

  const int32_t concatDim = compositeView.getDim();
  auto outType =
      mlir::cast<RankedTensorType>(compositeView.getResult().getType());
  const bool isTiled = mlir::isa<ttcore::TileType>(outType.getElementType());

  SmallVector<int64_t> scalarCompositePaddingTileShape;
  ArrayRef<int64_t> effectivePaddingTileShape = paddingTileShape;
  if (!isTiled && effectivePaddingTileShape.empty()) {
    scalarCompositePaddingTileShape =
        llvm::to_vector(ttcore::TileType::getDefaultShape());
    effectivePaddingTileShape = scalarCompositePaddingTileShape;
  }

  RankedTensorType materializedOutType = utils::tensorWithOptimalGrid(
      outType, targetGrid, ttnnMode, selectedGrid, effectivePaddingTileShape);

  for (Value input : compositeView.getInputs()) {
    CompositeInputGridInfo info;
    info.input = input;

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    info.materializedType = inputType;

    auto inputLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
    if (!inputLayout) {
      inputInfos.push_back(std::move(info));
      continue;
    }

    if (isTiled || !effectivePaddingTileShape.empty()) {
      // Match the composite output's padding on non-concat dimensions, but
      // keep the concat dimension tile-only so individual inputs do not each
      // inflate their contribution. Scalar inputs reuse the already-selected
      // composite grid; reselecting here can pick a different legal grid for
      // the padded shape and break the one-grid contract for the concat.
      auto materializedOutLayout = mlir::cast<ttcore::MetalLayoutAttr>(
          materializedOutType.getEncoding());
      SmallVector<int64_t> inputAlignments(
          materializedOutLayout.getDimAlignments().begin(),
          materializedOutLayout.getDimAlignments().end());

      ArrayRef<int64_t> paddingShape = effectivePaddingTileShape;
      if (isTiled) {
        auto tileType =
            mlir::cast<ttcore::TileType>(inputType.getElementType());
        paddingShape = tileType.getShape();
      }
      int64_t logicalRank = inputLayout.getLogicalShape().size();
      const int64_t tileHWIdx = concatDim - (logicalRank - 2);
      inputAlignments[concatDim] =
          (tileHWIdx >= 0) ? paddingShape[tileHWIdx] : 1;

      auto materializedLayout = ttcore::MetalLayoutAttr::get(
          input.getContext(), inputLayout.getLogicalShape(),
          inputLayout.getMemorySpace(), inputLayout.getMemoryLayout(),
          inputLayout.getCollapsedIntervals(), inputAlignments);

      ArrayRef<int64_t> elementTileShape =
          isTiled ? paddingShape : ArrayRef<int64_t>();
      auto inputPhysShape =
          materializedLayout.getPhysicalShape(elementTileShape);
      info.selectedGrid =
          isTiled
              ? utils::computeOptimalGrid(inputType, inputPhysShape, targetGrid)
              : capCompositeScalarInputGrid(materializedLayout, inputAlignments,
                                            selectedGrid, targetGrid, concatDim,
                                            paddingShape);

      auto deviceShape = materializedLayout.getDeviceShape(info.selectedGrid,
                                                           elementTileShape);
      info.materializedType = RankedTensorType::get(
          deviceShape, inputType.getElementType(), materializedLayout);
    } else {
      auto inputPhysShape =
          utils::computePhysicalShape(input, targetGrid, ttnnMode);
      info.selectedGrid =
          utils::computeOptimalGrid(inputType, inputPhysShape, targetGrid);
      info.materializedType = utils::tensorWithOptimalGrid(
          inputType, targetGrid, ttnnMode, info.selectedGrid);
    }

    inputInfos.push_back(std::move(info));
  }

  return inputInfos;
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

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    TT_assertv(operandLayout,
               "GridSelection expects GenericOp inputs/outputs to have "
               "MetalLayoutAttr");

    ArrayRef<int64_t> targetGrid = perOperandTargetGrids[operandIndex];
    llvm::SmallVector<int64_t> physShape =
        utils::computePhysicalShape(operand, targetGrid, ttnnMode);
    physicalShapes.push_back(physShape);
    auto optimalGrid =
        utils::computeOptimalGrid(operandType, physShape, targetGrid);
    optimalOperandGrids.push_back(optimalGrid);

    OperandGridInfo info;
    info.operand = operand;
    info.selectedGrid = optimalGrid; // Will be updated after normalization.
    info.targetGrid = perOperandTargetGrids[operandIndex];

    unsigned outputBegin = genericOp.getOutputs().getBeginOperandIndex();
    if (operandIndex >= outputBegin &&
        !mlir::isa<ttcore::TileType>(operandType.getElementType())) {
      unsigned resultIndex = operandIndex - outputBegin;
      if (resultIndex < genericOp.getNumResults()) {
        info.paddingTileShape =
            getTiledConsumerTileShape(genericOp.getResult(resultIndex));
      }
    }
    if (!mlir::isa<ttcore::TileType>(operandType.getElementType()) &&
        info.paddingTileShape.empty() &&
        operand.getDefiningOp<d2m::CompositeViewOp>()) {
      info.paddingTileShape =
          llvm::to_vector(ttcore::TileType::getDefaultShape());
    }

    // If the operand is a view over a ToLayoutOp (not fronting a TTNN cast),
    // pre-compute the ToLayoutOp's own optimal grid so it can be updated
    // independently at apply time.
    if (auto toLayoutOp = getToLayoutProducerBehindViews(operand)) {
      if (!toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        Value toLayoutResult = toLayoutOp.getResult(0);
        auto inputType =
            mlir::cast<mlir::RankedTensorType>(toLayoutResult.getType());
        llvm::SmallVector<int64_t> inputPhysShape =
            utils::computePhysicalShape(toLayoutResult, targetGrid, ttnnMode);
        info.viewSourceGrid =
            utils::computeOptimalGrid(inputType, inputPhysShape, targetGrid);
      }
    }

    result.operandInfos.push_back(std::move(info));
  }

  llvm::SmallVector<int64_t> genericPaddingTileShape;
  for (const OperandGridInfo &info : result.operandInfos) {
    if (!info.paddingTileShape.empty()) {
      genericPaddingTileShape = info.paddingTileShape;
      break;
    }
  }
  if (!genericPaddingTileShape.empty()) {
    for (OperandGridInfo &info : result.operandInfos) {
      auto operandType = mlir::cast<RankedTensorType>(info.operand.getType());
      if (!mlir::isa<ttcore::TileType>(operandType.getElementType()) &&
          info.paddingTileShape.empty()) {
        info.paddingTileShape = genericPaddingTileShape;
      }
    }
  }

  // Normalize the operand grids for the generic operation.
  result.normalizedOperandGrids = normalizeOperandGridsForGeneric(
      genericOp, optimalOperandGrids, physicalShapes);

  // Propagate normalized grids back to non-reinterpret ViewLayout operands
  // only. Other operand kinds use their independently computed optimal
  // grids — normalization may produce grids that exceed what the individual
  // tensor can physically support.
  for (unsigned idx = 0; idx < result.operandInfos.size(); ++idx) {
    OperandGridInfo &info = result.operandInfos[idx];
    if (isTTNNOperand(info.operand)) {
      continue;
    }
    auto view = info.operand.getDefiningOp<d2m::ViewLayoutOp>();
    if (view && !view.getReinterpretLayout()) {
      info.selectedGrid = result.normalizedOperandGrids[idx];
    }
  }

  for (OperandGridInfo &info : result.operandInfos) {
    auto compositeView = info.operand.getDefiningOp<d2m::CompositeViewOp>();
    if (!compositeView) {
      continue;
    }
    info.compositeInputInfos = computeCompositeInputGridInfos(
        compositeView, info.targetGrid, info.selectedGrid, ttnnMode,
        info.paddingTileShape);
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
