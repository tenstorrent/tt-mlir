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

#include <algorithm>

namespace mlir::tt::d2m {

bool GridAnalysis::isTTNNOperand(Value operand) {
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    operand = view.getInput();
  }
  return operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>() != nullptr;
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

// Locate the collapsed interval that owns a given logical-shape dim. Exactly
// one interval covers it for a valid layout; returns -1 otherwise.
static int64_t findIntervalIdxForDim(ttcore::MetalLayoutAttr layout,
                                     int64_t dim) {
  auto intervals = layout.getNormalizedIntervals();
  const int64_t tensorGridRank = intervals.size() / 2;
  for (int64_t i = 0; i < tensorGridRank; ++i) {
    if (dim >= intervals[i * 2] && dim < intervals[i * 2 + 1]) {
      return i;
    }
  }
  return -1;
}

// Compute the per-input concat-axis grid for a scalar composite_view input.
//
// Inputs share the composite operand's grid on every axis except the concat
// axis (alignments must match for layout compatibility). On the concat axis,
// each input only owns `inputTileCount` tiles, so its grid there is clamped
// to that count — walking downward for the largest divisor of the input's
// concat-axis interval that still yields a worker-grid-legal volume.
static int64_t computeScalarConcatGridDim(int64_t compositeConcatGridDim,
                                          int64_t inputTileCount,
                                          int64_t intervalSize,
                                          ArrayRef<int64_t> candidateGrid,
                                          int64_t concatIntervalIdx,
                                          ArrayRef<int64_t> targetGrid) {
  const int64_t upperBound = std::min(compositeConcatGridDim, inputTileCount);
  llvm::SmallVector<int64_t> trial(candidateGrid.begin(), candidateGrid.end());
  for (int64_t v = upperBound; v > 0; --v) {
    if (intervalSize % v != 0) {
      continue;
    }
    trial[concatIntervalIdx] = v;
    if (!utils::findLegalPhysicalGridForVolume(ttmlir::utils::volume(trial),
                                               targetGrid)
             .empty()) {
      return v;
    }
  }
  return 1;
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

static int64_t getTileSizeForLogicalDim(int64_t logicalRank, int64_t dim,
                                        ArrayRef<int64_t> tileShape) {
  const int64_t tileIdx = dim - (logicalRank - 2);
  if (tileIdx < 0 || static_cast<size_t>(tileIdx) >= tileShape.size()) {
    return 1;
  }
  return tileShape[tileIdx];
}

static ArrayRef<int64_t>
getEffectiveCompositePaddingTileShape(bool isTiled,
                                      ArrayRef<int64_t> paddingTileShape,
                                      SmallVector<int64_t> &storage) {
  if (isTiled || !paddingTileShape.empty()) {
    return paddingTileShape;
  }

  storage = llvm::to_vector(ttcore::TileType::getDefaultShape());
  return storage;
}

static SmallVector<int64_t>
deriveCompositeInputAlignments(ttcore::MetalLayoutAttr inputLayout,
                               ttcore::MetalLayoutAttr materializedOutLayout,
                               int64_t concatDim,
                               ArrayRef<int64_t> paddingShape) {
  SmallVector<int64_t> inputAlignments(
      materializedOutLayout.getDimAlignments().begin(),
      materializedOutLayout.getDimAlignments().end());
  inputAlignments[concatDim] = getTileSizeForLogicalDim(
      inputLayout.getLogicalShape().size(), concatDim, paddingShape);
  return inputAlignments;
}

static ttcore::MetalLayoutAttr
makeCompositeInputLayout(Value input, RankedTensorType materializedOutType,
                         int64_t concatDim, ArrayRef<int64_t> paddingShape) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto inputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
  auto materializedOutLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(materializedOutType.getEncoding());
  SmallVector<int64_t> inputAlignments = deriveCompositeInputAlignments(
      inputLayout, materializedOutLayout, concatDim, paddingShape);

  return ttcore::MetalLayoutAttr::get(
      input.getContext(), inputLayout.getLogicalShape(),
      inputLayout.getMemorySpace(), inputLayout.getMemoryLayout(),
      inputLayout.getCollapsedIntervals(), inputAlignments);
}

static SmallVector<int64_t> computeScalarCompositeInputGrid(
    ttcore::MetalLayoutAttr materializedLayout, int64_t concatDim,
    ArrayRef<int64_t> selectedGrid, ArrayRef<int64_t> paddingShape,
    ArrayRef<int64_t> targetGrid) {
  SmallVector<int64_t> inputGrid(selectedGrid.begin(), selectedGrid.end());
  int64_t concatIntervalIdx =
      findIntervalIdxForDim(materializedLayout, concatDim);
  if (concatIntervalIdx < 0) {
    return inputGrid;
  }

  ArrayRef<int64_t> logicalShape = materializedLayout.getLogicalShape();
  const int64_t tileSize =
      getTileSizeForLogicalDim(logicalShape.size(), concatDim, paddingShape);
  const int64_t inputTileCount =
      ttmlir::utils::alignUp(logicalShape[concatDim], tileSize) / tileSize;

  auto intervals = materializedLayout.getNormalizedIntervals();
  SmallVector<int64_t> alignments(materializedLayout.getDimAlignments().begin(),
                                  materializedLayout.getDimAlignments().end());
  const int64_t intervalSize = utils::computeCollapsedIntervalSize(
      logicalShape, alignments, intervals[concatIntervalIdx * 2],
      intervals[concatIntervalIdx * 2 + 1]);

  inputGrid[concatIntervalIdx] = computeScalarConcatGridDim(
      selectedGrid[concatIntervalIdx], inputTileCount, intervalSize, inputGrid,
      concatIntervalIdx, targetGrid);
  return inputGrid;
}

static SmallVector<int64_t> computeCompositeInputSelectedGrid(
    RankedTensorType inputType, ttcore::MetalLayoutAttr materializedLayout,
    bool isTiled, int64_t concatDim, ArrayRef<int64_t> selectedGrid,
    ArrayRef<int64_t> paddingShape, ArrayRef<int64_t> targetGrid) {
  if (isTiled) {
    SmallVector<int64_t> inputPhysShape =
        materializedLayout.getPhysicalShape(paddingShape);
    return utils::computeOptimalGrid(inputType, inputPhysShape, targetGrid);
  }

  return computeScalarCompositeInputGrid(
      materializedLayout, concatDim, selectedGrid, paddingShape, targetGrid);
}

static CompositeInputGridInfo computePaddedCompositeInputGridInfo(
    Value input, RankedTensorType materializedOutType,
    ArrayRef<int64_t> targetGrid, ArrayRef<int64_t> selectedGrid, bool isTiled,
    int64_t concatDim, ArrayRef<int64_t> paddingShape) {
  CompositeInputGridInfo info;
  info.input = input;

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  ttcore::MetalLayoutAttr materializedLayout = makeCompositeInputLayout(
      input, materializedOutType, concatDim, paddingShape);

  info.selectedGrid = computeCompositeInputSelectedGrid(
      inputType, materializedLayout, isTiled, concatDim, selectedGrid,
      paddingShape, targetGrid);

  ArrayRef<int64_t> elementTileShape =
      isTiled ? paddingShape : ArrayRef<int64_t>();
  SmallVector<int64_t> deviceShape =
      materializedLayout.getDeviceShape(info.selectedGrid, elementTileShape);
  info.materializedType = RankedTensorType::get(
      deviceShape, inputType.getElementType(), materializedLayout);
  return info;
}

static CompositeInputGridInfo computeCompositeInputGridInfo(
    Value input, RankedTensorType materializedOutType,
    ArrayRef<int64_t> targetGrid, ArrayRef<int64_t> selectedGrid, bool isTiled,
    int64_t concatDim, ArrayRef<int64_t> effectivePaddingShape) {
  CompositeInputGridInfo info;
  info.input = input;
  info.materializedType = mlir::cast<RankedTensorType>(input.getType());

  if (!mlir::isa<ttcore::MetalLayoutAttr>(
          info.materializedType.getEncoding())) {
    return info;
  }

  if (isTiled) {
    auto tileType =
        mlir::cast<ttcore::TileType>(info.materializedType.getElementType());
    SmallVector<int64_t> tileShape = llvm::to_vector(tileType.getShape());
    return computePaddedCompositeInputGridInfo(input, materializedOutType,
                                               targetGrid, selectedGrid,
                                               isTiled, concatDim, tileShape);
  }

  TT_assert(!effectivePaddingShape.empty());
  return computePaddedCompositeInputGridInfo(input, materializedOutType,
                                             targetGrid, selectedGrid, isTiled,
                                             concatDim, effectivePaddingShape);
}

static llvm::SmallVector<CompositeInputGridInfo>
computeCompositeInputGridInfos(d2m::CompositeViewOp compositeView,
                               ArrayRef<int64_t> targetGrid,
                               ArrayRef<int64_t> selectedGrid, bool ttnnMode,
                               ArrayRef<int64_t> paddingTileShape = {}) {
  auto outType =
      mlir::cast<RankedTensorType>(compositeView.getResult().getType());
  const bool isTiled = mlir::isa<ttcore::TileType>(outType.getElementType());

  SmallVector<int64_t> scalarPaddingStorage;
  ArrayRef<int64_t> effectivePaddingShape =
      getEffectiveCompositePaddingTileShape(isTiled, paddingTileShape,
                                            scalarPaddingStorage);
  RankedTensorType materializedOutType = utils::tensorWithOptimalGrid(
      outType, ttnnMode, selectedGrid, effectivePaddingShape);

  llvm::SmallVector<CompositeInputGridInfo> inputInfos;
  inputInfos.reserve(compositeView.getInputs().size());
  for (Value input : compositeView.getInputs()) {
    inputInfos.push_back(computeCompositeInputGridInfo(
        input, materializedOutType, targetGrid, selectedGrid, isTiled,
        compositeView.getDim(), effectivePaddingShape));
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
        info.paddingTileShape = utils::findDownstreamTiledToLayoutTileShape(
            genericOp.getResult(resultIndex));
      }
    }

    // If the operand is a view over a ToLayoutOp (not fronting a TTNN cast),
    // pre-compute the ToLayoutOp's own optimal grid so it can be updated
    // independently at apply time.
    if (auto toLayoutOp = utils::getToLayoutProducerBehindViews(operand)) {
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

  // Pick the padding tile shape from the output-derived consumer info first
  // (set by the per-operand loop above) so it wins over any defaulted shape.
  // If no output operand has a downstream tiled consumer but the generic does
  // feed a composite-view-style scalar pipeline, fall back to the default tile
  // shape so the eventual tile bridge has a consistent padding contract.
  llvm::SmallVector<int64_t> genericPaddingTileShape;
  for (const OperandGridInfo &info : result.operandInfos) {
    if (!info.paddingTileShape.empty()) {
      genericPaddingTileShape = info.paddingTileShape;
      break;
    }
  }
  if (genericPaddingTileShape.empty()) {
    for (const OperandGridInfo &info : result.operandInfos) {
      auto operandType = mlir::cast<RankedTensorType>(info.operand.getType());
      if (!mlir::isa<ttcore::TileType>(operandType.getElementType()) &&
          info.operand.getDefiningOp<d2m::CompositeViewOp>()) {
        genericPaddingTileShape =
            llvm::to_vector(ttcore::TileType::getDefaultShape());
        break;
      }
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

  // Must run after normalization above has finalized `info.selectedGrid`:
  // composite input grids are derived from the parent operand's *final*
  // selected grid, not its pre-normalization optimal grid.
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
