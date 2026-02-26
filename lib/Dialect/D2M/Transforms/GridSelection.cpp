// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/AffineMapAnalysis.h"
#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir::tt::d2m {

//--------------------------------------------------------
// Virtual Grid
//--------------------------------------------------------

static std::pair<unsigned, double>
findMaxDimAndAspectRatio(ArrayRef<int64_t> physicalShape) {

  // Find max aspect ratio between any dim and the other dims combined.
  double aspectRatio = 1.0;
  unsigned maxDimIndex = 0;
  for (size_t i = 0; i < physicalShape.size(); ++i) {
    double ratio = physicalShape[i];
    for (size_t j = 0; j < physicalShape.size(); ++j) {
      if (i == j) {
        continue;
      }
      ratio /= physicalShape[j];
    }
    if (ratio > aspectRatio) {
      aspectRatio = ratio;
      maxDimIndex = i;
    }
  }
  return {maxDimIndex, aspectRatio};
}

static llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetSquareGridShape);

static llvm::SmallVector<int64_t>
computeOptimalVirtualGrid(ArrayRef<int64_t> physicalShape,
                          ArrayRef<int64_t> targetSquareGridShape) {

  int64_t targetGridVolume = ttmlir::utils::volume(targetSquareGridShape);
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
        auto physGrid = utils::findLegalPhysicalGridForVolume(
            gridVolume, targetSquareGridShape);
        if (!physGrid.empty()) {

          bestGrid = grid;
          bestGridVolume = ttmlir::utils::volume<int64_t>(bestGrid);
        }
      }
    }
    return bestGrid;
  }

  // If not ND sharded, compute grid for 2D height or width sharding (Nx1, 1xN).
  auto [shardedDimIndex, aspectRatio] = findMaxDimAndAspectRatio(physicalShape);

  // Find the largest factor of the sharded dimension that fits within the
  // target grid volume.
  int64_t bestFactor = 0;
  const auto factors =
      ttmlir::utils::getFactors(physicalShape[shardedDimIndex]);
  for (int64_t factor : llvm::reverse(factors)) {
    if (factor <= targetGridVolume) {
      auto physGrid =
          utils::findLegalPhysicalGridForVolume(factor, targetSquareGridShape);
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

//--------------------------------------------------------
// Virtual Grid END
//--------------------------------------------------------

// ----------------------------------------------------------------------------
// Grid optimization utilities
// ----------------------------------------------------------------------------

// Compute physical shape for a MetalLayoutAttr by first computing grid-aware
// dimension alignments and then deriving the physical shape (always
// tile-aligned).
static llvm::SmallVector<int64_t> computePhysicalShape(
    ttcore::MetalLayoutAttr layout, mlir::RankedTensorType tensorType,
    ArrayRef<int64_t> targetSquareGridShape, OpBuilder &builder) {
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
          layout.getLogicalShape(), targetSquareGridShape,
          layout.getNormalizedIntervals());

  auto tempLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), layout.getLogicalShape(), layout.getOobVal(),
      layout.getMemorySpace(), layout.getMemoryLayout(),
      layout.getCollapsedIntervals(), alignments);

  return tempLayout.getPhysicalShape(
      llvm::ArrayRef(tileShape.data(), tileShape.size()));
}

// Compute optimal grid shape for a given physical shape and target grid by
// finding the largest grid dimensions that evenly divide the physical shape.
// This ensures maximum utilization of available worker cores while maintaining
// even distribution of work.
static llvm::SmallVector<int64_t>
computeOptimalBlockShardedGrid(ArrayRef<int64_t> physicalShape,
                               ArrayRef<int64_t> targetSquareGridShape) {
  llvm::SmallVector<int64_t> grid(physicalShape.size(), 1);

  TT_assert(targetSquareGridShape.size() == 2u);
  TT_assert(physicalShape.size() >= targetSquareGridShape.size());

  // For tensors with rank > 2, only shard across the last two dimensions
  // (which correspond to the 2D worker grid).
  const size_t dimOffset = physicalShape.size() - targetSquareGridShape.size();

  for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
    const int64_t dim = physicalShape[dimOffset + i];
    TT_assert(dim > 0);
    // Search downward from the target grid size to find the largest divisor.
    for (int64_t g = targetSquareGridShape[i]; g > 0; g--) {
      if (dim % g == 0) {
        grid[dimOffset + i] = g;
        break;
      }
    }
  }

  TT_assert(grid.size() == physicalShape.size());
  return grid;
}

// The following is a simple heuristic that determines (A) if a tensor _can_
// be implemented as a virtual grid and (B) if it makes sense to do so based
// on low grid utilization with regular block sharding.
static bool
shouldImplementAsVirtualGrid(RankedTensorType tensorType,
                             ArrayRef<int64_t> physicalShape,
                             ArrayRef<int64_t> targetSquareGridShape) {

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

  auto regularShardedGridVolume = ttmlir::utils::volume<int64_t>(
      computeOptimalBlockShardedGrid(physicalShape, targetSquareGridShape));
  int64_t targetGridVolume =
      ttmlir::utils::volume<int64_t>(targetSquareGridShape);
  bool lowGridUtilization = regularShardedGridVolume < 0.5 * targetGridVolume;
  return lowGridUtilization;
}

static llvm::SmallVector<int64_t>
computeOptimalGrid(mlir::RankedTensorType tensorType,
                   ArrayRef<int64_t> physicalShape,
                   ArrayRef<int64_t> targetSquareGridShape) {
  if (shouldImplementAsVirtualGrid(tensorType, physicalShape,
                                   targetSquareGridShape)) {
    auto virtualGrid =
        computeOptimalVirtualGrid(physicalShape, targetSquareGridShape);
    if (!virtualGrid.empty()) {
      return virtualGrid;
    }
  }
  return computeOptimalBlockShardedGrid(physicalShape, targetSquareGridShape);
}

static ttcore::MetalLayoutAttr
layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                      ArrayRef<int64_t> targetGridShape,
                      ArrayRef<int64_t> targetSquareGridShape,
                      ArrayRef<int64_t> optimalGrid, OpBuilder &builder) {
  auto collapsedIntervals = oldLayout.getCollapsedIntervals();

  llvm::SmallVector<int64_t> newDimAlignments =
      ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
          oldLayout.getLogicalShape(), targetSquareGridShape,
          oldLayout.getNormalizedIntervals());

  return ttcore::MetalLayoutAttr::get(
      builder.getContext(), oldLayout.getLogicalShape(), oldLayout.getOobVal(),
      oldLayout.getMemorySpace(), oldLayout.getMemoryLayout(),
      collapsedIntervals, newDimAlignments);
}

static RankedTensorType
tensorWithOptimalGrid(RankedTensorType oldTensor,
                      ArrayRef<int64_t> targetGridShape,
                      ArrayRef<int64_t> targetSquareGridShape,
                      ArrayRef<int64_t> optimalGrid, OpBuilder &builder) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());

  llvm::SmallVector<int64_t> tileShape;
  Type elementType = oldTensor.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
  }

  ttcore::MetalLayoutAttr newLayout = layoutWithOptimalGrid(
      oldLayout, targetGridShape, targetSquareGridShape, optimalGrid, builder);

  llvm::SmallVector<int64_t> deviceShape = newLayout.getDeviceShape(
      optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

  Type newElementType =
      tileShape.empty()
          ? elementType
          : ttcore::TileType::get(elementType, llvm::ArrayRef(tileShape));
  return RankedTensorType::get(deviceShape, newElementType, newLayout);
}

// Verify that a StreamLayoutOp result is used by a single GenericOp only.
// Returns the GenericOp if valid, nullptr otherwise.
static d2m::GenericOp
verifyStreamLayoutUsedBySingleGeneric(d2m::StreamLayoutOp streamLayoutOp) {
  d2m::GenericOp parentGeneric = nullptr;

  for (auto &streamUse : streamLayoutOp.getResult().getUses()) {
    mlir::Operation *streamUser = streamUse.getOwner();

    d2m::GenericOp streamUseGeneric =
        mlir::dyn_cast<d2m::GenericOp>(streamUser);
    if (!streamUseGeneric) {
      streamUseGeneric = streamUser->getParentOfType<d2m::GenericOp>();
    }

    TT_assertv(streamUseGeneric,
               "StreamLayout (fed by ToLayout) must be used by a GenericOp");

    if (!parentGeneric) {
      parentGeneric = streamUseGeneric;
    } else if (parentGeneric != streamUseGeneric) {
      TT_assertv(false, "StreamLayout (fed by ToLayout) should only be "
                        "used within one GenericOp");
    }
  }

  return parentGeneric;
}

// Update a ToLayoutOp and its associated EmptyOp to use a specified grid by
// recreating the MetalLayoutAttr with the given grid and proper dimension
// alignments.
static void optimizeToLayoutGrid(d2m::ToLayoutOp toLayoutOp,
                                 ArrayRef<int64_t> targetGridShape,
                                 ArrayRef<int64_t> targetSquareGridShape,
                                 ArrayRef<int64_t> optimalGrid,
                                 OpBuilder &builder) {
  auto emptyOp = toLayoutOp.getOutput().getDefiningOp<d2m::EmptyOp>();
  if (!emptyOp) {
    return;
  }

  // Check if we're already at the target grid.
  auto emptyType = mlir::cast<mlir::RankedTensorType>(emptyOp.getType());
  if (emptyType.getShape().take_front(2) == llvm::ArrayRef(optimalGrid)) {
    return;
  }

  auto outputType = mlir::cast<mlir::RankedTensorType>(toLayoutOp.getType(0));
  auto oldLayout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(outputType.getEncoding());
  if (!oldLayout) {
    return;
  }

  bool needsOptimization = false;
  for (int64_t g : optimalGrid) {
    if (g > 1) {
      needsOptimization = true;
      break;
    }
  }

  if (!needsOptimization) {
    return;
  }

  RankedTensorType newTensorType = tensorWithOptimalGrid(
      outputType, targetGridShape, targetSquareGridShape, optimalGrid, builder);
  builder.setInsertionPoint(emptyOp);

  // Determine if the chosen grid is virtual (exceeds 2D device bounds or is
  // ND). On main, this was the isVirtualGrid bool threaded from
  // computeOptimalGrid. Block sharding always produces 2D grids within device
  // bounds, so any grid exceeding bounds or >2D came from the virtual path.
  mlir::AffineMapAttr virtualGridMapping;
  auto device = ttcore::lookupDevice(toLayoutOp);
  auto workerGridShape = device.getWorkerGrid().getShape();
  bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
      optimalGrid, workerGridShape);
  if (isVirtual) {
    auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
        ttmlir::utils::volume<int64_t>(optimalGrid), targetSquareGridShape);
    TT_assertv(!physicalGridShape.empty(),
               "Unable to find 2D rect that can fit virtual grid {} within "
               "device grid {}",
               ttmlir::utils::formatIterable(optimalGrid, "x"),
               ttmlir::utils::formatIterable(targetSquareGridShape, "x"));
    auto [forwardMap, inverseMap] =
        ttmlir::d2m::utils::grids::createCoreVirtMaps(
            builder.getContext(), optimalGrid, physicalGridShape);
    virtualGridMapping = AffineMapAttr::get(inverseMap);
  }

  auto newEmptyOp = builder.create<d2m::EmptyOp>(
      emptyOp.getLoc(), newTensorType, virtualGridMapping);

  builder.setInsertionPoint(toLayoutOp);
  auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
      toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);

  // Reblock it back to original shape to preserve IR correctness.
  // The view chain that applyViews composes through depends on this
  // ViewLayoutOp existing between the optimal-grid ToLayout and downstream
  // StreamLayoutOps / GenericOps.
  auto viewOutputType =
      utils::reblockTensor(newTensorType, oldLayout.getGridShape(outputType));
  auto reblockMap = ttmlir::utils::calculateReblockMap(
      newTensorType.getShape(), viewOutputType.getShape(),
      builder.getContext());
  auto view = builder.create<d2m::ViewLayoutOp>(
      toLayoutOp.getLoc(), viewOutputType, newToLayoutOp.getResult(0),
      reblockMap, /*reinterpretLayout=*/false);

  // We expect the ToLayout to be used in one of two ways:
  // 1. Directly by a single GenericOp (or operations within its region)
  // 2. By a stream_layout operation, where the stream_layout result is then
  //    used by a single GenericOp
  d2m::GenericOp parentGeneric = nullptr;

  for (auto &use : toLayoutOp.getResult(0).getUses()) {
    mlir::Operation *user = use.getOwner();

    // Check if this use is by a stream_layout operation
    if (auto streamLayoutOp = mlir::dyn_cast<d2m::StreamLayoutOp>(user)) {
      // Verify that the stream_layout result is used by a single GenericOp
      d2m::GenericOp streamParentGeneric =
          verifyStreamLayoutUsedBySingleGeneric(streamLayoutOp);

      // Track the GenericOp that uses the stream_layout
      if (streamParentGeneric) {
        if (!parentGeneric) {
          parentGeneric = streamParentGeneric;
        } else if (parentGeneric != streamParentGeneric) {
          TT_assertv(false,
                     "ToLayout should only be used within one GenericOp");
        }
      }
      continue;
    }

    // Find the parent GenericOp for this use.
    // The user might be the GenericOp itself (if it's an operand), or
    // it might be an operation nested within the GenericOp's regions.
    d2m::GenericOp useGeneric = mlir::dyn_cast<d2m::GenericOp>(user);
    if (!useGeneric) {
      useGeneric = user->getParentOfType<d2m::GenericOp>();
    }

    TT_assertv(useGeneric,
               "ToLayout result must be used by a single GenericOp or a single "
               "StreamLayout that is an input to a single GenericOp");

    if (!parentGeneric) {
      parentGeneric = useGeneric;
    } else if (parentGeneric != useGeneric) {
      // Use is within a different GenericOp
      TT_assertv(false, "ToLayout should only be used within one GenericOp");
    }
  }
  toLayoutOp.getResult(0).replaceAllUsesWith(view.getResult());

  toLayoutOp.erase();
  if (emptyOp.getResult().use_empty()) {
    emptyOp.erase();
  }
}

struct ToLayoutUpdateInfo {
  d2m::ToLayoutOp op;
  llvm::SmallVector<int64_t> grid;
};

struct StreamLayoutUpdateInfo {
  d2m::StreamLayoutOp op;
  llvm::SmallVector<int64_t> grid;
};

struct EmptyUpdateInfo {
  d2m::EmptyOp op;
  llvm::SmallVector<int64_t> grid;
};

// This function normalizes the operand grids for a generic operation by
// ensuring that the grids are consistent across all operands that share the
// same loop dimension. We also need to make sure that the grids respect any
// constraints implied by the outputs' grids. If multiple operands participate
// in the same loop dimension, the corresponding grid extents must agree.
static llvm::SmallVector<llvm::SmallVector<int64_t>>
normalizeOperandGridsForGeneric(
    d2m::GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return {};
  }

  TT_assert(optimalOperandGrids.size() == genericOp.getNumOperands());

  // First, normalize input operand grids for operands that share loop
  // dimensions. For example, in a matmul, the two inputs share the reduction
  // dimension. If their independently chosen optimal grids differ along that
  // dimension, promote the grid factor for that *dimension only* to the
  // maximum across all inputs that share it.
  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids(
      optimalOperandGrids.begin(), optimalOperandGrids.end());

  uint64_t numInputs = genericOp.getNumDpsInputs();
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

  // For each loop dimension that is used by multiple inputs, promote the grid
  // size associated with that loop dimension to the maximum across those
  // inputs.
  for (auto &it : dimToInputOperandDims) {
    auto &entries = it.second;
    if (entries.size() < 2) {
      continue;
    }

    int64_t maxFactor = 0;
    TT_assertv(
        entries.size() <= normalizedOperandGrids.size(),
        "adjusted operand grids size does not match dim-operand mapping size");
    for (auto [operandIndex, operandDimIdx] : entries) {
      TT_assertv(operandDimIdx < normalizedOperandGrids[operandIndex].size(),
                 "operand dim index out of bounds on adjusted operand grids");
      maxFactor = std::max(maxFactor,
                           normalizedOperandGrids[operandIndex][operandDimIdx]);
    }
    for (auto [operandIndex, operandDimIdx] : entries) {
      TT_assertv(operandDimIdx < normalizedOperandGrids[operandIndex].size(),
                 "operand dim index out of bounds on adjusted operand grids");
      normalizedOperandGrids[operandIndex][operandDimIdx] = maxFactor;
    }
  }

  // Compute grid dim constraints implied by the generic's outputs. These
  // constraints describe which loop dimensions must agree across operands.
  auto outputIndexingMap =
      genericOp.getIndexingMapsValue()[genericOp.getOutputs()
                                           .getBeginOperandIndex()];
  auto outputShape =
      optimalOperandGrids[genericOp.getOutputs().getBeginOperandIndex()];
  std::optional<SmallVector<int64_t>> outputConstraints =
      utils::computeDimConstraints(
          llvm::ArrayRef<AffineMap>(outputIndexingMap),
          llvm::ArrayRef<SmallVector<int64_t>>(outputShape));

  // Ensure that input operand grid shapes respect any constraints implied by
  // the outputs' grids. If multiple operands participate in the same loop
  // dimension, the corresponding grid extents must agree.
  if (outputConstraints) {
    for (auto [operandIndex, operand] :
         llvm::enumerate(genericOp->getOpOperands())) {
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
          normalizedOperandGrids[operandIndex][resultIdx] = constraint;
        }
      }
    }
  }

  return normalizedOperandGrids;
}

// Phase 1: Analyze each operand of a GenericOp and compute optimal grids.
// We compute grids independently per operand to mirror the old TTIRToD2M
// behavior, ensuring compatibility with existing grid assignment logic.
static std::tuple<llvm::SmallVector<llvm::SmallVector<int64_t>>,
                  llvm::SmallVector<ToLayoutUpdateInfo>,
                  llvm::SmallVector<StreamLayoutUpdateInfo>,
                  llvm::SmallVector<EmptyUpdateInfo>>
analyzeOperandsAndComputeGrids(d2m::GenericOp genericOp,
                               ArrayRef<int64_t> targetGridShape,
                               ArrayRef<int64_t> targetSquareGridShape) {
  OpBuilder builder(genericOp->getContext());
  SmallVector<SmallVector<int64_t>> optimalOperandGrids;
  llvm::SmallVector<ToLayoutUpdateInfo> toLayoutsToUpdate;
  llvm::SmallVector<StreamLayoutUpdateInfo> streamLayoutsToUpdate;
  llvm::SmallVector<EmptyUpdateInfo> emptyOpsToUpdate;

  for (Value operand : genericOp.getOperands()) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    if (!operandLayout) {
      continue;
    }

    // Compute physical shape and find the optimal grid that evenly divides
    // it.
    llvm::SmallVector<int64_t> physShape = computePhysicalShape(
        operandLayout, operandType, targetSquareGridShape, builder);

    // Interleaved tensors do not support virtual grids
    auto optimalGrid =
        computeOptimalGrid(operandType, physShape, targetSquareGridShape);

    optimalOperandGrids.push_back(optimalGrid);

    // Identify which operations need updating based on the operand type.
    if (auto streamLayout = operand.getDefiningOp<d2m::StreamLayoutOp>()) {
      // For stream_layout ops, the output optimal grid (already computed) will
      // be used for the storage. The input needs its own grid computed
      // independently based on its own shape.
      streamLayoutsToUpdate.push_back({streamLayout, optimalGrid});
      if (auto toLayoutOp =
              streamLayout.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
        if (!toLayoutOp.getInput()
                 .getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
          // Compute the input's grid independently based on its own shape.
          auto inputType = mlir::cast<mlir::RankedTensorType>(
              streamLayout.getInput().getType());
          auto inputLayout =
              mlir::cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());

          llvm::SmallVector<int64_t> inputPhysShape = computePhysicalShape(
              inputLayout, inputType, targetSquareGridShape, builder);
          auto inputOptimalGrid = computeOptimalGrid(inputType, inputPhysShape,
                                                     targetSquareGridShape);

          toLayoutsToUpdate.push_back({toLayoutOp, inputOptimalGrid});
        }
      }
    } else if (auto toLayoutOp = operand.getDefiningOp<d2m::ToLayoutOp>()) {
      // Skip TTNN tensors as their grids are already correctly set.
      if (toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      toLayoutsToUpdate.push_back({toLayoutOp, optimalGrid});
    } else if (auto emptyOp = operand.getDefiningOp<d2m::EmptyOp>()) {
      emptyOpsToUpdate.push_back({emptyOp, optimalGrid});
    }
  }

  // Normalize the operand grids for the generic operation - see the comment on
  // this function for details.
  optimalOperandGrids =
      normalizeOperandGridsForGeneric(genericOp, optimalOperandGrids);

  return {optimalOperandGrids, toLayoutsToUpdate, streamLayoutsToUpdate,
          emptyOpsToUpdate};
}

// Phase 2: Update ToLayoutOps with their optimal grids.
static void updateToLayoutOps(ArrayRef<ToLayoutUpdateInfo> toLayoutsToUpdate,
                              ArrayRef<int64_t> targetGridShape,
                              ArrayRef<int64_t> targetSquareGridShape) {
  if (toLayoutsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(toLayoutsToUpdate.front().op->getContext());
  for (auto &info : toLayoutsToUpdate) {
    optimizeToLayoutGrid(info.op, targetGridShape, targetSquareGridShape,
                         info.grid, builder);
  }
}

// Phase 3: Update StreamLayoutOps by recreating their storage with the new
// grid. StreamLayoutOps perform reblocking and may have index_maps that
// transpose dimensions, requiring special handling.
static void
updateStreamLayoutOps(ArrayRef<StreamLayoutUpdateInfo> streamLayoutsToUpdate,
                      ArrayRef<int64_t> targetSquareGridShape,
                      d2m::GenericOp genericOp) {
  if (streamLayoutsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(streamLayoutsToUpdate.front().op->getContext());
  for (const auto &info : streamLayoutsToUpdate) {
    auto streamLayout = info.op;
    auto optimalGrid = info.grid;
    auto storageEmpty = streamLayout.getStorage().getDefiningOp<d2m::EmptyOp>();
    if (!storageEmpty) {
      continue;
    }

    auto storageType =
        mlir::cast<mlir::RankedTensorType>(storageEmpty.getType());
    auto storageLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(storageType.getEncoding());

    llvm::SmallVector<int64_t> storageDimAlignments =
        ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
            storageLayout.getLogicalShape(), targetSquareGridShape,
            storageLayout.getNormalizedIntervals());

    auto newStorageLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), storageLayout.getLogicalShape(),
        storageDimAlignments, storageLayout.getCollapsedIntervals(),
        storageLayout.getOobVal(), storageLayout.getMemorySpace(),
        storageLayout.getMemoryLayout());

    llvm::SmallVector<int64_t> tileShape;
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(storageType.getElementType())) {
      tileShape = llvm::to_vector(tileType.getShape());
    }
    llvm::SmallVector<int64_t> newStorageShape =
        newStorageLayout.getDeviceShape(
            optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

    // The application of a grid shape to the unit grid stream storage above
    // could result in an increase of dimAlignments, this breaks the reblock map
    // calculation from the old unit grid storage shape to the new optimal grid
    // storage shape (different strides & volumes). In this case, align-up the
    // old storage shape by overwriting it with the new storage's shape on a
    // unit grid.
    llvm::SmallVector<int64_t> oldStorageShape(storageType.getShape());
    if (!llvm::equal(storageLayout.getDimAlignments(),
                     newStorageLayout.getDimAlignments())) {
      TT_assert(
          ttmlir::utils::volume(storageLayout.getGridShape(storageType)) == 1);
      oldStorageShape = newStorageLayout.getDeviceShape(
          storageLayout.getGridShape(storageType), tileShape);
    }
    TT_assert(ttmlir::utils::volume<int64_t>(oldStorageShape) ==
              ttmlir::utils::volume<int64_t>(newStorageShape));

    builder.setInsertionPoint(storageEmpty);
    Type elementType = tileShape.empty()
                           ? storageType.getElementType()
                           : ttcore::TileType::get(storageType.getElementType(),
                                                   llvm::ArrayRef(tileShape));
    // Propagate virtualGridMapping if the old storage had one, or compute
    // a new one if the storage grid is virtual.
    mlir::AffineMapAttr virtualGridMapping =
        storageEmpty.getVirtualGridMappingAttr();
    if (!virtualGridMapping) {
      auto device = ttcore::lookupDevice(storageEmpty);
      auto workerGridShape = device.getWorkerGrid().getShape();
      bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
          optimalGrid, workerGridShape);
      if (isVirtual) {
        auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
            ttmlir::utils::volume<int64_t>(optimalGrid), targetSquareGridShape);
        TT_assertv(!physicalGridShape.empty(),
                   "Unable to find 2D rect that can fit virtual grid");
        auto [forwardMap, inverseMap] =
            ttmlir::d2m::utils::grids::createCoreVirtMaps(
                builder.getContext(), optimalGrid, physicalGridShape);
        virtualGridMapping = AffineMapAttr::get(inverseMap);
      }
    }

    auto newStorageEmpty = builder.create<d2m::EmptyOp>(
        storageEmpty.getLoc(),
        RankedTensorType::get(newStorageShape, elementType, newStorageLayout),
        virtualGridMapping);

    auto outputStreamType =
        mlir::cast<RankedTensorType>(streamLayout.getResult().getType());
    auto outputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(outputStreamType.getEncoding());
    mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
        oldStorageShape, newStorageShape, builder.getContext());
    auto newOutputMap = streamLayout.getRemapping().compose(reblockMap);

    auto newOutputLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), outputLayout.getLogicalShape(),
        storageDimAlignments, outputLayout.getCollapsedIntervals(),
        outputLayout.getOobVal(), outputLayout.getMemorySpace(),
        outputLayout.getMemoryLayout());

    auto newStreamOutputType = RankedTensorType::get(
        newStorageShape, outputStreamType.getElementType(), newOutputLayout);

    builder.setInsertionPoint(streamLayout);
    auto newStreamLayout = builder.create<d2m::StreamLayoutOp>(
        streamLayout.getLoc(), newStreamOutputType, streamLayout.getInput(),
        AffineMapAttr::get(newOutputMap), newStorageEmpty);

    // We expect the StreamLayout to be used only by the GenericOp we're
    // optimizing. Check that all uses are either the GenericOp itself or
    // operations nested within the GenericOp's region.
    assert(llvm::all_of(streamLayout.getResult().getUsers(),
                        [&](Operation *user) {
                          return user == genericOp ||
                                 genericOp->isAncestor(user);
                        }) &&
           "StreamLayout should only be used by the GenericOp being "
           "optimized or operations within its region");
    streamLayout.getResult().replaceAllUsesWith(newStreamLayout.getResult());
    streamLayout.erase();

    if (storageEmpty.use_empty()) {
      storageEmpty.erase();
    }
  }
}

static void updateEmptyOps(ArrayRef<EmptyUpdateInfo> emptyOpsToUpdate,
                           ArrayRef<int64_t> targetGridShape,
                           ArrayRef<int64_t> targetSquareGridShape) {
  if (emptyOpsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(emptyOpsToUpdate.front().op->getContext());
  for (auto &info : emptyOpsToUpdate) {
    EmptyOp emptyOp = info.op;
    auto emptyType =
        mlir::cast<mlir::RankedTensorType>(emptyOp.getResult().getType());
    RankedTensorType newTensorType = tensorWithOptimalGrid(
        emptyType, targetGridShape, targetSquareGridShape, info.grid, builder);
    builder.setInsertionPoint(info.op);

    // Propagate virtualGridMapping if the old EmptyOp had one, or compute
    // a new one if the grid is virtual.
    mlir::AffineMapAttr virtualGridMapping =
        emptyOp.getVirtualGridMappingAttr();
    if (!virtualGridMapping) {
      auto device = ttcore::lookupDevice(emptyOp);
      auto workerGridShape = device.getWorkerGrid().getShape();
      bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
          info.grid, workerGridShape);
      if (isVirtual) {
        auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
            ttmlir::utils::volume<int64_t>(info.grid), targetSquareGridShape);
        TT_assertv(!physicalGridShape.empty(),
                   "Unable to find 2D rect that can fit virtual grid");
        auto [forwardMap, inverseMap] =
            ttmlir::d2m::utils::grids::createCoreVirtMaps(
                builder.getContext(), info.grid, physicalGridShape);
        virtualGridMapping = AffineMapAttr::get(inverseMap);
      }
    }

    auto newEmptyOp = builder.create<d2m::EmptyOp>(
        emptyOp.getLoc(), newTensorType, virtualGridMapping);
    emptyOp.getResult().replaceAllUsesWith(newEmptyOp.getResult());
    emptyOp.erase();
  }
}

// Phase 4: Recreate the d2m.generic with updated operands.
// After updating all ToLayout and StreamLayout ops, the generic's operands
// now have new types with optimized grids. We must recreate the generic to
// reflect these type changes, including updating the region body and any
// nested linalg.generic result types.
static void
recreateGenericOp(d2m::GenericOp genericOp,
                  ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return;
  }

  TT_assert(optimalOperandGrids.size() == genericOp.getNumOperands());

  OpBuilder builder(genericOp);
  llvm::SmallVector<Value> newOperands;

  for (const auto &[optimalGrid, operand] :
       llvm::zip(optimalOperandGrids, genericOp->getOpOperands())) {

    auto definingView = operand.get().getDefiningOp<d2m::ViewLayoutOp>();
    if (!definingView) {
      newOperands.push_back(operand.get());
      continue;
    }

    if (genericOp.isDpsInit(&operand) && definingView) {
      // This is a workaround to avoid type checking errors during/after
      // canonicalization.  There is an offline proposal being discussed to
      // address this more holistically.  The short of it is that we need to
      // just reach through the view to get to the original to_layout operand so
      // that view_layout folding doesn't need to be applied in the first place.
      // View layout folding can cause the index_map inside of the metal_layout
      // to differ from the generic op's result type, leading to type-checking
      // errors.
      newOperands.push_back(definingView.getInput());
      continue;
    }

    auto tensorType =
        mlir::cast<mlir::RankedTensorType>(operand.get().getType());
    auto viewTensorType = utils::reblockTensor(tensorType, optimalGrid);
    auto reblockMap = ttmlir::utils::calculateReblockMap(
        tensorType.getShape(), viewTensorType.getShape(), builder.getContext());
    auto view = builder.create<d2m::ViewLayoutOp>(
        genericOp.getLoc(), viewTensorType, operand.get(), reblockMap,
        /*reinterpretLayout=*/false);
    newOperands.push_back(view.getResult());
  }

  {
    auto numInputs = genericOp.getNumDpsInputs();

    llvm::SmallVector<Value> newInputs(newOperands.begin(),
                                       newOperands.begin() + numInputs);
    llvm::SmallVector<Value> newOutputs(newOperands.begin() + numInputs,
                                        newOperands.end());

    Region &oldRegion = genericOp.getRegion(0);

    auto newGenericOp = builder.create<d2m::GenericOp>(
        genericOp.getLoc(), newInputs, newOutputs, genericOp.getIndexingMaps(),
        genericOp.getIteratorTypes(),
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          IRMapping mapping;

          // Map old operands to new operands for ops that capture external
          // values (e.g., DMAs that reference views outside the region).
          for (auto [oldOp, newOp] :
               llvm::zip(genericOp.getOperands(), newOperands)) {
            mapping.map(oldOp, newOp);
          }

          // Map block arguments.
          Block &oldBlock = oldRegion.front();
          for (auto [oldArg, newArg] :
               llvm::zip(oldBlock.getArguments(), blockArgs)) {
            mapping.map(oldArg, newArg);
          }
          for (Operation &op : oldBlock) {
            Operation *clonedOp = b.clone(op, mapping);

            // For nested linalg.generic ops, update result types to match the
            // new output operand types (which have changed due to grid
            // updates).
            if (auto remoteLoadOp =
                    llvm::dyn_cast<d2m::RemoteLoadOp>(clonedOp)) {
              // RemoteLoadOp must be in implicit form at this point in the
              // pipeline. GridSelection runs before conversion to explicit CB
              // form.
              TT_assertv(
                  remoteLoadOp.isImplicitForm(),
                  "RemoteLoadOp must be in implicit form during GridSelection");

              // Result exists - get shard shape from the remote tensor.
              // GridSelection operates in tensor space (before bufferization).
              auto tensorType = mlir::cast<RankedTensorType>(
                  remoteLoadOp.getMemref().getType());
              auto deviceLayout =
                  ttcore::getDeviceLayout(remoteLoadOp.getMemref());
              if (deviceLayout) {
                auto shardShape = deviceLayout.getShardShape(tensorType);
                auto elementType = tensorType.getElementType();
                auto shardType = RankedTensorType::get(shardShape, elementType);
                remoteLoadOp.getResult().setType(shardType);

                // Also update the localBuffer's defining operation's result
                // type to match the shard shape.
                Value localBuffer = remoteLoadOp.getLocalBuffer();
                if (localBuffer) {
                  if (auto *defOp = localBuffer.getDefiningOp()) {
                    if (defOp->getNumResults() == 1) {
                      defOp->getResult(0).setType(shardType);
                    }
                  }
                }
              }
            } else if (auto remoteStoreOp =
                           llvm::dyn_cast<d2m::RemoteStoreOp>(clonedOp)) {
              // RemoteStoreOp must have result form at this point in the
              // pipeline. GridSelection runs before conversion to explicit CB
              // form.
              TT_assertv(
                  remoteStoreOp.hasResultForm(),
                  "RemoteStoreOp must have result form during GridSelection");

              auto tensorType = mlir::cast<RankedTensorType>(
                  remoteStoreOp.getMemref().getType());
              remoteStoreOp.getResult().setType(tensorType);
            } else if (auto dstOp = llvm::dyn_cast<DestinationStyleOpInterface>(
                           clonedOp)) {
              int numIns = dstOp.getNumDpsInputs();
              int numOuts = clonedOp->getNumResults();
              for (int i = 0; i < numOuts; ++i) {
                auto outputOperandType =
                    clonedOp->getOperand(numIns + i).getType();
                clonedOp->getResult(i).setType(outputOperandType);
              }
            } else if (auto tensorEmptyOp =
                           llvm::dyn_cast<mlir::tensor::EmptyOp>(clonedOp)) {
              // Update tensor.empty result type to match the associated operand
              // CB Use the original operation (&op) to find the associated
              // operand and CB from the old generic op.
              if (auto originalEmptyOp =
                      mlir::dyn_cast<mlir::tensor::EmptyOp>(&op)) {
                Value associatedOperand =
                    d2m::GenericOp::findAssocOperand(originalEmptyOp);
                if (associatedOperand) {
                  // Find the CB block argument associated with this operand in
                  // the old generic.
                  Value oldCB = d2m::GenericOp::findAssocCBByOperand(
                      &op, associatedOperand);
                  if (oldCB) {
                    // Find the index of the old CB in the old block arguments
                    auto oldBlockArgs = oldBlock.getArguments();
                    unsigned cbIndex = UINT_MAX;
                    for (unsigned i = 0; i < oldBlockArgs.size(); ++i) {
                      if (oldBlockArgs[i] == oldCB) {
                        cbIndex = i;
                        break;
                      }
                    }
                    // Use the index to get the new CB from blockArgs
                    if (cbIndex != UINT_MAX && cbIndex < blockArgs.size()) {
                      Value newCB = blockArgs[cbIndex];
                      // Extract the tensor type from the CB type
                      if (auto cbType =
                              mlir::dyn_cast<d2m::CBType>(newCB.getType())) {
                        auto underlyingType = cbType.getUnderlying();
                        if (auto tensorType = mlir::dyn_cast<RankedTensorType>(
                                underlyingType)) {
                          tensorEmptyOp.getResult().setType(tensorType);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        /*singleThreadType=*/genericOp.getRegionThreadType(0));

    // Preserve scratch_inputs attribute if present.
    if (auto scratchInputs = genericOp.getScratchInputsAttr()) {
      newGenericOp.setScratchInputsAttr(scratchInputs);
    }

    genericOp.replaceAllUsesWith(newGenericOp);
    genericOp.erase();
  }
}

static bool hasTTNNOperands(d2m::GenericOp genericOp) {
  for (Value operand : genericOp.getOperands()) {
    if (operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      return true;
    }
    // Check if view operand's input is the result of a TTNNMetalLayoutCastOp.
    if (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>();
        view && view.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      return true;
    }
  }
  return false;
}

// Computes the expected TTNN generic output grid shape for the given tensor.
static llvm::SmallVector<llvm::SmallVector<int64_t>>
computeTTNNGenericGridShapes(GenericOp genericOp,
                             ArrayRef<int64_t> targetSquareGridShape) {

  auto optimalOperandGrids =
      llvm::SmallVector<llvm::SmallVector<int64_t>>(genericOp.getNumOperands());

  // Determine dim size constraints based on L1 operands. L1 operands are
  // assumed fixed and already legal; DRAM operand streams are aligned to match
  // L1 shapes.
  auto maybeConstrainedDims = genericOp.computeGridDimConstraints(
      [&](ttcore::MetalLayoutAttr baseMetalLayout, bool isOutputOperand) {
        return baseMetalLayout.getMemorySpace() ==
               ttcore::MemorySpace::DeviceL1;
      });
  // this should be guaranteed if GenericOp verification is working.
  TT_assertv(maybeConstrainedDims.has_value(),
             "GenericOp dim constraints are cannot be satisfied.");
  auto constrainedDims = maybeConstrainedDims.value();

  auto indexingMaps = genericOp.getIndexingMapsValue();
  auto getConstrainedDims = [&](int64_t operandIdx) {
    return indexingMaps[operandIdx].compose(constrainedDims);
  };
  auto allDimsConstrained = [&](int64_t operandIdx) {
    return llvm::all_of(getConstrainedDims(operandIdx),
                        [](int64_t dim) { return dim != 0; });
  };

  // Set all grid shapes according to constraints
  OpBuilder builder(genericOp->getContext());
  for (auto [operandIdx, operand] : llvm::enumerate(genericOp.getOperands())) {

    auto constrainedDims = getConstrainedDims(operandIdx);
    // if all dims are constrained, use the constrained dims.
    if (allDimsConstrained(operandIdx)) {
      optimalOperandGrids[operandIdx] = getConstrainedDims(operandIdx);
    } else {
      // if not all dims are constrained, shard to an optimal grid.
      auto metalTensorType =
          mlir::cast<mlir::RankedTensorType>(operand.getType());
      auto baseMetalLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(metalTensorType.getEncoding());
      auto constrainedDims = getConstrainedDims(operandIdx);

      // Compute constrained target grid shape as min of targetSquareGridShape
      // and constrainedDims (if constrainedDim > 0), else use
      // targetSquareGridShape.
      llvm::SmallVector<int64_t> constrainedTargetGridShape =
          llvm::to_vector(targetSquareGridShape);
      if (constrainedDims.size() == targetSquareGridShape.size()) {
        for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
          if (constrainedDims[i] > 0) {
            constrainedTargetGridShape[i] =
                std::min(constrainedDims[i], targetSquareGridShape[i]);
          }
        }
      }

      auto physicalShape =
          computePhysicalShape(baseMetalLayout, metalTensorType,
                               constrainedTargetGridShape, builder);
      optimalOperandGrids[operandIdx] = computeOptimalGrid(
          metalTensorType, physicalShape, constrainedTargetGridShape);
    }
  }

  return optimalOperandGrids;
}

// Finds and erases all unit reblocking views inserted by TTIRToD2M,
// passing each view's input as the new operands.
static void eraseUnitGridReblockingViews(d2m::GenericOp genericOp) {
  // Use vector here to avoid invalidating iterator with erasures.
  auto operands = llvm::to_vector(genericOp.getOperands());
  for (Value operand : operands) {
    if (auto viewOp = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
      auto originalOperand = viewOp.getInput();
      viewOp.getResult().replaceAllUsesWith(originalOperand);
      viewOp.erase();
    }
  }
}

// TTNN DRAM interleaved tensors are represented as having a 1x1 grid. This
// leads to the genericOp having a worker grid of 1x1 since it must match the
// output tensor grid. This is obviously not optimal. We match genericOps that
// have TTNN DRAM interleaved tensors as operands and:
// 1. Compute the "optimal" grid for the tensor as if it were a regular Metal
// sharded tensor.
// 2. Insert a stream layout op with a mock storage tensor to represent the
// tensor with the "optimal" grid.
// 3. Update the genericOp to use the stream output as an operand.
//
// Note the cast op is NOT erased as it represents the canonical layout mapping
// between TTNN and Metal layouts.
//
// For a given TTNN DRAM interleaved tensor, we end up with the following
// representations:
// 1. The canonical translation of the TTNN tensor to a Metal tensor, having
// a metal layout, DRAM memory space, and a 1x1 grid.
//
// 2. The "reblocked" version of tensor 1, having a metal layout, DRAM memory
// space, an inferred grid, and an index map to index into the original
// tensor.
//
// 3. A storage tensor required by the stream_layout op to represent the
// mapping of tensor 1 to tensor 2. This tensor has the inferred grid and L1
// memory space.
static llvm::SmallVector<llvm::SmallVector<int64_t>>
insertTTNNDRAMStreams(d2m::GenericOp genericOp,
                      ArrayRef<int64_t> targetSquareGridShape) {

  eraseUnitGridReblockingViews(genericOp);

  auto optimalOperandGrids =
      computeTTNNGenericGridShapes(genericOp, targetSquareGridShape);

  OpBuilder builder(genericOp->getContext());
  for (auto [operandIdx, operand] : llvm::enumerate(genericOp.getOperands())) {
    auto metalTensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto baseMetalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());
    if (baseMetalLayout.getMemorySpace() != ttcore::MemorySpace::DeviceDRAM) {
      continue;
    }

    // Do not "restream" metal -> ttnn -> metal sequences. This happens when the
    // output of a generic is the input to another generic. The output is
    // already streamed, but the cast back to ttnn silently erases the index
    // map. Instead, we just forward the already streamed metal tensor to the
    // current generic.
    auto castOp = operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
    TT_assertv(
        castOp,
        "If one d2m.generic operand is from TTNN, they must all be from TTNN.");
    auto producerCastOp =
        castOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
    if (producerCastOp) {
      castOp.getResult().replaceAllUsesExcept(producerCastOp.getInput(),
                                              producerCastOp);
      continue;
    }

    llvm::SmallVector<int64_t> unshardedShape =
        baseMetalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    // TTNN DRAM interleaved tensors are represented as having a 1x1 grid.
    llvm::SmallVector<int64_t> unitGridShape{1, 1};
    llvm::SmallVector<int64_t> unShardedShapeWithGrid =
        baseMetalLayout.getDeviceShape(unitGridShape,
                                       ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> fakeShardedShape =
        baseMetalLayout.getDeviceShape(optimalOperandGrids[operandIdx],
                                       ttcore::TileType::getDefaultShape());

    auto reblockMap = ttmlir::utils::calculateReblockMap(
        unShardedShapeWithGrid, fakeShardedShape, builder.getContext());
    auto streamOutputLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), baseMetalLayout.getLogicalShape(),
        baseMetalLayout.getOobVal(), ttcore::MemorySpace::DeviceDRAM,
        ttcore::TensorMemoryLayout::Interleaved,
        baseMetalLayout.getCollapsedIntervals(),
        baseMetalLayout.getDimAlignments());

    auto streamOutputTensor = mlir::RankedTensorType::get(
        fakeShardedShape, metalTensor.getElementType(), streamOutputLayout);

    auto storageLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), baseMetalLayout.getLogicalShape(),
        baseMetalLayout.getOobVal(), ttcore::MemorySpace::DeviceL1,
        ttcore::TensorMemoryLayout::Sharded,
        baseMetalLayout.getCollapsedIntervals(),
        baseMetalLayout.getDimAlignments());

    auto storageTensor = mlir::RankedTensorType::get(
        fakeShardedShape, metalTensor.getElementType(), storageLayout);

    builder.setInsertionPointAfter(castOp);
    auto storageOp =
        builder.create<d2m::EmptyOp>(castOp.getLoc(), storageTensor,
                                     /*virtualGridMapping=*/nullptr);
    auto streamOp = builder.create<d2m::StreamLayoutOp>(
        castOp.getLoc(), streamOutputTensor, castOp.getResult(),
        AffineMapAttr::get(reblockMap), storageOp);
    castOp.getResult().replaceAllUsesExcept(streamOp.getResult(), streamOp);
  }

  TT_assertv(llvm::all_of(optimalOperandGrids,
                          [](const llvm::SmallVector<int64_t> &grid) {
                            return !grid.empty();
                          }),
             "Optimal grids must be populated for all operands.");
  return optimalOperandGrids;
}

// Assign optimized grids to all ToLayoutOps feeding into a GenericOp by
// computing the optimal grid per tensor independently, mirroring the old
// TTIRToD2M behavior.
static void assignGrids(d2m::GenericOp genericOp,
                        ArrayRef<int64_t> targetGridShape,
                        ArrayRef<int64_t> targetSquareGridShape) {
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  if (!hasTTNNOperands(genericOp)) {
    llvm::SmallVector<ToLayoutUpdateInfo> toLayoutsToUpdate;
    llvm::SmallVector<StreamLayoutUpdateInfo> streamLayoutsToUpdate;
    llvm::SmallVector<EmptyUpdateInfo> emptyOpsToUpdate;
    std::tie(optimalOperandGrids, toLayoutsToUpdate, streamLayoutsToUpdate,
             emptyOpsToUpdate) =
        analyzeOperandsAndComputeGrids(genericOp, targetGridShape,
                                       targetSquareGridShape);

    updateToLayoutOps(toLayoutsToUpdate, targetGridShape,
                      targetSquareGridShape);

    updateStreamLayoutOps(streamLayoutsToUpdate, targetSquareGridShape,
                          genericOp);

    updateEmptyOps(emptyOpsToUpdate, targetGridShape, targetSquareGridShape);
  } else {
    optimalOperandGrids =
        insertTTNNDRAMStreams(genericOp, targetSquareGridShape);
  }

  recreateGenericOp(genericOp, optimalOperandGrids);
}

// ----------------------------------------------------------------------------
// Pass implementation
// ----------------------------------------------------------------------------

#define GEN_PASS_DEF_D2MGRIDSELECTION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGridSelectionPass final
    : public impl::D2MGridSelectionBase<D2MGridSelectionPass> {
public:
  using Base = impl::D2MGridSelectionBase<D2MGridSelectionPass>;

  D2MGridSelectionPass() = default;

  D2MGridSelectionPass(const D2MGridSelectionOptions &options) : Base() {
    this->overrideDeviceShape = llvm::to_vector(options.overrideDeviceShape);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();
    llvm::SmallVector<int64_t> targetSquareGridShape =
        d2m::utils::getSquareTargetGrid(targetGridShape);

    module.walk([&](d2m::GenericOp genericOp) {
      // Skip explicit datamovement form - users manage grids manually
      if (genericOp.isExplicitDatamovementForm()) {
        return;
      }
      assignGrids(genericOp, targetGridShape, targetSquareGridShape);
    });
  }

private:
  llvm::SmallVector<int64_t> getTargetGridShape() {
    if (!overrideDeviceShape.empty()) {
      return llvm::to_vector(overrideDeviceShape);
    }

    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }
};
} // namespace

} // namespace mlir::tt::d2m
