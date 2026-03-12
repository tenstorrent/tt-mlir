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

struct GridSelectionConfig {
  ArrayRef<int64_t> targetGridShape;
  ArrayRef<int64_t> targetSquareGridShape;
  bool ttnnMode;
};

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

// Compute physical shape for a MetalLayoutAttr. In TTNN mode, returns the raw
// physical shape without alignment adjustments. Otherwise, computes grid-aware
// dimension alignments and derives the physical shape (always tile-aligned).
static llvm::SmallVector<int64_t>
computePhysicalShape(Value operand, const GridSelectionConfig &config,
                     OpBuilder &builder) {
  // Pass through the unit-reblocking view, if present.
  if (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    operand = view.getInput();
  }

  auto tensorType = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());

  // TTNN tensors do not require dim-alignment.
  if (config.ttnnMode) {
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
          layout.getLogicalShape(), config.targetSquareGridShape,
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
static bool shouldImplementAsVirtualGrid(RankedTensorType tensorType,
                                         ArrayRef<int64_t> physicalShape,
                                         const GridSelectionConfig &config) {

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
  auto blockShardedGrid = computeOptimalBlockShardedGrid(
      physicalShape, config.targetSquareGridShape);
  auto blockShardedGridVolume =
      ttmlir::utils::volume<int64_t>(blockShardedGrid);
  int64_t targetGridVolume =
      ttmlir::utils::volume<int64_t>(config.targetSquareGridShape);
  bool lowGridUtilization = blockShardedGridVolume < 0.5 * targetGridVolume;
  return lowGridUtilization;
}

static llvm::SmallVector<int64_t>
computeOptimalGrid(mlir::RankedTensorType tensorType,
                   ArrayRef<int64_t> physicalShape,
                   const GridSelectionConfig &config) {
  if (shouldImplementAsVirtualGrid(tensorType, physicalShape, config)) {
    auto virtualGrid =
        computeOptimalVirtualGrid(physicalShape, config.targetSquareGridShape);
    if (!virtualGrid.empty()) {
      return virtualGrid;
    }
  }
  return computeOptimalBlockShardedGrid(physicalShape,
                                        config.targetSquareGridShape);
}

static ttcore::MetalLayoutAttr
layoutWithOptimalGrid(ttcore::MetalLayoutAttr oldLayout,
                      const GridSelectionConfig &config,
                      ArrayRef<int64_t> optimalGrid, OpBuilder &builder) {
  llvm::SmallVector<int64_t> newDimAlignments;
  if (config.ttnnMode) {
    // TTNN tensors use simple tile-aligned dim alignments without grid-aware
    // padding adjustments.
    newDimAlignments.assign(oldLayout.getLogicalShape().size(), 1);
    auto defaultTileShape = ttcore::TileType::getDefaultShape();
    newDimAlignments[newDimAlignments.size() - 1] = defaultTileShape[1];
    newDimAlignments[newDimAlignments.size() - 2] = defaultTileShape[0];
  } else {
    newDimAlignments = ttcore::MetalLayoutAttr::computeGridAwareDimAlignments(
        oldLayout.getLogicalShape(), config.targetSquareGridShape,
        oldLayout.getNormalizedIntervals());
  }

  return ttcore::MetalLayoutAttr::get(
      builder.getContext(), oldLayout.getLogicalShape(), oldLayout.getOobVal(),
      oldLayout.getMemorySpace(), oldLayout.getMemoryLayout(),
      oldLayout.getCollapsedIntervals(), newDimAlignments);
}

static RankedTensorType tensorWithOptimalGrid(RankedTensorType oldTensor,
                                              const GridSelectionConfig &config,
                                              ArrayRef<int64_t> optimalGrid,
                                              OpBuilder &builder) {
  auto oldLayout = mlir::cast<ttcore::MetalLayoutAttr>(oldTensor.getEncoding());

  llvm::SmallVector<int64_t> tileShape;
  Type elementType = oldTensor.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
  }

  ttcore::MetalLayoutAttr newLayout =
      layoutWithOptimalGrid(oldLayout, config, optimalGrid, builder);

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
                                 const GridSelectionConfig &config,
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

  RankedTensorType newTensorType =
      tensorWithOptimalGrid(outputType, config, optimalGrid, builder);
  builder.setInsertionPoint(emptyOp);

  // Determine if the chosen grid is virtual (exceeds 2D device bounds or is
  // ND). Note: VGM is NOT propagated from the to_layout's input here — the
  // output EmptyOp has its own grid/shard strategy. VGM for DMA addresses
  // is traced through the stream's input at DMA lowering time.
  mlir::AffineMapAttr virtualGridInverseMapping;
  mlir::AffineMapAttr virtualGridForwardMapping;
  auto device = ttcore::lookupDevice(toLayoutOp);
  auto workerGridShape = device.getWorkerGrid().getShape();
  bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
      optimalGrid, workerGridShape);
  if (isVirtual) {
    auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
        ttmlir::utils::volume<int64_t>(optimalGrid),
        config.targetSquareGridShape);
    TT_assertv(
        !physicalGridShape.empty(),
        "Unable to find 2D rect that can fit virtual grid {} within "
        "device grid {}",
        ttmlir::utils::formatIterable(optimalGrid, "x"),
        ttmlir::utils::formatIterable(config.targetSquareGridShape, "x"));
    auto [forwardMap, inverseMap] =
        ttmlir::d2m::utils::grids::createCoreVirtMaps(
            builder.getContext(), optimalGrid, physicalGridShape);
    virtualGridInverseMapping = AffineMapAttr::get(inverseMap);
    virtualGridForwardMapping = AffineMapAttr::get(forwardMap);
  }

  auto newEmptyOp = builder.create<d2m::EmptyOp>(
      emptyOp.getLoc(), newTensorType, virtualGridInverseMapping,
      virtualGridForwardMapping);

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

static bool isTTNNOperand(Value operand) {
  while (auto view = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    operand = view.getInput();
  }
  return operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>() != nullptr;
}

static void insertViewForTTNNDRAMTensor(Value operand,
                                        const GridSelectionConfig &config,
                                        ArrayRef<int64_t> optimalGrid,
                                        OpBuilder &builder) {
  while (auto viewOp = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
    auto originalOperand = viewOp.getInput();
    viewOp.getResult().replaceAllUsesWith(originalOperand);
    viewOp.erase();
    operand = originalOperand;
  }
  // Do not "restream" metal -> ttnn -> metal sequences. This happens when the
  // output of a generic is the input to another generic. The output is
  // already streamed, but the cast back to ttnn silently erases the index
  // map. Instead, we just forward the already streamed metal tensor to the
  // current generic.
  auto castOp = operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
  auto producerCastOp =
      castOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
  if (producerCastOp) {
    castOp.getResult().replaceAllUsesExcept(producerCastOp.getInput(),
                                            producerCastOp);
    return;
  }

  auto metalTensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
  auto baseMetalLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());

  // TTNN DRAM interleaved tensors are represented as having a 1x1 grid.
  llvm::SmallVector<int64_t> unitGridShape{1, 1};
  llvm::SmallVector<int64_t> unShardedShapeWithGrid =
      baseMetalLayout.getDeviceShape(unitGridShape,
                                     ttcore::TileType::getDefaultShape());

  llvm::SmallVector<int64_t> fakeShardedShape = baseMetalLayout.getDeviceShape(
      optimalGrid, ttcore::TileType::getDefaultShape());

  AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      unShardedShapeWithGrid, fakeShardedShape, builder.getContext());

  auto viewOutputLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), baseMetalLayout.getLogicalShape(),
      baseMetalLayout.getOobVal(), ttcore::MemorySpace::DeviceDRAM,
      ttcore::TensorMemoryLayout::Interleaved,
      baseMetalLayout.getCollapsedIntervals(),
      baseMetalLayout.getDimAlignments());

  auto viewOutputTensor = mlir::RankedTensorType::get(
      fakeShardedShape, metalTensor.getElementType(), viewOutputLayout);

  builder.setInsertionPointAfter(castOp);
  auto viewOp = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), viewOutputTensor, castOp.getResult(),
      AffineMapAttr::get(reblockMap));
  castOp.getResult().replaceAllUsesExcept(viewOp.getResult(), viewOp);
}

static void optimizeTTNNMetalLayoutCastOpGrid(
    ttir::TTNNMetalLayoutCastOp castOp, const GridSelectionConfig &config,
    ArrayRef<int64_t> optimalGrid, OpBuilder &builder) {
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(castOp.getResult().getType());
  auto outputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(outputType.getEncoding());

  if (optimalGrid == outputLayout.getGridShape(outputType)) {
    // Already at target grid shape.
    return;
  }

  auto newTensorType = utils::reblockTensor(outputType, optimalGrid);

  mlir::AffineMapAttr gridRemapping =
      AffineMapAttr::get(ttmlir::utils::calculateReblockMap(
          outputType.getShape(), newTensorType.getShape(),
          builder.getContext()));

  builder.setInsertionPointAfter(castOp);

  auto newViewLayoutOp = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), newTensorType, castOp.getResult(), gridRemapping);

  // Reblock it back to original shape to preserve IR correctness.
  auto viewOutputType = utils::reblockTensor(
      newTensorType, outputLayout.getGridShape(outputType));
  auto reblockMap = ttmlir::utils::calculateReblockMap(
      newTensorType.getShape(), viewOutputType.getShape(),
      builder.getContext());
  auto revertingView = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), viewOutputType, newViewLayoutOp.getResult(), reblockMap,
      /*reinterpretLayout=*/false);

  castOp.getResult().replaceAllUsesExcept(revertingView.getResult(),
                                          newViewLayoutOp);
}

struct ToLayoutUpdateInfo {
  d2m::ToLayoutOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct TTNNTensorUpdateInfo {
  Value operand;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct StreamLayoutUpdateInfo {
  d2m::StreamLayoutOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct EmptyUpdateInfo {
  d2m::EmptyOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct GridAnalysisResult {
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  llvm::SmallVector<ToLayoutUpdateInfo> toLayouts;
  llvm::SmallVector<TTNNTensorUpdateInfo> ttnnTensors;
  llvm::SmallVector<StreamLayoutUpdateInfo> streamLayouts;
  llvm::SmallVector<EmptyUpdateInfo> emptyOps;
};

// This function normalizes the operand grids for a generic operation by
// preserving independently selected per-operand grids. GridSelection now relies
// on withParallelization legalization to insert reblocking views when operand
// grids differ from generic execution requirements.
static llvm::SmallVector<llvm::SmallVector<int64_t>>
normalizeOperandGridsForGeneric(
    d2m::GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return {};
  }
  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());
  return llvm::to_vector(optimalOperandGrids);
}

// Phase 1: Analyze each operand of a GenericOp and compute optimal grids.
// We compute grids independently per operand to mirror the old TTIRToD2M
// behavior, ensuring compatibility with existing grid assignment logic.
static GridAnalysisResult
analyzeOperandsAndComputeGrids(d2m::GenericOp genericOp,
                               const GridSelectionConfig &config) {
  OpBuilder builder(genericOp->getContext());
  GridAnalysisResult result;

  for (auto [operandIndex, operand] :
       llvm::enumerate(genericOp.getInputsAndOutputs())) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    TT_assertv(operandLayout,
               "GridSelection expects GenericOp inputs/outputs to have "
               "MetalLayoutAttr");

    unsigned idx = static_cast<unsigned>(operandIndex);
    llvm::SmallVector<int64_t> physShape =
        computePhysicalShape(operand, config, builder);
    auto optimalGrid = computeOptimalGrid(operandType, physShape, config);
    result.optimalOperandGrids.push_back(optimalGrid);

    if (isTTNNOperand(operand)) {
      result.ttnnTensors.push_back({operand, idx, optimalGrid});
    } else if (auto streamLayout =
                   operand.getDefiningOp<d2m::StreamLayoutOp>()) {
      // For stream_layout ops, the output optimal grid (already computed)
      // will be used for the storage. The input needs its own grid computed
      // independently based on its own shape.
      result.streamLayouts.push_back({streamLayout, idx, optimalGrid});
      if (auto toLayoutOp =
              streamLayout.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
        if (!toLayoutOp.getInput()
                 .getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
          auto inputType = mlir::cast<mlir::RankedTensorType>(
              streamLayout.getInput().getType());

          llvm::SmallVector<int64_t> inputPhysShape =
              computePhysicalShape(streamLayout.getInput(), config, builder);
          auto inputOptimalGrid =
              computeOptimalGrid(inputType, inputPhysShape, config);

          result.toLayouts.push_back({toLayoutOp, idx, inputOptimalGrid});
        }
      }
    } else if (auto toLayoutOp = operand.getDefiningOp<d2m::ToLayoutOp>()) {
      result.toLayouts.push_back({toLayoutOp, idx, optimalGrid});
    } else if (auto emptyOp = operand.getDefiningOp<d2m::EmptyOp>()) {
      result.emptyOps.push_back({emptyOp, idx, optimalGrid});
    }
  }

  // Normalize the operand grids for the generic operation - see the comment on
  // this function for details.
  result.optimalOperandGrids =
      normalizeOperandGridsForGeneric(genericOp, result.optimalOperandGrids);

  return result;
}

// Phase 2: Update ToLayoutOps with their optimal grids.
static void updateToLayoutOps(ArrayRef<ToLayoutUpdateInfo> toLayoutsToUpdate,
                              const GridSelectionConfig &config) {
  if (toLayoutsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(toLayoutsToUpdate.front().op->getContext());
  for (auto &info : toLayoutsToUpdate) {
    optimizeToLayoutGrid(info.op, config, info.grid, builder);
  }
}

// Phase 3: Update TTNN tensors with their optimal grids.
static void
updateTTNNTensors(ArrayRef<TTNNTensorUpdateInfo> TTNNTensorsToUpdate,
                  const GridSelectionConfig &config) {
  if (TTNNTensorsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(TTNNTensorsToUpdate.front().operand.getContext());
  for (auto &info : TTNNTensorsToUpdate) {
    auto metalTensor =
        mlir::cast<mlir::RankedTensorType>(info.operand.getType());
    auto metalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());
    if (metalLayout.getMemorySpace() == ttcore::MemorySpace::DeviceDRAM) {
      insertViewForTTNNDRAMTensor(info.operand, config, info.grid, builder);
    } else if (auto castOp =
                   info.operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      optimizeTTNNMetalLayoutCastOpGrid(castOp, config, info.grid, builder);
    } else if (auto viewOp = info.operand.getDefiningOp<d2m::ViewLayoutOp>()) {
      // Erase the view op and directly operate on the defining cast.
      auto originalOperand = viewOp.getInput();
      auto castOp =
          originalOperand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
      TT_assertv(
          castOp,
          "Expected a TTNNMetalLayoutCastOp as the input of the view op.");
      viewOp.getResult().replaceAllUsesWith(originalOperand);
      viewOp.erase();
      optimizeTTNNMetalLayoutCastOpGrid(castOp, config, info.grid, builder);
    } else {
      llvm_unreachable("Expected a TTNNMetalLayoutCastOp or a ViewLayoutOp");
    }
  }
}

// Phase 4: Update StreamLayoutOps by recreating their storage with the new
// grid. StreamLayoutOps perform reblocking and may have index_maps that
// transpose dimensions, requiring special handling.
static void
updateStreamLayoutOps(ArrayRef<StreamLayoutUpdateInfo> streamLayoutsToUpdate,
                      const GridSelectionConfig &config,
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
            storageLayout.getLogicalShape(), config.targetSquareGridShape,
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
    // Propagate virtualGridInverseMapping if the old storage had one, or
    // compute a new one if the storage grid is virtual.
    mlir::AffineMapAttr virtualGridInverseMapping =
        storageEmpty.getVirtualGridInverseMappingAttr();
    mlir::AffineMapAttr virtualGridForwardMapping =
        storageEmpty.getVirtualGridForwardMappingAttr();
    if (!virtualGridInverseMapping) {
      auto device = ttcore::lookupDevice(storageEmpty);
      auto workerGridShape = device.getWorkerGrid().getShape();
      bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
          optimalGrid, workerGridShape);
      if (isVirtual) {
        auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
            ttmlir::utils::volume<int64_t>(optimalGrid),
            config.targetSquareGridShape);
        TT_assertv(!physicalGridShape.empty(),
                   "Unable to find 2D rect that can fit virtual grid");
        auto [forwardMap, inverseMap] =
            ttmlir::d2m::utils::grids::createCoreVirtMaps(
                builder.getContext(), optimalGrid, physicalGridShape);
        virtualGridInverseMapping = AffineMapAttr::get(inverseMap);
        virtualGridForwardMapping = AffineMapAttr::get(forwardMap);
      }
    }

    auto newStorageEmpty = builder.create<d2m::EmptyOp>(
        storageEmpty.getLoc(),
        RankedTensorType::get(newStorageShape, elementType, newStorageLayout),
        virtualGridInverseMapping, virtualGridForwardMapping);

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
                           const GridSelectionConfig &config) {
  if (emptyOpsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(emptyOpsToUpdate.front().op->getContext());
  for (auto &info : emptyOpsToUpdate) {
    EmptyOp emptyOp = info.op;
    auto emptyType =
        mlir::cast<mlir::RankedTensorType>(emptyOp.getResult().getType());
    RankedTensorType newTensorType =
        tensorWithOptimalGrid(emptyType, config, info.grid, builder);
    builder.setInsertionPoint(info.op);

    // Propagate virtualGridInverseMapping if the old EmptyOp had one, or
    // compute a new one if the grid is virtual.
    mlir::AffineMapAttr virtualGridInverseMapping =
        emptyOp.getVirtualGridInverseMappingAttr();
    mlir::AffineMapAttr virtualGridForwardMapping =
        emptyOp.getVirtualGridForwardMappingAttr();
    if (!virtualGridInverseMapping) {
      auto device = ttcore::lookupDevice(emptyOp);
      auto workerGridShape = device.getWorkerGrid().getShape();
      bool isVirtual = ttmlir::d2m::utils::grids::requiresVirtualGrid(
          info.grid, workerGridShape);
      if (isVirtual) {
        auto physicalGridShape = utils::findLegalPhysicalGridForVolume(
            ttmlir::utils::volume<int64_t>(info.grid),
            config.targetSquareGridShape);
        TT_assertv(!physicalGridShape.empty(),
                   "Unable to find 2D rect that can fit virtual grid");
        auto [forwardMap, inverseMap] =
            ttmlir::d2m::utils::grids::createCoreVirtMaps(
                builder.getContext(), info.grid, physicalGridShape);
        virtualGridInverseMapping = AffineMapAttr::get(inverseMap);
        virtualGridForwardMapping = AffineMapAttr::get(forwardMap);
      }
    }

    auto newEmptyOp = builder.create<d2m::EmptyOp>(
        emptyOp.getLoc(), newTensorType, virtualGridInverseMapping,
        virtualGridForwardMapping);
    emptyOp.getResult().replaceAllUsesWith(newEmptyOp.getResult());
    emptyOp.erase();
  }
}

// Derive grid (including virtual grid mapping) and block factors from the
// optimized operand grids selected by GridSelection, mirroring
// GenericOp::build.
static ttcore::GridAttr deriveGridAttrForOutput(Value output,
                                                ArrayRef<int64_t> gridShape,
                                                OpBuilder &builder) {
  auto layout = ttcore::getDeviceLayout(cast<ShapedType>(output.getType()));
  auto metalLayout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(layout);
  if (!metalLayout) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  if (auto invMap = utils::getVirtualGridInverseMapping(output)) {
    return builder.getAttr<ttcore::GridAttr>(gridShape, *invMap);
  }

  auto existingRemapping = utils::getAssociatedRemapping(output);
  if (!existingRemapping.has_value() || existingRemapping->isEmpty() ||
      existingRemapping->isIdentity()) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  auto indexMap = *existingRemapping;
  constexpr size_t kExpectedDimsFor2DDeviceShape = 2 * 2;
  bool is2DPermutation =
      indexMap.isPermutation() &&
      indexMap.getNumResults() == kExpectedDimsFor2DDeviceShape &&
      indexMap.getNumInputs() == kExpectedDimsFor2DDeviceShape;
  if (!is2DPermutation) {
    return builder.getAttr<ttcore::GridAttr>(gridShape);
  }

  auto invMap = ttmlir::utils::createGridInverseMapFor2DPermutation(
      indexMap, gridShape.size(), builder.getContext());
  return builder.getAttr<ttcore::GridAttr>(gridShape, invMap);
}

static std::pair<ttcore::GridAttr, SmallVector<int64_t>>
deriveGridAndBlockFactors(
    d2m::GenericOp genericOp,
    ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
    OpBuilder &builder) {
  auto inputOutputOperands = llvm::to_vector(genericOp.getInputsAndOutputs());
  Value output = genericOp.getOutputs().front();
  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  ArrayRef<int64_t> gridShape = optimalOperandGrids[outputOperandIndex];
  ttcore::GridAttr grid = deriveGridAttrForOutput(output, gridShape, builder);

  // Derive block factors using concatInversePermutationMap, mirroring
  // GenericOp::build.
  auto maps = genericOp.getIndexingMapsValue();
  auto flatInverseMap =
      ttmlir::utils::concatInversePermutationMap(maps, /*reverse=*/true);

  SmallVector<int64_t> flattenedOperandGridShapes;
  for (ArrayRef<int64_t> operandGridShape :
       llvm::reverse(optimalOperandGrids)) {
    flattenedOperandGridShapes.append(operandGridShape.begin(),
                                      operandGridShape.end());
  }

  for (std::size_t i = 0; i < grid.getShape().size(); ++i) {
    flattenedOperandGridShapes[i] /= grid.getShape()[i];
  }

  SmallVector<int64_t> blockFactors =
      flatInverseMap.compose(flattenedOperandGridShapes);
  return {grid, blockFactors};
}

// Phase 5: Recreate the d2m.generic with updated operands.
// After updating all ToLayout and StreamLayout ops, the generic's operands
// now have new types with optimized grids. We must recreate the generic to
// reflect these type changes. We derive the grid and block factors from the
// output operand, then use withParallelization to create ViewLayoutOps that
// make each operand compatible with the generic's grid.
static void
recreateGenericOp(d2m::GenericOp genericOp,
                  ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return;
  }

  OpBuilder builder(genericOp);
  auto [grid, blockFactors] =
      deriveGridAndBlockFactors(genericOp, optimalOperandGrids, builder);
  auto ret = genericOp.withParallelization(builder, grid, blockFactors,
                                           /*generateReturnView=*/false);
  if (failed(ret)) {
    return;
  }

  genericOp->replaceAllUsesWith(ret->genericOp);
  genericOp.erase();
}

// Assign optimized grids to all ToLayoutOps feeding into a GenericOp by
// computing the optimal grid per tensor independently, mirroring the old
// TTIRToD2M behavior.
static void assignGrids(d2m::GenericOp genericOp,
                        const GridSelectionConfig &config) {
  GridAnalysisResult analysis =
      analyzeOperandsAndComputeGrids(genericOp, config);

  updateToLayoutOps(analysis.toLayouts, config);

  updateTTNNTensors(analysis.ttnnTensors, config);

  updateStreamLayoutOps(analysis.streamLayouts, config, genericOp);

  updateEmptyOps(analysis.emptyOps, config);

  recreateGenericOp(genericOp, analysis.optimalOperandGrids);
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
    // Setting TTNN mode to true ensures we do not implicitly pad or wrap-around
    // when sharding. Any grid decisions in this mode are representable
    // using a TTNNLayoutAttr and can be created with a single ttnn.empty()
    // call. This can be removed only when we implement support for creating
    // padded tensors in D2MToTTNN pass.
    this->ttnnMode = options.ttnnMode;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();
    llvm::SmallVector<int64_t> targetSquareGridShape =
        d2m::utils::getSquareTargetGrid(targetGridShape);
    GridSelectionConfig config{targetGridShape, targetSquareGridShape,
                               ttnnMode};

    module.walk([&](d2m::GenericOp genericOp) {
      // Skip explicit datamovement form - users manage grids manually
      if (genericOp.isExplicitDatamovementForm()) {
        return;
      }
      assignGrids(genericOp, config);
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
