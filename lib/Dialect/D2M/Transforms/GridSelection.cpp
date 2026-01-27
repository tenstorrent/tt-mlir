// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

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
#include <optional>

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
  auto viewOutputType = mlir::cast<RankedTensorType>(utils::reblockShapedType(
      newTensorType, oldLayout.getGridShape(outputType)));
  auto reblockMap = ttmlir::utils::calculateReblockMap(
      newTensorType.getShape(), viewOutputType.getShape(),
      builder.getContext());
  auto view = builder.create<d2m::ViewLayoutOp>(
      toLayoutOp.getLoc(), viewOutputType, newToLayoutOp.getResult(0),
      reblockMap, /*reinterpretLayout=*/false);

  // We expect the ToLayout to be used in one of two ways:
  // 1. Directly by a single GenericOp (or operations within its region)
  // 2. By a stream_layout or view_layout operation, where the result is then
  //    used by a single GenericOp
  d2m::GenericOp parentGeneric = nullptr;

  for (auto &use : toLayoutOp.getResult(0).getUses()) {
    mlir::Operation *user = use.getOwner();

    // Check if this use is by a view_layout operation (e.g., tensor
    // manipulation ops that express data rearrangement as a view).
    if (auto viewLayoutOp = mlir::dyn_cast<d2m::ViewLayoutOp>(user)) {
      // Walk through view_layout users to find the parent GenericOp.
      for (auto &viewUse : viewLayoutOp.getResult().getUses()) {
        mlir::Operation *viewUser = viewUse.getOwner();
        d2m::GenericOp viewUseGeneric =
            mlir::dyn_cast<d2m::GenericOp>(viewUser);
        if (!viewUseGeneric) {
          viewUseGeneric = viewUser->getParentOfType<d2m::GenericOp>();
        }
        if (viewUseGeneric) {
          if (!parentGeneric) {
            parentGeneric = viewUseGeneric;
          } else if (parentGeneric != viewUseGeneric) {
            TT_assertv(false,
                       "ToLayout should only be used within one GenericOp");
          }
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
               "ToLayout result must be used by a single GenericOp or a "
               "single StreamLayout/ViewLayout that feeds a single GenericOp");

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

  auto newTensorType = mlir::cast<RankedTensorType>(
      utils::reblockShapedType(outputType, optimalGrid));

  mlir::AffineMapAttr gridRemapping =
      AffineMapAttr::get(ttmlir::utils::calculateReblockMap(
          outputType.getShape(), newTensorType.getShape(),
          builder.getContext()));

  builder.setInsertionPointAfter(castOp);

  auto newViewLayoutOp = builder.create<d2m::ViewLayoutOp>(
      castOp.getLoc(), newTensorType, castOp.getResult(), gridRemapping);

  // Reblock it back to original shape to preserve IR correctness.
  auto viewOutputType = mlir::cast<RankedTensorType>(utils::reblockShapedType(
      newTensorType, outputLayout.getGridShape(outputType)));
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

struct EmptyUpdateInfo {
  d2m::EmptyOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct ViewLayoutUpdateInfo {
  d2m::ViewLayoutOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct CompositeViewUpdateInfo {
  d2m::CompositeViewOp op;
  unsigned operandIndex;
  llvm::SmallVector<int64_t> grid;
};

struct GridAnalysisResult {
  llvm::SmallVector<llvm::SmallVector<int64_t>> optimalOperandGrids;
  llvm::SmallVector<ToLayoutUpdateInfo> toLayouts;
  llvm::SmallVector<TTNNTensorUpdateInfo> ttnnTensors;
  llvm::SmallVector<EmptyUpdateInfo> emptyOps;
  llvm::SmallVector<ViewLayoutUpdateInfo> viewLayouts;
  llvm::SmallVector<CompositeViewUpdateInfo> compositeViews;
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

  TT_assert(optimalOperandGrids.size() ==
            genericOp.getInputsAndOutputs().size());

  // First, normalize input operand grids for operands that share loop
  // dimensions. For example, in a matmul, the two inputs share the reduction
  // dimension. If their independently chosen optimal grids differ along that
  // dimension, promote the grid factor for that *dimension only* to the
  // maximum across all inputs that share it.
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
    } else if (auto viewLayout = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
      // Track non-reinterpret views so their output type can be updated to
      // match the normalized grid. Reinterpret views are just type casts and
      // their grid must match the input — don't update them.
      if (!viewLayout.getReinterpretLayout()) {
        result.viewLayouts.push_back({viewLayout, idx, optimalGrid});
      }

      // If the view's input is a ToLayoutOp, also compute and apply the
      // optimal grid for that ToLayoutOp independently.
      if (auto toLayoutOp =
              viewLayout.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
        if (!toLayoutOp.getInput()
                 .getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
          auto inputType = mlir::cast<mlir::RankedTensorType>(
              viewLayout.getInput().getType());

          llvm::SmallVector<int64_t> inputPhysShape =
              computePhysicalShape(viewLayout.getInput(), config, builder);
          auto inputOptimalGrid =
              computeOptimalGrid(inputType, inputPhysShape, config);

          result.toLayouts.push_back({toLayoutOp, idx, inputOptimalGrid});
        }
      }
    } else if (auto compositeViewOp =
                   operand.getDefiningOp<d2m::CompositeViewOp>()) {
      result.compositeViews.push_back({compositeViewOp, idx, optimalGrid});
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

  // Propagate normalized grids back to view layout update infos, since their
  // grids may have been adjusted during normalization.
  for (auto &info : result.viewLayouts) {
    info.grid = result.optimalOperandGrids[info.operandIndex];
  }

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

static void
updateCompositeViewOps(ArrayRef<CompositeViewUpdateInfo> compositeViewsToUpdate,
                       const GridSelectionConfig &config) {
  if (compositeViewsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(compositeViewsToUpdate.front().op->getContext());
  for (const auto &info : compositeViewsToUpdate) {
    auto compositeView = info.op;

    // The composite_view handles its own inputs instead of relying on
    // updateToLayoutOps, and does not use computePhysicalShape to recreate the
    // input with its grid-aligned shape.
    // Views don't own the data and we want to stack other views on top of the
    // composite_view, it might be difficult to update the upstream to_layout:
    // e.g. one input is from a slicing view.
    SmallVector<Value> reblockedInputs;
    for (Value input : compositeView.getInputs()) {
      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      auto inputLayout =
          mlir::dyn_cast<ttcore::MetalLayoutAttr>(inputType.getEncoding());
      if (!inputLayout) {
        reblockedInputs.push_back(input);
        continue;
      }

      auto tileType = mlir::cast<ttcore::TileType>(inputType.getElementType());
      auto inputPhysShape = inputLayout.getPhysicalShape(tileType.getShape());
      auto inputOptimalGrid =
          computeOptimalGrid(inputType, inputPhysShape, config);

      auto currentGrid = inputLayout.getGridShape(inputType);
      if (llvm::ArrayRef(currentGrid) == llvm::ArrayRef(inputOptimalGrid)) {
        reblockedInputs.push_back(input);
        continue;
      }

      auto viewTensorType = mlir::cast<RankedTensorType>(
          utils::reblockShapedType(inputType, inputOptimalGrid));
      builder.setInsertionPoint(compositeView);
      auto view = builder.create<d2m::ViewLayoutOp>(compositeView.getLoc(),
                                                    viewTensorType, input);
      reblockedInputs.push_back(view.getResult());
    }

    auto outType =
        mlir::cast<RankedTensorType>(compositeView.getResult().getType());
    RankedTensorType newOutType =
        tensorWithOptimalGrid(outType, config, info.grid, builder);

    builder.setInsertionPoint(compositeView);
    auto newCompositeView = builder.create<d2m::CompositeViewOp>(
        compositeView.getLoc(), newOutType, reblockedInputs,
        compositeView.getDim());

    compositeView.getResult().replaceAllUsesWith(newCompositeView.getResult());
    compositeView.erase();
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

// Derive grid (including virtual grid mapping) from the
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

static ttcore::GridAttr
deriveGenericGridAttr(d2m::GenericOp genericOp,
                      ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
                      OpBuilder &builder) {
  Value output = genericOp.getOutputs().front();
  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  ArrayRef<int64_t> gridShape = optimalOperandGrids[outputOperandIndex];
  return deriveGridAttrForOutput(output, gridShape, builder);
}

// Update ViewLayoutOps by recreating them with a new output type that matches
// the normalized grid. The remapping is composed with a reblock map that
// accounts for the shape change from the old grid to the new grid.
static void
updateViewLayoutOps(ArrayRef<ViewLayoutUpdateInfo> viewLayoutsToUpdate,
                    const GridSelectionConfig &config) {
  if (viewLayoutsToUpdate.empty()) {
    return;
  }

  OpBuilder builder(viewLayoutsToUpdate.front().op->getContext());
  for (auto &info : viewLayoutsToUpdate) {
    d2m::ViewLayoutOp viewOp = info.op;
    auto oldResultType =
        mlir::cast<RankedTensorType>(viewOp.getResult().getType());
    auto oldLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(oldResultType.getEncoding());
    RankedTensorType newResultType =
        tensorWithOptimalGrid(oldResultType, config, info.grid, builder);

    // Compose the original remapping with a reblock map that maps from the
    // old output shape to the new output shape. This mirrors what
    // updateStreamLayoutOps did for stream_layout storage.
    llvm::SmallVector<int64_t> oldShape(oldResultType.getShape());
    llvm::SmallVector<int64_t> newShape(newResultType.getShape());

    // If dim alignments changed, align-up the old shape to match.
    auto newLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(newResultType.getEncoding());
    if (!llvm::equal(oldLayout.getDimAlignments(),
                     newLayout.getDimAlignments())) {
      llvm::SmallVector<int64_t> tileShape;
      if (auto tileType = mlir::dyn_cast<ttcore::TileType>(
              oldResultType.getElementType())) {
        tileShape = llvm::to_vector(tileType.getShape());
      }
      TT_assert(ttmlir::utils::volume(oldLayout.getGridShape(oldResultType)) ==
                1);
      oldShape = newLayout.getDeviceShape(oldLayout.getGridShape(oldResultType),
                                          tileShape);
    }

    mlir::AffineMap newRemapping = viewOp.getRemapping();
    if (!llvm::equal(oldShape, newShape)) {
      TT_assert(ttmlir::utils::volume<int64_t>(oldShape) ==
                ttmlir::utils::volume<int64_t>(newShape));
      mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
          oldShape, newShape, builder.getContext());
      newRemapping = newRemapping.compose(reblockMap);
    }

    builder.setInsertionPoint(viewOp);
    auto newViewOp = builder.create<d2m::ViewLayoutOp>(
        viewOp.getLoc(), newResultType, viewOp.getInput(), newRemapping,
        viewOp.getReinterpretLayout());
    viewOp.getResult().replaceAllUsesWith(newViewOp.getResult());
    viewOp.erase();
  }
}

// Phase 5: Recreate the d2m.generic with updated operands.
// After updating ToLayout and StreamLayout ops, the generic's operands have
// new types with the selected grids. The generic grid is still anchored by the
// output operand's chosen grid, but we must re-materialize the generic attrs
// from the selected operand grids after those rewrites so the rebuilt op stays
// consistent with the new operand types and the derived block factors.
static void
recreateGenericOp(d2m::GenericOp genericOp,
                  ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids) {
  if (optimalOperandGrids.empty()) {
    return;
  }

  OpBuilder builder(genericOp);
  unsigned outputOperandIndex = genericOp.getOutputs().getBeginOperandIndex();
  ArrayRef<int64_t> outputGridShape = optimalOperandGrids[outputOperandIndex];
  ttcore::GridAttr grid =
      deriveGenericGridAttr(genericOp, optimalOperandGrids, builder);
  SmallVector<int64_t> blockFactors = utils::deriveBlockFactorsFromOperandGrids(
      genericOp.getIndexingMapsValue(), optimalOperandGrids, outputGridShape);
  auto ret = genericOp.withParallelization(builder, grid, blockFactors,
                                           /*generateReturnView=*/false);
  if (failed(ret)) {
    genericOp.emitOpError()
        << "failed to recreate generic op with withParallelization";
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

  updateEmptyOps(analysis.emptyOps, config);

  updateCompositeViewOps(analysis.compositeViews, config);

  updateViewLayoutOps(analysis.viewLayouts, config);

  recreateGenericOp(genericOp, analysis.optimalOperandGrids);
}

// Resolve to a value that dominates the spatial op by following view_layout
// chains defined inside the spatial's regions (region-border value).
static Value resolveToRegionBorderValue(Value operand,
                                        d2m::SpatialOp spatialOp) {
  auto inSpatialRegion = [&](Value val) {
    Operation *def = val.getDefiningOp();
    if (!def) {
      return false;
    }
    Region *parent = def->getBlock()->getParent();
    return llvm::any_of(spatialOp->getRegions(),
                        [parent](Region &r) { return &r == parent; });
  };
  Value current = operand;
  while (inSpatialRegion(current)) {
    if (auto viewOp = current.getDefiningOp<d2m::ViewLayoutOp>()) {
      current = viewOp.getInput();
    } else {
      break;
    }
  }
  return current;
}

// Rebuild d2m.spatial's ins and outs from the operands actually used by
// d2m.generic ops in each region, and set result types from the collected outs.
static void reconstructSpatialOperands(d2m::SpatialOp spatialOp) {
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<mlir::Value> outputs;
  for (Region &region : spatialOp->getRegions()) {
    if (region.empty()) {
      continue;
    }
    for (d2m::GenericOp genericOp : region.front().getOps<d2m::GenericOp>()) {
      for (mlir::Value input : genericOp.getInputs()) {
        inputs.push_back(resolveToRegionBorderValue(input, spatialOp));
      }
      for (mlir::Value output : genericOp.getOutputs()) {
        outputs.push_back(resolveToRegionBorderValue(output, spatialOp));
      }
    }
  }
  spatialOp.getInputsMutable().assign(inputs);
  spatialOp.getOutputsMutable().assign(outputs);
  if (spatialOp->getNumResults() == outputs.size()) {
    for (auto [result, outVal] : llvm::zip(spatialOp->getResults(), outputs)) {
      result.setType(outVal.getType());
    }
  }
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

    module.walk([&](d2m::GenericOp genericOp) {
      // Skip explicit datamovement form - users manage grids manually
      if (genericOp.isExplicitDatamovementForm()) {
        return;
      }
      if (genericOp->hasAttr("d2m.skip_grid_selection")) {
        return;
      }
      llvm::SmallVector<int64_t> targetGridShape =
          getTargetGridShape(genericOp);
      llvm::SmallVector<int64_t> targetSquareGridShape =
          d2m::utils::getSquareTargetGrid(targetGridShape);
      GridSelectionConfig config{targetGridShape, targetSquareGridShape,
                                 this->ttnnMode};
      assignGrids(genericOp, config);
    });

    // Rebuild each SpatialOp's ins/outs from the operands actually used by
    // generics in its regions (e.g. after stream_layout splits one cast into
    // multiple operands per region).
    module.walk([&](d2m::SpatialOp spatialOp) {
      reconstructSpatialOperands(spatialOp);
    });
  }

private:
  // Returns the device-wide grid shape (worker grid or override).
  llvm::SmallVector<int64_t> getDeviceGridShape() {
    if (!overrideDeviceShape.empty()) {
      return llvm::to_vector(overrideDeviceShape);
    }

    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }

  // Returns the target grid shape for genericOp. If it is inside a
  // d2m.spatial region, uses that region's grid_ranges entry; otherwise
  // returns the device-wide grid shape.
  llvm::SmallVector<int64_t> getTargetGridShape(d2m::GenericOp genericOp) {
    mlir::Region *region = genericOp->getParentRegion();
    if (auto spatialOp =
            mlir::dyn_cast<d2m::SpatialOp>(region->getParentOp())) {
      auto coreRanges = spatialOp.getGridRanges().getCoreRanges();
      unsigned regionIndex = region->getRegionNumber();
      if (regionIndex < coreRanges.size()) {
        ttcore::CoreRangeAttr range = coreRanges[regionIndex];
        return {range.getEndCoord().getY() - range.getStartCoord().getY() + 1,
                range.getEndCoord().getX() - range.getStartCoord().getX() + 1};
      }
    }
    return getDeviceGridShape();
  }
};
} // namespace

} // namespace mlir::tt::d2m
