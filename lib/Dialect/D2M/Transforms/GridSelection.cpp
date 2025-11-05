// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::tt::d2m {

// Compute dimension alignments for a MetalLayoutAttr that align to the worker
// grid shape for a given operation. This extends basic tile alignment (32x32)
// with grid-aware strategies to maximize worker utilization.
//
// For example, tensor<4x43x7> with grid<8x8> and tile<32x32>:
// - Basic tile alignments: 1x32x32
// - Tile-aligned logical shape: 4x64x32
// - Collapsed physical shape: 256x32
// - Grid & shard: 256x32 / 32x32 = 8x1
// - Result: 8 workers, each with a 1x1 tile shard
// This achieves good worker utilization with padding at tensor buffer ends,
// minimizing memory access stride issues.
//
// However, consider tensor<9x43x7> with the same grid and tile:
// - Collapsed physical shape: 576x32
// - Without grid alignment: 576x32 with 8x8 grid gives uneven distribution
// - With grid alignment (256x32x32): forces shape to 768x32
// - Grid & shard: 768x32 / 32x32 = 24x1
// - Result: More even distribution, though creates 'unnatural' shards
//
// For a collapsed shape of 31x33x447 with grid<8x8>:
// - Basic alignments: 1x32x32 (from TTIRToD2M)
// - Last dim exceeds tile*grid threshold (32*8=256)
// - Grid-aware alignments: 1x32x256
// - Forces shape: 31x64x512 to saturate the 8x8 worker grid
//
// This strategy trades some padding overhead for better worker utilization and
// more predictable NoC traffic patterns.
static llvm::SmallVector<int64_t>
computeGridAwareDimAlignments(ArrayRef<int64_t> logicalShape,
                              ArrayRef<int64_t> deviceGridShape,
                              ArrayRef<int64_t> normalizedIntervals) {
  constexpr std::array<int64_t, 2> tileShape =
      ttcore::TileType::getDefaultShape();

  const int64_t logicalRank = logicalShape.size();
  const int64_t deviceGridRank = deviceGridShape.size();
  const int64_t tensorGridRank = normalizedIntervals.size() / 2;

  assert(logicalRank >= 2);
  assert(deviceGridRank == 2);
  assert(normalizedIntervals.size() % 2 == 0);
  assert(deviceGridRank <= tensorGridRank);

  llvm::SmallVector<int64_t> alignments(logicalRank, 1);

  // Process the last two intervals (which map to the 2D tile shape) and apply
  // grid-aware alignments to saturate the worker grid when possible.
  for (int64_t idx = -1; idx >= -2; idx--) {
    const int64_t tileIdx = tileShape.size() + idx;
    const int64_t tileDim = tileShape[tileIdx];

    const int64_t gridIdx = deviceGridRank + idx;
    const int64_t gridDim = deviceGridShape[gridIdx];

    const int64_t intvIdx = tensorGridRank + idx;

    const int64_t gridAlignmentThreshold = gridDim * tileDim;

    const int64_t intervalStart = normalizedIntervals[intvIdx * 2];
    const int64_t intervalEnd = normalizedIntervals[intvIdx * 2 + 1] - 1;

    // Calculate the collapsed size for this interval by multiplying dimensions
    // within the interval, applying tile alignment to the last two logical
    // dims.
    int64_t collapsedSize = 1;
    for (int64_t i = intervalEnd; i >= intervalStart; i--) {
      if (i >= logicalRank - 2) {
        collapsedSize *= ttmlir::utils::alignUp(logicalShape[i], tileDim);
      } else {
        collapsedSize *= logicalShape[i];
      }
    }

    // If the collapsed size exceeds the grid threshold, align to the grid
    // boundary to distribute work evenly across cores; otherwise just align
    // to tile boundaries.
    const bool alignToGrid = collapsedSize > gridAlignmentThreshold;
    const int64_t alignment = alignToGrid ? gridAlignmentThreshold : tileDim;

    // Apply the alignment to the appropriate dimension(s) in the interval.
    // Assumes collapsed intervals are always <[[0, N-2], [N-1, N]]>.
    if (intervalStart == intervalEnd) {
      alignments[intervalEnd] = alignment;
    } else {
      assert(idx == -2);
      assert(intervalEnd == logicalRank - 2);
      alignments[intervalEnd] = tileDim;
      // For multi-dimension intervals, apply grid alignment to the leading
      // dimension to avoid redundant alignments (e.g., [32x32]x32 ->
      // [1x32]x32).
      if (alignToGrid) {
        alignments[intervalStart] = alignment;
      }
    }
  }
  assert(alignments[logicalRank - 1] % tileShape[1] == 0);
  assert(alignments[logicalRank - 2] % tileShape[0] == 0);
  return alignments;
}

// ----------------------------------------------------------------------------
// Grid optimization utilities
// ----------------------------------------------------------------------------

// Compute optimal grid shape for a given physical shape and target grid by
// finding the largest grid dimensions that evenly divide the physical shape.
// This ensures maximum utilization of available worker cores while maintaining
// even distribution of work.
static llvm::SmallVector<int64_t>
computeOptimalGrid(ArrayRef<int64_t> physicalShape,
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

  llvm::SmallVector<int64_t> tileShape;
  Type elementType = outputType.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
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

  auto collapsedIntervals = oldLayout.getCollapsedIntervals();

  llvm::SmallVector<int64_t> newDimAlignments = computeGridAwareDimAlignments(
      oldLayout.getLogicalShape(), targetSquareGridShape,
      oldLayout.getNormalizedIntervals());

  auto newLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), oldLayout.getLogicalShape(), oldLayout.getOobVal(),
      oldLayout.getMemorySpace(), oldLayout.getMemoryLayout(),
      collapsedIntervals, newDimAlignments);

  llvm::SmallVector<int64_t> shardedShape = newLayout.getDeviceShape(
      optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

  builder.setInsertionPoint(emptyOp);
  Type newElementType =
      tileShape.empty()
          ? elementType
          : ttcore::TileType::get(elementType, llvm::ArrayRef(tileShape));
  auto newEmptyOp = builder.create<d2m::EmptyOp>(emptyOp.getLoc(), shardedShape,
                                                 newElementType, newLayout);

  builder.setInsertionPoint(toLayoutOp);
  auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
      toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);

  // We expect the ToLayout to be used only by the GenericOp we're optimizing.
  // Assert this assumption to catch unexpected sharing.
  assert(toLayoutOp.getResult(0).hasOneUse() &&
         "ToLayout should only be used by the GenericOp being optimized");
  toLayoutOp.getResult(0).replaceAllUsesWith(newToLayoutOp.getResult(0));

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

// Phase 1: Analyze each operand of a GenericOp and compute optimal grids.
// We compute grids independently per operand to mirror the old TTIRToD2M
// behavior, ensuring compatibility with existing grid assignment logic.
static std::pair<llvm::SmallVector<ToLayoutUpdateInfo>,
                 llvm::SmallVector<StreamLayoutUpdateInfo>>
analyzeOperandsAndComputeGrids(d2m::GenericOp genericOp,
                               ArrayRef<int64_t> targetGridShape,
                               ArrayRef<int64_t> targetSquareGridShape) {
  OpBuilder builder(genericOp->getContext());
  llvm::SmallVector<ToLayoutUpdateInfo> toLayoutsToUpdate;
  llvm::SmallVector<StreamLayoutUpdateInfo> streamLayoutsToUpdate;

  for (Value operand : genericOp.getOperands()) {
    auto operandType = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto operandLayout =
        mlir::dyn_cast<ttcore::MetalLayoutAttr>(operandType.getEncoding());
    if (!operandLayout) {
      continue;
    }

    llvm::SmallVector<int64_t> tileShape;
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(operandType.getElementType())) {
      tileShape = llvm::to_vector(tileType.getShape());
    }

    // Compute alignments assuming the target square grid, then get the physical
    // shape that would result from those alignments. The logical shape is
    // already correct from TTIRToD2M (transposed if needed).
    llvm::SmallVector<int64_t> targetAlignments = computeGridAwareDimAlignments(
        operandLayout.getLogicalShape(), targetSquareGridShape,
        operandLayout.getNormalizedIntervals());

    auto tempLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), operandLayout.getLogicalShape(),
        operandLayout.getOobVal(), operandLayout.getMemorySpace(),
        operandLayout.getMemoryLayout(), operandLayout.getCollapsedIntervals(),
        targetAlignments);

    llvm::SmallVector<int64_t> physShape = tempLayout.getPhysicalShape(
        llvm::ArrayRef(tileShape.data(), tileShape.size()));

    // Find the optimal grid that evenly divides the physical shape.
    llvm::SmallVector<int64_t> optimalGrid =
        computeOptimalGrid(physShape, targetSquareGridShape);

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

          llvm::SmallVector<int64_t> inputTileShape;
          if (auto tileType = mlir::dyn_cast<ttcore::TileType>(
                  inputType.getElementType())) {
            inputTileShape = llvm::to_vector(tileType.getShape());
          }

          llvm::SmallVector<int64_t> inputAlignments =
              computeGridAwareDimAlignments(
                  inputLayout.getLogicalShape(), targetSquareGridShape,
                  inputLayout.getNormalizedIntervals());

          auto inputTempLayout = ttcore::MetalLayoutAttr::get(
              builder.getContext(), inputLayout.getLogicalShape(),
              inputLayout.getOobVal(), inputLayout.getMemorySpace(),
              inputLayout.getMemoryLayout(),
              inputLayout.getCollapsedIntervals(), inputAlignments);

          llvm::SmallVector<int64_t> inputPhysShape =
              inputTempLayout.getPhysicalShape(
                  llvm::ArrayRef(inputTileShape.data(), inputTileShape.size()));

          llvm::SmallVector<int64_t> inputOptimalGrid =
              computeOptimalGrid(inputPhysShape, targetSquareGridShape);

          toLayoutsToUpdate.push_back({toLayoutOp, inputOptimalGrid});
        }
      }
    } else if (auto toLayoutOp = operand.getDefiningOp<d2m::ToLayoutOp>()) {
      // Skip TTNN tensors as their grids are already correctly set.
      if (toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      toLayoutsToUpdate.push_back({toLayoutOp, optimalGrid});
    }
  }

  return {toLayoutsToUpdate, streamLayoutsToUpdate};
}

// Phase 2: Update ToLayoutOps with their optimal grids.
static void updateToLayoutOps(ArrayRef<ToLayoutUpdateInfo> toLayoutsToUpdate,
                              ArrayRef<int64_t> targetGridShape,
                              ArrayRef<int64_t> targetSquareGridShape) {
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
                      ArrayRef<int64_t> targetSquareGridShape) {
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
        computeGridAwareDimAlignments(storageLayout.getLogicalShape(),
                                      targetSquareGridShape,
                                      storageLayout.getNormalizedIntervals());

    auto newStorageLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), storageLayout.getLogicalShape(),
        storageDimAlignments, storageLayout.getCollapsedIntervals(),
        storageLayout.getOobVal(), storageLayout.getMemorySpace(),
        storageLayout.getMemoryLayout(), storageLayout.getIndexAffineMap());

    llvm::SmallVector<int64_t> tileShape;
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(storageType.getElementType())) {
      tileShape = llvm::to_vector(tileType.getShape());
    }
    llvm::SmallVector<int64_t> newStorageShape =
        newStorageLayout.getDeviceShape(
            optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

    builder.setInsertionPoint(storageEmpty);
    Type elementType = tileShape.empty()
                           ? storageType.getElementType()
                           : ttcore::TileType::get(storageType.getElementType(),
                                                   llvm::ArrayRef(tileShape));
    auto newStorageEmpty = builder.create<d2m::EmptyOp>(
        storageEmpty.getLoc(), newStorageShape, elementType, newStorageLayout);

    builder.setInsertionPoint(streamLayout);
    auto newStreamLayout = builder.create<d2m::StreamLayoutOp>(
        streamLayout.getLoc(), newStorageEmpty.getType(),
        streamLayout.getInput(), newStorageEmpty);

    // We expect the StreamLayout to be used only by the GenericOp we're
    // optimizing. Assert this assumption to catch unexpected sharing.
    assert(streamLayout.getResult().hasOneUse() &&
           "StreamLayout should only be used by the GenericOp being optimized");
    streamLayout.getResult().replaceAllUsesWith(newStreamLayout.getResult());
    streamLayout.erase();

    if (storageEmpty.use_empty()) {
      storageEmpty.erase();
    }
  }
}

// Phase 4: Recreate the d2m.generic with updated operands.
// After updating all ToLayout and StreamLayout ops, the generic's operands
// now have new types with optimized grids. We must recreate the generic to
// reflect these type changes, including updating the region body and any
// nested linalg.generic result types.
static void recreateGenericOp(d2m::GenericOp genericOp) {
  OpBuilder builder(genericOp->getContext());
  llvm::SmallVector<Value> newOperands;
  for (Value operand : genericOp.getOperands()) {
    newOperands.push_back(operand);
  }

  {
    auto numInputs = genericOp.getNumDpsInputs();

    llvm::SmallVector<Value> newInputs(newOperands.begin(),
                                       newOperands.begin() + numInputs);
    llvm::SmallVector<Value> newOutputs(newOperands.begin() + numInputs,
                                        newOperands.end());

    Region &oldRegion = genericOp.getRegion(0);

    builder.setInsertionPoint(genericOp);
    auto newGenericOp = builder.create<d2m::GenericOp>(
        genericOp.getLoc(), newInputs, newOutputs, genericOp.getIndexingMaps(),
        genericOp.getIteratorTypes(),
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          IRMapping mapping;
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
            if (llvm::isa<DestinationStyleOpInterface>(clonedOp)) {
              auto numInputs = clonedOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
                  "operandSegmentSizes");
              if (numInputs && numInputs.size() >= 2) {
                int32_t numIns = numInputs[0];
                int32_t numOuts = numInputs[1];

                for (uint32_t i = 0; static_cast<int32_t>(i) < numOuts &&
                                     i < clonedOp->getNumResults();
                     ++i) {
                  auto outputOperandType =
                      clonedOp->getOperand(numIns + i).getType();
                  clonedOp->getResult(i).setType(outputOperandType);
                }
              }
            } else if (llvm::isa<d2m::WaitOp, d2m::ReserveOp>(clonedOp)) {
              assert(clonedOp->getNumOperands() == 1);
              assert(clonedOp->getNumResults() == 1);
              clonedOp->getResult(0).setType(
                  mlir::cast<d2m::CBType>(clonedOp->getOperand(0).getType())
                      .getUnderlying());
            }
          }
        },
        /*singleThreadType=*/genericOp.getRegionThreadType(0));

    genericOp.replaceAllUsesWith(newGenericOp);
    genericOp.erase();
  }
}

static bool hasTTNNOperands(d2m::GenericOp genericOp) {
  for (Value operand : genericOp.getOperands()) {
    if (operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      return true;
    }
  }
  return false;
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
static void insertTTNNDRAMStreams(d2m::GenericOp genericOp,
                                  ArrayRef<int64_t> targetSquareGridShape) {
  OpBuilder builder(genericOp->getContext());
  for (Value operand : genericOp.getOperands()) {
    auto metalTensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto baseMetalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());
    if (baseMetalLayout.getMemorySpace() != ttcore::MemorySpace::DeviceDRAM) {
      continue;
    }

    llvm::SmallVector<int64_t> unshardedShape =
        baseMetalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    // TTNN DRAM interleaved tensors are represented as having a 1x1 grid.
    llvm::SmallVector<int64_t> unitGridShape{1, 1};
    llvm::SmallVector<int64_t> unShardedShapeWithGrid =
        baseMetalLayout.getDeviceShape(unitGridShape,
                                       ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> workerGrid =
        computeOptimalGrid(unshardedShape, targetSquareGridShape);
    llvm::SmallVector<int64_t> fakeShardedShape =
        baseMetalLayout.getDeviceShape(workerGrid,
                                       ttcore::TileType::getDefaultShape());

    auto streamOutputLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), baseMetalLayout.getLogicalShape(),
        baseMetalLayout.getDimAlignments(),
        baseMetalLayout.getCollapsedIntervals(), baseMetalLayout.getOobVal(),
        ttcore::MemorySpace::DeviceDRAM,
        ttcore::TensorMemoryLayout::Interleaved,
        mlir::tt::d2m::utils::calculateReblockMap(
            unShardedShapeWithGrid, fakeShardedShape, builder.getContext()));

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

    auto castOp = operand.getDefiningOp<ttir::TTNNMetalLayoutCastOp>();
    builder.setInsertionPointAfter(castOp);
    auto storageOp =
        builder.create<d2m::EmptyOp>(castOp.getLoc(), storageTensor);
    auto streamOp = builder.create<d2m::StreamLayoutOp>(
        castOp.getLoc(), streamOutputTensor, castOp.getResult(), storageOp);
    castOp.getResult().replaceAllUsesExcept(streamOp.getResult(), streamOp);
  }
}

// Assign optimized grids to all ToLayoutOps feeding into a GenericOp by
// computing the optimal grid per tensor independently, mirroring the old
// TTIRToD2M behavior.
static void assignGrids(d2m::GenericOp genericOp,
                        ArrayRef<int64_t> targetGridShape,
                        ArrayRef<int64_t> targetSquareGridShape) {
  if (!hasTTNNOperands(genericOp)) {
    auto [toLayoutsToUpdate, streamLayoutsToUpdate] =
        analyzeOperandsAndComputeGrids(genericOp, targetGridShape,
                                       targetSquareGridShape);

    if (toLayoutsToUpdate.empty() && streamLayoutsToUpdate.empty()) {
      return;
    }

    if (!toLayoutsToUpdate.empty()) {
      updateToLayoutOps(toLayoutsToUpdate, targetGridShape,
                        targetSquareGridShape);
    }

    if (!streamLayoutsToUpdate.empty()) {
      updateStreamLayoutOps(streamLayoutsToUpdate, targetSquareGridShape);
    }
  } else {
    insertTTNNDRAMStreams(genericOp, targetSquareGridShape);
  }

  recreateGenericOp(genericOp);
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
