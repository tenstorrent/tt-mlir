// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

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

// ----------------------------------------------------------------------------
// Grid optimization utilities
// ----------------------------------------------------------------------------

// Compute optimal grid shape for a given physical shape and target grid.
// Finds the largest grid dimensions that evenly divide the physical shape.
static llvm::SmallVector<int64_t>
computeOptimalGrid(ArrayRef<int64_t> physicalShape,
                   ArrayRef<int64_t> targetSquareGridShape) {
  llvm::SmallVector<int64_t> grid;

  assert(physicalShape.size() == targetSquareGridShape.size());

  for (size_t i = 0; i < physicalShape.size(); ++i) {
    const int64_t dim = physicalShape[i];
    assert(dim > 0);
    // Find largest grid dimension that divides evenly.
    for (int64_t g = targetSquareGridShape[i]; g > 0; g--) {
      if (dim % g == 0) {
        grid.push_back(g);
        break;
      }
    }
  }

  assert(grid.size() == physicalShape.size());
  return grid;
}

// Update a ToLayoutOp and its associated EmptyOp to use a specified grid.
// This recreates the MetalLayoutAttr with the given grid and proper
// dimAlignments.
static void optimizeToLayoutGrid(d2m::ToLayoutOp toLayoutOp,
                                 ArrayRef<int64_t> targetGridShape,
                                 ArrayRef<int64_t> targetSquareGridShape,
                                 ArrayRef<int64_t> optimalGrid,
                                 OpBuilder &builder,
                                 bool swapAlignments = false) {
  // Get the output EmptyOp
  auto emptyOp = toLayoutOp.getOutput().getDefiningOp<d2m::EmptyOp>();
  if (!emptyOp) {
    return; // ToLayoutOp doesn't have an EmptyOp, skip
  }

  // Get the output tensor type and layout
  auto outputType = mlir::cast<mlir::RankedTensorType>(toLayoutOp.getType(0));
  auto oldLayout =
      mlir::dyn_cast<ttcore::MetalLayoutAttr>(outputType.getEncoding());
  if (!oldLayout) {
    return; // No layout to optimize
  }

  // Determine the tile shape
  llvm::SmallVector<int64_t> tileShape;
  Type elementType = outputType.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType)) {
    tileShape = llvm::to_vector(tileType.getShape());
    elementType = tileType.getElementType();
  }

  // If the grid is already optimal (all 1s means no sharding yet), update it
  bool needsOptimization = false;
  for (int64_t g : optimalGrid) {
    if (g > 1) {
      needsOptimization = true;
      break;
    }
  }

  if (!needsOptimization) {
    return; // Already at 1x1 grid, nothing to optimize
  }

  // Create new layout with only the grid changed, keep everything else the same
  auto collapsedIntervals = oldLayout.getCollapsedIntervals();
  auto dimAlignments = oldLayout.getDimAlignments();

  // Swap alignments if requested (for transpose case)
  llvm::SmallVector<int64_t> newDimAlignments(dimAlignments.begin(),
                                              dimAlignments.end());
  if (swapAlignments && newDimAlignments.size() >= 2) {
    std::swap(newDimAlignments[newDimAlignments.size() - 2],
              newDimAlignments[newDimAlignments.size() - 1]);
  }

  auto newLayout = ttcore::MetalLayoutAttr::get(
      builder.getContext(), oldLayout.getLogicalShape(), targetGridShape,
      oldLayout.getOobVal(), oldLayout.getMemorySpace(),
      oldLayout.getMemoryLayout(), collapsedIntervals, newDimAlignments);

  // Get optimal sharded shape with new layout
  llvm::SmallVector<int64_t> shardedShape = newLayout.getDeviceShape(
      optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

  // Create a new EmptyOp with the optimal sharded shape and new layout
  builder.setInsertionPoint(emptyOp);
  Type newElementType =
      tileShape.empty()
          ? elementType
          : ttcore::TileType::get(elementType, llvm::ArrayRef(tileShape));
  auto newEmptyOp = builder.create<d2m::EmptyOp>(emptyOp.getLoc(), shardedShape,
                                                 newElementType, newLayout);

  // Create a new ToLayoutOp with the correct output type
  builder.setInsertionPoint(toLayoutOp);
  auto newToLayoutOp = builder.create<d2m::ToLayoutOp>(
      toLayoutOp.getLoc(), toLayoutOp.getInput(), newEmptyOp);

  // Replace all uses of the old ToLayoutOp with the new one
  toLayoutOp.getResult(0).replaceAllUsesWith(newToLayoutOp.getResult(0));

  // Erase the old ops
  toLayoutOp.erase();
  if (emptyOp.getResult().use_empty()) {
    emptyOp.erase();
  }
}

// Assign optimized grids to all ToLayoutOps feeding into a GenericOp.
// Mirrors the old TTIRToD2M behavior: compute optimal grid per tensor
// independently.
static void assignGrids(d2m::GenericOp genericOp,
                        ArrayRef<int64_t> targetGridShape,
                        ArrayRef<int64_t> targetSquareGridShape) {
  OpBuilder builder(genericOp->getContext());

  // Collect operations that need grid optimization
  struct ToLayoutUpdateInfo {
    d2m::ToLayoutOp op;
    llvm::SmallVector<int64_t> grid;
    bool swapLogicalShape;
  };
  llvm::SmallVector<ToLayoutUpdateInfo> toLayoutsToUpdate;
  llvm::SmallVector<std::pair<d2m::StreamLayoutOp, llvm::SmallVector<int64_t>>>
      streamLayoutsToUpdate;

  for (Value operand : genericOp.getOperands()) {
    // Get the actual operand type to compute optimal grid
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
    llvm::SmallVector<int64_t> physShape = operandLayout.getPhysicalShape(
        llvm::ArrayRef(tileShape.data(), tileShape.size()));

    // Compute optimal grid for this operand's physical shape
    llvm::SmallVector<int64_t> optimalGrid =
        computeOptimalGrid(physShape, targetSquareGridShape);

    // Now figure out what operation to update
    if (auto streamLayout = operand.getDefiningOp<d2m::StreamLayoutOp>()) {
      // Check if storage has an index_map
      auto storageType = mlir::cast<mlir::RankedTensorType>(
          streamLayout.getStorage().getType());
      auto storageLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(storageType.getEncoding());
      bool hasIndexMap = static_cast<bool>(storageLayout.getIndexAffineMap());

      // Update both storage and input
      streamLayoutsToUpdate.push_back({streamLayout, optimalGrid});
      if (auto toLayoutOp =
              streamLayout.getInput().getDefiningOp<d2m::ToLayoutOp>()) {
        if (!toLayoutOp.getInput()
                 .getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
          // If storage has index_map, input needs swapped logical shape
          toLayoutsToUpdate.push_back({toLayoutOp, optimalGrid, hasIndexMap});
        }
      }
    } else if (auto toLayoutOp = operand.getDefiningOp<d2m::ToLayoutOp>()) {
      // Skip TTNN tensors
      if (toLayoutOp.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        continue;
      }
      toLayoutsToUpdate.push_back({toLayoutOp, optimalGrid, false});
    }
  }

  if (toLayoutsToUpdate.empty() && streamLayoutsToUpdate.empty()) {
    return; // Nothing to optimize
  }

  // Update each ToLayoutOp with its own optimal grid
  for (auto &info : toLayoutsToUpdate) {
    optimizeToLayoutGrid(info.op, targetGridShape, targetSquareGridShape,
                         info.grid, builder, info.swapLogicalShape);
  }

  // Update each StreamLayoutOp by recreating its storage with the new grid
  for (auto [streamLayout, optimalGrid] : streamLayoutsToUpdate) {
    auto storageEmpty = streamLayout.getStorage().getDefiningOp<d2m::EmptyOp>();
    if (!storageEmpty) {
      continue;
    }

    auto storageType =
        mlir::cast<mlir::RankedTensorType>(storageEmpty.getType());
    auto storageLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(storageType.getEncoding());

    // For storage with index_map that transposes, use transposed logical shape
    llvm::SmallVector<int64_t> storageLogicalShape;
    bool hasIndexMap = static_cast<bool>(storageLayout.getIndexAffineMap());
    if (hasIndexMap && storageLayout.getLogicalShape().size() == 2) {
      // Transpose the logical shape for 2D layouts with index_map
      storageLogicalShape = {storageLayout.getLogicalShape()[1],
                             storageLayout.getLogicalShape()[0]};
    } else {
      storageLogicalShape = llvm::to_vector(storageLayout.getLogicalShape());
    }

    // Compute new dim_alignments for the optimal grid
    llvm::SmallVector<int64_t> normalizedIntervals;
    auto collapsedIntervals = storageLayout.getCollapsedIntervals();
    if (collapsedIntervals) {
      auto intervalsArray = collapsedIntervals.getValues<int64_t>();
      normalizedIntervals.assign(intervalsArray.begin(), intervalsArray.end());
    } else {
      int64_t rank = storageLogicalShape.size();
      if (rank >= 2) {
        normalizedIntervals = {0, rank - 1, rank - 1, rank};
      } else {
        normalizedIntervals = {0, rank};
      }
    }

    llvm::SmallVector<int64_t> storageDimAlignments =
        ttcore::MetalLayoutAttr::computeAlignments(
            storageLogicalShape, optimalGrid, normalizedIntervals);

    // Create new storage layout with optimal grid and alignments
    auto newStorageLayout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), storageLogicalShape, targetGridShape,
        storageLayout.getOobVal(), storageLayout.getMemorySpace(),
        storageLayout.getMemoryLayout(), collapsedIntervals,
        storageDimAlignments);

    // If there's an index_map, add it
    if (hasIndexMap) {
      newStorageLayout = ttcore::MetalLayoutAttr::get(
          builder.getContext(), storageLogicalShape, targetGridShape,
          storageLayout.getOobVal(), storageLayout.getMemorySpace(),
          storageLayout.getMemoryLayout(), storageLayout.getIndexAffineMap());
    }

    // Compute new storage shape
    llvm::SmallVector<int64_t> tileShape;
    if (auto tileType =
            mlir::dyn_cast<ttcore::TileType>(storageType.getElementType())) {
      tileShape = llvm::to_vector(tileType.getShape());
    }
    llvm::SmallVector<int64_t> newStorageShape =
        newStorageLayout.getDeviceShape(
            optimalGrid, llvm::ArrayRef(tileShape.data(), tileShape.size()));

    // Create new storage EmptyOp
    builder.setInsertionPoint(storageEmpty);
    Type elementType = tileShape.empty()
                           ? storageType.getElementType()
                           : ttcore::TileType::get(storageType.getElementType(),
                                                   llvm::ArrayRef(tileShape));
    auto newStorageEmpty = builder.create<d2m::EmptyOp>(
        storageEmpty.getLoc(), newStorageShape, elementType, newStorageLayout);

    // Create new stream_layout
    builder.setInsertionPoint(streamLayout);
    auto newStreamLayout = builder.create<d2m::StreamLayoutOp>(
        streamLayout.getLoc(), newStorageEmpty.getType(),
        streamLayout.getInput(), newStorageEmpty);

    streamLayout.getResult().replaceAllUsesWith(newStreamLayout.getResult());
    streamLayout.erase();

    if (storageEmpty.use_empty()) {
      storageEmpty.erase();
    }
  }

  // Collect the updated operands from the generic
  llvm::SmallVector<Value> newOperands;
  for (Value operand : genericOp.getOperands()) {
    newOperands.push_back(operand);
  }

  // Recreate the generic with the new operands
  {
    auto numInputs = genericOp.getNumDpsInputs();

    llvm::SmallVector<Value> newInputs(newOperands.begin(),
                                       newOperands.begin() + numInputs);
    llvm::SmallVector<Value> newOutputs(newOperands.begin() + numInputs,
                                        newOperands.end());

    // Clone the region body
    Region &oldRegion = genericOp.getRegion(0);

    builder.setInsertionPoint(genericOp);
    auto newGenericOp = builder.create<d2m::GenericOp>(
        genericOp.getLoc(), newInputs, newOutputs, genericOp.getIndexingMaps(),
        genericOp.getIteratorTypes(),
        [&](OpBuilder &b, Location loc, ValueRange blockArgs) {
          // Clone the old region body into the new generic
          IRMapping mapping;
          Block &oldBlock = oldRegion.front();
          for (auto [oldArg, newArg] :
               llvm::zip(oldBlock.getArguments(), blockArgs)) {
            mapping.map(oldArg, newArg);
          }
          // Clone all operations including the terminator
          for (Operation &op : oldBlock) {
            Operation *clonedOp = b.clone(op, mapping);

            // For linalg.generic, update result types to match output operand
            // types
            if (clonedOp->getName().getStringRef() == "linalg.generic") {
              auto numInputs = clonedOp->getAttrOfType<mlir::DenseI32ArrayAttr>(
                  "operandSegmentSizes");
              if (numInputs && numInputs.size() >= 2) {
                int32_t numIns = numInputs[0];
                int32_t numOuts = numInputs[1];

                // Update result types to match output operand types
                for (uint32_t i = 0; static_cast<int32_t>(i) < numOuts &&
                                     i < clonedOp->getNumResults();
                     ++i) {
                  auto outputOperandType =
                      clonedOp->getOperand(numIns + i).getType();
                  clonedOp->getResult(i).setType(outputOperandType);
                }
              }
            }
          }
        },
        mlir::cast<d2m::ThreadAttr>(genericOp.getThreads()[0]).getThreadType(),
        /*grid=*/nullptr, // Let it infer from output
        llvm::to_vector(
            llvm::map_range(genericOp.getBlockFactors(), [](Attribute attr) {
              return mlir::cast<mlir::IntegerAttr>(attr).getInt();
            })));

    // Replace uses and erase old generic
    genericOp.replaceAllUsesWith(newGenericOp);
    genericOp.erase();
  }
}

// ----------------------------------------------------------------------------
// Pass implementation
// ----------------------------------------------------------------------------

#define GEN_PASS_DEF_D2MGRIDOPTIMIZATION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGridOptimizationPass final
    : public impl::D2MGridOptimizationBase<D2MGridOptimizationPass> {
public:
  using Base = impl::D2MGridOptimizationBase<D2MGridOptimizationPass>;

  D2MGridOptimizationPass() = default;

  D2MGridOptimizationPass(const D2MGridOptimizationOptions &options) : Base() {
    this->overrideDeviceShape = llvm::to_vector(options.overrideDeviceShape);
    this->enableDebugLogging = options.enableDebugLogging;
    this->ttnnMode = options.ttnnMode;
  }

  void runOnOperation() override {
    // Skip grid optimization in TTNN mode - grids are already set by TTNN
    if (ttnnMode) {
      return;
    }

    ModuleOp module = getOperation();

    // Get target grid shape from device or override
    llvm::SmallVector<int64_t> targetGridShape = getTargetGridShape();
    llvm::SmallVector<int64_t> targetSquareGridShape =
        d2m::utils::getSquareTargetGrid(targetGridShape);

    // Walk all GenericOps and optimize their grids
    module.walk([&](d2m::GenericOp genericOp) {
      assignGrids(genericOp, targetGridShape, targetSquareGridShape);
    });
  }

private:
  // Helper to get defined device shape if an override is not provided.
  llvm::SmallVector<int64_t> getTargetGridShape() {
    if (!overrideDeviceShape.empty()) {
      return llvm::to_vector(overrideDeviceShape);
    }

    // Get from device if no override given.
    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
  }
};
} // namespace

} // namespace mlir::tt::d2m
