// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTLOADSTOREOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is remote (i.e., is a stream op)
static bool isRemote(Value operand) {
  // Remote operands are those that come from stream_layout ops
  return mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp());
}

// Calculate multicast iterators based on the grid, operand indexing map, and
// iterator types. Returns an empty vector if all dimensions are parallel
// (no multicast needed).
static SmallVector<ttcore::IteratorType>
calculateMcastIterators(ttcore::GridAttr grid, AffineMap operandIndexingMap,
                        ArrayAttr iteratorTypes) {
  assert(grid.getShape().size() == operandIndexingMap.getNumResults());

  bool allParallel = true;
  SmallVector<ttcore::IteratorType> mcastIterators;
  mcastIterators.reserve(grid.getShape().size());
  for (unsigned dim = 0; dim < grid.getShape().size(); dim++) {
    AffineExpr result = operandIndexingMap.getResult(dim);
    assert(mlir::isa<AffineDimExpr>(result) ||
           mlir::isa<AffineConstantExpr>(result));

    ttcore::IteratorType iteratorType;
    if (auto constant = mlir::dyn_cast<AffineConstantExpr>(result)) {
      assert(constant.getValue() == 0);
      iteratorType = ttcore::IteratorType::Parallel;
    } else {
      auto dimId = mlir::cast<AffineDimExpr>(result);
      unsigned dimPosition = dimId.getPosition();

      iteratorType =
          mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[dimPosition])
              .getValue();
    }
    mcastIterators.push_back(iteratorType);

    // If the grid dimension is 1, we can special case it and always safely
    // fallback to mode parallel. Reduction implies multicast, and while
    // it'll be functionally correct, a multicast with a single core to
    // itself is a redundant copy and more complicated than necessary.
    bool singleCore = grid.getShape()[dim] == 1;
    allParallel &=
        (iteratorType == ttcore::IteratorType::Parallel) || singleCore;
  }

  return allParallel ? SmallVector<ttcore::IteratorType>() : mcastIterators;
}

// Struct to hold multicast arguments for gather-multicast pattern.
struct McastArguments {
  SmallVector<Value> mcastStartIndex;
  SmallVector<Value> mcastShape;
};

// Calculate gather-multicast arguments for RemoteLoadOp.
// For the gather-multicast pattern:
// - Core 0 along each reduction dimension is the sender
// - Sender multicasts to all other cores starting at position 1
// - Multicast shape is grid dimension - 1 to exclude sender from destination
// range
static McastArguments
calculateGatherMcastArguments(OpBuilder &builder, Location loc,
                              ttcore::GridAttr grid,
                              ArrayRef<ttcore::IteratorType> mcastIterators) {
  Value one = builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                                builder.getIndexAttr(1));

  McastArguments args;
  args.mcastStartIndex.reserve(grid.getShape().size());
  args.mcastShape.reserve(grid.getShape().size());

  for (auto [dim, iteratorType] : llvm::enumerate(mcastIterators)) {
    Value core = builder.create<CoreIndexOp>(loc, builder.getIndexType(),
                                             builder.getI64IntegerAttr(dim));
    if (iteratorType == ttcore::IteratorType::Parallel) {
      // Parallel dimension: mcast to self only (start=core, shape=1)
      args.mcastStartIndex.push_back(core);
      args.mcastShape.push_back(one);
    } else {
      // Reduction dimension: mcast to all other cores (start=1,
      // shape=gridDim-1) The sender is at position 0, so multicast starts at
      // position 1 and spans gridDim-1 cores to avoid overlapping with sender
      assert(iteratorType == ttcore::IteratorType::Reduction);
      int64_t numDests = grid.getShape()[dim] - 1;
      Value gridDimMinusOne = builder.create<arith::ConstantOp>(
          loc, builder.getIndexType(), builder.getIndexAttr(numDests));
      args.mcastStartIndex.push_back(one);
      args.mcastShape.push_back(gridDimMinusOne);
    }
  }

  return args;
}

// Helper function to get generic operand and indexing map from CB block
// argument
static std::optional<std::pair<Value, AffineMap>>
getGenericOperandAndIndexingMap(Operation *op, Value cbValue) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return std::nullopt;
  }

  // Skip if generic op is in explicit datamovement form (no indexing maps)
  if (generic.isExplicitDatamovementForm()) {
    return std::nullopt;
  }

  // Skip if generic op is in DMA-only form (should not have automatic
  // insertion of reserve/pop/push/wait ops)
  if (generic.isDMAOnlyForm()) {
    return std::nullopt;
  }

  BlockArgument cbArg = dyn_cast<BlockArgument>(cbValue);
  if (!cbArg) {
    return std::nullopt;
  }

  unsigned operandIndex = cbArg.getArgNumber();
  Value genericOperand = generic->getOperand(operandIndex);

  // Verify operand is a memref (post-bufferization)
  if (!isa<MemRefType>(genericOperand.getType())) {
    return std::nullopt;
  }

  // Get the indexing map
  AffineMap indexingMap = generic.getIndexingMap(operandIndex);
  return std::make_pair(genericOperand, indexingMap);
}

class D2MInsertLoadStoreOps
    : public impl::D2MInsertLoadStoreOpsBase<D2MInsertLoadStoreOps> {
public:
  using impl::D2MInsertLoadStoreOpsBase<
      D2MInsertLoadStoreOps>::D2MInsertLoadStoreOpsBase;

  void runOnOperation() final {
    // Check for illegal DMAOp operations
    WalkResult result = getOperation()->walk([&](DMAOp dmaOp) {
      dmaOp.emitOpError("d2m.dma operations are not supported by this pass");
      return WalkResult::interrupt();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(&getContext());

    // Build a map of pre-existing remote_load and remote_store operations
    // associated with each CB to avoid inserting duplicates
    DenseSet<Value> cbsWithRemoteLoad;
    DenseSet<Value> cbsWithRemoteStore;

    getOperation()->walk([&](RemoteLoadOp remoteLoadOp) {
      Value cbValue = remoteLoadOp.getCb();
      cbsWithRemoteLoad.insert(cbValue);
    });

    getOperation()->walk([&](RemoteStoreOp remoteStoreOp) {
      Value cbValue = remoteStoreOp.getCb();
      cbsWithRemoteStore.insert(cbValue);
    });

    // Collect remote reserve operations that need remote_store at end of block
    struct RemoteStoreInfo {
      Block *block;
      Location loc;
      Value remoteMemref;
      AffineMap indexingMap;
      Value cbValue;
    };
    SmallVector<RemoteStoreInfo> remoteStores;

    // PHASE 1: Process all REMOTE operands first
    // This ensures remote_load/remote_store operations are inserted before
    // processing local operands, avoiding confusion with existing wait/reserve
    // operations

    // Process WaitOp operations for remote operands
    getOperation()->walk([&](WaitOp waitOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = waitOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = waitOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(waitOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;

      if (isRemote(genericOperand)) {
        // Remote case: insert remote_load before wait only if one doesn't
        // already exist for this CB
        if (!cbsWithRemoteLoad.contains(cbValue)) {
          GenericOp generic = waitOp->getParentOfType<GenericOp>();
          builder.setInsertionPoint(waitOp);
          Location loc = waitOp.getLoc();
          SmallVector<Value> indices =
              utils::buildGridIndices(builder, loc, indexingMap);

          // Check if multicast is needed based on iterator types
          SmallVector<ttcore::IteratorType> mcastIterators =
              calculateMcastIterators(generic.getGrid(), indexingMap,
                                      generic.getIteratorTypes());

          if (!mcastIterators.empty()) {
            // Multicast needed: calculate mcast arguments and create
            // multicast variant
            McastArguments mcastArgs = calculateGatherMcastArguments(
                builder, loc, generic.getGrid(), mcastIterators);
            builder.create<RemoteLoadOp>(loc, cbValue, genericOperand, indices,
                                         mcastArgs.mcastStartIndex,
                                         mcastArgs.mcastShape);
          } else {
            // No multicast needed: create simple remote_load
            builder.create<RemoteLoadOp>(loc, cbValue, genericOperand, indices);
          }
        }
      }
      // Local case handled in PHASE 2
    });

    // Process ReserveOp operations for remote operands
    getOperation()->walk([&](ReserveOp reserveOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = reserveOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = reserveOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(reserveOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;
      Location loc = reserveOp.getLoc();

      if (isRemote(genericOperand)) {
        // Remote case: collect info to insert remote_store at end of block
        // only if one doesn't already exist for this CB
        if (!cbsWithRemoteStore.contains(cbValue)) {
          // Use the stream_layout result (genericOperand) as the remote memref
          remoteStores.push_back({reserveOp->getBlock(), loc, genericOperand,
                                  indexingMap, cbValue});
        }
      }
      // Local case handled in PHASE 2
    });

    // Insert remote_store operations at the end of their respective blocks
    // Note: multicast inference is not well defined for remote store
    // operations, so we only create simple (non-multicast) remote_store ops
    // here.
    for (const auto &info : remoteStores) {
      builder.setInsertionPoint(info.block, info.block->end());
      SmallVector<Value> indices =
          utils::buildGridIndices(builder, info.loc, info.indexingMap);
      builder.create<RemoteStoreOp>(info.loc, info.remoteMemref, indices,
                                    info.cbValue);
    }

    // PHASE 2: Process all LOCAL operands
    // Now that all remote operations have been converted, process local
    // operands without confusion

    // Collect local reserve operations that need wait and pop at end of block
    struct LocalWaitPopInfo {
      Block *block;
      Location loc;
      Value cbValue;
    };
    SmallVector<LocalWaitPopInfo> localWaitPops;

    // Process WaitOp operations for local operands
    getOperation()->walk([&](WaitOp waitOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = waitOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = waitOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(waitOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;

      if (!isRemote(genericOperand)) {
        // Local case: insert reserve and push before wait
        builder.setInsertionPoint(waitOp);
        Location loc = waitOp.getLoc();

        // Check if reserve already exists before this wait
        bool hasReserve = false;
        for (auto &op : waitOp->getBlock()->getOperations()) {
          if (&op == waitOp.getOperation()) {
            break;
          }
          if (auto existingReserve = dyn_cast<ReserveOp>(op)) {
            if (existingReserve.getCb() == cbValue) {
              hasReserve = true;
              break;
            }
          }
        }
        // Check if push already exists before this wait
        bool hasPush = false;
        for (auto &op : waitOp->getBlock()->getOperations()) {
          if (&op == waitOp.getOperation()) {
            break;
          }
          if (auto existingPush = dyn_cast<PushOp>(op)) {
            if (existingPush.getCb() == cbValue) {
              hasPush = true;
              break;
            }
          }
        }
        // Insert operations: reserve first, then push, so they appear as
        // reserve -> push -> wait
        if (!hasReserve) {
          builder.create<ReserveOp>(loc, cbValue);
        }
        if (!hasPush) {
          builder.create<PushOp>(loc, cbValue);
        }
      }
    });

    // Process ReserveOp operations for local operands
    getOperation()->walk([&](ReserveOp reserveOp) {
      // Only process memref types (skip tensor types)
      auto memrefType = reserveOp.getCbType().getUnderlyingAs<MemRefType>();
      if (!memrefType) {
        return;
      }

      Value cbValue = reserveOp.getCb();
      auto genericInfo = getGenericOperandAndIndexingMap(reserveOp, cbValue);
      if (!genericInfo) {
        return;
      }

      auto [genericOperand, indexingMap] = *genericInfo;
      Location loc = reserveOp.getLoc();

      if (!isRemote(genericOperand)) {
        // Local case: collect info to insert wait and pop at end of block
        localWaitPops.push_back({reserveOp->getBlock(), loc, cbValue});
      }
    });

    // Insert wait, pop, and push operations at the end of blocks for local
    // reserve ops Order: wait -> pop -> push
    for (const auto &info : localWaitPops) {
      // Check if push already exists in the block
      bool hasPush = false;
      for (auto &op : info.block->getOperations()) {
        if (auto existingPush = dyn_cast<PushOp>(op)) {
          if (existingPush.getCb() == info.cbValue) {
            hasPush = true;
            break;
          }
        }
      }
      // Check if wait already exists in the block
      bool hasWait = false;
      for (auto &op : info.block->getOperations()) {
        if (auto existingWait = dyn_cast<WaitOp>(op)) {
          if (existingWait.getCb() == info.cbValue) {
            hasWait = true;
            break;
          }
        }
      }
      // Check if pop already exists in the block
      bool hasPop = false;
      for (auto &op : info.block->getOperations()) {
        if (auto existingPop = dyn_cast<PopOp>(op)) {
          if (existingPop.getCb() == info.cbValue) {
            hasPop = true;
            break;
          }
        }
      }
      // Insert operations at end: wait -> pop -> push
      builder.setInsertionPoint(info.block, info.block->end());
      if (!hasWait) {
        builder.create<WaitOp>(info.loc, info.cbValue);
      }
      if (!hasPop) {
        builder.create<PopOp>(info.loc, info.cbValue);
      }
      if (!hasPush) {
        builder.create<PushOp>(info.loc, info.cbValue);
      }
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
