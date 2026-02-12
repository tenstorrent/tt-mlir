// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MCONVERTLOCALLOADSTOREOPSTOALIASEDCBS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Helper function to check if an operand is local (i.e., NOT a stream op
// and NOT in a DMA-only generic op)
static bool isLocalOperand(Value operand, Operation *op) {
  // Check if operand comes from stream_layout op
  if (mlir::isa_and_nonnull<StreamLayoutOp>(operand.getDefiningOp())) {
    return false;
  }

  // Check if the operation is inside a DMA-only generic op
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (generic && generic.isDMAOnlyForm()) {
    return false;
  }

  return true;
}

// Helper function to find the CB block argument that corresponds to a memref
// operand in a generic op. Returns the CB block argument if found, null
// otherwise.
// Assumes that the operand index in the generic op equals the CB block arg
// index.
static Value findAssociatedCB(Operation *op, Value memrefOperand) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find which operand index this memref corresponds to.
  std::optional<unsigned> operandIndex;
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == memrefOperand) {
      operandIndex = i;
      break;
    }
  }

  if (!operandIndex.has_value()) {
    return Value();
  }

  // Find the generic op's thread region that contains this operation
  // If there's only one region, use it directly. Otherwise, use the utility
  // function
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }

  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  // Get the first block of the generic region (thread region block)
  Block *threadBlock = &genericRegion->front();

  // The CB block arguments are in the same order as the generic operands.
  // The operand index equals the CB block arg index (confirmed by user).
  if (threadBlock->getNumArguments() > *operandIndex) {
    return threadBlock->getArgument(*operandIndex);
  }

  return Value();
}

// Helper function to recursively walk a block and find the last operation that
// uses a value, including uses in nested regions. Returns the operation after
// which to insert the pop, or null if no uses found.
//
// This function only considers top-level operations in the given block. If a
// value is used anywhere within a nested region (e.g., inside an scf.for or
// scf.if), the function returns the parent operation that contains that region,
// not the operation inside the nested region.
//
// This function also tracks indirect uses through view-like operations (e.g.,
// memref.collapse_shape, memref.expand_shape). If the original value is used to
// create a view, and that view is used elsewhere, this function considers those
// uses as uses of the original value.
static Operation *findLastUse(Value value, Block *block) {
  Operation *lastUse = nullptr;

  // Build a set of all values that are aliases of the input value
  // This includes the value itself and any views derived from it
  llvm::SmallPtrSet<Value, 8> aliasedValues;
  aliasedValues.insert(value);

  // First pass: collect all aliased values (views of the original value)
  // We need to iterate until no new aliases are found, since aliases can be
  // chained (e.g., alloc -> collapse_shape -> cast)
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : *block) {
      // Check if this operation takes an aliased value as input
      bool takesAliasedInput = false;
      for (OpOperand &operand : op.getOpOperands()) {
        if (aliasedValues.contains(operand.get())) {
          takesAliasedInput = true;
          break;
        }
      }

      // If the operation takes an aliased value as input and is a view-like
      // operation, add its results to the alias set
      if (takesAliasedInput && mlir::isa<mlir::ViewLikeOpInterface>(op)) {
        for (Value result : op.getResults()) {
          if (aliasedValues.insert(result).second) {
            changed = true;
          }
        }
      }
    }
  }

  // Helper to check if any aliased value is used anywhere in a region
  // (recursively)
  std::function<bool(Region &)> isUsedInRegion = [&](Region &region) -> bool {
    for (Block &regionBlock : region) {
      for (Operation &op : regionBlock) {
        // Check if this op uses any aliased value
        for (OpOperand &operand : op.getOpOperands()) {
          if (aliasedValues.contains(operand.get())) {
            return true;
          }
        }

        // Recursively check nested regions
        for (Region &nestedRegion : op.getRegions()) {
          if (isUsedInRegion(nestedRegion)) {
            return true;
          }
        }
      }
    }
    return false;
  };

  // Second pass: walk top-level operations to find the last use
  for (Operation &op : *block) {
    bool opUsesValue = false;

    // Check if op directly uses any aliased value
    for (OpOperand &operand : op.getOpOperands()) {
      if (aliasedValues.contains(operand.get())) {
        opUsesValue = true;
        break;
      }
    }

    // Check if any nested regions use any aliased value
    if (!opUsesValue) {
      for (Region &region : op.getRegions()) {
        if (isUsedInRegion(region)) {
          opUsesValue = true;
          break;
        }
      }
    }

    // If this operation (or its nested regions) uses any aliased value, update
    // lastUse
    if (opUsesValue) {
      lastUse = &op;
    }
  }

  return lastUse;
}

// Helper function to find the memref.alloc operation that produces a given
// value, potentially through a chain of operations. Returns the alloc op if
// found, null otherwise.
static memref::AllocOp findAllocOp(Value value) {
  if (!value) {
    return nullptr;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return nullptr;
  }

  // Direct case: value is directly produced by memref.alloc
  if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(definingOp)) {
    return allocOp;
  }

  // Trace through operations that might pass the buffer through
  // (e.g., view-like ops, cast ops, etc.)
  for (Value operand : definingOp->getOperands()) {
    if (auto allocOp = findAllocOp(operand)) {
      return allocOp;
    }
  }

  return nullptr;
}

class D2MConvertLocalLoadStoreOpsToAliasedCBs
    : public impl::D2MConvertLocalLoadStoreOpsToAliasedCBsBase<
          D2MConvertLocalLoadStoreOpsToAliasedCBs> {
public:
  using impl::D2MConvertLocalLoadStoreOpsToAliasedCBsBase<
      D2MConvertLocalLoadStoreOpsToAliasedCBs>::
      D2MConvertLocalLoadStoreOpsToAliasedCBsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect remote_load operations to convert
    SmallVector<RemoteLoadOp> remoteLoadsToConvert;
    moduleOp->walk([&](RemoteLoadOp remoteLoad) {
      Value memref = remoteLoad.getMemref();
      if (isLocalOperand(memref, remoteLoad.getOperation())) {
        // Skip if multicast is present (shouldn't happen for local operands,
        // but verify)
        if (remoteLoad.isMcast()) {
          remoteLoad.emitWarning(
              "remote_load with local operand has multicast parameters, "
              "skipping conversion");
          return;
        }
        remoteLoadsToConvert.push_back(remoteLoad);
      }
    });

    // Convert remote_load operations
    for (RemoteLoadOp remoteLoad : remoteLoadsToConvert) {
      Location loc = remoteLoad.getLoc();
      Value memref = remoteLoad.getMemref();
      Value assocCb = findAssociatedCB(remoteLoad.getOperation(), memref);

      if (!assocCb) {
        remoteLoad.emitWarning(
            "could not find associated CB block argument, skipping conversion");
        continue;
      }

      // Get the local buffer that the remote_load writes to
      Value localBuffer = remoteLoad.getLocalBuffer();
      if (!localBuffer) {
        remoteLoad.emitWarning(
            "remote_load does not have a local buffer operand, skipping "
            "conversion");
        continue;
      }

      // Find the memref.alloc that produces the local buffer
      memref::AllocOp allocOp = findAllocOp(localBuffer);
      if (!allocOp) {
        remoteLoad.emitWarning(
            "could not find memref.alloc for local buffer operand, skipping "
            "conversion");
        continue;
      }

      // Find the last use of the alloc result or remote_load result BEFORE we
      // modify the IR
      Block *block = allocOp->getBlock();
      Operation *lastUseOfAlloc = findLastUse(allocOp.getResult(), block);
      Operation *lastUseOfRemoteLoad =
          findLastUse(remoteLoad.getResult(), block);

      // Exclude the remote_load itself from being considered the last use
      if (lastUseOfAlloc == remoteLoad.getOperation()) {
        lastUseOfAlloc = nullptr;
      }

      // Determine the actual last use (the one that appears later in the
      // block).
      Operation *lastUse = nullptr;
      if (lastUseOfAlloc && lastUseOfRemoteLoad) {
        // Both have uses, find which one is later
        lastUse = lastUseOfAlloc->isBeforeInBlock(lastUseOfRemoteLoad)
                      ? lastUseOfRemoteLoad
                      : lastUseOfAlloc;
      } else {
        lastUse = lastUseOfAlloc ? lastUseOfAlloc : lastUseOfRemoteLoad;
      }

      // Insert reserve, push, and wait where the alloc was, so they dominate
      // all uses of the buffer (including view operations like collapse_shape
      // that may occur before the remote_load)
      rewriter.setInsertionPoint(allocOp);

      // Create reserve, push, and wait operations
      rewriter.create<ReserveOp>(loc, assocCb);
      rewriter.create<PushOp>(loc, assocCb);
      auto waitOp = rewriter.create<WaitOp>(loc, assocCb);

      // Replace all uses of the alloc result and remote_load result with the
      // wait result
      rewriter.replaceAllUsesWith(allocOp.getResult(), waitOp.getResult());
      rewriter.replaceAllUsesWith(remoteLoad.getResult(), waitOp.getResult());

      // Erase the alloc operation
      rewriter.eraseOp(allocOp);

      // Insert pop after the last use
      if (lastUse) {
        rewriter.setInsertionPointAfter(lastUse);
      } else {
        // No uses found, insert pop immediately after wait
        rewriter.setInsertionPointAfter(waitOp);
      }
      rewriter.create<PopOp>(loc, assocCb);

      // Erase the original remote_load operation
      rewriter.eraseOp(remoteLoad);
    }

    // Collect remote_store operations to convert
    SmallVector<RemoteStoreOp> remoteStoresToConvert;
    moduleOp->walk([&](RemoteStoreOp remoteStore) {
      Value memref = remoteStore.getMemref();
      if (isLocalOperand(memref, remoteStore.getOperation())) {
        remoteStoresToConvert.push_back(remoteStore);
      }
    });

    // Convert remote_store operations
    for (RemoteStoreOp remoteStore : remoteStoresToConvert) {
      Location loc = remoteStore.getLoc();
      Value memref = remoteStore.getMemref();
      Value assocCb = findAssociatedCB(remoteStore.getOperation(), memref);

      if (!assocCb) {
        remoteStore.emitWarning(
            "could not find associated CB block argument, skipping conversion");
        continue;
      }

      // Get the local buffer being stored (either explicit or from CB form)
      Value localBuffer = remoteStore.getLocalBuffer();
      if (!localBuffer) {
        remoteStore.emitWarning(
            "remote_store does not have a local buffer operand, skipping "
            "conversion");
        continue;
      }

      // Find the memref.alloc that produces the local buffer being stored
      memref::AllocOp allocOp = findAllocOp(localBuffer);

      if (!allocOp) {
        remoteStore.emitWarning(
            "could not find memref.alloc for local buffer operand, skipping "
            "conversion");
        continue;
      }

      // Replace memref.alloc with reserve
      rewriter.setInsertionPoint(allocOp);
      auto reserveOp = rewriter.create<ReserveOp>(loc, assocCb);
      rewriter.replaceAllUsesWith(allocOp.getResult(), reserveOp.getResult());
      rewriter.eraseOp(allocOp);

      // At remote_store location, insert: push, wait, pop
      rewriter.setInsertionPoint(remoteStore);
      rewriter.create<PushOp>(loc, assocCb);
      rewriter.create<WaitOp>(loc, assocCb);
      rewriter.create<PopOp>(loc, assocCb);

      // Erase the original remote_store operation
      rewriter.eraseOp(remoteStore);
    }

    // Convert get_scratch_from_cb operations to reserve ops.
    SmallVector<GetScratchFromCBOp> scratchOpsToConvert;
    moduleOp->walk([&](GetScratchFromCBOp getScratchOp) {
      scratchOpsToConvert.push_back(getScratchOp);
    });

    for (GetScratchFromCBOp getScratchOp : scratchOpsToConvert) {
      rewriter.setInsertionPoint(getScratchOp);
      auto reserveOp = rewriter.create<ReserveOp>(getScratchOp.getLoc(),
                                                  getScratchOp.getCb());
      rewriter.replaceAllUsesWith(getScratchOp.getResult(),
                                  reserveOp.getResult());
      rewriter.eraseOp(getScratchOp);
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
