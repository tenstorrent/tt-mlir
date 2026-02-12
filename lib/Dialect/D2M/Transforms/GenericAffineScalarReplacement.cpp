// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GenericAffineUtils.h"
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "D2MGenericAffineScalarReplacement"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAFFINESCALARREPLACEMENT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static constexpr llvm::StringLiteral kAffineFusedAttr = "d2m.affine_fused";
static constexpr llvm::StringLiteral kScratchSlotAttr = "d2m.scratch_slot";

struct IntermediateCandidate {
  unsigned inputIdx;
  Value operandVal;
};

/// Check if a GenericOp input operand is an intermediate that can be
/// internalized. An input is intermediate if the only top-level operation
/// that uses it is the GenericOp itself; remote_load/remote_store uses
/// inside generic ops don't count as top-level uses.
static bool isIntermediateInput(GenericOp genericOp, Value operandVal) {
  for (Operation *user : operandVal.getUsers()) {
    if (user == genericOp.getOperation()) {
      continue;
    }
    if (mlir::isa<RemoteLoadOp, RemoteStoreOp>(user) &&
        user->getParentOfType<GenericOp>()) {
      continue;
    }
    return false;
  }
  return true;
}

/// Check that every remote_load from the intermediate inside the generic is
/// dominated by a remote_store to it. This ensures scalrep can forward all
/// loads. If any load appears before a store (e.g., accumulation patterns),
/// internalization is unsafe.
static bool allLoadsStoreDominated(GenericOp genericOp, Value operandVal,
                                   DominanceInfo &domInfo) {
  SmallVector<RemoteStoreOp> stores;
  SmallVector<RemoteLoadOp> loads;

  genericOp.getRegion(0).walk([&](Operation *op) {
    if (auto store = mlir::dyn_cast<RemoteStoreOp>(op)) {
      if (store.getMemref() == operandVal) {
        stores.push_back(store);
      }
      return;
    }

    if (auto load = mlir::dyn_cast<RemoteLoadOp>(op)) {
      if (load.getMemref() == operandVal) {
        loads.push_back(load);
      }
    }
  });

  for (RemoteLoadOp load : loads) {
    bool hasDominatingStore = false;
    for (RemoteStoreOp store : stores) {
      if (domInfo.dominates(store, load)) {
        hasDominatingStore = true;
        break;
      }
    }
    if (!hasDominatingStore) {
      return false;
    }
  }
  return true;
}

static SmallVector<GenericOp> collectFusedGenericOps(func::FuncOp funcOp) {
  SmallVector<GenericOp> genericOps;
  funcOp.walk([&](GenericOp op) {
    if (op->hasAttr(kAffineFusedAttr)) {
      genericOps.push_back(op);
    }
  });
  return genericOps;
}

static SmallVector<IntermediateCandidate>
collectIntermediateCandidates(GenericOp genericOp, DominanceInfo &domInfo) {
  if (genericOp.getRegions().empty() || genericOp.getRegion(0).empty()) {
    return {};
  }

  Block &body = genericOp.getRegion(0).front();
  unsigned numInputs = genericOp.getInputs().size();
  TT_assertv(body.getNumArguments() >= numInputs,
             "generic body has fewer block arguments ({}) than inputs ({})",
             body.getNumArguments(), numInputs);

  SmallVector<IntermediateCandidate> intermediates;
  intermediates.reserve(numInputs);

  for (unsigned i = 0; i < numInputs; ++i) {
    Value operandVal = genericOp.getInputs()[i];
    if (!mlir::isa_and_nonnull<memref::AllocOp>(operandVal.getDefiningOp())) {
      continue;
    }
    if (!isIntermediateInput(genericOp, operandVal)) {
      continue;
    }
    if (!allLoadsStoreDominated(genericOp, operandVal, domInfo)) {
      continue;
    }

    BlockArgument cbArg = body.getArgument(i);
    if (!cbArg.use_empty()) {
      continue;
    }

    intermediates.push_back({i, operandVal});
  }

  return intermediates;
}

static void
materializeIntermediateAllocs(GenericOp genericOp,
                              ArrayRef<IntermediateCandidate> intermediates,
                              OpBuilder &builder) {
  if (intermediates.empty()) {
    return;
  }
  Block &body = genericOp.getRegion(0).front();
  for (const IntermediateCandidate &candidate : intermediates) {
    Value operandVal = candidate.operandVal;
    TT_assertv(mlir::isa<MemRefType>(operandVal.getType()),
               "expected memref intermediate operand type");

    builder.setInsertionPointToStart(&body);
    auto allocOp = builder.create<memref::AllocOp>(
        genericOp.getLoc(), mlir::cast<MemRefType>(operandVal.getType()));
    allocOp->setAttr(
        kScratchSlotAttr,
        builder.getI64IntegerAttr(static_cast<int64_t>(candidate.inputIdx)));

    operandVal.replaceUsesWithIf(allocOp.getResult(), [&](OpOperand &use) {
      return genericOp->isAncestor(use.getOwner());
    });
  }
}

static void
eraseInternalizedInputsAndMaps(GenericOp genericOp,
                               ArrayRef<IntermediateCandidate> intermediates,
                               OpBuilder &builder) {
  if (intermediates.empty()) {
    return;
  }

  Block &body = genericOp.getRegion(0).front();
  unsigned numInputs = genericOp.getInputs().size();
  TT_assertv(body.getNumArguments() >= numInputs,
             "generic body has fewer block arguments ({}) than inputs ({})",
             body.getNumArguments(), numInputs);

  SmallVector<Attribute> maps(genericOp.getIndexingMaps().getValue());
  TT_assertv(maps.size() >= numInputs,
             "indexing_maps has fewer entries ({}) than inputs ({})",
             maps.size(), numInputs);

  SmallVector<unsigned> indices;
  indices.reserve(intermediates.size());
  for (const IntermediateCandidate &candidate : intermediates) {
    indices.push_back(candidate.inputIdx);
  }
  llvm::sort(indices, std::greater<unsigned>());

  for (unsigned idx : indices) {
    TT_assertv(idx < body.getNumArguments(),
               "cannot erase block argument {} with {} arguments", idx,
               body.getNumArguments());
    TT_assertv(idx < maps.size(), "cannot erase indexing_map {} with {} maps",
               idx, maps.size());
    TT_assertv(idx < genericOp.getInputs().size(),
               "cannot erase input {} with {} inputs", idx,
               genericOp.getInputs().size());

    body.eraseArgument(idx);
    maps.erase(maps.begin() + idx);
    genericOp.getInputsMutable().erase(idx);
  }

  genericOp.setIndexingMapsAttr(builder.getArrayAttr(maps));
}

/// Internalize eligible intermediate generic inputs into local allocs and erase
/// their corresponding input operands/CB args/maps. This eliminates the chance
/// of aliasing outside of the generic op, so affine scalar replacement can
/// freely eliminate remote store ops that operate on internalized buffers.
static void internalizeIntermediateOperands(GenericOp genericOp,
                                            DominanceInfo &domInfo,
                                            OpBuilder &builder) {
  SmallVector<IntermediateCandidate> intermediates =
      collectIntermediateCandidates(genericOp, domInfo);
  if (intermediates.empty()) {
    return;
  }
  materializeIntermediateAllocs(genericOp, intermediates, builder);
  eraseInternalizedInputsAndMaps(genericOp, intermediates, builder);
}

/// Temporarily convert remote_load from DPS-like init passing to SSA-direct
/// style. Swaps uses of localBuffer with the load result to build use-chains
/// needed for affine scalar replacement to work properly.
static void convertRemoteLoadToDirectStyle(func::FuncOp funcOp) {
  funcOp.walk([&](RemoteLoadOp loadOp) {
    Value localBuf = loadOp.getLocalBuffer();
    Value result = loadOp.getResult();
    if (!localBuf || !result) {
      return;
    }

    localBuf.replaceUsesWithIf(result, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      if (user == loadOp.getOperation()) {
        return false;
      }
      // Only replace uses that come after the load in the same block.
      return loadOp->getBlock() == user->getBlock() &&
             loadOp->isBeforeInBlock(user);
    });
  });
}

/// Convert RemoteLoadOps from direct style back to destination-passing
/// style: replace post-load uses of the result with the localBuffer
/// operand.
static void convertRemoteLoadToDPSStyle(func::FuncOp funcOp) {
  funcOp.walk([&](RemoteLoadOp loadOp) {
    Value localBuf = loadOp.getLocalBuffer();
    Value result = loadOp.getResult();
    if (!localBuf || !result) {
      return;
    }

    result.replaceUsesWithIf(localBuf, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      if (user == loadOp.getOperation()) {
        return false;
      }
      return loadOp->getBlock() == user->getBlock() &&
             loadOp->isBeforeInBlock(user);
    });
  });
}

/// Return true if the value participates in any remote load/store path
/// reachable through simple memref view/cast forwarding ops.
static bool isRelatedToRemoteLoadStore(Value value) {
  SmallVector<Value> worklist{value};
  llvm::DenseSet<Value> visited;
  visited.insert(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(user)) {
        if (loadOp.getMemref() == current ||
            loadOp.getLocalBuffer() == current) {
          return true;
        }
      }
      if (auto storeOp = mlir::dyn_cast<RemoteStoreOp>(user)) {
        if (storeOp.getMemref() == current ||
            storeOp.getLocalBuffer() == current) {
          return true;
        }
      }

      if (mlir::isa<memref::CastOp, memref::SubViewOp,
                    memref::ReinterpretCastOp, memref::ViewOp>(user)) {
        for (Value result : user->getResults()) {
          if (visited.insert(result).second) {
            worklist.push_back(result);
          }
        }
      }
    }
  }
  return false;
}

struct ScratchAllocCandidate {
  memref::AllocOp allocOp;
  int64_t slot;
};

static int64_t computeNextScratchSlot(GenericOp genericOp) {
  int64_t maxSlot = -1;
  genericOp.walk([&](Operation *op) {
    if (auto scratchOp = mlir::dyn_cast<ScratchAllocateOp>(op)) {
      maxSlot = std::max(maxSlot, static_cast<int64_t>(scratchOp.getSlot()));
      return;
    }
    if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(op)) {
      if (auto slot = allocOp->getAttrOfType<IntegerAttr>(kScratchSlotAttr)) {
        maxSlot = std::max(maxSlot, slot.getInt());
      }
    }
  });
  return maxSlot + 1;
}

/// Replace generic-local intermediate memref.alloc ops with
/// d2m.scratch_allocate. Internalized intermediates (tagged with
/// d2m.scratch_slot) are always lowered. Additionally, untagged generic-local
/// intermediates that are not related to any remote load/store are lowered as
/// scratch allocations. Scratch allocs are inserted at the top of the generic
/// unified region.
static void replaceGenericIntermediateAllocsWithScratch(func::FuncOp funcOp,
                                                        OpBuilder &builder) {
  for (GenericOp genericOp : collectFusedGenericOps(funcOp)) {
    if (genericOp.getRegions().empty() || genericOp.getRegion(0).empty()) {
      continue;
    }

    int64_t nextSlot = computeNextScratchSlot(genericOp);
    SmallVector<ScratchAllocCandidate> candidates;
    SmallVector<memref::AllocOp> deadAllocs;
    genericOp.getRegion(0).walk([&](memref::AllocOp allocOp) {
      if (allocOp.getResult().use_empty()) {
        deadAllocs.push_back(allocOp);
        return;
      }
      if (auto slot = allocOp->getAttrOfType<IntegerAttr>(kScratchSlotAttr)) {
        candidates.push_back({allocOp, slot.getInt()});
        return;
      }
      if (!isRelatedToRemoteLoadStore(allocOp.getResult())) {
        candidates.push_back({allocOp, nextSlot++});
      }
    });

    if (candidates.empty() && deadAllocs.empty()) {
      continue;
    }

    Block &body = genericOp.getRegion(0).front();
    SmallVector<Value> replacementValues;
    replacementValues.reserve(candidates.size());
    for (ScratchAllocCandidate &candidate : candidates) {
      builder.setInsertionPointToStart(&body);
      auto scratchOp = builder.create<ScratchAllocateOp>(
          candidate.allocOp.getLoc(), candidate.allocOp.getType(),
          builder.getI64IntegerAttr(candidate.slot));
      replacementValues.push_back(scratchOp.getResult());
    }

    for (auto [candidate, replacement] :
         llvm::zip_equal(candidates, replacementValues)) {
      candidate.allocOp.replaceAllUsesWith(replacement);
      candidate.allocOp.erase();
    }

    for (memref::AllocOp deadAlloc : deadAllocs) {
      if (deadAlloc.getResult().use_empty()) {
        deadAlloc.erase();
      }
    }
  }
}

static void internalizeFusedGenericIntermediates(func::FuncOp funcOp,
                                                 DominanceInfo &domInfo,
                                                 OpBuilder &builder) {
  for (GenericOp op : collectFusedGenericOps(funcOp)) {
    internalizeIntermediateOperands(op, domInfo, builder);
  }
}

static void convertFusedGenericsToAffineCompatibilityForm(func::FuncOp funcOp,
                                                          OpBuilder &builder) {
  for (GenericOp op : collectFusedGenericOps(funcOp)) {
    utils::convertToAffineCompatibilityForm(op, builder);
  }
}

static void
restoreFusedGenericsFromAffineCompatibilityForm(func::FuncOp funcOp,
                                                OpBuilder &builder) {
  for (GenericOp op : collectFusedGenericOps(funcOp)) {
    utils::convertFromAffineCompatibilityForm(op, builder);
  }
}

class D2MGenericAffineScalarReplacement
    : public impl::D2MGenericAffineScalarReplacementBase<
          D2MGenericAffineScalarReplacement> {
public:
  using D2MGenericAffineScalarReplacementBase::
      D2MGenericAffineScalarReplacementBase;

  void runOnOperation() final {
    if (!enable) {
      return;
    }
    getOperation()->walk([&](func::FuncOp funcOp) {
      OpBuilder builder(funcOp.getContext());

      auto &domInfo = getAnalysis<DominanceInfo>();

      // Internalize generic op inputs that are only referenced locally into
      // local memref.alloc
      internalizeFusedGenericIntermediates(funcOp, domInfo, builder);

      // Rewrite all fused generics into an affine-compatible form and move
      // remote_load into direct style so scalrep can forward through SSA.
      convertFusedGenericsToAffineCompatibilityForm(funcOp, builder);
      convertRemoteLoadToDirectStyle(funcOp);

      // Run the core affine scalar replacement transform.
      auto &postDomInfo = getAnalysis<PostDominanceInfo>();
      auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
      affine::affineScalarReplace(funcOp, domInfo, postDomInfo, aliasAnalysis);

      // Restore remote_load back to destination-passing style and then rewrite
      // generics back into regular generic op form.
      convertRemoteLoadToDPSStyle(funcOp);
      restoreFusedGenericsFromAffineCompatibilityForm(funcOp, builder);

      // Lower generic-local intermediate allocs to d2m.scratch_allocate.
      replaceGenericIntermediateAllocsWithScratch(funcOp, builder);
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
