// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPTIMIZEDMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers for DMA barrier ordering analysis.
// These are shared across all DMA scheduling optimizations.
//===----------------------------------------------------------------------===//

// Trace a local buffer Value back to its CB through the SSA def chain.
// Handles both producer-side (reserve → CB) and consumer-side (wait → CB).
static Value getCBFromLocalBuffer(Value localBuffer) {
  if (!localBuffer) {
    return nullptr;
  }
  if (auto reserveOp = localBuffer.getDefiningOp<ReserveOp>()) {
    return reserveOp.getCb();
  }
  if (auto cbWaitOp = localBuffer.getDefiningOp<WaitOp>()) {
    return cbWaitOp.getCb();
  }
  return nullptr;
}

// Get the CB associated with any DMA or CB-protocol op by tracing through
// the SSA def chain. For DMA ops, follows: dma_read/write → local buffer →
// reserve/wait → CB. For DMAWaitOp, follows the extra hop through mem_tx.
static Value getCBForOp(Operation *op) {
  if (!op) {
    return nullptr;
  }
  if (auto reserveOp = dyn_cast<ReserveOp>(op)) {
    return reserveOp.getCb();
  }
  if (auto pushOp = dyn_cast<PushOp>(op)) {
    return pushOp.getCb();
  }
  if (auto popOp = dyn_cast<PopOp>(op)) {
    return popOp.getCb();
  }
  if (auto waitOp = dyn_cast<WaitOp>(op)) {
    return waitOp.getCb();
  }
  if (auto dmaRead = dyn_cast<DMAReadOp>(op)) {
    return getCBFromLocalBuffer(dmaRead.getDst());
  }
  if (auto dmaWrite = dyn_cast<DMAWriteOp>(op)) {
    return getCBFromLocalBuffer(dmaWrite.getSrc());
  }
  if (auto dmaWaitOp = dyn_cast<DMAWaitOp>(op)) {
    return getCBForOp(dmaWaitOp.getMemTx().getDefiningOp());
  }
  return nullptr;
}

// Returns true only when both CBs can be resolved and are definitively
// different. Returns false when either is null (conservative: unknown CBs
// might alias, so treat as potentially conflicting).
static bool areDifferentCBs(Value a, Value b) { return a && b && a != b; }

static bool isReadBarrier(DMAWaitOp waitOp) {
  return isa_and_nonnull<DMAReadOp>(waitOp.getMemTx().getDefiningOp());
}

//===----------------------------------------------------------------------===//
// canSinkPast - core ordering legality check.
//
// Returns true if `toSink` can be moved past `sinkOver` in block order without
// violating DMA ordering constraints. This is the shared primitive for all
// DMA scheduling optimizations (barrier sinking, write barrier deferral, etc.).
//
// Dispatches to per-type helpers below. Each helper returns true (safe to sink)
// unless `sinkOver` matches a specific blocker for that op type.
//===----------------------------------------------------------------------===//

// Barrier (dma_wait) can't sink past:
//   - another dma_wait (preserves barrier ordering, prevents oscillation)
//   - dma_read/write of same CB (data hazard)
//   - read barrier: push of same CB, any semaphore op
//   - write barrier: pop of same CB, semaphore_set
static bool canBarrierSinkPast(DMAWaitOp dmaWait, Operation *sinkOver,
                               Value sinkCB) {
  if (isa<DMAWaitOp>(sinkOver)) {
    return false;
  }
  if (isa<DMAReadOp, DMAWriteOp>(sinkOver)) {
    return areDifferentCBs(getCBForOp(sinkOver), sinkCB);
  }
  if (isReadBarrier(dmaWait)) {
    if (auto pushOp = dyn_cast<PushOp>(sinkOver)) {
      return areDifferentCBs(pushOp.getCb(), sinkCB);
    }
    if (isa<SemaphoreSetOp, SemaphoreWaitOp, SemaphoreIncOp>(sinkOver)) {
      return false;
    }
  } else {
    if (auto popOp = dyn_cast<PopOp>(sinkOver)) {
      return areDifferentCBs(popOp.getCb(), sinkCB);
    }
    if (isa<SemaphoreSetOp>(sinkOver)) {
      return false;
    }
  }
  return true;
}

// Push can't sink past:
//   - reserve of same CB (reserve needs push to free the slot)
//   - another push of same CB
//   - dma_wait of same CB (barrier must complete before push signals readiness)
static bool canPushSinkPast(Operation *sinkOver, Value sinkCB) {
  if (auto reserveOp = dyn_cast<ReserveOp>(sinkOver)) {
    return areDifferentCBs(reserveOp.getCb(), sinkCB);
  }
  if (auto otherPush = dyn_cast<PushOp>(sinkOver)) {
    return areDifferentCBs(otherPush.getCb(), sinkCB);
  }
  if (isa<DMAWaitOp>(sinkOver)) {
    return areDifferentCBs(getCBForOp(sinkOver), sinkCB);
  }
  return true;
}

// Pop can't sink past:
//   - wait (CB) of same CB (wait needs pop to release the slot)
//   - another pop of same CB
//   - dma_wait of same CB (barrier must complete before pop releases the slot)
static bool canPopSinkPast(Operation *sinkOver, Value sinkCB) {
  if (auto waitOp = dyn_cast<WaitOp>(sinkOver)) {
    return areDifferentCBs(waitOp.getCb(), sinkCB);
  }
  if (auto otherPop = dyn_cast<PopOp>(sinkOver)) {
    return areDifferentCBs(otherPop.getCb(), sinkCB);
  }
  if (isa<DMAWaitOp>(sinkOver)) {
    return areDifferentCBs(getCBForOp(sinkOver), sinkCB);
  }
  return true;
}

static bool canSinkPast(Operation *toSink, Operation *sinkOver) {
  if (sinkOver->getNumRegions() > 0) {
    return false;
  }

  for (Value result : sinkOver->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (user == toSink) {
        return false;
      }
    }
  }

  Value sinkCB = getCBForOp(toSink);
  if (!sinkCB) {
    return false;
  }

  if (auto dmaWait = dyn_cast<DMAWaitOp>(toSink)) {
    return canBarrierSinkPast(dmaWait, sinkOver, sinkCB);
  }
  if (isa<PushOp>(toSink)) {
    return canPushSinkPast(sinkOver, sinkCB);
  }
  if (isa<PopOp>(toSink)) {
    return canPopSinkPast(sinkOver, sinkCB);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// sinkBarriers - barrier coalescing via paired fence sinking.
//
// Sinks each dma_wait together with its associated push/pop as an atomic unit.
// This groups NOC transactions together before barriers fire, allowing the
// hardware to overlap multiple transfers, while ensuring push/pop fires as
// soon as its barrier completes so compute can begin immediately.
//
// Desired block order after sinking:
//   reserves/waits_cb → dma_reads/writes → (dma_wait + push/pop) pairs
//===----------------------------------------------------------------------===//

static void sinkBarriers(Block &block) {
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<DMAWaitOp> barriers;
    for (Operation &op : block) {
      if (auto wait = dyn_cast<DMAWaitOp>(&op)) {
        barriers.push_back(wait);
      }
    }

    for (DMAWaitOp barrier : barriers) {
      // Check if this barrier has a matching push/pop for the same CB
      // immediately after it. They sink together as a unit.
      Operation *cbPushPop = barrier->getNextNode();
      Value barrierCB = getCBForOp(barrier);
      bool hasCBPushPop = cbPushPop && isa<PushOp, PopOp>(cbPushPop) &&
                          barrierCB && getCBForOp(cbPushPop) == barrierCB;

      Operation *next =
          hasCBPushPop ? cbPushPop->getNextNode() : barrier->getNextNode();
      if (!next || next->hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }

      // Both the barrier and its push/pop must be able to sink past next.
      if (!canSinkPast(barrier, next)) {
        continue;
      }
      if (hasCBPushPop && !canSinkPast(cbPushPop, next)) {
        continue;
      }

      barrier->moveAfter(next);
      if (hasCBPushPop) {
        cbPushPop->moveAfter(barrier);
      }
      changed = true;
    }
  }
}

// Defer write barriers from the current loop iteration to the next iteration.
static void deferWriteBarriers(scf::ForOp forOp) {
  // TODO(vtang): Implement write barrier deferral (Optimization 2).
}

// Defer mcast write barriers from the current loop iteration to the next.
static void deferMcastWriteBarriers(scf::ForOp forOp) {
  // TODO(vtang): Implement mcast write barrier deferral (Optimization 3).
}

class D2MOptimizeDMA : public impl::D2MOptimizeDMABase<D2MOptimizeDMA> {
public:
  using impl::D2MOptimizeDMABase<D2MOptimizeDMA>::D2MOptimizeDMABase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([](GenericOp generic) {
      for (Region &region : generic.getRegions()) {
        if (region.empty()) {
          continue;
        }
        if (generic.getRegionThreadType(region.getRegionNumber()) !=
            ThreadType::Datamovement) {
          continue;
        }

        for (Block &block : region) {
          sinkBarriers(block);
        }
        region.walk([](Operation *op) {
          for (Region &r : op->getRegions()) {
            for (Block &b : r) {
              sinkBarriers(b);
            }
          }
        });

        region.walk([](scf::ForOp forOp) {
          deferWriteBarriers(forOp);
          deferMcastWriteBarriers(forOp);
        });
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
