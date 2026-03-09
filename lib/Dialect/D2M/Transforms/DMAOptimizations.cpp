// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MOPTIMIZEDMA
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

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

static bool areDifferentCBs(Value a, Value b) { return a && b && a != b; }

static bool isReadBarrier(DMAWaitOp waitOp) {
  return isa_and_nonnull<DMAReadOp>(waitOp.getMemTx().getDefiningOp());
}

static bool isWriteBarrier(DMAWaitOp waitOp) {
  return isa_and_nonnull<DMAWriteOp>(waitOp.getMemTx().getDefiningOp());
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

struct WriteBarrierGroup {
  DMAWaitOp barrier;
  Operation *companion; // PopOp or SemaphoreSetOp
  Value writeCB;
};

// Walk backwards from the terminator to find a write barrier group:
// (DMAWaitOp for a DMAWriteOp, PopOp) at the end of a block.
static std::optional<WriteBarrierGroup> findWriteBarrierGroup(Block &body) {
  if (body.empty()) {
    return std::nullopt;
  }
  Operation &lastOp = body.back();
  if (!lastOp.hasTrait<OpTrait::IsTerminator>()) {
    return std::nullopt;
  }

  // Walk backwards: expect pop/semaphore_set then dma_wait right before it.
  Operation *companion = lastOp.getPrevNode();
  if (!companion) {
    return std::nullopt;
  }

  // Non-mcast: companion is PopOp.
  if (!isa<PopOp>(companion)) {
    return std::nullopt;
  }

  Operation *barrierOp = companion->getPrevNode();
  if (!barrierOp) {
    return std::nullopt;
  }

  auto barrier = dyn_cast<DMAWaitOp>(barrierOp);
  if (!barrier || !isWriteBarrier(barrier)) {
    return std::nullopt;
  }

  Value writeCB = getCBForOp(barrier);
  if (!writeCB) {
    return std::nullopt;
  }

  // Verify the companion's CB matches the barrier's CB.
  Value companionCB = getCBForOp(companion);
  if (!companionCB || companionCB != writeCB) {
    return std::nullopt;
  }

  return WriteBarrierGroup{barrier, companion, writeCB};
}

static bool canDeferWriteBarrier(Block &body, Value writeCB) {
  // Write CB must not alias any read CB (RAW hazard).
  auto readCheck = body.walk([&](DMAReadOp dmaRead) -> WalkResult {
    Value readCB = getCBFromLocalBuffer(dmaRead.getDst());
    if (readCB && readCB == writeCB) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (readCheck.wasInterrupted()) {
    return false;
  }

  // Write CB must be unique among writes.
  unsigned writeCount = 0;
  auto writeCheck = body.walk([&](DMAWriteOp dmaWrite) -> WalkResult {
    Value otherCB = getCBFromLocalBuffer(dmaWrite.getSrc());
    if (otherCB && otherCB == writeCB && ++writeCount > 1) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (writeCheck.wasInterrupted()) {
    return false;
  }

  return true;
}

static Operation *findDeferralInsertionPoint(Block &body, Value writeCB) {
  Operation *lastRead = nullptr;
  for (Operation &op : body) {
    if (auto waitOp = dyn_cast<WaitOp>(&op)) {
      if (waitOp.getCb() == writeCB) {
        return &op;
      }
    }
    if (isa<DMAReadOp>(&op)) {
      lastRead = &op;
    }
  }
  return lastRead ? lastRead->getNextNode() : &body.front();
}

//===----------------------------------------------------------------------===//
// Defer one write barrier group from the end of a loop body to the next
// iteration. Adds an iter_arg for the deferred mem_tx, inserts an
// scf.if(iv != lb) guard, and adds an epilogue.
//
// Returns true if a barrier was deferred (forOp is updated to the new ForOp).
// Returns false if no deferrable write barrier was found.
//
// Assumes the loop executes at least once (lb < ub). For a
// zero-iteration loop, the epilogue would incorrectly pop from an unfilled CB.
//===----------------------------------------------------------------------===//
static bool deferOneWriteBarrier(scf::ForOp &forOp) {
  if (auto tripCount = forOp.getStaticTripCount()) {
    TT_assertv(tripCount->isStrictlyPositive(),
               "write barrier deferral requires loop to execute at least once");
  }

  Block &body = *forOp.getBody();

  auto group = findWriteBarrierGroup(body);
  if (!group) {
    return false;
  }

  if (!canDeferWriteBarrier(body, group->writeCB)) {
    return false;
  }

  Operation *insertBefore = findDeferralInsertionPoint(body, group->writeCB);
  IRRewriter rewriter(forOp->getContext());
  Location loc = forOp.getLoc();

  rewriter.setInsertionPoint(forOp);
  Value nullTx = rewriter.create<NullTxOp>(loc);

  // Capture the write's tx and CB before erasing the barrier group.
  Value currentTx = group->barrier.getMemTx();
  Value companionCB = cast<PopOp>(group->companion).getCb();

  group->companion->erase();
  group->barrier->erase();

  auto result = forOp.replaceWithAdditionalYields(
      rewriter, /*newInitOperands=*/{nullTx},
      /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
        return SmallVector<Value>{currentTx};
      });
  auto newFor = cast<scf::ForOp>(*result);
  forOp = newFor;

  Value prevTx = newFor.getBody()->getArguments().back();

  // Insert the deferred guard.
  rewriter.setInsertionPoint(insertBefore);
  Value iv = newFor.getInductionVar();
  Value lb = newFor.getLowerBound();
  Value notFirst =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, iv, lb);
  auto ifOp =
      rewriter.create<scf::IfOp>(loc, notFirst, /*withElseRegion=*/false);

  rewriter.setInsertionPointToStart(ifOp.thenBlock());
  rewriter.create<DMAWaitOp>(loc, prevTx);
  rewriter.create<PopOp>(loc, companionCB);

  // Epilogue: handle the final iteration's deferred write barrier.
  rewriter.setInsertionPointAfter(newFor);
  Value finalTx = newFor.getResults().back();
  rewriter.create<DMAWaitOp>(loc, finalTx);
  rewriter.create<PopOp>(loc, companionCB);

  return true;
}

// Defer write barriers from the current loop iteration to the next iteration.
// Terminates because each iteration removes exactly one barrier group from the
// block until no more barriers are found.
static void deferWriteBarriers(scf::ForOp forOp) {
  while (deferOneWriteBarrier(forOp)) {
  }
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

        // Collect ForOps before processing to avoid walk invalidation
        // (deferWriteBarriers erases and replaces the ForOp).
        SmallVector<scf::ForOp> forOps;
        region.walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });
        for (scf::ForOp forOp : forOps) {
          deferWriteBarriers(forOp);
        }
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::d2m
