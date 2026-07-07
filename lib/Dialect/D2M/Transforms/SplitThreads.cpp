// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MTraits.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITTHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static constexpr StringRef kThreadAttrName = "d2m.thread";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// The CB port a shard DMA op synchronizes on, form-agnostic. Post-
// d2m-assign-threads most DMA ops are in explicit CB form (getCBPort() works),
// but *aliased* remote_load/store ops are intentionally left implicit -- they
// carry no transfer and are turned into compute-side CB obligations later by
// d2m-insert-compute-cb. getCBPort() asserts on implicit form, so for those we
// derive the port from the generic operand the op references: the memref (for
// remote ops) / dst (for local_copy) operand is itself a d2m.generic operand,
// and its operand index is the CB port.
static unsigned getDMACBPort(GenericOp generic, Operation *op) {
  return llvm::TypeSwitch<Operation *, unsigned>(op)
      .Case<RemoteLoadOp, RemoteStoreOp>([&](auto dma) -> unsigned {
        if (dma.isExplicitCBForm()) {
          return dma.getCBPort();
        }
        return generic.getOperandIndex(dma.getMemref());
      })
      .Case<LocalCopyOp>([&](LocalCopyOp copy) -> unsigned {
        if (copy.isExplicitCBForm()) {
          return copy.getCBPort();
        }
        return generic.getOperandIndex(copy.getDst());
      })
      .Default([](Operation *) -> unsigned {
        llvm_unreachable("unexpected ShardDMAOpInterface op");
      });
}

// Verify each CB's synchronization scope is well-formed. A sync scope is a
// property of each CB, not the whole region: the wait/pop (or reserve/push)
// cadence is set by the loop block its DMA marker lives in. The only
// unsupported case is a *single* CB whose markers sit in distinct loop nests --
// one per-block sync pair can't balance that and would deadlock. Different CBs
// in different nests are fine (e.g. matmul: A/B stream in the K-loop while C is
// stored once per output tile in the persistent loop).
static LogicalResult checkComputeSyncScope(GenericOp generic) {
  // Parents of synchronizable ops bound the raw-span climb below: a compute op
  // never climbs past a loop that itself carries a DMA marker.
  DenseSet<Operation *> opsWithSynchronizableOps;
  generic.getRegion(0).walk([&](Operation *op) {
    if (dyn_cast<SynchronizableOpInterface>(op)) {
      opsWithSynchronizableOps.insert(op->getParentOp());
    }
  });

  // A CB whose markers span more than one block lives in distinct loop nests, which we can't synchronize. 
  llvm::MapVector<unsigned, DenseSet<Block *>> cbMarkerBlocks;
  generic.getRegion(0).walk([&](ShardDMAOpInterface dma) {
    cbMarkerBlocks[getDMACBPort(generic, dma)].insert(dma->getBlock());
  });
  
  for (auto &[cbPort, blocks] : cbMarkerBlocks) {
    if (blocks.size() > 1) {
      return generic.emitOpError()
             << "circular buffer (port " << cbPort
             << ") is used across distinct loop nests; cross-nest fan-out is "
                "not yet supported (would deadlock on a wait/pop cadence "
                "mismatch)";
    }
  }

  LogicalResult result = success();
  generic.getRegion(0).walk([&](Operation *op) {
    if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      return WalkResult::advance();
    }
    Operation *outermostOp = op;
    while (outermostOp->getParentOp() != generic.getOperation() &&
           !opsWithSynchronizableOps.contains(outermostOp->getParentOp())) {
      outermostOp = outermostOp->getParentOp();
      if (!mlir::isa<scf::ForOp>(outermostOp) &&
          !mlir::isa<linalg::GenericOp>(outermostOp)) {
        outermostOp->emitOpError("Parent ops containing compute ops must be "
                                 "scf.for or linalg.generic");
        result = failure();
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

// ---------------------------------------------------------------------------
// DMA thread cleanup
// ---------------------------------------------------------------------------

// Recursively collect non-DMA ops to erase from the datamovement thread.
static void collectComputeOpsToErase(Block *block,
                                     DenseSet<Operation *> &eraseSet) {
  for (Operation &op : block->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      collectComputeOpsToErase(forOp.getBody(), eraseSet);
      continue;
    }
    bool isDMAOp = isa<ShardDMAOpInterface, DeviceSynchronizeOp>(&op);
    bool isReplicated = isa<SemaphoreWaitOp>(&op);
    if (!isDMAOp && !isReplicated) {
      eraseSet.insert(&op);
    }
  }
}

// Iteratively erase compute (non-DMA) ops from the datamovement thread until
// fixpoint. Side-effecting compute ops (memref.store into #dst etc.) are not
// trivially dead, so this targets them by thread membership rather than
// dead-code analysis.
static void eraseComputeOpsInDMThread(RewriterBase &rewriter, Block *dmBlock) {
  bool changed = true;
  while (changed) {
    changed = false;
    DenseSet<Operation *> eraseSet;
    collectComputeOpsToErase(dmBlock, eraseSet);
    SmallVector<Operation *> toErase;
    for (Operation *op : eraseSet) {
      if (op->use_empty()) {
        toErase.push_back(op);
        changed = true;
      }
    }
    for (Operation *op : llvm::reverse(toErase)) {
      rewriter.eraseOp(op);
    }
  }
}

// ---------------------------------------------------------------------------
// Compute thread cleanup
// ---------------------------------------------------------------------------

// Drop the data-movement-tagged ops (streaming remote_load/store, local_copy,
// DMA) from the compute thread, then DCE their now-dead structural feeders.
// Aliased remote markers are tagged compute, so they are preserved as
// compute-side CB obligations for d2m-insert-compute-cb.
static void eraseDMOpsInComputeThread(RewriterBase &rewriter,
                                      Block *computeBlock) {
  SmallVector<Operation *> dmOps;
  computeBlock->walk([&](Operation *op) {
    if (auto thread = op->getAttrOfType<ThreadAttr>(kThreadAttrName)) {
      if (thread.getThreadType() == ThreadType::Datamovement) {
        dmOps.push_back(op);
      }
    }
  });
  for (Operation *op : dmOps) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  // Drop trivially-dead structural feeders. Erase one at a time so erasing a
  // dead parent never dangles a dead child collected earlier.
  bool changed = true;
  while (changed) {
    changed = false;
    Operation *dead = nullptr;
    computeBlock->walk([&](Operation *op) {
      if (isOpTriviallyDead(op)) {
        dead = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (dead) {
      rewriter.eraseOp(dead);
      changed = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Main rewriter
// ---------------------------------------------------------------------------

class D2MSplitThreadsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (generic.getNumRegions() != 1) {
      return failure();
    }
    if (generic.getRegionThreadType(0) != ThreadType::Unified) {
      return failure();
    }

    if (failed(checkComputeSyncScope(generic))) {
      return failure();
    }

    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }
    Block *originalBlock = &originalRegion.front();

    if (failed(utils::checkForIllegalSemaphoreOps(originalBlock))) {
      return failure();
    }

    // Create new 2-region GenericOp: datamovement + compute.
    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(),
        rewriter.getArrayAttr(
            {rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement),
             rewriter.getAttr<ThreadAttr>(ThreadType::Compute)}),
        generic.getFabricConnectionConfigAttr(),
        /*numRegions*/ 2);

    Block *dmBlock = &newGeneric.getRegion(0).emplaceBlock();
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();

    // Map semaphore block arguments to both new blocks.
    IRMapping dmMapping, computeMapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      BlockArgument arg = originalBlock->getArgument(i);
      assert(mlir::isa<d2m::LocalSemaphoreType>(arg.getType()) &&
             "region block arguments must be of local semaphore type");
      dmMapping.map(arg, dmBlock->addArgument(arg.getType(), generic.getLoc()));
      computeMapping.map(
          arg, computeBlock->addArgument(arg.getType(), generic.getLoc()));
    }

    // Clone all ops into both regions, then drop each thread's non-members.
    rewriter.setInsertionPointToStart(dmBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, dmMapping);
    }
    rewriter.setInsertionPointToStart(computeBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, computeMapping);
    }
    if (originalBlock->mightHaveTerminator()) {
      Operation *term = originalBlock->getTerminator();
      rewriter.setInsertionPointToEnd(dmBlock);
      rewriter.clone(*term, dmMapping);
      rewriter.setInsertionPointToEnd(computeBlock);
      rewriter.clone(*term, computeMapping);
    }

    // Datamovement thread: drop all compute ops; keep DMA (including aliased
    // remote markers, which d2m-insert-compute-cb consumes) + replicated ops.
    eraseComputeOpsInDMThread(rewriter, dmBlock);

    // Compute thread: drop all data-movement ops (streaming and aliased remote
    // ops alike -- the verifier requires remote ops to live on the datamovement
    // thread).
    eraseDMOpsInComputeThread(rewriter, computeBlock);

    // Strip the thread-assignment annotations; they are an internal handoff.
    newGeneric->walk([](Operation *op) { op->removeAttr(kThreadAttrName); });

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitThreads : public impl::D2MSplitThreadsBase<D2MSplitThreads> {
public:
  using impl::D2MSplitThreadsBase<D2MSplitThreads>::D2MSplitThreadsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitThreadsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
