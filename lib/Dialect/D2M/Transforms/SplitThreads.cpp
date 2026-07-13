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
// derive the port from the CB the op references: the local buffer (for remote
// ops) / dst (for local_copy) operand is itself a d2m.generic operand, and its
// operand index is the CB port -- matching the operand index the compute side
// keys its CBs on.
static unsigned getDMACBPort(GenericOp generic, Operation *op) {
  return llvm::TypeSwitch<Operation *, unsigned>(op)
      .Case<RemoteLoadOp, RemoteStoreOp>([&](auto dma) -> unsigned {
        if (dma.isExplicitCBForm()) {
          return dma.getCBPort();
        }
        return generic.getOperandIndex(dma.getLocalBuffer());
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

// The top-level loop nest `op` belongs to: the outermost ancestor that is a
// direct child of the region block, if that ancestor is an scf.for. Accesses at
// the region top level (a one-time CB init, or a bare tile op) are not in a
// loop nest and return null -- they sit at depth 0, compatible with any single
// nest.
static Operation *topLevelNest(Operation *op, Block *regionBlock) {
  Operation *a = op;
  while (a->getBlock() != regionBlock) {
    a = a->getParentOp();
  }
  return isa<scf::ForOp>(a) ? a : nullptr;
}

// Does `operand` read (consume) the memref it references? Only reads impose the
// wait/pop obligation whose scope we are checking; writes are produced under a
// separate reserve/push and may legally span sibling nests.
static bool isCBReadOperand(Operation *op, OpOperand &operand) {
  if (isa<memref::LoadOp>(op)) {
    return operand.getOperandNumber() == 0; // source memref
  }
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    return dps.isDpsInput(&operand); // linalg.generic ins
  }
  if (auto sync = dyn_cast<SynchronizableOpInterface>(op)) {
    return sync.isConsumer(operand); // tile ops
  }
  return false;
}

// Verify that each CB's synchronization scope is well-formed for
// thread-splitting. Three invariants:
//
//  1. DMA-side confinement: a CB's data-movement markers occupy a single block.
//     A marker per block is a discrete transfer with its own reserve/push (or
//     wait/pop); a CB whose markers straddle two loop nests can't be balanced
//     by one per-block sync pair and would deadlock.
//  2. Consumer-side confinement: a CB *consumed* (read) by compute must have
//  all
//     its reads in a single top-level nest. A consumed CB gets one wait/pop
//     pair; reads spread across sibling nests can't be dominated/post-dominated
//     by it and would deadlock on a wait/pop cadence mismatch.
//  3. Nesting well-formedness: only scf.for / linalg.generic sit between a
//     compute op and its sync scope, so d2m-insert-compute-cb can place the
//     sync at a well-defined loop depth.
//
// Only *reads* are confined in (2): producing a CB across sibling nests is
// legal and common (a block-grid reduction writes its data region in one nest
// and identity-fills the padding region in another; matmul accumulates in the
// K-loop and copies out in a sibling loop). A single reserve/push brackets
// those.
static LogicalResult checkComputeSyncScope(GenericOp generic) {
  Region &region = generic.getRegion(0);
  Operation *genericOp = generic.getOperation();
  Block *regionBlock = &region.front();

  // Map every CB-derived value -- each additionalArg CB and the
  // collapse_shape/subview results viewing it -- to its port, so any op reading
  // a CB is attributable in one walk. Scratch and reduction-scaler
  // additionalArgs are not CBs.
  DenseMap<Value, unsigned> cbValueToPort;
  for (Value cb : generic.getAdditionalArgs()) {
    Operation *def = cb.getDefiningOp();
    if (def && (def->hasAttr("d2m.scratch_buffer") ||
                utils::isReductionScalerBuffer(def))) {
      continue;
    }
    unsigned port = generic.getOperandIndex(cb);
    SmallVector<Value> worklist{cb};
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      cbValueToPort.try_emplace(v, port);
      for (Operation *user : v.getUsers()) {
        if (isa<memref::CollapseShapeOp, memref::SubViewOp>(user)) {
          worklist.push_back(user->getResult(0));
        }
      }
    }
  }

  // One walk gathers: per-CB DMA blocks, per-CB top-level nests that *read* the
  // CB, the parents of synchronizable ops (which bound the nesting climb), and
  // the compute ops to validate.
  llvm::MapVector<unsigned, DenseSet<Block *>> dmaBlocks;
  llvm::MapVector<unsigned, DenseSet<Operation *>> consumerNests;
  DenseSet<Operation *> synchronizableParents;
  SmallVector<Operation *> computeOps;
  region.walk([&](Operation *op) {
    if (isa<SynchronizableOpInterface>(op)) {
      synchronizableParents.insert(op->getParentOp());
    }
    if (isa<memref::CollapseShapeOp, memref::SubViewOp>(op)) {
      return; // a view, not an access
    }
    if (isa<ShardDMAOpInterface>(op)) {
      dmaBlocks[getDMACBPort(generic, op)].insert(op->getBlock());
      return;
    }
    if (Operation *nest = topLevelNest(op, regionBlock)) {
      for (OpOperand &operand : op->getOpOperands()) {
        auto it = cbValueToPort.find(operand.get());
        if (it != cbValueToPort.end() && isCBReadOperand(op, operand)) {
          consumerNests[it->second].insert(nest);
        }
      }
    }
    if (op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      computeOps.push_back(op);
    }
  });

  // (1) DMA-side confinement: each CB's markers occupy a single block.
  for (auto &[port, blocks] : dmaBlocks) {
    if (blocks.size() > 1) {
      return generic.emitOpError()
             << "circular buffer (port " << port
             << ") has data-movement accesses across distinct loop nests; "
                "cross-nest fan-out is not yet supported (would deadlock on a "
                "cadence mismatch)";
    }
  }

  // (2) Consumer-side confinement: each CB's reads occupy a single top-level
  // nest.
  for (auto &[port, nests] : consumerNests) {
    if (nests.size() > 1) {
      return generic.emitOpError()
             << "circular buffer (port " << port
             << ") is consumed by compute across distinct loop nests; "
                "cross-nest "
                "fan-out is not yet supported (would deadlock on a wait/pop "
                "cadence mismatch)";
    }
  }

  // (3) Only scf.for / linalg.generic between a compute op and its sync scope.
  for (Operation *op : computeOps) {
    Operation *a = op;
    while (a->getParentOp() != genericOp &&
           !synchronizableParents.contains(a->getParentOp())) {
      a = a->getParentOp();
      if (!isa<scf::ForOp, linalg::GenericOp>(a)) {
        return a->emitOpError("parent ops containing compute ops must be "
                              "scf.for or linalg.generic");
      }
    }
  }

  return success();
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
