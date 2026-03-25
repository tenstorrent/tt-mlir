// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Recursively erase dead operations from a block after thread splitting.
// Standard type-based filtering only removes remote/DMA ops, leaving behind
// orphaned CB management ops (wait, pop, get_cb, etc.) whose results are
// unused. This function performs targeted DCE to clean them up.
static void eraseDeadOpsAfterSplit(PatternRewriter &rewriter, Block *block) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toErase;

    std::function<void(Block *)> collect = [&](Block *b) {
      for (Operation &op : b->getOperations()) {
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          collect(forOp.getBody());
          continue;
        }

        // Pure ops (GetCBOp, CoreIndexOp, arith.constant,
        // memref.collapse_shape, etc.) with no uses are trivially dead.
        if (isOpTriviallyDead(&op)) {
          toErase.push_back(&op);
          changed = true;
          continue;
        }

        // WaitOp, ReserveOp, and AcquireDstOp declare memory effects for
        // the CB semaphore protocol, but are semantically dead when their
        // results are unused in this thread. Ops like tile_tilize_block
        // also have unused results but perform real in-place writes and
        // must NOT be erased.
        if (op.use_empty() && isa<WaitOp, ReserveOp, AcquireDstOp>(&op)) {
          toErase.push_back(&op);
          changed = true;
          continue;
        }

        // Pop/push have no results, so standard DCE can't detect them as dead.
        // Check if the CB they reference has any real users (wait, reserve,
        // l1_copy, remote_load/store) — if not, the pop/push is orphaned.
        Value cb;
        if (auto popOp = dyn_cast<PopOp>(&op)) {
          cb = popOp.getCb();
        } else if (auto pushOp = dyn_cast<PushOp>(&op)) {
          cb = pushOp.getCb();
        }
        if (cb) {
          bool hasRealUsers =
              llvm::any_of(cb.getUsers(), [&op](Operation *user) {
                return user != &op && !isa<PopOp, PushOp>(user);
              });
          if (!hasRealUsers) {
            toErase.push_back(&op);
            changed = true;
          }
        }
      }
    };

    collect(block);

    for (Operation *op : llvm::reverse(toErase)) {
      rewriter.eraseOp(op);
    }
  }
}

class D2MSplitUnifiedThreadRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Only match GenericOp with a single region (unified compute thread form)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    // Check if the single region is a unified thread
    if (generic.getRegionThreadType(0) != ThreadType::Unified) {
      return failure();
    }

    // Create a new GenericOp with 2 regions: datamovement first, then compute
    SmallVector<Attribute> threads;
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
        generic.getBlockFactors(), generic.getIndexingMaps(),
        generic.getIteratorTypes(), rewriter.getArrayAttr(threads),
        generic.getScratchInputsAttr(),
        /*numRegions*/ 2);

    // Get the original region
    Region &originalRegion = generic.getRegion(0);
    if (originalRegion.empty()) {
      return failure();
    }
    Block *originalBlock = &originalRegion.front();

    // Check that there are no illegal semaphore ops in the unified thread.
    // Replicating these across two threads would create a race condition on the
    // shared semaphore.
    if (failed(utils::checkForIllegalSemaphoreOps(originalBlock))) {
      return failure();
    }

    // Create blocks for both new regions with the same arguments
    Block *datamovementBlock = &newGeneric.getRegion(0).emplaceBlock();
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();

    // Copy semaphore block arguments to both new blocks.
    IRMapping datamovementMapping;
    IRMapping computeMapping;
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      BlockArgument origArg = originalBlock->getArgument(i);
      assert(mlir::isa<d2m::SemaphoreType>(origArg.getType()) &&
             "region block arguments must be of semaphore type");
      auto dmArg =
          datamovementBlock->addArgument(origArg.getType(), generic.getLoc());
      auto cmpArg =
          computeBlock->addArgument(origArg.getType(), generic.getLoc());
      datamovementMapping.map(origArg, dmArg);
      computeMapping.map(origArg, cmpArg);
    }

    // Clone all operations to both regions (excluding terminators for now)
    rewriter.setInsertionPointToStart(datamovementBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, datamovementMapping);
    }

    rewriter.setInsertionPointToStart(computeBlock);
    for (Operation &op : originalBlock->without_terminator()) {
      rewriter.clone(op, computeMapping);
    }

    // Clone terminators if they exist
    if (originalBlock->mightHaveTerminator()) {
      Operation *terminator = originalBlock->getTerminator();
      rewriter.setInsertionPointToEnd(datamovementBlock);
      rewriter.clone(*terminator, datamovementMapping);
      rewriter.setInsertionPointToEnd(computeBlock);
      rewriter.clone(*terminator, computeMapping);
    }

    // Helper function to recursively collect all operations in a block and its
    // nested regions (excluding scf.for ops themselves, but including their
    // body contents)
    std::function<void(Block *, DenseSet<Operation *> &, bool)>
        collectOpsToErase = [&](Block *block, DenseSet<Operation *> &eraseSet,
                                bool keepRemoteOps) {
          for (Operation &op : block->getOperations()) {
            Operation *opPtr = &op;

            // Skip terminators
            if (opPtr->hasTrait<OpTrait::IsTerminator>()) {
              continue;
            }

            // Never erase scf.for operations - they have nested regions.
            // Recursively process their body instead.
            if (auto forOp = dyn_cast<scf::ForOp>(opPtr)) {
              collectOpsToErase(forOp.getBody(), eraseSet, keepRemoteOps);
              continue;
            }

            bool isRemoteOp = isa<RemoteLoadOp, RemoteStoreOp, L1CopyOp>(opPtr);
            // Semaphore waits are replicated: preserved in both threads.
            bool isReplicatedOp = isa<SemaphoreWaitOp>(opPtr);
            if (keepRemoteOps) {
              // In datamovement region: keep RemoteLoadOp, RemoteStoreOp, and
              // replicated ops; erase everything else.
              if (!isRemoteOp && !isReplicatedOp) {
                eraseSet.insert(opPtr);
              }
            } else {
              // In compute region: remove RemoteLoadOp and RemoteStoreOp, keep
              // everything else (including replicated ops).
              if (isRemoteOp) {
                eraseSet.insert(opPtr);
              }
            }
          }
        };

    // Helper function to iteratively erase operations that should be removed
    // We keep erasing operations with no uses (or uses only by ops we're
    // erasing) until no more can be erased
    auto eraseOpsIteratively = [&](Block *block, bool keepRemoteOps) {
      bool changed = true;
      while (changed) {
        changed = false;
        DenseSet<Operation *> eraseSet;
        SmallVector<Operation *> toErase;

        // First pass: recursively identify all operations that should be
        // erased (based on type), walking into all nested scf.for loops
        collectOpsToErase(block, eraseSet, keepRemoteOps);

        // Second pass: only erase operations that have no uses
        // Operations with uses will be handled by canonicalization
        for (Operation *opPtr : eraseSet) {
          // Only erase operations that have no uses
          // Operations with uses (like wait/reserve used by tile_matmul_block)
          // will be handled by canonicalization after their users are erased
          if (opPtr->use_empty()) {
            toErase.push_back(opPtr);
            changed = true;
          }
        }

        // Erase operations in reverse order
        for (Operation *op : llvm::reverse(toErase)) {
          rewriter.eraseOp(op);
        }
      }
    };

    // Filter operations in datamovement region: keep only RemoteLoadOp and
    // RemoteStoreOp (preserve loops and terminators)
    eraseOpsIteratively(datamovementBlock, /*keepRemoteOps=*/true);

    // Filter operations in compute region: remove RemoteLoadOp and
    // RemoteStoreOp (preserve loops and terminators)
    eraseOpsIteratively(computeBlock, /*keepRemoteOps=*/false);

    // Clean up orphaned CB ops (dead waits, pops, get_cbs) left behind
    // after type-based filtering removed their consumers.
    eraseDeadOpsAfterSplit(rewriter, datamovementBlock);
    eraseDeadOpsAfterSplit(rewriter, computeBlock);

    rewriter.replaceOp(generic, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MSplitUnifiedThread
    : public impl::D2MSplitUnifiedThreadBase<D2MSplitUnifiedThread> {
public:
  using impl::D2MSplitUnifiedThreadBase<
      D2MSplitUnifiedThread>::D2MSplitUnifiedThreadBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MSplitUnifiedThreadRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
