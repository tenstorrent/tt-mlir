// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MSPLITUNIFIEDTHREAD
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MSplitUnifiedThreadRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    // Only match GenericOp with a single region (unified compute thread form)
    if (generic.getNumRegions() != 1) {
      return failure();
    }

    // Check if the single region is a compute thread
    if (generic.getRegionThreadType(0) != ThreadType::Compute) {
      return failure();
    }

    // Create a new GenericOp with 2 regions: datamovement first, then compute
    SmallVector<Attribute> threads;
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));

    auto newGeneric = rewriter.create<GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        generic.getOutputs(), generic.getGrid(), generic.getBlockFactors(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        rewriter.getArrayAttr(threads), /*numRegions*/ 2);

    // Get the original region
    Region &originalRegion = generic.getRegion(0);
    Block *originalBlock = &originalRegion.front();

    // Create blocks for both new regions with the same arguments
    Block *datamovementBlock = &newGeneric.getRegion(0).emplaceBlock();
    Block *computeBlock = &newGeneric.getRegion(1).emplaceBlock();

    // Add arguments to both blocks matching the original region
    SmallVector<Type> argTypes(originalBlock->getArgumentTypes().begin(),
                               originalBlock->getArgumentTypes().end());
    SmallVector<Location> argLocs(argTypes.size(), generic.getLoc());
    datamovementBlock->addArguments(argTypes, argLocs);
    computeBlock->addArguments(argTypes, argLocs);

    // Create IR mappings for both regions
    IRMapping datamovementMapping;
    IRMapping computeMapping;

    // Map block arguments
    for (unsigned i = 0; i < originalBlock->getNumArguments(); ++i) {
      datamovementMapping.map(originalBlock->getArgument(i),
                              datamovementBlock->getArgument(i));
      computeMapping.map(originalBlock->getArgument(i),
                         computeBlock->getArgument(i));
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

    // Helper function to iteratively erase operations that should be removed
    // We keep erasing operations with no uses (or uses only by ops we're
    // erasing) until no more can be erased
    auto eraseOpsIteratively = [&](Block *block, bool keepRemoteOps) {
      bool changed = true;
      while (changed) {
        changed = false;
        DenseSet<Operation *> eraseSet;
        SmallVector<Operation *> toErase;

        // First pass: identify all operations that should be erased (based on
        // type)
        block->walk([&](Operation *op) {
          // Skip terminators
          if (op->hasTrait<OpTrait::IsTerminator>()) {
            return;
          }
          // Preserve loops (they have d2m.outer_loop attribute) - but filter
          // their contents
          if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            if (forOp->hasAttr("d2m.outer_loop")) {
              // Don't erase the loop itself, but continue walking to filter its
              // contents
              return;
            }
          }
          bool isRemoteOp = isa<RemoteLoadOp, RemoteStoreOp>(op);
          if (keepRemoteOps) {
            // In datamovement region: keep RemoteLoadOp and RemoteStoreOp,
            // erase everything else
            if (!isRemoteOp) {
              eraseSet.insert(op);
            }
          } else {
            // In compute region: remove RemoteLoadOp and RemoteStoreOp, keep
            // everything else
            if (isRemoteOp) {
              eraseSet.insert(op);
            }
          }
        });

        // Second pass: only erase operations that have no uses
        // Operations with uses will be handled by canonicalization
        block->walk([&](Operation *op) {
          if (!eraseSet.contains(op)) {
            return;
          }
          // Only erase operations that have no uses
          // Operations with uses (like wait/reserve used by tile_matmul_block)
          // will be handled by canonicalization after their users are erased
          if (op->use_empty()) {
            toErase.push_back(op);
            changed = true;
          }
        });

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
