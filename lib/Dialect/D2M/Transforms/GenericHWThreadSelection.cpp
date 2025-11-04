// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICHWTHREADSELECTION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericMergeDatamovementThreadsRewritePattern
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  unsigned numDMAHWThreads;

  D2MGenericMergeDatamovementThreadsRewritePattern(MLIRContext *context,
                                                   unsigned numDMAHWThreads)
      : OpRewritePattern<GenericOp>(context), numDMAHWThreads(numDMAHWThreads) {
    assert(numDMAHWThreads > 0 && "numDMAHWThreads must be greater than 0");
  }

  // Returns true if the region contains only a d2m.wait
  // associated with the output CB.
  static bool isLocalPopForOutputCB(Region &region,
                                    unsigned outputOperandsIndex) {
    assert(region.getBlocks().size() == 1);
    auto &block = region.front();
    if (block.getOperations().size() != 1) {
      return false;
    }

    d2m::WaitOp WaitOp = dyn_cast<d2m::WaitOp>(block.front());
    if (!WaitOp) {
      return false;
    }

    BlockArgument blockArg = mlir::dyn_cast<BlockArgument>(WaitOp.getCb());
    if (!blockArg) {
      return false;
    }
    assert(blockArg.getOwner() == &block);

    return blockArg.getArgNumber() == outputOperandsIndex;
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {

    // Ops with # threads <= # DMA HW Threads don't need to be merged.
    size_t numDatamovementThreads =
        llvm::count_if(op.getThreads(), [](Attribute threadAttr) {
          return mlir::cast<ThreadAttr>(threadAttr).getThreadType() ==
                 ThreadType::Datamovement;
        });
    if (numDatamovementThreads <= numDMAHWThreads) {
      return failure();
    }

    // Check if the last thread in op has ThreadType::Compute
    ThreadAttr lastThreadAttr =
        mlir::cast<ThreadAttr>(op.getThreads()[op.getThreads().size() - 1]);
    bool hasComputeThread =
        lastThreadAttr && lastThreadAttr.getThreadType() == ThreadType::Compute;

    // The final merged region will always have numDMAHWThreads datamovement
    // threads and (if present in original op) a compute thread
    SmallVector<Attribute> threads(
        numDMAHWThreads,
        rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    if (hasComputeThread) {
      threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));
    }

    size_t numNewRegions = threads.size();

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threads), numNewRegions);

    // Helper function to wrap operations in a block with scf.execute_region.
    auto wrapBlockOpsInExecuteRegion = [&](Block &block, Location loc) {
      // Collect all operations in the block.
      SmallVector<Operation *> opsToWrap;
      for (Operation &blockOp : llvm::make_early_inc_range(block)) {
        opsToWrap.push_back(&blockOp);
      }

      if (opsToWrap.empty()) {
        return;
      }

      // Create scf.execute_region at the start of the block.
      rewriter.setInsertionPointToStart(&block);
      auto executeRegionOp =
          rewriter.create<scf::ExecuteRegionOp>(loc, TypeRange{});
      executeRegionOp->setAttr("no_inline", rewriter.getUnitAttr());

      // Create block in execute_region WITHOUT arguments.
      // The operations will use the parent block's arguments directly.
      Block *executeRegionBlock = &executeRegionOp.getRegion().emplaceBlock();

      // Move all operations into the execute_region.
      for (Operation *opToMove : opsToWrap) {
        opToMove->moveBefore(executeRegionBlock, executeRegionBlock->end());
      }

      // Add yield operation to terminate the execute_region.
      rewriter.setInsertionPointToEnd(executeRegionBlock);
      rewriter.create<scf::YieldOp>(loc);
    };

    // Copy compute region from op to newGeneric, if it exists.
    // Do NOT wrap it yet - only wrap if blocks get merged into it.
    if (hasComputeThread) {
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegions().back().takeBody(op.getRegions().back());
      });
    }

    // Copy initial dma threads to newGeneric WITHOUT wrapping.
    // We'll wrap them on-demand when blocks are merged into them.
    unsigned regionIndex = 0;
    for (unsigned i = 0; i < numDMAHWThreads; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegion(regionIndex++).takeBody(op.getRegion(i));
      });
    }

    // Track which regions have already had their initial ops wrapped.
    llvm::DenseSet<unsigned> wrappedRegions;

    // Merge remaining DMA threads into existing DMA regions in newGeneric.
    // IMPORTANT: output DMA blocks MUST be merged after all input DMA blocks in
    // a region.
    unsigned dmaInputThreadsMerged = 0;
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    for (unsigned i = numDMAHWThreads; i <= outputOperandsIndex; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      Block *mergeSrcBlock = &op.getRegion(i).front();

      unsigned dmaMergeTargetIndex = 0;
      if (hasComputeThread &&
          isLocalPopForOutputCB(op.getRegion(i), outputOperandsIndex)) {
        // Merge trivial output DMA into compute region, if it exists.
        dmaMergeTargetIndex = newGeneric->getNumRegions() - 1;
      } else if (i == outputOperandsIndex) {
        // Always merge output to last DMA region (associated with NOC1).
        dmaMergeTargetIndex = numDMAHWThreads - 1;
      } else {
        // Merge input DMA threads to alternate DMA regions.
        dmaMergeTargetIndex = dmaInputThreadsMerged % numDMAHWThreads;
        dmaInputThreadsMerged++;
      }
      Block *mergeDestBlock =
          &newGeneric.getRegion(dmaMergeTargetIndex).front();

      // If this is the first merge into this region, wrap the existing ops
      // first.
      if (!wrappedRegions.contains(dmaMergeTargetIndex)) {
        wrapBlockOpsInExecuteRegion(*mergeDestBlock, op.getLoc());
        wrappedRegions.insert(dmaMergeTargetIndex);
      }

      // Wrap the source block's operations in execute_region before merging.
      rewriter.setInsertionPointToEnd(mergeDestBlock);
      auto executeRegionOp =
          rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), TypeRange{});
      executeRegionOp->setAttr("no_inline", rewriter.getUnitAttr());

      // Create block in execute_region WITHOUT arguments.
      // The operations will use the parent block's arguments directly.
      Block *executeRegionBlock = &executeRegionOp.getRegion().emplaceBlock();

      // Merge the source block into the execute_region block, mapping the
      // source block arguments to the parent (destination) block arguments.
      rewriter.mergeBlocks(mergeSrcBlock, executeRegionBlock,
                           mergeDestBlock->getArguments());

      // Add yield operation to terminate the execute_region.
      rewriter.setInsertionPointToEnd(executeRegionBlock);
      rewriter.create<scf::YieldOp>(op.getLoc());
    }

    rewriter.replaceOp(op, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class D2MGenericHWThreadSelection
    : public impl::D2MGenericHWThreadSelectionBase<
          D2MGenericHWThreadSelection> {
public:
  using impl::D2MGenericHWThreadSelectionBase<
      D2MGenericHWThreadSelection>::D2MGenericHWThreadSelectionBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
        mlir::tt::ttcore::SystemDescAttr::name);
    auto chipDesc = systemDesc.getChipDescs().front();

    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericMergeDatamovementThreadsRewritePattern>(
        &getContext(), chipDesc.getNumDatamovementThreads());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    moduleOp.walk([&](GenericOp op) {
      // Assert that the op has a valid HW thread selection.
      if (op.getNumRegions() > (chipDesc.getNumComputeThreads() +
                                chipDesc.getNumDatamovementThreads())) {
        op.emitError("invalid number of regions (")
            << op.getNumRegions() << "), expected at most ("
            << (chipDesc.getNumComputeThreads() +
                chipDesc.getNumDatamovementThreads())
            << ")";
        signalPassFailure();
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
