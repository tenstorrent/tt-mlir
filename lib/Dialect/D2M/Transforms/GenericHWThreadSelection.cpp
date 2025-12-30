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

  D2MGenericMergeDatamovementThreadsRewritePattern(MLIRContext *context)
      : OpRewritePattern<GenericOp>(context) {}

  // Returns true if the region contains only a d2m.wait
  // associated with the output CB.
  static bool isLocalPopForOutputCB(Region &region,
                                    unsigned outputOperandsIndex) {
    // Only check trivial regions (single block, single operation).
    if (region.getBlocks().size() != 1) {
      return false;
    }
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

    // Count datamovement threads - merge if there's more than 1.
    size_t numDatamovementThreads =
        llvm::count_if(op.getThreads(), [](Attribute threadAttr) {
          return mlir::cast<ThreadAttr>(threadAttr).getThreadType() ==
                 ThreadType::Datamovement;
        });
    if (numDatamovementThreads <= 1) {
      return failure();
    }

    // Check if the last thread in op has ThreadType::Compute
    ThreadAttr lastThreadAttr =
        mlir::cast<ThreadAttr>(op.getThreads()[op.getThreads().size() - 1]);
    bool hasComputeThread =
        lastThreadAttr && lastThreadAttr.getThreadType() == ThreadType::Compute;

    // The final merged region will always have 1 datamovement thread
    // and (if present in original op) a compute thread
    SmallVector<Attribute> threads(
        1, rewriter.getAttr<ThreadAttr>(ThreadType::Datamovement));
    if (hasComputeThread) {
      threads.push_back(rewriter.getAttr<ThreadAttr>(ThreadType::Compute));
    }

    size_t numNewRegions = threads.size();

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threads), numNewRegions);

    // Copy compute region from op to newGeneric, if it exists.
    if (hasComputeThread) {
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegions().back().takeBody(op.getRegions().back());
      });
    }

    // Copy the first DMA thread to newGeneric. Merging to an empty block
    // results in discarding cb block args, so we must copy the first block
    // instead of uniformly merging everything.
    rewriter.modifyOpInPlace(
        op, [&] { newGeneric.getRegion(0).takeBody(op.getRegion(0)); });

    // Track whether we've wrapped the DMA region's initial ops.
    bool wrappedDmaRegion = false;

    // Merge all remaining DMA threads into the single DMA region.
    // IMPORTANT: output DMA blocks MUST be merged after all input DMA blocks.
    //
    // When merging threads, ensure the ops from each thread are scoped in an
    // scf.execute_region op. This greatly simplifies the logic for inserting CB
    // ops and avoids deadlocks in reader-writer kernels.
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    // Iterate through all remaining DMA thread regions (skip region 0 which we
    // already copied).
    for (unsigned i = 1; i < numDatamovementThreads; ++i) {
      // Skip if region doesn't exist or is empty.
      if (i >= op.getNumRegions() || op.getRegion(i).empty()) {
        continue;
      }
      // Get the entry block of the source region (may have multiple blocks
      // after lower-dmas pass).
      Block *mergeSrcBlock = &op.getRegion(i).front();

      unsigned dmaMergeTargetIndex = 0;
      if (hasComputeThread &&
          isLocalPopForOutputCB(op.getRegion(i), outputOperandsIndex)) {
        // Merge trivial output DMA into compute region, if it exists.
        dmaMergeTargetIndex = newGeneric->getNumRegions() - 1;
      } else {
        // Always merge into the single DMA region (index 0).
        dmaMergeTargetIndex = 0;
      }
      Block *mergeDestBlock =
          &newGeneric.getRegion(dmaMergeTargetIndex).front();

      // If this is the first merge into the DMA region, wrap the existing ops
      // first.
      if (dmaMergeTargetIndex == 0 && !wrappedDmaRegion) {
        wrapBlockOpsInExecuteRegion(rewriter, *mergeDestBlock, op.getLoc());
        wrappedDmaRegion = true;
      }

      // Wrap the source block's operations in execute_region before merging.
      rewriter.setInsertionPointToEnd(mergeDestBlock);
      auto executeRegionOp =
          rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), TypeRange{});
      // Prevent canonicalization from inlining the execute_region op.
      executeRegionOp->setAttr("no_inline", rewriter.getUnitAttr());

      // Merge the source block into the execute_region block, mapping the
      // source block arguments to the parent (destination) block arguments.
      // The scf.execute_region will have access to any dominating block
      // arguments.
      Block *executeRegionBlock = &executeRegionOp.getRegion().emplaceBlock();
      rewriter.mergeBlocks(mergeSrcBlock, executeRegionBlock,
                           mergeDestBlock->getArguments());
      rewriter.setInsertionPointToEnd(executeRegionBlock);
      rewriter.create<scf::YieldOp>(op.getLoc());
    }

    rewriter.replaceOp(op, newGeneric.getResults());

    return success();
  }

  void wrapBlockOpsInExecuteRegion(PatternRewriter &rewriter, Block &block,
                                   Location loc) const {
    Block *executeRegionBlock = rewriter.splitBlock(&block, block.begin());
    rewriter.setInsertionPointToStart(&block);
    auto executeRegionOp = rewriter.create<scf::ExecuteRegionOp>(
        loc, TypeRange{}, /*no_inline=*/rewriter.getUnitAttr());

    // splitBlock creates the executeRegionBlock in the parent region of
    // "block".
    executeRegionBlock->getParent()->getBlocks().remove(executeRegionBlock);
    executeRegionOp.getRegion().push_back(executeRegionBlock);

    rewriter.setInsertionPointToEnd(executeRegionBlock);
    rewriter.create<scf::YieldOp>(loc);
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
        &getContext());
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
