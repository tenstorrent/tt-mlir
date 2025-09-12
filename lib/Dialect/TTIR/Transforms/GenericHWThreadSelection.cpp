// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICHWTHREADSELECTION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericMergeDatamovementThreadsRewritePattern
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;
  unsigned numDMAHWThreads;

  TTIRGenericMergeDatamovementThreadsRewritePattern(MLIRContext *context,
                                                    unsigned numDMAHWThreads)
      : OpRewritePattern<GenericOp>(context), numDMAHWThreads(numDMAHWThreads) {
    assert(numDMAHWThreads > 0 && "numDMAHWThreads must be greater than 0");
  }

  // Returns true if the region contains only a ttir.await
  // associated with the output CB.
  static bool isLocalAwaitForOutputCB(Region &region,
                                      unsigned outputOperandsIndex) {
    assert(region.getBlocks().size() == 1);
    auto &block = region.front();
    if (block.getOperations().size() != 1) {
      return false;
    }
    ttir::AwaitOp awaitOp = dyn_cast<ttir::AwaitOp>(block.front());
    if (!awaitOp) {
      return false;
    }

    if (!llvm::all_of(awaitOp.getOperands(), [&](Value operand) {
          return mlir::dyn_cast<BlockArgument>(operand) &&
                 mlir::cast<BlockArgument>(operand).getOwner() == &block &&
                 mlir::cast<BlockArgument>(operand).getArgNumber() ==
                     outputOperandsIndex;
        })) {
      return false;
    }

    return true;
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {

    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    unsigned numInputs = op.getInputs().size();
    unsigned numOutputs = op.getOutputs().size();

    // Ops with # operands <= # DMA HW Threads don't need to be merged.
    if (numInputs + numOutputs <= numDMAHWThreads) {
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

    // Copy compute region from op to newGeneric, if it exists.
    if (hasComputeThread) {
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegions().back().takeBody(op.getRegions().back());
      });
    }

    // Copy initial dma threads to newGeneric. Merging to an empty block
    // results in discarding cb block args, so we must copy the first
    // numDMAHWThreads blocks instead of uniformly merging everything.
    unsigned regionIndex = 0;
    for (unsigned i = 0; i < numDMAHWThreads; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegion(regionIndex++).takeBody(op.getRegion(i));
      });
    }

    // Merge remaining DMA threads into existing DMA regions in newGeneric.
    // IMPORTANT: output DMA blocks MUST be merged after all input DMA blocks in
    // a region.
    unsigned dmaInputThreadsMerged = 0;
    for (unsigned i = numDMAHWThreads; i <= outputOperandsIndex; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      Block *mergeSrcBlock = &op.getRegion(i).front();

      unsigned dmaMergeTargetIndex = 0;
      if (hasComputeThread &&
          isLocalAwaitForOutputCB(op.getRegion(i), outputOperandsIndex)) {
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

      rewriter.mergeBlocks(mergeSrcBlock, mergeDestBlock,
                           mergeDestBlock->getArguments());
    }

    rewriter.replaceOp(op, newGeneric.getResults());

    return success();
  }
};
} // namespace

namespace {
class TTIRGenericHWThreadSelection
    : public impl::TTIRGenericHWThreadSelectionBase<
          TTIRGenericHWThreadSelection> {
public:
  using impl::TTIRGenericHWThreadSelectionBase<
      TTIRGenericHWThreadSelection>::TTIRGenericHWThreadSelectionBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
        mlir::tt::ttcore::SystemDescAttr::name);
    auto chipDesc = systemDesc.getChipDescs().front();

    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericMergeDatamovementThreadsRewritePattern>(
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

} // namespace mlir::tt::ttir
