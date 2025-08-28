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

  // returns true if the region contains only a ttir.await
  // associated with the output CB
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

    auto device = ttcore::lookupDevice(op);
    auto systemDesc = ttcore::getCurrentScopeSystemDesc(op);
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    unsigned kInputs = op.getInputs().size();
    unsigned kOutputs = op.getOutputs().size();
    unsigned kDMAHWThreads = chipDesc.getNumDatamovementThreads();

    // handling special case of single datamovement region is very clumsy and
    // achieves nothing, so skip it
    if (kInputs + kOutputs < 2) {
      return failure();
    }

    if (outputOperandsIndex >= op.getNumRegions()) {
      return failure();
    }

    llvm::SmallDenseSet<unsigned> mergedIndices;
    for (unsigned i = kDMAHWThreads; i <= outputOperandsIndex; ++i) {
      mergedIndices.insert(i);
    }

    SmallVector<Attribute> threads;
    for (unsigned idx = 0; idx < op.getThreads().size(); ++idx) {
      if (!mergedIndices.contains(idx)) {
        threads.push_back(op.getThreads()[idx]);
      }
    }
    size_t numNewRegions = threads.size();

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threads), numNewRegions);

    // Copy compute region from op to newGeneric.
    rewriter.modifyOpInPlace(op, [&] {
      newGeneric.getRegions().back().takeBody(op.getRegions().back());
    });

    // Copy initial dma threads to newGeneric. Merging to an empty block
    // results in discarding cb block args, so we must copy the first
    // kDMAHWThreads blocks instead of uniformly merge everything
    unsigned regionIndex = 0;
    for (unsigned i = 0; i < kDMAHWThreads; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegion(regionIndex++).takeBody(op.getRegion(i));
      });
    }

    // Merge remaining dma threads into existing dma regions in newGeneric.
    // IMPORTANT: output dma blocks MUST be merged after all input dma blocks in
    // a region.
    unsigned dmaInputThreadsMerged = 0;
    for (unsigned i = kDMAHWThreads; i <= outputOperandsIndex; ++i) {
      assert(op.getRegion(i).getBlocks().size() == 1 &&
             "all datamovement regions should have exactly one block");
      Block *mergeSrcBlock = &op.getRegion(i).front();

      unsigned dmaMergeTargetIndex = 0;
      if (isLocalAwaitForOutputCB(op.getRegion(i), outputOperandsIndex)) {
        // Merge trivial output dma into compute region
        dmaMergeTargetIndex = newGeneric->getNumRegions() - 1;
      } else if (i == outputOperandsIndex) {
        // Always merge output to last dma region (associated with NOC1).
        dmaMergeTargetIndex = kDMAHWThreads - 1;
      } else {
        // Merge input dma threads to alternate dma regions.
        dmaMergeTargetIndex = dmaInputThreadsMerged % kDMAHWThreads;
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
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericMergeDatamovementThreadsRewritePattern>(
        &getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    ModuleOp moduleOp = getOperation();
    auto systemDesc = moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
        mlir::tt::ttcore::SystemDescAttr::name);
    auto chipDesc = systemDesc.getChipDescs().front();
    moduleOp.walk([&](GenericOp op) {
      // assert that the op has a valid HW thread selection
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
