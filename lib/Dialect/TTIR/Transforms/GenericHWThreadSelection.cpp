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

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    unsigned inputOperandsLength = op.getInputs().size();
    unsigned outputOperandsLength = op.getOutputs().size();
    assert(inputOperandsLength <= 2 &&
           "only 2 or less input operands are supported");
    assert(outputOperandsLength == 1 && "only one output operand is supported");
    // note: If only one input + one output operand, dma is already in optimal
    // form
    if (inputOperandsLength < 2 || outputOperandsIndex >= op.getNumRegions()) {
      return failure();
    }
    Region &outputRegion = op.getRegion(outputOperandsIndex);
    if (outputRegion.getBlocks().size() == 0) {
      return failure();
    }
    assert(outputRegion.getBlocks().size() == 1 &&
           "output datamovement region should have exactly one block");
    Block &outputBlock = outputRegion.front();

    SmallVector<Attribute> threads(op.getThreads().getValue());
    // Skip the output operands block
    threads.erase(threads.begin() + outputOperandsIndex);
    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threads), op.getNumRegions() - 1);

    unsigned regionIndex = 0;
    for (mlir::Region &region : op.getRegions()) {
      if (region.getRegionNumber() == outputOperandsIndex) {
        continue;
      }
      rewriter.modifyOpInPlace(
          op, [&] { newGeneric.getRegion(regionIndex++).takeBody(region); });
    }

    // ??? isn't the last region a compute region?
    unsigned lastInputRegionIndex = inputOperandsLength - 1;
    Block *newBlock = &newGeneric.getRegion(lastInputRegionIndex).back();
    rewriter.mergeBlocks(&outputBlock, newBlock, newBlock->getArguments());
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
