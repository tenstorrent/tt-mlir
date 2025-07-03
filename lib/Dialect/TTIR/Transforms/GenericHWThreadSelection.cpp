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
class TTIRGenericMoveTrivialOutputThreadToComputeRewritePattern
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  static Block *getIfTrivialBlock(Region &region,
                                  unsigned outputOperandsIndex) {
    assert(region.getBlocks().size() == 1);
    auto &block = region.front();
    if (block.getOperations().size() != 1) {
      return nullptr;
    }
    ttir::AwaitOp awaitOp = dyn_cast<ttir::AwaitOp>(block.front());
    if (!awaitOp) {
      return nullptr;
    }

    if (!llvm::all_of(awaitOp.getOperands(), [&](Value operand) {
          return mlir::dyn_cast<BlockArgument>(operand) &&
                 mlir::cast<BlockArgument>(operand).getOwner() == &block &&
                 mlir::cast<BlockArgument>(operand).getArgNumber() ==
                     outputOperandsIndex;
        })) {
      return nullptr;
    }

    return &block;
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    unsigned outputOperandsIndex = op.getOutputs().getBeginOperandIndex();
    unsigned outputOperandsLength = op.getOutputs().size();
    assert(outputOperandsLength == 1);
    if (outputOperandsIndex >= op.getNumRegions()) {
      return failure();
    }
    Region &outputRegion = op.getRegion(outputOperandsIndex);
    Block *outputBlock = getIfTrivialBlock(outputRegion, outputOperandsIndex);
    if (!outputBlock) {
      return failure();
    }

    SmallVector<Attribute> threads(op.getThreads().getValue());
    // Skip the output operands block
    threads.erase(threads.begin() + outputOperandsIndex);
    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getSubblockFactors(), op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threads), op.getNumRegions() - 1);

    unsigned regionIndex = 0;
    for (mlir::Region &region : op.getRegions()) {
      if (region.getRegionNumber() == outputOperandsIndex) {
        continue;
      }
      rewriter.modifyOpInPlace(
          op, [&] { newGeneric.getRegion(regionIndex++).takeBody(region); });
    }

    Block *newBlock = &newGeneric.getRegions().back().front();
    rewriter.mergeBlocks(outputBlock, newBlock, newBlock->getArguments());
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
    patterns.add<TTIRGenericMoveTrivialOutputThreadToComputeRewritePattern>(
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
