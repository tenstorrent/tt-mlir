// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

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

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getIndexingMaps(),
        op.getIteratorTypes(), op.getNumRegions() - 1);

    unsigned regionIndex = 0;
    for (mlir::Region &region : op.getRegions()) {
      if (region.getRegionNumber() == outputOperandsIndex) {
        continue;
      }
      newGeneric.getRegion(regionIndex++).takeBody(region);
    }

    outputBlock->dump();
    newGeneric.getRegions().back().front().dump();
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
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }

    ModuleOp module = getOperation();
    auto systemDesc =
        module->getAttrOfType<SystemDescAttr>(SystemDescAttr::name);
    auto chipDesc = systemDesc.getChipDescs().front();
    getOperation().walk([&](GenericOp op) {
      // assert that the op has a valid HW thread selection
      if (op.getNumRegions() >= (chipDesc.getNumComputeThreads() +
                                 chipDesc.getNumDatamovementThreads())) {
        op.emitError("Invalid number of regions, expected at most ")
            << (chipDesc.getNumComputeThreads() +
                chipDesc.getNumDatamovementThreads());
        return;
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
