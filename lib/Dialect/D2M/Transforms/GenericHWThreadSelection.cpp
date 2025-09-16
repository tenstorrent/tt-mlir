// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICHWTHREADSELECTION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
class D2MGenericMoveTrivialOutputThreadToComputeRewritePattern
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
    d2m::AwaitOp awaitOp = dyn_cast<d2m::AwaitOp>(block.front());
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

    ArrayAttr threadsAttr = op.getThreads();
    SmallVector<Attribute> threadAttrs(threadsAttr.getValue().begin(),
                                       threadsAttr.getValue().end());
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
         regionIndex++) {
      auto threadAttr = mlir::cast<ThreadAttr>(threadAttrs[regionIndex]);
      if (threadAttr.getThreadType() != ThreadType::Datamovement) {
        continue;
      }

      Region &region = op.getRegion(regionIndex);
      Block *trivialBlock = getIfTrivialBlock(region, outputOperandsIndex);
      if (!trivialBlock) {
        continue;
      }

      // Move the trivial datamovement thread to compute thread.
      threadAttrs[regionIndex] =
          rewriter.getAttr<ThreadAttr>(ThreadType::Compute);
      modified = true;
    }

    if (!modified) {
      return failure();
    }

    // Use the low-level build method that matches the D2M GenericOp signature
    auto newOp = rewriter.create<GenericOp>(
        op.getLoc(), TypeRange(op.getResults()), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getBlockFactors(),
        op.getIndexingMaps(), op.getIteratorTypes(),
        rewriter.getArrayAttr(threadAttrs), op.getNumRegions());

    // Transfer regions
    for (unsigned i = 0; i < op.getNumRegions(); ++i) {
      newOp.getRegion(i).takeBody(op.getRegion(i));
    }

    rewriter.replaceOp(op, newOp.getResults());

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
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MGenericMoveTrivialOutputThreadToComputeRewritePattern>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
