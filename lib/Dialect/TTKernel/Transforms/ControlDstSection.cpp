// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELCONTROLDSTSECTION
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

class TTKernelTileRegsRewriter : public OpRewritePattern<ttkernel::PackTileOp> {
public:
  using OpRewritePattern<ttkernel::PackTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::PackTileOp op,
                                PatternRewriter &rewriter) const final {
    Block *acquireBlock = findBlockContaining<ttkernel::TileRegsAcquireOp>(op);
    Operation *parent = parentOpAtBlock(op, acquireBlock);

    // Check if this parent already has been wrapped by looking at its immediate
    // neighbors. The parent should have commit/wait before it and release after.
    Operation *immediatePrev = parent->getPrevNode();
    Operation *immediatePrevPrev = immediatePrev ? immediatePrev->getPrevNode() : nullptr;
    Operation *immediateNext = parent->getNextNode();

    // Pattern: <...> commit wait <parent> release <...>
    if (immediatePrevPrev && isa<ttkernel::TileRegsCommitOp>(immediatePrevPrev) &&
        immediatePrev && isa<ttkernel::TileRegsWaitOp>(immediatePrev) &&
        immediateNext && isa<ttkernel::TileRegsReleaseOp>(immediateNext)) {
      // Already wrapped
      return failure();
    }

    // Insert tile_regs management around this parent operation
    rewriter.setInsertionPoint(parent);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc());

    // Remove any existing TileRegsReleaseOp that comes after the parent in the
    // same block. This handles the case where d2m.release_dst was converted to
    // ttkernel.tile_regs_release before this pass runs - we don't want duplicate
    // releases as they cause semaphore errors on the simulator.
    SmallVector<ttkernel::TileRegsReleaseOp> toErase;
    for (Operation &blockOp : *acquireBlock) {
      if (&blockOp == parent || blockOp.isBeforeInBlock(parent)) {
        continue;
      }
      if (auto releaseOp = dyn_cast<ttkernel::TileRegsReleaseOp>(&blockOp)) {
        // Check if this is not the one we just inserted (different location)
        if (releaseOp.getLoc() != op->getLoc()) {
          toErase.push_back(releaseOp);
        }
      }
    }
    for (auto releaseOp : toErase) {
      rewriter.eraseOp(releaseOp);
    }

    return success();
  };

  template <typename ConcreteOp>
  static Block *findBlockContaining(Operation *op) {
    Block *block = op->getBlock();
    while (block->getOps<ConcreteOp>().empty()) {
      block = block->getParentOp()->getBlock();
    }
    return block;
  }

  static Operation *parentOpAtBlock(Operation *child, Block *atBlock) {
    Operation *parent = child;
    while (parent->getBlock() != atBlock) {
      parent = parent->getParentOp();
      assert(parent);
    }
    return parent;
  }
};

} // namespace

namespace {
class TTKernelControlDstSection
    : public impl::TTKernelControlDstSectionBase<TTKernelControlDstSection> {
public:
  using impl::TTKernelControlDstSectionBase<
      TTKernelControlDstSection>::TTKernelControlDstSectionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTKernelTileRegsRewriter>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
