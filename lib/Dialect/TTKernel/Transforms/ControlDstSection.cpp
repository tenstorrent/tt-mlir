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

    // Guard against re-application: check for an existing commit between the
    // nearest preceding acquire and `parent` in acquireBlock. If one exists,
    // this pack has already been handled.
    if (hasPrecedingCommit(parent)) {
      return failure();
    }

    rewriter.setInsertionPoint(parent);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc());

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

  // Returns true if a TileRegsCommitOp exists between the most recent
  // TileRegsAcquireOp before `op` (in `op`'s block) and `op` itself.
  // Used to prevent double-insertion on re-application.
  static bool hasPrecedingCommit(Operation *op) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<ttkernel::TileRegsCommitOp>(it)) {
        return true;
      }
      if (isa<ttkernel::TileRegsAcquireOp>(it)) {
        return false;
      }
    }
    return false;
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
