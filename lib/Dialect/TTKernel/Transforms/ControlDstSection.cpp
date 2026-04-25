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

template <typename PackOp>
class TTKernelTileRegsRewriter : public OpRewritePattern<PackOp> {
public:
  using OpRewritePattern<PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PackOp op,
                                PatternRewriter &rewriter) const final {
    Block *acquireBlock = findBlockContaining<ttkernel::TileRegsAcquireOp>(op);
    Operation *parent = parentOpAtBlock(op, acquireBlock);
    Operation *firstParent = findFirstContiguousPackParent(parent);
    Operation *lastParent = findLastContiguousPackParent(parent);

    // Guard against re-application: check for an existing commit between the
    // nearest preceding acquire and `firstParent` in acquireBlock. If one
    // exists, this pack has already been handled.
    if (hasPrecedingCommit(firstParent)) {
      return failure();
    }

    rewriter.setInsertionPoint(firstParent);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());
    rewriter.setInsertionPointAfter(lastParent);
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

  static bool containsPackOp(Operation *op) {
    bool found = false;
    op->walk([&](PackOp) {
      found = true;
      return WalkResult::interrupt();
    });
    return found;
  }

  static Operation *findFirstContiguousPackParent(Operation *parent) {
    Operation *firstParent = parent;
    for (Operation *prev = parent->getPrevNode(); prev != nullptr;
         prev = prev->getPrevNode()) {
      if (!containsPackOp(prev)) {
        break;
      }
      firstParent = prev;
    }
    return firstParent;
  }

  static Operation *findLastContiguousPackParent(Operation *parent) {
    Operation *lastParent = parent;
    for (Operation *next = parent->getNextNode(); next != nullptr;
         next = next->getNextNode()) {
      if (!containsPackOp(next)) {
        break;
      }
      lastParent = next;
    }
    return lastParent;
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
    patterns.add<TTKernelTileRegsRewriter<ttkernel::PackTileOp>>(&getContext());
    patterns.add<TTKernelTileRegsRewriter<ttkernel::PackTileBlockOp>>(
        &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
