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

    // Walk backwards from the pack point to find whether this specific
    // DST section has already been processed.
    for (Operation *it = parent->getPrevNode(); it; it = it->getPrevNode()) {
      if (isa<ttkernel::TileRegsCommitOp>(it)) {
        // Has a commit, already processed.
        return failure();
      }
      if (isa<ttkernel::TileRegsAcquireOp>(it)) {
        break;
      }
    }

    rewriter.setInsertionPoint(parent);
    ttkernel::TileRegsCommitOp::create(rewriter, op->getLoc());
    ttkernel::TileRegsWaitOp::create(rewriter, op->getLoc());
    rewriter.setInsertionPointAfter(parent);
    ttkernel::TileRegsReleaseOp::create(rewriter, op->getLoc());

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
