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
    if (!acquireBlock->getOps<ttkernel::TileRegsCommitOp>().empty()) {
      return failure();
    }

    Operation *parent = parentOpAtBlock(op, acquireBlock);
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
