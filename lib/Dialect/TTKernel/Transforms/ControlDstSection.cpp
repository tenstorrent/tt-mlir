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
    // Skip if this pack already has commit/wait before it.
    if (op->hasAttr("ttkernel.dst_managed")) {
      return failure();
    }

    // Find the nearest preceding TileRegsAcquireOp in the same block or
    // ancestor blocks. We need to find the specific acquire that governs
    // this pack, not just any acquire.
    ttkernel::TileRegsAcquireOp acquire = findPrecedingAcquire(op);
    if (!acquire) {
      // No acquire found - this pack doesn't have DST management.
      return failure();
    }

    // Check if there's already a commit between the acquire and this pack.
    // If the acquire has a "committed" marker, we've already processed it.
    if (acquire->hasAttr("ttkernel.committed")) {
      return failure();
    }

    // Find the loop/parent op that contains the pack and is at the same level
    // as the acquire. We want to wrap the entire loop with commit/release.
    Operation *packParent = findLoopOrParent(op, acquire);

    // Insert commit/wait before the parent op.
    rewriter.setInsertionPoint(packParent);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());

    // Insert release after the parent op.
    rewriter.setInsertionPointAfter(packParent);
    rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc());

    // Mark the acquire as committed so we don't process it again.
    acquire->setAttr("ttkernel.committed", rewriter.getUnitAttr());

    // Mark this pack as managed.
    op->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());

    return success();
  };

  // Find the nearest preceding TileRegsAcquireOp that governs this op.
  static ttkernel::TileRegsAcquireOp findPrecedingAcquire(Operation *op) {
    // Walk backwards in the block to find an acquire.
    Block *block = op->getBlock();
    Operation *current = op;

    while (block) {
      // Walk backwards from current op in this block.
      for (auto it = Block::reverse_iterator(current); it != block->rend();
           ++it) {
        if (auto acquire = dyn_cast<ttkernel::TileRegsAcquireOp>(&*it)) {
          return acquire;
        }
      }

      // Move to parent block.
      Operation *parentOp = block->getParentOp();
      if (!parentOp) {
        return nullptr;
      }
      current = parentOp;
      block = parentOp->getBlock();
    }
    return nullptr;
  }

  // Find the loop or parent op that should be wrapped with commit/release.
  static Operation *findLoopOrParent(Operation *pack,
                                     ttkernel::TileRegsAcquireOp acquire) {
    Operation *parent = pack;
    Block *acquireBlock = acquire->getBlock();

    // Walk up until we find an op in the same block as the acquire.
    while (parent->getBlock() != acquireBlock) {
      parent = parent->getParentOp();
      assert(parent && "pack should be nested under acquire's block");
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
