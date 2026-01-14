// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELCONTROLDSTSECTION
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

// Check if a loop contains multiple pack_tile operations (indicating reuse)
// static bool hasMultiplePacksInLoop(scf::ForOp forOp) {
//   int packCount = 0;
//   forOp.walk([&](ttkernel::PackTileOp) {
//     packCount++;
//   });
//   return packCount > 1;
// }

// Pattern to handle loops with DST register reuse
class TTKernelLoopDstRewriter : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Skip if already processed
    if (forOp->hasAttr("ttkernel.dst_managed")) {
      return failure();
    }

    // Check if this loop has pack operations
    SmallVector<ttkernel::PackTileOp> packOps;
    forOp.walk([&](ttkernel::PackTileOp packOp) { packOps.push_back(packOp); });

    if (packOps.empty()) {
      return failure();
    }

    // Find preceding acquire
    ttkernel::TileRegsAcquireOp acquire = findPrecedingAcquire(forOp);
    if (!acquire) {
      return failure();
    }

    // Check if already committed
    if (acquire->hasAttr("ttkernel.committed")) {
      return failure();
    }

    // Determine if we need per-iteration sync
    // Conservative approach: always use per-iteration sync for loops containing
    // packs This handles:
    // - Multiple packs in one iteration
    // - Single pack but register reused across iterations
    // - Nested loops where inner loop reuses registers
    bool needsPerIterationSync = true;

    if (needsPerIterationSync) {
      // Insert acquire at the start of the loop body
      Block *loopBody = forOp.getBody();
      rewriter.setInsertionPointToStart(loopBody);
      rewriter.create<ttkernel::TileRegsAcquireOp>(forOp->getLoc());

      // Insert commit/wait/release around EACH pack inside the loop
      for (auto packOp : packOps) {
        if (packOp->hasAttr("ttkernel.dst_managed")) {
          continue;
        }

        rewriter.setInsertionPoint(packOp);
        rewriter.create<ttkernel::TileRegsCommitOp>(packOp->getLoc());
        rewriter.create<ttkernel::TileRegsWaitOp>(packOp->getLoc());

        rewriter.setInsertionPointAfter(packOp);
        rewriter.create<ttkernel::TileRegsReleaseOp>(packOp->getLoc());

        packOp->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());
      }

      // Only erase the acquire if it's in the immediate parent block
      // (not if it's an outer acquire governing multiple loops)
      if (acquire->getBlock() == forOp->getBlock()) {
        rewriter.eraseOp(acquire);
      } else {
        // Mark it so we don't try to use it again for other loops
        acquire->setAttr("ttkernel.committed", rewriter.getUnitAttr());
      }

      // Mark loop as processed
      forOp->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());

    } else {
      // Single pack in loop - wrap the entire loop
      rewriter.setInsertionPoint(forOp);
      rewriter.create<ttkernel::TileRegsCommitOp>(forOp->getLoc());
      rewriter.create<ttkernel::TileRegsWaitOp>(forOp->getLoc());

      rewriter.setInsertionPointAfter(forOp);
      rewriter.create<ttkernel::TileRegsReleaseOp>(forOp->getLoc());

      // Mark everything as processed
      acquire->setAttr("ttkernel.committed", rewriter.getUnitAttr());
      forOp->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());
      for (auto packOp : packOps) {
        packOp->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());
      }
    }

    return success();
  }

private:
  static ttkernel::TileRegsAcquireOp findPrecedingAcquire(Operation *op) {
    Block *block = op->getBlock();
    Operation *current = op;

    while (block) {
      for (auto it = Block::reverse_iterator(current); it != block->rend();
           ++it) {
        if (auto acquire = dyn_cast<ttkernel::TileRegsAcquireOp>(&*it)) {
          return acquire;
        }
      }

      Operation *parentOp = block->getParentOp();
      if (!parentOp) {
        return nullptr;
      }
      current = parentOp;
      block = parentOp->getBlock();
    }
    return nullptr;
  }
};

// Fallback pattern for packs outside of loops
class TTKernelTileRegsRewriter : public OpRewritePattern<ttkernel::PackTileOp> {
public:
  using OpRewritePattern<ttkernel::PackTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::PackTileOp op,
                                PatternRewriter &rewriter) const final {
    // Skip if already managed
    if (op->hasAttr("ttkernel.dst_managed")) {
      return failure();
    }

    // Find the nearest preceding acquire
    ttkernel::TileRegsAcquireOp acquire = findPrecedingAcquire(op);
    if (!acquire) {
      return failure();
    }

    // Skip if already committed
    if (acquire->hasAttr("ttkernel.committed")) {
      return failure();
    }

    // Check if this pack is inside a loop - if so, let the loop pattern handle
    // it
    Operation *parent = op->getParentOp();
    while (parent && !isa<func::FuncOp>(parent)) {
      if (isa<scf::ForOp>(parent)) {
        // Inside a loop, let TTKernelLoopDstRewriter handle it
        return failure();
      }
      parent = parent->getParentOp();
    }

    // Not in a loop - wrap just this pack operation
    rewriter.setInsertionPoint(op);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());

    rewriter.setInsertionPointAfter(op);
    rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc());

    // Mark as processed
    acquire->setAttr("ttkernel.committed", rewriter.getUnitAttr());
    op->setAttr("ttkernel.dst_managed", rewriter.getUnitAttr());

    return success();
  }

private:
  static ttkernel::TileRegsAcquireOp findPrecedingAcquire(Operation *op) {
    Block *block = op->getBlock();
    Operation *current = op;

    while (block) {
      for (auto it = Block::reverse_iterator(current); it != block->rend();
           ++it) {
        if (auto acquire = dyn_cast<ttkernel::TileRegsAcquireOp>(&*it)) {
          return acquire;
        }
      }

      Operation *parentOp = block->getParentOp();
      if (!parentOp) {
        return nullptr;
      }
      current = parentOp;
      block = parentOp->getBlock();
    }
    return nullptr;
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

    // Process loops first, then individual packs
    patterns.add<TTKernelLoopDstRewriter>(&getContext());
    patterns.add<TTKernelTileRegsRewriter>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
