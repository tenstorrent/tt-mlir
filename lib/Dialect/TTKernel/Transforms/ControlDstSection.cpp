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
    // Check if this PackTileOp already has commit/wait directly before it.
    Operation *immediatePrev = op->getPrevNode();
    Operation *immediatePrevPrev =
        immediatePrev ? immediatePrev->getPrevNode() : nullptr;

    // Pattern: <...> commit wait <pack> <...>
    if (immediatePrevPrev &&
        isa<ttkernel::TileRegsCommitOp>(immediatePrevPrev) && immediatePrev &&
        isa<ttkernel::TileRegsWaitOp>(immediatePrev)) {
      // Already wrapped
      return failure();
    }

    // Insert commit and wait directly before the pack operation.
    // This ensures they are inside any loops containing the pack,
    // which is necessary for multiblock processing where we need
    // commit/wait per tile, not once before all loops.
    // Note: We don't insert release here because InsertDstRegisterGC
    // already handles acquire/release placement.
    rewriter.setInsertionPoint(op);
    rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc());
    rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());

    return success();
  };
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
