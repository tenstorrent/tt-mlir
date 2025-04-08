// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttmetal {
#define GEN_PASS_DEF_TTMETALCONTROLDSTSECTION
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

namespace {

class TTMetalTileRegsRewriter : public OpRewritePattern<ttkernel::PackTileOp> {
public:
  using OpRewritePattern<ttkernel::PackTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttkernel::PackTileOp op,
                                PatternRewriter &rewriter) const final {
    if (!op->getBlock()->getOps<ttkernel::TileRegsCommitOp>().empty()) {
      return failure();
    }

    rewriter.moveOpAfter(
        rewriter.create<ttkernel::TileRegsReleaseOp>(op->getLoc()), op);
    auto regsWait = rewriter.create<ttkernel::TileRegsWaitOp>(op->getLoc());
    rewriter.moveOpBefore(regsWait, op);
    rewriter.moveOpBefore(
        rewriter.create<ttkernel::TileRegsCommitOp>(op->getLoc()), regsWait);

    rewriter.moveOpAfter(
        rewriter.create<ttkernel::TileRegsAcquireOp>(op->getLoc()),
        op->getBlock(), op->getBlock()->begin());

    return success();
  };
};

} // namespace

namespace {
class TTMetalControlDstSection
    : public impl::TTMetalControlDstSectionBase<TTMetalControlDstSection> {
public:
  using impl::TTMetalControlDstSectionBase<
      TTMetalControlDstSection>::TTMetalControlDstSectionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTMetalTileRegsRewriter>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttmetal
