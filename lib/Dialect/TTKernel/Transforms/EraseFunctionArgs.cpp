// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELERASEFUNCTIONARGS
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {
class TTKernelFunctionArgsRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {

    // Only erase function args for kernel functions.
    if (!op->hasAttr("ttir.thread_type")) { // TODO(nsmith/jdesousa): String
                                            // constant somewhere for this?
      return failure();
    }

    // TODO(jdesousa): Handle semaphore args, this condition should be
    // op.getArguments().size() == 0
    if (std::all_of(
            op.getArgumentTypes().begin(), op.getArgumentTypes().end(),
            [](Type type) { return mlir::isa<ttir::SemaphoreType>(type); })) {
      return failure();
    }

    rewriter.setInsertionPointToStart(&op.getBody().front());
    for (auto funcArg : op.getArguments()) {
      if (auto cbType = mlir::dyn_cast<CBType>(funcArg.getType())) {
        auto getCBOp = rewriter.create<ttkernel::GetCBOp>(op.getLoc(), cbType);
        rewriter.replaceAllUsesWith(funcArg, getCBOp);
      }
      // TODO(jdesousa): Handle semaphore/other function args
    }

    while (mlir::isa<CBType>(op.getArgumentTypes().front())) {
      rewriter.modifyOpInPlace(op, [&]() { op.eraseArgument(0); });
    }

    return success();
  }
};
} // namespace

namespace {
class TTKernelEraseFunctionArgs
    : public impl::TTKernelEraseFunctionArgsBase<TTKernelEraseFunctionArgs> {
public:
  using impl::TTKernelEraseFunctionArgsBase<
      TTKernelEraseFunctionArgs>::TTKernelEraseFunctionArgsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTKernelFunctionArgsRewriter>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
