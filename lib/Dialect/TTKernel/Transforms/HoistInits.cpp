// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DEF_TTKERNELHOISTINITS
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

namespace {

class TTKernelFunctionRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    DenseMap<scf::ForOp, SmallVector<OperationName>> initOps;

    op.walk([&initOps](Operation *op) {
      if (op->hasTrait<ttkernel::TTKernelInitOpTrait>()) {
        auto forParent = op->getParentOfType<scf::ForOp>();
        while (forParent) {
          auto forInitOps = initOps.lookup(forParent);
          initOps[forParent].push_back(op->getName());
          forParent = forParent->getParentOfType<scf::ForOp>();
        }
      }
    });

    op.walk([&](Operation *op) {
      if (op->hasTrait<ttkernel::TTKernelInitOpTrait>()) {
        Operation *highestLiftableLoop = nullptr;
        scf::ForOp curr = op->getParentOfType<scf::ForOp>();
        while (curr) {
          assert(initOps.contains(curr) &&
                 "Init op's parent loop should be in the initOps map.");
          auto currLoopInitOps = initOps.lookup(curr);
          assert(std::find(currLoopInitOps.begin(), currLoopInitOps.end(),
                           op->getName()) != currLoopInitOps.end() &&
                 "Init op should be inside the parent loop's initOps map.");

          // This condition should be smarter, in the sense that we should have
          // a lookup table of conflicting inits and detect whether we can keep
          // going. For now, assume all inits conflict.
          if (currLoopInitOps.size() == 1) {
            highestLiftableLoop = curr;
          } else {
            break;
          }
          curr = curr->getParentOfType<scf::ForOp>();
        }
        if (highestLiftableLoop) {
          rewriter.moveOpBefore(op, highestLiftableLoop);
        }
      }
    });
    return success();
  }
};

} // namespace

namespace {
class TTKernelHoistInits
    : public impl::TTKernelHoistInitsBase<TTKernelHoistInits> {
public:
  using impl::TTKernelHoistInitsBase<
      TTKernelHoistInits>::TTKernelHoistInitsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTKernelFunctionRewriter>(&getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttkernel
