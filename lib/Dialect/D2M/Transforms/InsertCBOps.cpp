// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MREMOVESYNCHRONIZEDREGIONS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MRemoveSynchronizedRegions
    : public impl::D2MRemoveSynchronizedRegionsBase<
          D2MRemoveSynchronizedRegions> {
public:
  using impl::D2MRemoveSynchronizedRegionsBase<
      D2MRemoveSynchronizedRegions>::D2MRemoveSynchronizedRegionsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](func::FuncOp funcOp) {
      if (failed(removeSynchronizedRegions(rewriter, funcOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

private:
  LogicalResult removeSynchronizedRegions(IRRewriter &rewriter,
                                          func::FuncOp funcOp) {
    funcOp->walk([&](Operation *op) {
      // Remove synchronized region ops, and hoist it's ops up parent level
      if (auto synchronizedOp = dyn_cast<SynchronizedRegionOp>(op)) {
        rewriter.setInsertionPoint(op);
        // hoist the top level ops in the region up parent level, replacing the
        // block args with the synchronized region's operands
        rewriter.setInsertionPoint(synchronizedOp);
        IRMapping mapping;
        for (auto [blockArg, operand] :
             llvm::zip(synchronizedOp.getRegion().front().getArguments(),
                       synchronizedOp->getOperands())) {
          mapping.map(blockArg, operand);
        }

        for (Operation &op :
             synchronizedOp.getRegion().front().getOperations()) {
          if (!mlir::isa<d2m::YieldOp>(op)) {
            rewriter.clone(op, mapping);
          }
        }
        // remove the synchronized region op
        rewriter.eraseOp(synchronizedOp);

        // TODO:need to replace the results as well??
      }
      return WalkResult::advance();
    });
    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
