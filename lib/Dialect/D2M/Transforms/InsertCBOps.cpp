// SPDX-FileCopyrightText: (c) 2026wr Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <llvm/ADT/STLExtras.h>

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTCBOPS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MInsertCBOps : public impl::D2MInsertCBOpsBase<D2MInsertCBOps> {
public:
  using impl::D2MInsertCBOpsBase<D2MInsertCBOps>::D2MInsertCBOpsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk(
        [&](func::FuncOp funcOp) { 
            if (failed(insertCBOps(rewriter, funcOp))) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
  }

  private:

  LogicalResult insertCBOps(IRRewriter &rewriter, func::FuncOp funcOp) {
    // look at every double buffered alloc (cb layout attribute) and change to
    // convert every syncrhonized to non synchronized form by inserting get cb, reserve/wait/push/pop 
    funcOp->walk(
        [&](Operation *op) { 
            //if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
            //    if (allocOp.hasCBLayoutAttr()) {
            //        // insert get cb, reserve/wait/push/pop
            //        rewriter.setInsertionPoint(allocOp);
            //        auto getCBOp = rewriter.create<d2m::GetCBOp>(allocOp->getLoc(), allocOp.getResult());
            //        // insert get cb at the top
            //        // this requires the cb to have been hoisted correctly, currently this is tied to operands, but we should tie it to the alloc instead
            //        // each alloc should be assigned a CB port number, and this should be used to get the correct CB
            //        
            //        // on producer side, remove cb as operand and instead insert a reserve before op, pass its result to our 
            //        // syncrhonized op as operand instead converting to non syncrhonized form, and and insert push after op
            //        auto reserveOp = rewriter.create<d2m::ReserveOp>(allocOp->getLoc(), allocOp.getResult());
            //        auto pushOp = rewriter.create<d2m::PushOp>(allocOp->getLoc(), allocOp.getResult());
            //
            //        // on consumer side, similar behaviour
            //        auto waitOp = rewriter.create<d2m::WaitOp>(allocOp->getLoc(), allocOp.getResult());
            //        auto popOp = rewriter.create<d2m::PopOp>(allocOp->getLoc(), allocOp.getResult());
            //    }
            //}

            // to start testing things, start with just eliminating syncrhonized region ops, and hositing it's ops up parent level
            if (auto synchronizedOp = dyn_cast<SynchronizedRegionOp>(op)) {
              rewriter.setInsertionPoint(op);
              // hoist the top level ops in the region up parent level, replacing the block args with the synchronized region's operands
              rewriter.setInsertionPoint(synchronizedOp);
              IRMapping mapping;
              for (auto [blockArg, operand] : llvm::zip(synchronizedOp.getRegion().front().getArguments(), synchronizedOp->getOperands())) {
                mapping.map(blockArg, operand);
              }

              for (Operation &op : synchronizedOp.getRegion().front().getOperations()) {
                if (!mlir::isa<d2m::YieldOp>(op)) {
                  rewriter.clone(op, mapping);
                }
              }
              // remove the synchronized region op
              rewriter.eraseOp(synchronizedOp);

              // TODO:need to replace the results as well??
            }
            return WalkResult::advance();

            // this will will be needed for other ops, so we should have an interface function: convertToNonSynchronizedForm
        });
    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m