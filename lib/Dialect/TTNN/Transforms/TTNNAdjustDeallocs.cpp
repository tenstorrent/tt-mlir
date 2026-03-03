// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNADJUSTDEALLOCS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNAdjustDeallocs
    : public impl::TTNNAdjustDeallocsBase<TTNNAdjustDeallocs> {

public:
  using impl::TTNNAdjustDeallocsBase<
      TTNNAdjustDeallocs>::TTNNAdjustDeallocsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Walk through all functions in the module
    //
    moduleOp->walk([&](func::FuncOp funcOp) {
      // Get the set of constant and parameter arguments for this function.
      //
      llvm::SmallPtrSet<mlir::BlockArgument, 4> constsAndParams =
          ttcore::getConstsAndParams(funcOp);

      // Collect deallocate ops that need to be removed.
      //
      llvm::SmallVector<DeallocateOp> deallocsToErase;

      funcOp->walk([&](DeallocateOp deallocOp) {
        Value input = deallocOp.getInput();

        // Check if the input comes from a parameter or constant argument.
        //
        if (auto blockArg = mlir::dyn_cast<BlockArgument>(input)) {
          if (constsAndParams.contains(blockArg)) {
            deallocsToErase.push_back(deallocOp);
          }
        } else if (auto loadCachedOp = mlir::dyn_cast<ttcore::LoadCachedOp>(
                       input.getDefiningOp())) {
          deallocsToErase.push_back(deallocOp);
        }
      });

      // Erase the collected ttnn.deallocate operations.
      //
      for (DeallocateOp deallocOp : deallocsToErase) {
        rewriter.eraseOp(deallocOp);
      }
    });
  }
};

} // namespace mlir::tt::ttnn
