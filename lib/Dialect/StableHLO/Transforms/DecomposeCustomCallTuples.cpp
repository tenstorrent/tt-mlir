// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_DECOMPOSECUSTOMCALLTUPLESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

struct DecomposeCustomCallTuplesPass
    : public impl::DecomposeCustomCallTuplesPassBase<
          DecomposeCustomCallTuplesPass> {
public:
  using impl::DecomposeCustomCallTuplesPassBase<
      DecomposeCustomCallTuplesPass>::DecomposeCustomCallTuplesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    // Collect all custom_call ops that return a tuple type.
    SmallVector<mlir::stablehlo::CustomCallOp> tupleCustomCalls;
    module.walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getNumResults() == 1 &&
          isa<TupleType>(op.getResult(0).getType())) {
        tupleCustomCalls.push_back(op);
      }
    });

    for (auto customCallOp : tupleCustomCalls) {
      auto tupleType = cast<TupleType>(customCallOp.getResult(0).getType());
      SmallVector<Type> elementTypes(tupleType.getTypes());

      // Create a new custom_call that returns multiple results instead of a
      // tuple.
      rewriter.setInsertionPoint(customCallOp);
      auto newOp = rewriter.create<mlir::stablehlo::CustomCallOp>(
          customCallOp.getLoc(), TypeRange(elementTypes),
          customCallOp.getInputs(), customCallOp->getAttrs());

      // Replace all get_tuple_element users with the corresponding result
      // of the new multi-result custom_call.
      Value tupleResult = customCallOp.getResult(0);
      SmallVector<Operation *> toErase;

      for (OpOperand &use : llvm::make_early_inc_range(tupleResult.getUses())) {
        auto getTupleOp =
            dyn_cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner());
        if (!getTupleOp) {
          // Unexpected use of the tuple result - skip this custom_call.
          continue;
        }

        int64_t index = getTupleOp.getIndex();
        getTupleOp.getResult().replaceAllUsesWith(newOp.getResult(index));
        toErase.push_back(getTupleOp);
      }

      // Erase the get_tuple_element ops and the original custom_call.
      for (auto *op : toErase) {
        rewriter.eraseOp(op);
      }

      // Only erase the original if all uses have been replaced.
      if (tupleResult.use_empty()) {
        rewriter.eraseOp(customCallOp);
      }
    }
  }
};
} // namespace mlir::tt::stablehlo
