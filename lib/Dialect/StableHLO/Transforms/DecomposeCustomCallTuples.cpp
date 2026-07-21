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
    IRRewriter rewriter(getOperation().getContext());

    // Rewrite 1: stablehlo.tuple ops - forward operands to
    // get_tuple_element users directly.
    SmallVector<mlir::stablehlo::TupleOp> tupleOps;
    getOperation().walk(
        [&](mlir::stablehlo::TupleOp op) { tupleOps.push_back(op); });

    for (auto op : tupleOps) {
      for (OpOperand &use :
           llvm::make_early_inc_range(op.getResult().getUses())) {
        auto getTupleOp =
            dyn_cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner());
        if (!getTupleOp) {
          continue;
        }
        rewriter.replaceOp(getTupleOp, op.getOperand(getTupleOp.getIndex()));
      }
      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
      }
    }

    // Shardy propagation can't traverse tuple types, so flatten multi-output
    // custom_calls (XLA encodes them as one tuple result) to multi-result form
    // before it runs. Rewrite 1 misses these: the tuple is the custom_call's
    // own result, not a separate stablehlo.tuple op.
    SmallVector<mlir::stablehlo::CustomCallOp> tupleCustomCalls;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getNumResults() != 1) {
        return;
      }
      if (!isa<mlir::TupleType>(op.getResult(0).getType())) {
        return;
      }
      // Only decompose when every use is a get_tuple_element; anything that
      // consumes the tuple directly would need it rematerialized.
      for (mlir::Operation *user : op.getResult(0).getUsers()) {
        if (!isa<mlir::stablehlo::GetTupleElementOp>(user)) {
          return;
        }
      }
      tupleCustomCalls.push_back(op);
    });

    for (auto op : tupleCustomCalls) {
      auto tupleType = cast<mlir::TupleType>(op.getResult(0).getType());
      SmallVector<mlir::Type> newResultTypes(tupleType.begin(),
                                             tupleType.end());

      rewriter.setInsertionPoint(op);
      mlir::OperationState state(op.getLoc(), op->getName().getStringRef());
      state.addOperands(op->getOperands());
      state.addTypes(newResultTypes);
      state.addAttributes(op->getAttrs());
      mlir::Operation *newOp = rewriter.create(state);

      for (OpOperand &use :
           llvm::make_early_inc_range(op.getResult(0).getUses())) {
        auto getTupleOp =
            cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner());
        rewriter.replaceOp(getTupleOp, newOp->getResult(getTupleOp.getIndex()));
      }
      rewriter.eraseOp(op);
    }
  }
};
} // namespace mlir::tt::stablehlo
