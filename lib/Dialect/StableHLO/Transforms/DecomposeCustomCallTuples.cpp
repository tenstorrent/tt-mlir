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

    // Rewrite 2: custom_call ops returning a tuple - replace with a
    // multi-result custom_call.
    SmallVector<mlir::stablehlo::CustomCallOp> tupleCustomCalls;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getNumResults() == 1 &&
          isa<TupleType>(op.getResult(0).getType())) {
        tupleCustomCalls.push_back(op);
      }
    });

    for (auto op : tupleCustomCalls) {
      auto tupleType = cast<TupleType>(op.getResult(0).getType());

      rewriter.setInsertionPoint(op);
      auto newOp = rewriter.create<mlir::stablehlo::CustomCallOp>(
          op.getLoc(), tupleType.getTypes(), op.getInputs(), op->getAttrs());

      for (OpOperand &use :
           llvm::make_early_inc_range(op.getResult(0).getUses())) {
        auto getTupleOp =
            dyn_cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner());
        if (!getTupleOp) {
          continue;
        }

        rewriter.replaceOp(getTupleOp, newOp.getResult(getTupleOp.getIndex()));
      }

      if (op.getResult(0).use_empty()) {
        rewriter.eraseOp(op);
      }
    }
  }
};
} // namespace mlir::tt::stablehlo
