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
  }
};
} // namespace mlir::tt::stablehlo
