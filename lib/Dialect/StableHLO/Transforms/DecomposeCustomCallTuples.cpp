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

    // Rewrite 2: stablehlo.custom_call ops whose sole result is a tuple.
    // A multi-output custom_call (e.g. tt.tt_lang_op with more than one "out"
    // operand) is emitted as a single value of tuple type, consumed by
    // get_tuple_element ops. Rewrite 1 does not cover this because the tuple
    // is the custom_call's own result rather than a stablehlo.tuple op. Shardy
    // cannot propagate through tuple types, so split the op into an equivalent
    // multi-result custom_call and forward each get_tuple_element user to the
    // matching result.
    SmallVector<mlir::stablehlo::CustomCallOp> tupleCustomCalls;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getNumResults() == 1 &&
          mlir::isa<mlir::TupleType>(op.getResult(0).getType())) {
        tupleCustomCalls.push_back(op);
      }
    });

    for (auto op : tupleCustomCalls) {
      auto tupleType = mlir::cast<mlir::TupleType>(op.getResult(0).getType());
      rewriter.setInsertionPoint(op);

      // Clone the op with one tensor result per tuple element, copying every
      // attribute (including the discardable mhlo.frontend_attributes that
      // carries kernel_id / arg_roles / version_tag). The operand/result
      // layout attributes are dropped: they are sized against the old single
      // tuple result, and stablehlo requires them to be present in pairs.
      OperationState state(op.getLoc(), op->getName().getStringRef());
      state.addOperands(op->getOperands());
      state.addTypes(llvm::to_vector(tupleType.getTypes()));
      for (mlir::NamedAttribute attr : op->getAttrs()) {
        if (attr.getName() == "operand_layouts" ||
            attr.getName() == "result_layouts") {
          continue;
        }
        state.addAttribute(attr.getName(), attr.getValue());
      }
      mlir::Operation *newCall = rewriter.create(state);

      mlir::Value oldResult = op.getResult(0);
      for (OpOperand &use :
           llvm::make_early_inc_range(oldResult.getUses())) {
        if (auto getTupleOp =
                dyn_cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner())) {
          rewriter.replaceOp(getTupleOp,
                             newCall->getResult(getTupleOp.getIndex()));
        }
      }
      // Any non-get_tuple_element user still expects a tuple value; rebuild
      // one from the split results so those uses stay well-typed.
      if (!oldResult.use_empty()) {
        auto rebuilt = rewriter.create<mlir::stablehlo::TupleOp>(
            op.getLoc(), tupleType, newCall->getResults());
        rewriter.replaceAllUsesWith(oldResult, rebuilt.getResult());
      }
      rewriter.eraseOp(op);
    }
  }
};
} // namespace mlir::tt::stablehlo
