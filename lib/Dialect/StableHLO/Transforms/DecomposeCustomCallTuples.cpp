// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_DECOMPOSECUSTOMCALLTUPLESPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral TTLangTargetName = "tt.tt_lang_op";
} // namespace

struct DecomposeCustomCallTuplesPass
    : public impl::DecomposeCustomCallTuplesPassBase<
          DecomposeCustomCallTuplesPass> {
public:
  using impl::DecomposeCustomCallTuplesPassBase<
      DecomposeCustomCallTuplesPass>::DecomposeCustomCallTuplesPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(getOperation().getContext());

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

    // Convert all tt.tt_lang_op custom calls with a tuple result into a
    // multi-result custom call.
    SmallVector<mlir::stablehlo::CustomCallOp> tupleCustomCalls;
    getOperation().walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getCallTargetName() == TTLangTargetName &&
          op.getNumResults() == 1 &&
          mlir::isa<mlir::TupleType>(op.getResult(0).getType())) {
        tupleCustomCalls.push_back(op);
      }
    });

    for (mlir::stablehlo::CustomCallOp op : tupleCustomCalls) {
      if (mlir::failed(decomposeTupleResult(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  static mlir::LogicalResult
  decomposeTupleResult(mlir::stablehlo::CustomCallOp op, IRRewriter &rewriter) {
    auto tupleType = mlir::cast<mlir::TupleType>(op.getResult(0).getType());
    mlir::Value oldResult = op.getResult(0);

    // Splitting flattens exactly one level, so a nested tuple would leave a
    // tuple-typed result behind and Shardy would still reject the op.
    for (mlir::Type elementType : tupleType.getTypes()) {
      if (mlir::isa<mlir::TupleType>(elementType)) {
        return op.emitError()
               << "cannot decompose nested tuple result " << tupleType
               << " of '" << TTLangTargetName << "'";
      }
    }

    for (mlir::Operation *user : oldResult.getUsers()) {
      if (!mlir::isa<mlir::stablehlo::GetTupleElementOp>(user)) {
        return op.emitError()
               << "cannot decompose tuple result of '" << TTLangTargetName
               << "': expected every user to be a "
                  "stablehlo.get_tuple_element, but found '"
               << user->getName() << "'";
      }
    }

    if (tupleType.size() == 1 && !op.getOutputOperandAliases().empty()) {
      return op.emitError()
             << "cannot decompose single-element tuple result of '"
             << TTLangTargetName
             << "' while `output_operand_aliases` is set: the alias indices "
                "address a tuple that the split removes";
    }

    rewriter.setInsertionPoint(op);

    // Rebuild with one tensor result per tuple element.
    auto newCall = rewriter.create<mlir::stablehlo::CustomCallOp>(
        op.getLoc(), llvm::to_vector(tupleType.getTypes()), op.getOperands(),
        op.getCallTargetNameAttr(), op.getHasSideEffectAttr(),
        op.getBackendConfigAttr(), op.getApiVersionAttr(),
        op.getCalledComputationsAttr(),
        /*operand_layouts=*/nullptr,
        /*result_layouts=*/nullptr, op.getOutputOperandAliasesAttr());

    newCall->setDiscardableAttrs(op->getDiscardableAttrDictionary());

    for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
      auto getTupleOp =
          mlir::cast<mlir::stablehlo::GetTupleElementOp>(use.getOwner());
      rewriter.replaceOp(getTupleOp, newCall->getResult(getTupleOp.getIndex()));
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};
} // namespace mlir::tt::stablehlo
