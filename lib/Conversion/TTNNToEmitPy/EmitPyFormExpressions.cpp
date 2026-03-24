// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYFORMEXPRESSIONS
#include "ttmlir/Conversion/Passes.h.inc"

// This pass packs certain op patterns into expression ops so that they can be
// inlined during code emission. Two transformations are applied:
//
// 1. foldSingleUseOps: Wraps single-use PyExpressionInterface ops into an
//    ExpressionOp.
//
// 2. absorbOperandsIntoExpressions: Walks operand def-use chains of existing
//    ExpressionOps and absorbs single-use ops into the expression body.
//

namespace {

static bool hasSingleUse(Operation *op) {
  return op->getNumResults() == 1 && op->getResult(0).hasOneUse();
}
static bool isCreateListCall(Operation *op) {
  auto callOp = dyn_cast<emitpy::CallOpaqueOp>(op);
  return callOp &&
         callOp.getCallee() == ttnn_to_emitpy::kCreateListFunctionName;
}

static bool isPyExprInterfaceSingleUseOp(Operation *op) {
  if (!hasSingleUse(op)) {
    return false;
  }
  if (!isa<emitpy::PyExpressionInterface>(op)) {
    return false;
  }
  // As for the CallOpaqueOp, only inline when it is a util_create_list call.
  if (dyn_cast<emitpy::CallOpaqueOp>(op) && !isCreateListCall(op)) {
    return false;
  }
  return true;
}

// Walk a def-use chain starting from the expression op to collect all
// ops to absorb.
SetVector<Operation *> collectOpsToAbsorb(emitpy::ExpressionOp exprOp) {
  SetVector<Operation *> worklist;
  SetVector<Operation *> opsToAbsorb;
  worklist.insert(exprOp.getOperation());
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (auto operand : op->getOperands()) {
      auto *operandDef = operand.getDefiningOp();
      if (operandDef && isPyExprInterfaceSingleUseOp(operandDef) &&
          opsToAbsorb.insert(operandDef)) {
        worklist.insert(operandDef);
      }
    }
  }
  return opsToAbsorb;
}

// Collect all values used by the expression or absorbed ops that aren't
// produced by another absorbed op. Those external operands will be
// passed as new operands of the expression op.
SetVector<Value> collectExternalOperands(emitpy::ExpressionOp exprOp,
                                         SetVector<Operation *> &opsToAbsorb) {
  SetVector<Value> externalOperands;
  auto addIfExternal = [&opsToAbsorb](Value operand,
                                      SetVector<Value> &externalOperands) {
    auto *operandDef = operand.getDefiningOp();
    if (!operandDef || !opsToAbsorb.contains(operandDef)) {
      externalOperands.insert(operand);
    }
  };

  for (auto *op : llvm::reverse(opsToAbsorb)) {
    for (auto operand : op->getOperands()) {
      addIfExternal(operand, externalOperands);
    }
  }
  for (auto operand : exprOp.getOperands()) {
    addIfExternal(operand, externalOperands);
  }
  return externalOperands;
}

static void
rewireExpressionOperandsAndArgs(emitpy::ExpressionOp exprOp,
                                DenseMap<Value, Value> &operandsToBlockArgsMap,
                                SetVector<Value> &newOperands) {
  auto *exprBody = exprOp.getBodyBlock();
  unsigned oldNumArgs = exprBody->getNumArguments();

  // Redirect old block args and erase them.
  for (unsigned i = 0; i < oldNumArgs; ++i) {
    auto it = operandsToBlockArgsMap.find(exprOp.getOperand(i));
    exprBody->getArgument(i).replaceAllUsesWith(
        (it != operandsToBlockArgsMap.end()) ? it->second
                                             : exprOp.getOperand(i));
  }
  for (int i = oldNumArgs - 1; i >= 0; --i) {
    exprBody->eraseArgument(i);
  }

  // Assign new operands to the expression op.
  exprOp.getOperandsMutable().assign(
      SmallVector<Value>(newOperands.begin(), newOperands.end()));
}

// Move absorbed ops into the expression op body and rewire their
// operands to block args.
static void rewireAbsorbedOps(emitpy::ExpressionOp exprOp,
                              DenseMap<Value, Value> &operandsToBlockArgsMap,
                              SetVector<Operation *> &opsToAbsorb) {
  Operation *insertPt = &exprOp.getBodyBlock()->front();
  for (auto *op : llvm::reverse(opsToAbsorb)) {
    op->moveBefore(insertPt);
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = operandsToBlockArgsMap.find(op->getOperand(i));
      if (it != operandsToBlockArgsMap.end()) {
        op->setOperand(i, it->second);
      }
    }
  }
}

class EmitPyFormExpressions
    : public impl::EmitPyFormExpressionsBase<EmitPyFormExpressions> {
public:
  using impl::EmitPyFormExpressionsBase<
      EmitPyFormExpressions>::EmitPyFormExpressionsBase;
  // Wrap single-use PyExpressionInterface ops into ExpressionOps.
  void foldSingleUseOps(func::FuncOp funcOp) {
    SmallVector<Operation *> candidates;
    funcOp.walk([&candidates](Operation *op) {
      if (isa<emitpy::ExpressionOp>(op->getParentOp())) {
        return;
      }
      if (!isPyExprInterfaceSingleUseOp(op)) {
        return;
      }
      candidates.push_back(op);
    });

    OpBuilder builder(&getContext());
    for (auto *op : candidates) {
      auto loc = op->getLoc();
      ValueRange operands = op->getOperands();

      builder.setInsertionPoint(op);
      auto exprOp = builder.create<emitpy::ExpressionOp>(
          loc, op->getResult(0).getType(), operands);

      auto *body = &exprOp.getBody().emplaceBlock();
      for (auto operand : operands) {
        body->addArgument(operand.getType(), operand.getLoc());
      }

      op->getResult(0).replaceAllUsesWith(exprOp.getResult());

      op->moveBefore(body, body->end());
      for (unsigned i = 0; i < operands.size(); ++i) {
        op->setOperand(i, body->getArgument(i));
      }

      builder.setInsertionPointToEnd(body);
      builder.create<emitpy::YieldOp>(loc, op->getResult(0));
    }
  }

  // Absorb single-use ops from the operand chain of expressions
  // into the expression body for more compact inlined output.
  void absorbOperandsIntoExpressions(func::FuncOp funcOp) {
    SmallVector<emitpy::ExpressionOp> exprs;
    funcOp.walk(
        [&exprs](emitpy::ExpressionOp exprOp) { exprs.push_back(exprOp); });

    for (auto exprOp : exprs) {
      SetVector<Operation *> opsToAbsorb = collectOpsToAbsorb(exprOp);

      if (opsToAbsorb.empty()) {
        continue;
      }

      SetVector<Value> externalOperands =
          collectExternalOperands(exprOp, opsToAbsorb);

      // Add new block arguments for external operands. Build a value map
      // from external operands to block arguments for rewiring.
      auto loc = exprOp.getLoc();
      auto *exprBody = exprOp.getBodyBlock();
      DenseMap<Value, Value> extOperandsToBlockArgsMap;
      for (auto extOperand : externalOperands) {
        extOperandsToBlockArgsMap[extOperand] =
            exprBody->addArgument(extOperand.getType(), loc);
      }

      rewireExpressionOperandsAndArgs(exprOp, extOperandsToBlockArgsMap,
                                      externalOperands);
      rewireAbsorbedOps(exprOp, extOperandsToBlockArgsMap, opsToAbsorb);
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      foldSingleUseOps(funcOp);
      absorbOperandsIntoExpressions(funcOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyFormExpressionsPass() {
  return std::make_unique<EmitPyFormExpressions>();
}

} // namespace mlir::tt
