// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/FunctionTypes.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYFORMEXPRESSIONS
#include "ttmlir/Conversion/Passes.h.inc"

// This pass packs certain patterns into expression ops so that they can be
// inlined. Patterns supported:
// 1. get dict value chain -> expression op   (from a forward function)
// 2. util_create_list call -> expression op  (from a consteval wrapper
// function)
// 3. absorb single-use operand chains into existing expressions (from a
// consteval wrapper function)

namespace {

using ttnn_to_emitpy::kCreateListFunctionName;
using ttnn_to_emitpy::kWrapperAttr;

static bool hasSingleUse(Operation *op) {
  return op->getNumResults() == 1 && op->getResult(0).hasOneUse();
}

static bool isCreateListCall(Operation *op) {
  auto callOp = dyn_cast<emitpy::CallOpaqueOp>(op);
  return callOp && callOp.getCallee() == kCreateListFunctionName;
}

static bool isPartOfSetDictValueChain(Operation *op) {
  if (!hasSingleUse(op)) {
    return false;
  }
  if (isa<emitpy::ConstantOp, emitpy::SubscriptOp, emitpy::LiteralOp>(op)) {
    return true;
  }
  return isCreateListCall(op);
}

class EmitPyFormExpressions
    : public impl::EmitPyFormExpressionsBase<EmitPyFormExpressions> {
public:
  using impl::EmitPyFormExpressionsBase<
      EmitPyFormExpressions>::EmitPyFormExpressionsBase;

  // Collect all ops from the get dict value chain into one
  // expression op.
  void foldDictGetPattern(func::FuncOp funcOp) {
    SmallVector<emitpy::SubscriptOp> candidates;
    funcOp.walk([&](emitpy::SubscriptOp subscriptOp) {
      if (!isa<emitpy::DictType>(subscriptOp.getContainer().getType())) {
        return;
      }
      auto *indexDef = subscriptOp.getIndex().getDefiningOp();
      if (!indexDef || !isa<emitpy::ConstantOp>(indexDef)) {
        return;
      }
      if (!isa<emitpy::StringType>(indexDef->getResult(0).getType())) {
        return;
      }
      if (!hasSingleUse(indexDef)) {
        return;
      }
      if (subscriptOp->getParentOfType<emitpy::ExpressionOp>()) {
        return;
      }
      candidates.push_back(subscriptOp);
    });

    OpBuilder builder(&getContext());
    for (auto &subscriptOp : candidates) {
      auto loc = subscriptOp.getLoc();
      auto constantOp =
          cast<emitpy::ConstantOp>(subscriptOp.getIndex().getDefiningOp());
      auto dict = subscriptOp.getContainer();

      builder.setInsertionPoint(constantOp);
      auto exprOp = builder.create<emitpy::ExpressionOp>(
          loc, subscriptOp.getResult().getType(), ValueRange{dict});

      auto *body = &exprOp.getBody().emplaceBlock();
      auto dictArg = body->addArgument(dict.getType(), dict.getLoc());

      subscriptOp.replaceAllUsesWith(exprOp.getResult());

      constantOp->moveBefore(body, body->end());
      subscriptOp->moveBefore(body, body->end());
      subscriptOp.getContainerMutable().assign(dictArg);

      // Add terminator at the end of the expression body.
      builder.setInsertionPointToEnd(body);
      builder.create<emitpy::YieldOp>(loc, subscriptOp.getResult());
    }
  }

  // Pack util_create_list call into an expression op so it can be inlined.
  void foldListIntoCallPattern(func::FuncOp funcOp) {
    SmallVector<emitpy::CallOpaqueOp> createListOp;
    funcOp.walk([&](emitpy::CallOpaqueOp listOp) {
      if (!isCreateListCall(listOp) || !hasSingleUse(listOp)) {
        return;
      }
      auto *user = *listOp->getResult(0).getUsers().begin();
      if (!isa<emitpy::CallOpaqueOp, emitpy::AssignOp>(user)) {
        return;
      }
      createListOp.push_back(listOp);
    });

    OpBuilder builder(&getContext());
    for (auto &listOp : createListOp) {
      auto loc = listOp.getLoc();
      ValueRange args = listOp.getOperands();

      builder.setInsertionPoint(listOp);
      auto exprOp = builder.create<emitpy::ExpressionOp>(
          loc, listOp.getResult(0).getType(), args);

      auto *body = &exprOp.getBody().emplaceBlock();
      for (auto arg : args) {
        body->addArgument(arg.getType(), arg.getLoc());
      }

      listOp.getResult(0).replaceAllUsesWith(exprOp.getResult());

      listOp->moveBefore(body, body->end());
      listOp.getOperandsMutable().assign(body->getArguments());

      builder.setInsertionPointToEnd(body);
      builder.create<emitpy::YieldOp>(loc, listOp.getResult(0));
    }
  }

  // Absorb single-use ops from the operand chain of expressions
  // into the expression body for more compact inlined output.
  void absorbOperandsIntoExpressions(func::FuncOp funcOp) {
    SmallVector<emitpy::ExpressionOp> exprs;
    funcOp.walk([&](emitpy::ExpressionOp exprOp) { exprs.push_back(exprOp); });

    for (auto exprOp : exprs) {
      // Walk a def-use chain starting from the expression op to collect all
      // ops to absorb.
      SetVector<Operation *> opsToAbsorb;
      SetVector<Operation *> worklist;
      worklist.insert(exprOp.getOperation());
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val();
        for (auto operand : op->getOperands()) {
          auto *operandDef = operand.getDefiningOp();
          if (operandDef && isPartOfSetDictValueChain(operandDef) &&
              opsToAbsorb.insert(operandDef)) {
            worklist.insert(operandDef);
          }
        }
      }

      if (opsToAbsorb.empty()) {
        continue;
      }

      SmallVector<Operation *> sortedOps(opsToAbsorb.begin(),
                                         opsToAbsorb.end());
      llvm::sort(sortedOps, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });

      // Collect all values used by the expression or absorbed ops that aren't
      // produced by another absorbed op. Those external operands will be
      // passed as new operands of the expression op.
      SetVector<Value> externalOperands;
      auto addIfExternal = [&](Value operand) {
        auto *operandDef = operand.getDefiningOp();
        if (!operandDef || !opsToAbsorb.contains(operandDef)) {
          externalOperands.insert(operand);
        }
      };

      for (auto *op : sortedOps) {
        for (auto operand : op->getOperands()) {
          addIfExternal(operand);
        }
      }
      for (auto operand : exprOp.getOperands()) {
        addIfExternal(operand);
      }

      auto loc = exprOp.getLoc();
      auto *exprBody = exprOp.getBodyBlock();
      unsigned oldNumArgs = exprBody->getNumArguments();

      // Add new block arguments for external operands. Build a value map
      // from external operands to block arguments for rewiring.
      DenseMap<Value, Value> extOperandsToBlockArgsMap;
      for (auto &extOperand : externalOperands) {
        extOperandsToBlockArgsMap[extOperand] =
            exprBody->addArgument(extOperand.getType(), loc);
      }

      // Redirect old block args and erase them.
      for (unsigned i = 0; i < oldNumArgs; ++i) {
        auto it = extOperandsToBlockArgsMap.find(exprOp.getOperand(i));
        exprBody->getArgument(i).replaceAllUsesWith(
            (it != extOperandsToBlockArgsMap.end()) ? it->second
                                                    : exprOp.getOperand(i));
      }
      for (int i = oldNumArgs - 1; i >= 0; --i) {
        exprBody->eraseArgument(i);
      }

      // Assign new operands to the expression op.
      exprOp.getOperandsMutable().assign(
          SmallVector<Value>(externalOperands.begin(), externalOperands.end()));

      // Move absorbed ops into the expression op body and rewire their
      // operands to block args.
      Operation *insertPt = &exprBody->front();
      for (auto *op : sortedOps) {
        op->moveBefore(insertPt);
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          auto it = extOperandsToBlockArgsMap.find(op->getOperand(i));
          if (it != extOperandsToBlockArgsMap.end()) {
            op->setOperand(i, it->second);
          }
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      if (ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        foldDictGetPattern(funcOp);
      }
      if (funcOp->hasAttr(kWrapperAttr)) {
        foldListIntoCallPattern(funcOp);
        absorbOperandsIntoExpressions(funcOp);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyFormExpressionsPass() {
  return std::make_unique<EmitPyFormExpressions>();
}

} // namespace mlir::tt
