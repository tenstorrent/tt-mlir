// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt {

#define GEN_PASS_DEF_REMOVEDEADEMITCEXPRESSIONS
#define GEN_PASS_DEF_FORMDEDUPLICATEDEMITCEXPRESSIONS
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

struct UniqueOperands {
  SmallVector<Value> values;
  SmallVector<unsigned> remappedIndices;
  bool hasDuplicates = false;
};

static UniqueOperands collectUniqueOperands(ValueRange operands) {
  UniqueOperands uniqueOperands;
  uniqueOperands.remappedIndices.reserve(operands.size());

  DenseMap<Value, unsigned> uniqueIndices;
  uniqueIndices.reserve(operands.size());
  for (Value operand : operands) {
    auto [it, inserted] =
        uniqueIndices.try_emplace(operand, uniqueOperands.values.size());
    if (inserted) {
      uniqueOperands.values.push_back(operand);
    } else {
      uniqueOperands.hasDuplicates = true;
    }
    uniqueOperands.remappedIndices.push_back(it->second);
  }

  return uniqueOperands;
}

static bool deduplicateExpressionOperands(emitc::ExpressionOp expression);

static bool isFormableExpressionOp(Operation *op) {
  auto expression = dyn_cast<emitc::CExpressionInterface>(op);
  if (!expression || isa<emitc::LiteralOp>(op)) {
    return false;
  }

  if (isa_and_nonnull<emitc::ExpressionOp>(op->getParentOp())) {
    return false;
  }

  return op->getNumResults() == 1 && op->getResult(0).hasOneUse();
}

static void formDeduplicatedExpression(Operation *op) {
  UniqueOperands uniqueOperands = collectUniqueOperands(op->getOperands());
  OpBuilder builder(op);
  auto expression = builder.create<emitc::ExpressionOp>(
      op->getLoc(), op->getResult(0).getType(), uniqueOperands.values);
  Block &body = expression.createBody();

  op->getResult(0).replaceAllUsesWith(expression.getResult());
  op->moveBefore(&body, body.end());

  for (auto [index, uniqueIndex] :
       llvm::enumerate(uniqueOperands.remappedIndices)) {
    op->setOperand(index, body.getArgument(uniqueIndex));
  }

  builder.setInsertionPointAfter(op);
  builder.create<emitc::YieldOp>(op->getLoc(), op->getResult(0));
}

static bool deduplicateExpressionOperands(emitc::ExpressionOp expression) {
  UniqueOperands uniqueOperands =
      collectUniqueOperands(expression->getOperands());
  if (!uniqueOperands.hasDuplicates) {
    return false;
  }

  OpBuilder builder(expression);
  auto newExpression = builder.create<emitc::ExpressionOp>(
      expression.getLoc(), expression.getResult().getType(),
      uniqueOperands.values, expression.getDoNotInlineAttr());
  Block &newBody = newExpression.createBody();

  IRMapping mapping;
  Block &oldBody = expression.getRegion().front();
  for (auto [index, oldArgument] : llvm::enumerate(oldBody.getArguments())) {
    mapping.map(oldArgument,
                newBody.getArgument(uniqueOperands.remappedIndices[index]));
  }

  builder.setInsertionPointToEnd(&newBody);
  for (Operation &op : oldBody.getOperations()) {
    builder.clone(op, mapping);
  }

  expression.getResult().replaceAllUsesWith(newExpression.getResult());
  expression.erase();
  return true;
}

static bool isDeadEmitCExpression(Operation *op);

static bool hasNoUses(Operation *op) {
  return op->getNumResults() > 0 &&
         llvm::all_of(op->getResults(),
                      [](OpResult result) { return result.use_empty(); });
}

static bool hasOnlyDeadAssignUses(emitc::VariableOp op) {
  return llvm::all_of(op.getResult().getUses(), [](OpOperand &use) {
    return use.getOperandNumber() == 0 && isa<emitc::AssignOp>(use.getOwner());
  });
}

static bool isDeadEmitCExpression(Operation *op) {
  auto expression = dyn_cast<emitc::CExpressionInterface>(op);
  if (expression) {
    return hasNoUses(op) && !expression.hasSideEffects();
  }
  auto expressionOp = dyn_cast<emitc::ExpressionOp>(op);
  return expressionOp && hasNoUses(op) && !expressionOp.hasSideEffects();
}

static bool isDeadEmitCVariable(Operation *op) {
  auto variable = dyn_cast<emitc::VariableOp>(op);
  return variable && hasOnlyDeadAssignUses(variable);
}

static bool isDeadEmitCOp(Operation *op) {
  return isDeadEmitCExpression(op) || isDeadEmitCVariable(op);
}

static void enqueueIfDead(Operation *op, SmallVectorImpl<Operation *> &worklist,
                          DenseSet<Operation *> &queued) {
  if (op && !queued.contains(op) && isDeadEmitCOp(op)) {
    worklist.push_back(op);
    queued.insert(op);
  }
}

static void eraseDeadExpression(Operation *op,
                                SmallVectorImpl<Operation *> &worklist,
                                DenseSet<Operation *> &queued) {
  SmallVector<Operation *> definingOps;
  for (Value operand : op->getOperands()) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      definingOps.push_back(definingOp);
    }
  }

  op->erase();
  for (Operation *definingOp : definingOps) {
    enqueueIfDead(definingOp, worklist, queued);
  }
}

static void eraseDeadVariable(emitc::VariableOp variable,
                              SmallVectorImpl<Operation *> &worklist,
                              DenseSet<Operation *> &queued) {
  SmallVector<emitc::AssignOp> assignOps;
  SmallVector<Operation *> valueDefiningOps;
  for (OpOperand &use : variable.getResult().getUses()) {
    auto assign = cast<emitc::AssignOp>(use.getOwner());
    assignOps.push_back(assign);
    if (Operation *definingOp = assign.getValue().getDefiningOp()) {
      valueDefiningOps.push_back(definingOp);
    }
  }

  for (emitc::AssignOp assign : assignOps) {
    assign.erase();
  }
  variable.erase();
  for (Operation *definingOp : valueDefiningOps) {
    enqueueIfDead(definingOp, worklist, queued);
  }
}

class FormDeduplicatedEmitCExpressions
    : public impl::FormDeduplicatedEmitCExpressionsBase<
          FormDeduplicatedEmitCExpressions> {
public:
  using impl::FormDeduplicatedEmitCExpressionsBase<
      FormDeduplicatedEmitCExpressions>::FormDeduplicatedEmitCExpressionsBase;

  void runOnOperation() final {
    SmallVector<Operation *> candidates;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isFormableExpressionOp(op)) {
        candidates.push_back(op);
      }
    });

    for (Operation *op : candidates) {
      formDeduplicatedExpression(op);
    }

    SmallVector<emitc::ExpressionOp> expressions;
    getOperation().walk([&](emitc::ExpressionOp expression) {
      expressions.push_back(expression);
    });
    for (emitc::ExpressionOp expression : expressions) {
      deduplicateExpressionOperands(expression);
    }
  }
};

class RemoveDeadEmitCExpressions
    : public impl::RemoveDeadEmitCExpressionsBase<RemoveDeadEmitCExpressions> {
public:
  using impl::RemoveDeadEmitCExpressionsBase<
      RemoveDeadEmitCExpressions>::RemoveDeadEmitCExpressionsBase;

  void runOnOperation() final {
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> queued;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (!isDeadEmitCOp(op)) {
        return WalkResult::advance();
      }
      worklist.push_back(op);
      queued.insert(op);
      return WalkResult::skip();
    });

    for (unsigned index = 0; index < worklist.size(); ++index) {
      Operation *op = worklist[index];
      queued.erase(op);
      if (!isDeadEmitCOp(op)) {
        continue;
      }

      if (auto variable = dyn_cast<emitc::VariableOp>(op)) {
        eraseDeadVariable(variable, worklist, queued);
      } else {
        eraseDeadExpression(op, worklist, queued);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveDeadEmitCExpressionsPass() {
  return std::make_unique<RemoveDeadEmitCExpressions>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
createFormDeduplicatedEmitCExpressionsPass() {
  return std::make_unique<FormDeduplicatedEmitCExpressions>();
}

} // namespace mlir::tt
