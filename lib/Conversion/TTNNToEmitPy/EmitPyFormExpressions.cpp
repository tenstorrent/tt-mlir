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

// This pass wraps single-use PyExpressionInterface ops into ExpressionOps
// for inline emission. Exceptions are callOpaqueOps that are not creating
// a list. Those are not inlined, to prevent from inlining of consteval
// functions calls and provide cleaner emission of ttnn ops calls.
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

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      foldSingleUseOps(funcOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyFormExpressionsPass() {
  return std::make_unique<EmitPyFormExpressions>();
}

} // namespace mlir::tt
