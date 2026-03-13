// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYCONSTEVALCACHING
#include "ttmlir/Conversion/Passes.h.inc"

// Post-conversion pass that wraps const-eval function calls and their dict
// caching ops in an "if not dict:" guard. This ensures that cached results are
// only computed on the first invocation.
//
namespace {

using ttnn_to_emitpy::kConstEvaledAttr;
using ttnn_to_emitpy::kNameAttr;

class EmitPyConstEvalCaching
    : public impl::EmitPyConstEvalCachingBase<EmitPyConstEvalCaching> {
public:
  using impl::EmitPyConstEvalCachingBase<
      EmitPyConstEvalCaching>::EmitPyConstEvalCachingBase;

  // Collect all ops in the def-use chain.
  // Forward pass: transitively collect users of each seed op's results.
  // Backward pass: collect operand-defining ops, excluding the cache dict
  // defining op which must remain outside the if-guard.
  // Returns the collected ops sorted by block position.
  llvm::SmallVector<Operation *>
  collectDefUseChain(llvm::SetVector<Operation *> &ops, Block &block,
                     Value cacheDict) {
    // Forward: collect transitive users.
    llvm::SmallVector<Operation *> worklist(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (auto result : op->getResults()) {
        for (auto *user : result.getUsers()) {
          if (user->getBlock() == &block && ops.insert(user)) {
            worklist.push_back(user);
          }
        }
      }
    }

    // Backward: collect operand-defining ops. Skipping:
    // - The cache dict defining op (GlobalStatementOp in no-split files case;
    // otherwise, a cacheDict is a block arg).
    // - Ops with users outside the collected set (shared between const-eval
    //   and main paths, must stay in the parent block for dominance).
    Operation *cacheDictDefOp = cacheDict.getDefiningOp();
    worklist.assign(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (auto operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp == cacheDictDefOp || defOp->getBlock() != &block) {
          continue;
        }
        if (llvm::any_of(defOp->getUsers(), [&](Operation *user) {
              return user->getBlock() == &block && !ops.contains(user);
            })) {
          continue;
        }
        if (ops.insert(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }

    llvm::SmallVector<Operation *> sorted(ops.begin(), ops.end());
    llvm::sort(sorted, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    return sorted;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());

    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }

      Block &body = funcOp.getBody().front();

      // Identify the cache dictionary value.
      // No-split files case: produced by a GlobalStatementOp with dict type
      // at the top of the forward function body.
      // Split files case: first argument of the wrapper function.
      Value cacheDict = nullptr;

      for (auto globalStmt : body.getOps<emitpy::GlobalStatementOp>()) {
        if (isa<emitpy::DictType>(globalStmt.getResult().getType())) {
          cacheDict = globalStmt.getResult();
          // Set the name attribute of the callOp (this is the consteval wrapper
          // function call) to the name of the cache dictionary for that
          // function.
          for (auto *user : cacheDict.getUsers()) {
            if (auto callOp = dyn_cast<func::CallOp>(user)) {
              callOp->setDiscardableAttr(
                  kNameAttr, builder.getStringAttr(globalStmt.getName()));
            }
          }
          break;
        }
      }

      if (!cacheDict && body.getNumArguments() > 0 &&
          isa<emitpy::DictType>(body.getArgument(0).getType())) {
        cacheDict = body.getArgument(0);
      }

      if (!cacheDict) {
        return;
      }

      // Collect caching ops to put in the if body.
      llvm::SetVector<Operation *> opsToGuard;
      for (auto &op : body) {
        auto callOp = dyn_cast<emitpy::CallOpaqueOp>(&op);
        if (!callOp || !callOp->hasAttr(kConstEvaledAttr)) {
          continue;
        }
        opsToGuard.insert(callOp);
        callOp->removeDiscardableAttr(kConstEvaledAttr);
      }

      if (opsToGuard.empty()) {
        return;
      }

      llvm::SmallVector<Operation *> opsToGuardChain =
          collectDefUseChain(opsToGuard, body, cacheDict);

      // Create if-guard and move the caching ops into the if body.
      builder.setInsertionPoint(opsToGuardChain.front());
      auto ifOp = emitpy::IfOp::create(builder, funcOp.getLoc(),
                                       builder.getStringAttr("not {}"),
                                       ValueRange{cacheDict});

      auto *ifBody = builder.createBlock(&ifOp.getThenRegion());
      for (Operation *op : opsToGuardChain) {
        op->moveBefore(ifBody, ifBody->end());
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createEmitPyConstEvalCachingPass() {
  return std::make_unique<EmitPyConstEvalCaching>();
}

} // namespace mlir::tt
