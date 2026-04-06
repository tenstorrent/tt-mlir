// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitPy/EmitPyConversion.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/EmitPy/IR/EmitPyOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt {

#define GEN_PASS_DEF_EMITPYCONSTEVALCACHING
#include "ttmlir/Conversion/Passes.h.inc"

// Post-conversion pass that wraps const-eval function calls from the consteval
// wrapper functions in an "if not dict:" guards. This ensures that cached
// results are only computed on the first invocation.
//
namespace {

using ttnn_to_emitpy::kConstEvaledAttr;
using ttnn_to_emitpy::kNameAttr;

constexpr const char *kCacheDictAttr = "cache_dict";

class EmitPyConstEvalCaching
    : public impl::EmitPyConstEvalCachingBase<EmitPyConstEvalCaching> {
public:
  using impl::EmitPyConstEvalCachingBase<
      EmitPyConstEvalCaching>::EmitPyConstEvalCachingBase;

  // Collect all ops in the def-use chain.
  // Forward pass: transitively collect users of the given ops.
  // Backward pass: collect operand-defining ops.
  // Returns the collected ops sorted by block position.
  llvm::SmallVector<Operation *>
  collectDefUseChain(llvm::SetVector<Operation *> &ops, Block &block) {
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

    // Backward: collect operand-defining ops. Skipping ops with users outside
    // the collected set.
    worklist.assign(ops.begin(), ops.end());
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      for (auto operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != &block) {
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

      // Set the `emitpy.name` attribute of the consteval wrapper function call
      // in the forward function body.
      if (ttmlir::utils::isForwardDeviceFunc(funcOp)) {
        for (auto globalStmt : body.getOps<emitpy::GlobalStatementOp>()) {
          if (globalStmt->hasAttr(kCacheDictAttr)) {
            Value cacheDict = globalStmt.getResult();
            for (auto *user : cacheDict.getUsers()) {
              if (auto callOp = dyn_cast<func::CallOp>(user)) {
                func::FuncOp callee = nullptr;
                moduleOp.walk([&](func::FuncOp funcOp) {
                  if (!funcOp.isDeclaration() &&
                      funcOp.getName() == callOp.getCallee()) {
                    callee = funcOp;
                    return WalkResult::interrupt();
                  }
                  return WalkResult::advance();
                });
                if (callee && ttmlir::utils::isConstEvalWrapperFunc(callee)) {
                  callOp->setDiscardableAttr(
                      kNameAttr, builder.getStringAttr(globalStmt.getName()));
                  return;
                }
              }
            }
            globalStmt->removeDiscardableAttr(kCacheDictAttr);
          }
        }
      }

      if (!ttmlir::utils::isConstEvalWrapperFunc(funcOp)) {
        return;
      }

      // Identify the cache dictionary value. It is the first argument of the
      // consteval wrapper function.
      Value cacheDict = funcOp.getArgument(0);
      assert(cacheDict && "Cache dictionary not found as an argument of the "
                          "consteval wrapper function");

      // Collect caching ops to put in the if body.
      llvm::SetVector<Operation *> opsToGuard;
      for (auto &op : body) {
        auto callOp = dyn_cast<emitpy::CallOpaqueOp>(&op);
        if (callOp && callOp->hasAttr(kConstEvaledAttr)) {
          opsToGuard.insert(callOp);
          callOp->removeDiscardableAttr(kConstEvaledAttr);
        }
      }

      assert(!opsToGuard.empty() && "No caching ops found in the consteval "
                                    "wrapper function body");

      llvm::SmallVector<Operation *> opsToGuardChain =
          collectDefUseChain(opsToGuard, body);

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
