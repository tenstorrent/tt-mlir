// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/IRHasher.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdlib>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNINSERTDISKCACHEOPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNInsertDiskCacheOps
    : public impl::TTNNInsertDiskCacheOpsBase<TTNNInsertDiskCacheOps> {
public:
  using impl::TTNNInsertDiskCacheOpsBase<
      TTNNInsertDiskCacheOps>::TTNNInsertDiskCacheOpsBase;

  void runOnOperation() final {
    // Check env var - if not set, skip this pass
    if (!std::getenv("TTMLIR_ENABLE_DISK_CACHE")) {
      return;
    }

    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](func::FuncOp func) {
      // Skip private and const-eval functions
      if (func.isPrivate() || ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      // Skip functions with no body
      if (func.getBody().empty()) {
        return;
      }

      // Compute program hash once per function
      std::string programHash = hashFuncOp(func);

      // Process each tensor argument
      for (auto [argIndex, arg] : llvm::enumerate(func.getArguments())) {
        if (!mlir::isa<RankedTensorType>(arg.getType())) {
          continue;
        }

        // Verify layout consistency among users
        if (failed(verifyLayoutConsistency(arg))) {
          signalPassFailure();
          return;
        }

        // Insert op at function entry
        rewriter.setInsertionPointToStart(&func.getBody().front());
        auto cacheOp = rewriter.create<ttcore::GetOrInsertIntoDiskCacheOp>(
            arg.getLoc(),
            arg.getType(), // result type
            arg,           // input
            programHash, static_cast<uint32_t>(argIndex));

        // Replace all uses of arg with cache op result (except the cache op
        // itself)
        arg.replaceAllUsesExcept(cacheOp.getResult(), cacheOp);
      }
    });
  }

private:
  // Verify all users expect the same layout
  LogicalResult verifyLayoutConsistency(BlockArgument arg) {
    auto tensorType = mlir::cast<RankedTensorType>(arg.getType());
    auto argLayout =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding());

    TTNNLayoutAttr expectedLayout = argLayout;

    for (Operation *user : arg.getUsers()) {
      // Skip the cache op we might be inserting
      if (mlir::isa<ttcore::GetOrInsertIntoDiskCacheOp>(user)) {
        continue;
      }

      // Get expected layout from user's operand type
      for (auto &operand : user->getOpOperands()) {
        if (operand.get() != arg) {
          continue;
        }

        auto operandType =
            mlir::dyn_cast<RankedTensorType>(operand.get().getType());
        if (!operandType) {
          continue;
        }

        auto userLayout =
            mlir::dyn_cast_or_null<TTNNLayoutAttr>(operandType.getEncoding());

        if (!expectedLayout) {
          expectedLayout = userLayout;
        } else if (userLayout && expectedLayout != userLayout) {
          return user->emitError("Layout mismatch: argument has conflicting "
                                 "layout requirements from different users");
        }
      }
    }

    return success();
  }
};

} // namespace

} // namespace mlir::tt::ttnn
