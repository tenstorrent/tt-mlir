// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/ConstevalForwardAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace mlir::tt::ttir {

ConstevalForwardAnalysis::ConstevalForwardAnalysis(mlir::Operation *root)
    : root_(root) {}

void ConstevalForwardAnalysis::rebuild() {
  cache_.clear();

  // Pre-order matches SSA dominance for the operand-uses-result relation,
  // so by the time we visit an op its operands are already cached.
  root_->walk([&](mlir::func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    for (mlir::BlockArgument arg : funcOp.getArguments()) {
      bool isConst = ttcore::isConstOrParamArg(arg, funcOp);
      cache_[arg] = {isConst, !isConst};
    }

    funcOp.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (op == funcOp.getOperation()) {
        return;
      }

      // Nested-region block args (scf.for iter_args, linalg.generic, ...)
      // are never function-level CONST/PARAM args.
      for (mlir::Region &region : op->getRegions()) {
        for (mlir::Block &block : region.getBlocks()) {
          for (mlir::BlockArgument innerArg : block.getArguments()) {
            cache_[innerArg] = {false, true};
          }
        }
      }

      Reachability r;
      for (mlir::Value operand : op->getOperands()) {
        auto it = cache_.find(operand);
        if (it == cache_.end()) {
          // Operand defined outside the walked region. Treat
          // pessimistically as non-const; callers will skip the
          // optimization rather than apply an incorrect one.
          r.hasNonConstOrParamArg = true;
          continue;
        }
        r.hasConstOrParamArg |= it->second.hasConstOrParamArg;
        r.hasNonConstOrParamArg |= it->second.hasNonConstOrParamArg;
      }

      for (mlir::Value result : op->getResults()) {
        cache_[result] = r;
      }
    });
  });

  dirty_ = false;
}

bool ConstevalForwardAnalysis::valueTracesToConstantArgs(mlir::Value v) {
  if (dirty_) {
    rebuild();
  }
  auto it = cache_.find(v);
  if (it == cache_.end()) {
    // Not expected in EIO -- every Value reachable from `root_` is cached
    // after rebuild. Fall back to the slow path so the answer stays
    // correct for any unexpected query.
    return ttcore::valueTracesToConstantArgs(v);
  }
  return it->second.hasConstOrParamArg && !it->second.hasNonConstOrParamArg;
}

} // namespace mlir::tt::ttir
