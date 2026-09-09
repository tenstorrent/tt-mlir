// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/ConstevalForwardAnalysis.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt::ttir {

ConstevalForwardAnalysis::ConstevalForwardAnalysis(mlir::Operation *root)
    : root_(root) {}

std::optional<ConstevalForwardAnalysis::Reachability>
ConstevalForwardAnalysis::computeFromOperands(mlir::Operation *op) {
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
      // Operand not cached: in rebuild() this never happens (pre-order
      // caches operands first); in the incremental path it means we cannot
      // reason about this op, so the caller must fall back to a full
      // rebuild rather than risk a wrong (favorable) answer.
      return std::nullopt;
    }
    r.hasConstOrParamArg |= it->second.hasConstOrParamArg;
    r.hasNonConstOrParamArg |= it->second.hasNonConstOrParamArg;
  }
  return r;
}

void ConstevalForwardAnalysis::propagateFrom(mlir::Value seed) {
  llvm::SmallVector<mlir::Value> work{seed};
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    mlir::Operation *def = v.getDefiningOp();
    if (!def) {
      // Block args are seeded separately and never change during rewriting.
      continue;
    }

    std::optional<Reachability> r = computeFromOperands(def);
    if (!r) {
      // Safe fallback: an operand is not cached, so rebuild on next query.
      dirty_ = true;
      return;
    }

    for (mlir::Value result : def->getResults()) {
      auto it = cache_.find(result);
      if (it == cache_.end() || it->second != *r) {
        cache_[result] = *r;
        // Const-ness flows strictly forward: only downstream users can flip.
        for (mlir::Operation *user : result.getUsers()) {
          for (mlir::Value userResult : user->getResults()) {
            work.push_back(userResult);
          }
        }
      }
    }
  }
}

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

      std::optional<Reachability> r = computeFromOperands(op);
      // Pre-order guarantees operands are cached, so r always has a value.
      assert(r.has_value() && "rebuild: operands must be cached in pre-order");
      Reachability reachability = *r;

      for (mlir::Value result : op->getResults()) {
        cache_[result] = reachability;
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
