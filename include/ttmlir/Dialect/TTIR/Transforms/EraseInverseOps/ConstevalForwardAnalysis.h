// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir::tt::ttir {

// Cache for ttcore::valueTracesToConstantArgs over a func::FuncOp.
// EraseInverseOps queries the predicate from nearly every pattern match
// attempt; without caching the pass is O(num_ops^2). One forward walk
// per FuncOp populates the cache; subsequent queries are O(1).
//
// Acts as a RewriterBase::Listener. Const-ness flows strictly forward along
// the use-def chain, so each IR mutation can only change the touched op's
// results and their transitive users. The listener updates exactly those
// entries incrementally instead of dropping the whole cache. Any situation
// the incremental path cannot reason about (an operand not yet cached, an
// uncertain notification ordering) falls back to `dirty_ = true`, so the next
// query rebuilds from scratch and correctness is never sacrificed.
class ConstevalForwardAnalysis : public mlir::RewriterBase::Listener {
public:
  explicit ConstevalForwardAnalysis(mlir::Operation *root);

  // Returns the same answer as ttcore::valueTracesToConstantArgs(value),
  // i.e. whether `value`'s use-def chain reaches at least one CONST/PARAM
  // function block argument and no non-CONST/PARAM block argument.
  bool valueTracesToConstantArgs(mlir::Value value);

  // Listener overrides. Const-ness only propagates forward, so each hook
  // re-derives the affected entries and pushes the change to downstream
  // users. If anything is uncertain the hook sets `dirty_` and the next
  // query rebuilds.
  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint) override {
    if (dirty_) {
      return;
    }
    // Operands are pre-existing (already cached); seed from this op's
    // results. A missing operand makes propagateFrom set dirty_.
    for (mlir::Value result : op->getResults()) {
      propagateFrom(result);
    }
  }
  void notifyOperationModified(mlir::Operation *op) override {
    if (dirty_) {
      return;
    }
    // Operands may have been rewired; recompute results and propagate.
    for (mlir::Value result : op->getResults()) {
      propagateFrom(result);
    }
  }
  void notifyOperationReplaced(mlir::Operation *op,
                               mlir::Operation *newOp) override {
    if (dirty_) {
      return;
    }
    // Old results are RAUW'd to the new op's results; drop the stale
    // entries and re-derive from the replacements.
    for (mlir::Value result : op->getResults()) {
      cache_.erase(result);
    }
    for (mlir::Value result : newOp->getResults()) {
      propagateFrom(result);
    }
  }
  void notifyOperationReplaced(mlir::Operation *op,
                               mlir::ValueRange newValues) override {
    if (dirty_) {
      return;
    }
    for (mlir::Value result : op->getResults()) {
      cache_.erase(result);
    }
    for (mlir::Value newValue : newValues) {
      propagateFrom(newValue);
    }
  }
  void notifyOperationErased(mlir::Operation *op) override {
    if (dirty_) {
      return;
    }
    // No users remain on an erased op's results.
    for (mlir::Value result : op->getResults()) {
      cache_.erase(result);
    }
  }

private:
  // Tracks both bits: a value traces to constant args iff its use-def chain
  // reaches at least one CONST/PARAM arg and no non-CONST/PARAM arg.
  struct Reachability {
    bool hasConstOrParamArg = false;
    bool hasNonConstOrParamArg = false;

    bool operator==(const Reachability &other) const {
      return hasConstOrParamArg == other.hasConstOrParamArg &&
             hasNonConstOrParamArg == other.hasNonConstOrParamArg;
    }
    bool operator!=(const Reachability &other) const {
      return !(*this == other);
    }
  };

  void rebuild();

  // Computes the Reachability for `op`'s results from its operands' cached
  // entries -- identical logic to the rebuild() inner loop, so there is
  // exactly one definition of the rule. Also seeds nested-region block args
  // (never function CONST/PARAM args). Returns std::nullopt if any operand is
  // not yet cached, so the incremental caller can fall back to rebuild().
  std::optional<Reachability> computeFromOperands(mlir::Operation *op);

  // Recomputes `seed`'s defining op entry; if it changed, pushes its users.
  // Const-ness flows strictly forward, so the frontier is bounded by the
  // touched value's downstream users. Sets `dirty_` (full rebuild next query)
  // if it hits an operand that is not cached.
  void propagateFrom(mlir::Value seed);

  mlir::Operation *root_;
  bool dirty_ = true;
  llvm::DenseMap<mlir::Value, Reachability> cache_;
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H
