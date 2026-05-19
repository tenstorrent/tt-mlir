// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::ttir {

// Cache for ttcore::valueTracesToConstantArgs over a func::FuncOp.
// EraseInverseOps queries the predicate from nearly every pattern match
// attempt; without caching the pass is O(num_ops^2). One forward walk
// per FuncOp populates the cache; subsequent queries are O(1).
//
// Acts as a RewriterBase::Listener: any IR mutation flips a dirty flag
// and the next query rebuilds from scratch, so cached Value pointers
// never go stale.
class ConstevalForwardAnalysis : public mlir::RewriterBase::Listener {
public:
  explicit ConstevalForwardAnalysis(mlir::Operation *root);

  // Returns the same answer as ttcore::valueTracesToConstantArgs(value),
  // i.e. whether `value`'s use-def chain reaches at least one CONST/PARAM
  // function block argument and no non-CONST/PARAM block argument.
  bool valueTracesToConstantArgs(mlir::Value value);

  // Listener overrides: any IR mutation invalidates the cache.
  void notifyOperationInserted(mlir::Operation *,
                               mlir::OpBuilder::InsertPoint) override {
    dirty_ = true;
  }
  void notifyOperationModified(mlir::Operation *) override { dirty_ = true; }
  void notifyOperationReplaced(mlir::Operation *, mlir::Operation *) override {
    dirty_ = true;
  }
  void notifyOperationReplaced(mlir::Operation *, mlir::ValueRange) override {
    dirty_ = true;
  }
  void notifyOperationErased(mlir::Operation *) override { dirty_ = true; }

private:
  void rebuild();

  // Tracks both bits: a value traces to constant args iff its use-def chain
  // reaches at least one CONST/PARAM arg and no non-CONST/PARAM arg.
  struct Reachability {
    bool hasConstOrParamArg = false;
    bool hasNonConstOrParamArg = false;
  };

  mlir::Operation *root_;
  bool dirty_ = true;
  llvm::DenseMap<mlir::Value, Reachability> cache_;
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_TRANSFORMS_ERASEINVERSEOPS_CONSTEVALFORWARDANALYSIS_H
