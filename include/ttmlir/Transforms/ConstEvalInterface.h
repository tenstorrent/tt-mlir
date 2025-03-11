// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TRANSFORMS_CONSTEVALINTERFACE_H
#define TTMLIR_TRANSFORMS_CONSTEVALINTERFACE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt {

/// Interface for dialects to implement constant evaluation hoisting.
class ConstEvalHoistingInterface
    : public DialectInterface::Base<ConstEvalHoistingInterface> {
public:
  ConstEvalHoistingInterface(Dialect *dialect) : Base(dialect) {}
  virtual ~ConstEvalHoistingInterface() = default;

  /// Determines if an operation can be hoisted for constant evaluation.
  /// Returns true if the operation can be hoisted.
  virtual bool canHoistOp(Operation *op) const { return false; }

  /// Determines if an operation is guaranteed to have the same result
  /// for the same inputs (i.e., is functionally pure).
  virtual bool isPure(Operation *op) const { return false; }

  /// Returns true if the specified operation is suitable for caching.
  virtual bool isCacheable(Operation *op) const { return false; }

  /// Generates a name for the hoisted function containing this operation.
  /// Returns an empty string if no specific name is suggested.
  virtual std::string getHoistedFuncName(Operation *op) const { return ""; }

  /// Returns the set of operands that should be passed to the hoisted function.
  /// Returns an empty vector if all operands should be passed.
  virtual llvm::SmallVector<Value, 4> getHoistingOperands(Operation *op) const {
    return {};
  }

  /// Returns true if the given operation should terminate hoisting
  /// (i.e., hoisting should not cross this operation boundary).
  virtual bool isHoistingBarrier(Operation *op) const { return false; }
};

/// Register all dialect interfaces for constant evaluation hoisting.
void registerConstEvalHoistingInterfaces(DialectRegistry &registry);

/// Creates a pass that performs constant evaluation hoisting.
std::unique_ptr<Pass> createConstEvalHoistingPass();

} // namespace mlir::tt

#endif // TTMLIR_TRANSFORMS_CONSTEVALINTERFACE_H
