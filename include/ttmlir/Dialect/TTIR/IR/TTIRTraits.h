// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace tt {
namespace ttir {
namespace OpTrait {

template <typename ConcreteType>
class TTIRInvolution
    : public mlir::TypeTrait::TraitBase<ConcreteType, TTIRInvolution> {
public:
  static mlir::LogicalResult foldTrait(mlir::Operation *op,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<OpFoldResult> &results) {
    if (isFoldableOperation(op)) {
      results.push_back(op->getOperand(0).getDefiningOp()->getOperand(0));
      return mlir::success();
    }
    return mlir::failure();
  }

private:
  // Op is foldable iff:
  // 1. argment and result types are the same.
  // 2. argument is defined by the same op.
  // 3. 1) is true for the producing op of the argument.
  // op(op(T a, T r0), T r1)
  static bool isFoldableOperation(mlir::Operation *op) {
    // Dependant trait of TTIRInvolution is DestionationStyleOpInterface, hence
    // operands include the result.
    auto operandAndResultSameType = [](Operation *op) {
      return llvm::all_equal(op->getOperandTypes());
    };
    if (!operandAndResultSameType(op)) {
      return false;
    }
    Operation *producerOp = op->getOperand(0).getDefiningOp();
    if (!producerOp || producerOp->getName() != op->getName()) {
      return false;
    }
    return operandAndResultSameType(producerOp);
  }
};

template <typename ConcreteType>
class TTIRIdempotence
    : public mlir::TypeTrait::TraitBase<ConcreteType, TTIRIdempotence> {
public:
  static mlir::LogicalResult foldTrait(mlir::Operation *op,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<OpFoldResult> &results) {
    if (isFoldableOperation(op)) {
      results.push_back(op->getOperand(0));
      return mlir::success();
    }
    return mlir::failure();
  }

private:
  // Op is foldable iff:
  // 1. argment and result types are the same.
  // 2. argument is defined by the same op.
  // 3. 1) is true for the producing op of the argument.
  // op(op(T a, T r0), T r1)
  static bool isFoldableOperation(mlir::Operation *op) {
    // Dependant trait of TTIRIdempotence is DestionationStyleOpInterface, hence
    // operands include the result.
    auto operandAndResultSameType = [](mlir::Operation *op) {
      return llvm::all_equal(op->getOperandTypes());
    };
    if (!operandAndResultSameType(op)) {
      return false;
    }
    mlir::Operation *producerOp = op->getOperand(0).getDefiningOp();
    if (!producerOp || producerOp->getName() != op->getName()) {
      return false;
    }
    return operandAndResultSameType(producerOp);
  }
};

} // namespace OpTrait
} // namespace ttir
} // namespace tt
} // namespace mlir

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H
