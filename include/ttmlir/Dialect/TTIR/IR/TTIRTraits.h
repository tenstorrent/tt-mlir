// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tt {
namespace ttir {
namespace OpTrait {

template <typename ConcreteType>
class ConditionalDPSInvolution
    : public mlir::TypeTrait::TraitBase<ConcreteType,
                                        ConditionalDPSInvolution> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    static_assert(ConcreteType::template hasTrait<
                      mlir::DestinationStyleOpInterface::Trait>(),
                  "expected destination passing style operation");
    static_assert(
        ConcreteType::template hasTrait<mlir::OpTrait::NOperands<2>::Impl>(),
        "expected operation to take two operands");
    // Operation does have the expected traits, foldability has to be
    // dinamically checked.
    return mlir::success();
  }

  static mlir::LogicalResult foldTrait(mlir::Operation *op,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<OpFoldResult> &results) {
    llvm::errs() << "FOLDING DPS INVOLUTION\n";
    if (isFoldableOperation(op)) {
      results.push_back(op->getOperand(0).getDefiningOp()->getOperand(0));
      return mlir::success();
    }
    return failure();
  }

private:
  // Op is foldable iff:
  // 1. argment and result types are the same.
  // 2. argument is defined by the same op.
  // 3. 1) is true for the producing op of the argument.
  // op(op(T a, T r0), T r1)

  // TODO (azecevic): refactor this into the separate function
  static bool isFoldableOperation(mlir::Operation *op) {
    // TODO (azecevic): LEAVE A COMMENT ABOUT OPERANDS(1) == RESULT(0) BECAUSE
    // OF DPS
    llvm::errs() << "FOLDABILITY CHECK\n";
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
class ConditionalDPSIdempotence
    : public mlir::TypeTrait::TraitBase<ConcreteType,
                                        ConditionalDPSIdempotence> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    static_assert(ConcreteType::template hasTrait<
                      mlir::DestinationStyleOpInterface::Trait>(),
                  "expected destination passing style operation");
    static_assert(
        ConcreteType::template hasTrait<mlir::OpTrait::NOperands<2>::Impl>(),
        "expected operation to take two operands");
    // Operation does have the expected traits, foldability has to be
    // dinamically checked.
    return mlir::success();
  }

  static mlir::LogicalResult foldTrait(mlir::Operation *op,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<OpFoldResult> &results) {
    llvm::errs() << "FOLDING DPS INVOLUTION\n";
    if (isFoldableOperation(op)) {
      results.push_back(op->getOperand(0));
      return mlir::success();
    }
    return failure();
  }

private:
  // Op is foldable iff:
  // 1. argment and result types are the same.
  // 2. argument is defined by the same op.
  // 3. 1) is true for the producing op of the argument.
  // op(op(T a, T r0), T r1)
  static bool isFoldableOperation(mlir::Operation *op) {
    // TODO (azecevic): LEAVE A COMMENT ABOUT OPERANDS(1) == RESULT(0) BECAUSE
    // OF DPS
    llvm::errs() << "FOLDABILITY CHECK\n";
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

} // namespace OpTrait
} // namespace ttir
} // namespace tt
} // namespace mlir

#endif // TTMLIR_TTMLIR_DIALECT_TTIR_TTIRTRAITS_H
