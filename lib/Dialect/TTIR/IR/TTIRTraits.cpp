// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"

// Check if all operands and result have the same type. Function assumes op has
// at least one operand and exactly one result.
static bool operandAndResultSameType(mlir::Operation *op) {
  return llvm::all_equal(op->getOperandTypes()) &&
         op->getOperand(0).getType() == op->getResult(0).getType();
}

// If Op has TTIRInvolution trait, then it's foldable if:
// 1. Argument and result types are the same.
// 2. Argument is defined by the same op.
// 3. 1) is true for the producing op of the argument.
// op(op(T a, T r0), T r1)
bool mlir::tt::ttir::OpTrait::impl::verifyInvolution(mlir::Operation *op) {
  if (!operandAndResultSameType(op)) {
    return false;
  }
  Operation *producerOp = op->getOperand(0).getDefiningOp();
  if (!producerOp || producerOp->getName() != op->getName()) {
    return false;
  }
  return operandAndResultSameType(producerOp);
}

// If Op has TTIRIdempotence trait, then it's foldable if:
// 1. Argument and result types are the same.
// 2. Argument is defined by the same op.
// 3. 1) is true for the producing op of the argument.
// op(op(T a, T r0), T r1)
bool mlir::tt::ttir::OpTrait::impl::verifyIdempotence(mlir::Operation *op) {
  if (!operandAndResultSameType(op)) {
    return false;
  }
  mlir::Operation *producerOp = op->getOperand(0).getDefiningOp();
  if (!producerOp || producerOp->getName() != op->getName()) {
    return false;
  }
  return operandAndResultSameType(producerOp);
}

// If Op has TTIRBinaryIdempotence trait, then it's foldable if:
// 1. Both inputs are the same.
// 2. Inputs and result types are the same.
bool mlir::tt::ttir::OpTrait::impl::verifyBinaryIdempotence(
    mlir::Operation *op) {
  if (op->getOperand(0) != op->getOperand(1)) {
    return false;
  }

  return op->getResult(0).getType() == op->getOperand(0).getType();
}

mlir::OpFoldResult
mlir::tt::ttir::OpTrait::impl::foldInvolution(mlir::Operation *op) {
  return op->getOperand(0).getDefiningOp()->getOperand(0);
}

mlir::OpFoldResult
mlir::tt::ttir::OpTrait::impl::foldIdempotence(mlir::Operation *op) {
  return op->getOperand(0);
}

mlir::OpFoldResult
mlir::tt::ttir::OpTrait::impl::foldBinaryIdempotence(mlir::Operation *op) {
  return op->getOperand(0);
}
