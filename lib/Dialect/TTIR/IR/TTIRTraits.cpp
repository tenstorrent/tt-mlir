// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

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
bool mlir::tt::ttir::impl::verifyInvolution(mlir::Operation *op) {
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
bool mlir::tt::ttir::impl::verifyIdempotence(mlir::Operation *op) {
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
bool mlir::tt::ttir::impl::verifyBinaryIdempotence(mlir::Operation *op) {
  if (op->getOperand(0) != op->getOperand(1)) {
    return false;
  }

  return op->getResult(0).getType() == op->getOperand(0).getType();
}

mlir::OpFoldResult mlir::tt::ttir::impl::foldInvolution(mlir::Operation *op) {
  return op->getOperand(0).getDefiningOp()->getOperand(0);
}

mlir::OpFoldResult mlir::tt::ttir::impl::foldIdempotence(mlir::Operation *op) {
  return op->getOperand(0);
}

mlir::OpFoldResult
mlir::tt::ttir::impl::foldBinaryIdempotence(mlir::Operation *op) {
  return op->getOperand(0);
}

static mlir::LogicalResult
verifyGenericRegionOpThreadType(mlir::Operation *op,
                                mlir::tt::ttir::ThreadType threadType) {
  mlir::Region *region =
      ttmlir::utils::getRegionWithParentOfType<mlir::tt::ttir::GenericOp>(op);
  if (!region) {
    // If not enclosed in a generic op then we forgo verification.
    return mlir::success();
  }
  mlir::tt::ttir::GenericOp genericOp =
      mlir::cast<mlir::tt::ttir::GenericOp>(region->getParentOp());
  if (genericOp.getRegionThreadType(region->getRegionNumber()) != threadType) {
    return op->emitOpError("expected to be in a ")
           << stringifyEnum(threadType) << " region";
  }
  return mlir::success();
}

mlir::LogicalResult
mlir::tt::ttir::impl::verifyGenericRegionComputeOp(mlir::Operation *op) {
  return verifyGenericRegionOpThreadType(op,
                                         ::mlir::tt::ttir::ThreadType::Compute);
}

mlir::LogicalResult
mlir::tt::ttir::impl::verifyGenericRegionDatamovementOp(mlir::Operation *op) {
  return verifyGenericRegionOpThreadType(
      op, ::mlir::tt::ttir::ThreadType::Datamovement);
}
