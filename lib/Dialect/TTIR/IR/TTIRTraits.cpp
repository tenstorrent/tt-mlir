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

mlir::LogicalResult
mlir::tt::ttir::impl::verifyBroadcastable(mlir::Operation *op) {
  assert(op->getNumResults() == 1 &&
         "Expected a single result for broadcastable operation");

  auto getShape = [](const Value val) {
    return mlir::cast<mlir::RankedTensorType>(val.getType()).getShape();
  };

  auto operands = op->getOperands();
  // DPS operands shouldn't affect the result shape.
  if (auto dpsOp = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op)) {
    assert(dpsOp.getNumDpsInits() == 1 &&
           "Expected a single dps init for broadcastable operation");
    operands = operands.drop_back(dpsOp.getNumDpsInits());
  }
  auto operandShapes = llvm::map_range(operands, getShape);
  llvm::SmallVector<int64_t> broadcastedShape;
  for (llvm::ArrayRef<int64_t> operandShape : operandShapes) {
    llvm::SmallVector<int64_t> prevBroadcastedShape = broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            prevBroadcastedShape, operandShape, broadcastedShape)) {
      return op->emitOpError()
             << "operand shape (" << operandShape
             << ") is not broadcast compatible with inferred operand shapes ("
             << prevBroadcastedShape << ")";
    }
  }

  // Check that the result shape matches the broadcasted shape of the operands.
  llvm::SmallVector<int64_t> resultShape(getShape(op->getResult(0)));
  if (broadcastedShape != resultShape) {
    return op->emitOpError()
           << "result shape (" << resultShape
           << ") doesn't match expected shape after broadcasting ("
           << broadcastedShape << ")";
  }

  return success();
}
