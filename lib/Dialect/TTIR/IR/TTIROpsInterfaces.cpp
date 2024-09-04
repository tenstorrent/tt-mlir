// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/ValueRange.h>

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

mlir::LogicalResult
mlir::tt::ttir::detail::verifyElementwiseOp(mlir::Operation *op) {
  llvm::SmallVector<int64_t, 4> broadcastedShape;
  mlir::OperandRange operands = op->getOperands();
  mlir::OperandRange::iterator operand_it = operands.begin();
  llvm::SmallVector<int64_t, 4> prevOperandShape(
      mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape());

  while (++operand_it != operands.end()) {
    llvm::SmallVector<int64_t, 4> nextOperandShape(
        mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape());

    if (!OpTrait::util::getBroadcastedShape(prevOperandShape, nextOperandShape,
                                            broadcastedShape)) {
      return op->emitOpError("Operands are not broadcast compatible");
    }
    prevOperandShape = broadcastedShape;
  }

  llvm::SmallVector<int64_t, 4> resultShape(
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
          .getShape());
  if (broadcastedShape != resultShape) {
    return op->emitOpError(
        "Result shape must match operand shapes after broadcasting");
  }

  TypeID expectedBaseTy = op->getResultTypes().front().getTypeID();
  if (!llvm::all_of(op->getOperandTypes(),
                    [&](Type t) { return t.getTypeID() == expectedBaseTy; })) {
    return op->emitOpError() << "All operands/results must have the same type";
  }

  return success();
}
