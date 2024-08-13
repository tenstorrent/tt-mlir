// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include <cstdint>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Traits.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

void mlir::tt::ttir::detail::inferBroadcastedShape(mlir::Operation *op) {
  llvm::SmallVector<int64_t> broadcastedShape;
  auto operands = op->getOperands();
  auto operand_it = operands.begin();
  llvm::SmallVector<int64_t> operandShape(
      mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape());

  while (++operand_it != operands.end()) {
    auto operandShape2 =
        mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape();
    OpTrait::util::getBroadcastedShape(operandShape, operandShape2,
                                       broadcastedShape);
    operandShape = broadcastedShape;
  }

  auto resultShape =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
  assert(broadcastedShape == resultShape);

  for (auto operand : operands) {
    operand.setType(RankedTensorType::get(
        broadcastedShape,
        mlir::cast<RankedTensorType>(operand.getType()).getElementType()));
  }
}

mlir::LogicalResult
mlir::tt::ttir::detail::verifyElementwiseOp(mlir::Operation *op) {
  llvm::SmallVector<int64_t> broadcastedShape;
  auto operands = op->getOperands();
  auto operand_it = operands.begin();
  llvm::SmallVector<int64_t> operandShape(
      mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape());

  while (++operand_it != operands.end()) {
    auto operandShape2 =
        mlir::cast<mlir::RankedTensorType>((*operand_it).getType()).getShape();

    if (!OpTrait::util::getBroadcastedShape(operandShape, operandShape2,
                                            broadcastedShape)) {
      return op->emitOpError("Operands are not broadcast compatible");
    }
    operandShape = broadcastedShape;
  }

  auto resultShape =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()).getShape();
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
