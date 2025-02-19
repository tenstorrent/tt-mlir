// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/ValueRange.h>

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

mlir::LogicalResult
mlir::tt::ttir::detail::verifyBroadcastable(mlir::Operation *op) {
  const auto getShape = [](const Value val) {
    return mlir::cast<mlir::RankedTensorType>(val.getType()).getShape();
  };

  const auto operandSegmentSizes =
      op->getAttrOfType<mlir::DenseI32ArrayAttr>("operandSegmentSizes");
  // DPS operands shouldn't affect the result shape.
  const auto outputSegmentSize =
      operandSegmentSizes[operandSegmentSizes.size() - 1];
  const auto operandShapes = llvm::map_range(op->getOperands(), getShape);
  llvm::SmallVector<int64_t, 4> broadcastedShape;
  for (const auto operandShape :
       llvm::drop_end(operandShapes, outputSegmentSize)) {
    const auto prevBroadcastedShape = broadcastedShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            prevBroadcastedShape, operandShape, broadcastedShape)) {
      return op->emitOpError("Operands are not broadcast compatible");
    }
  }

  // Check that the result shape matches the broadcasted shape of the operands.
  llvm::SmallVector<int64_t, 4> resultShape(getShape(op->getResults().front()));
  if (broadcastedShape != resultShape) {
    return op->emitOpError(
        "Result shape must match operand shapes after broadcasting");
  }

  return success();
}

mlir::LogicalResult
mlir::tt::ttir::detail::verifyGenericParent(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (llvm::dyn_cast<ttir::GenericOp>(parent)) {
      return success();
    }
    parent = parent->getParentOp();
  }

  return op->emitOpError("TTIR Generic Ops must be inside a generic region");
}
