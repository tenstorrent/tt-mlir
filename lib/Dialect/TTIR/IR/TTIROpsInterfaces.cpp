// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  return (op->getParentOfType<ttir::GenericOp>() ||
          op->getParentOfType<func::FuncOp>())
             ? success()
             : op->emitOpError(
                   "TTIR Generic Ops must be inside a generic region");
}

std::pair<mlir::MemRefType, mlir::AffineMap>
mlir::tt::ttir::applyViews(mlir::Operation *op) {
  auto viewOp = mlir::dyn_cast<ttir::ViewOpInterface>(op);
  auto resultMemref = mlir::cast<mlir::MemRefType>(op->getResult(0).getType());
  if (!viewOp) {
    return std::make_pair(
        resultMemref, mlir::AffineMap::getMultiDimIdentityMap(
                          resultMemref.getRank(), resultMemref.getContext()));
  }
  auto map =
      mlir::cast<tt::ViewLayoutAttr>(resultMemref.getLayout()).getAffineMap();
  Value input = viewOp.getInput();
  auto inputMemref = mlir::cast<mlir::MemRefType>(input.getType());
  while (mlir::isa<tt::ViewLayoutAttr>(inputMemref.getLayout())) {
    map = inputMemref.getLayout().getAffineMap().compose(map);
    viewOp = mlir::cast<ttir::ViewOpInterface>(input.getDefiningOp());
    input = viewOp.getInput();
    inputMemref = mlir::cast<mlir::MemRefType>(input.getType());
  }
  assert(mlir::isa<tt::ShardLayoutAttr>(inputMemref.getLayout()) &&
         "Expected ShardLayoutAttr");
  return std::make_pair(inputMemref, map);
}
