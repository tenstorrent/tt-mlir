// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

mlir::LogicalResult
mlir::tt::ttir::detail::verifyBroadcastable(mlir::Operation *op) {
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
  assert(op->getNumResults() == 1 &&
         "Expected a single result for broadcastable operation");
  llvm::SmallVector<int64_t> resultShape(getShape(op->getResult(0)));
  if (broadcastedShape != resultShape) {
    return op->emitOpError()
           << "result shape (" << resultShape
           << ") doesn't match expected shape after broadcasting ("
           << broadcastedShape << ")";
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

static mlir::AffineMap reblockViewWorkaround(mlir::MemRefType inputMemref,
                                             mlir::MemRefType resultMemref) {
  assert(inputMemref.getRank() == resultMemref.getRank());
  auto *ctx = inputMemref.getContext();
  int64_t rank = inputMemref.getRank();
  auto inputLayout =
      mlir::cast<mlir::tt::DeviceLayoutInterface>(inputMemref.getLayout());
  auto resultLayout =
      mlir::cast<mlir::tt::DeviceLayoutInterface>(resultMemref.getLayout());
  mlir::ArrayRef<int64_t> inputGridShape =
      inputLayout.getGridShape(inputMemref);
  mlir::ArrayRef<int64_t> inputShardShape =
      inputLayout.getShardShape(inputMemref);
  mlir::ArrayRef<int64_t> resultGridShape =
      resultLayout.getGridShape(resultMemref);
  mlir::ArrayRef<int64_t> resultShardShape =
      resultLayout.getShardShape(resultMemref);
  mlir::SmallVector<mlir::AffineExpr> mapExprs(rank);

  // Canonicalize.
  for (size_t i = 0; i < resultGridShape.size(); i++) {
    auto dG = getAffineDimExpr(i, ctx);
    mapExprs[i] = dG.floorDiv(resultGridShape[i]);
  }
  for (size_t i = 0; i < resultGridShape.size(); i++) {
    size_t j = i + resultGridShape.size();
    auto dG = getAffineDimExpr(i, ctx);
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[j] = dG * resultShardShape[i] + dS;
  }
  auto resultToCanonical = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  // Uncanonicalize.
  for (size_t i = 0; i < inputGridShape.size(); i++) {
    size_t j = i + inputGridShape.size();
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[i] = dS.floorDiv(inputShardShape[i]);
  }
  for (size_t i = 0; i < inputGridShape.size(); i++) {
    size_t j = i + inputGridShape.size();
    auto dS = getAffineDimExpr(j, ctx);
    mapExprs[j] = dS % inputShardShape[i];
  }
  auto canonicalToInput = mlir::AffineMap::get(rank, 0, mapExprs, ctx);

  return canonicalToInput.compose(resultToCanonical);
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
  assert(
      mlir::isa<tt::ShardLayoutAttr>(inputMemref.getLayout()) &&
      "Expected ShardLayoutAttr, only one level of view nesting is supported");
  auto reblockMap = reblockViewWorkaround(inputMemref, resultMemref);
  return std::make_pair(inputMemref, reblockMap.compose(map));
}
