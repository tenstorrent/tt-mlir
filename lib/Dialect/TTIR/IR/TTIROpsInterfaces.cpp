// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

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
