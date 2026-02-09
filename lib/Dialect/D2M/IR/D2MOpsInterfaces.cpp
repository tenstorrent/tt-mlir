// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::tt::d2m;

mlir::LogicalResult
mlir::tt::d2m::detail::verifyGenericParent(mlir::Operation *op) {
  return (op->getParentOfType<mlir::tt::d2m::GenericOp>() ||
          op->getParentOfType<func::FuncOp>())
             ? success()
             : op->emitOpError(
                   "D2M Generic Ops must be inside a generic region");
}

std::pair<mlir::MemRefType, mlir::AffineMap>
mlir::tt::d2m::applyViews(mlir::Operation *op) {
  auto viewOp = mlir::dyn_cast<mlir::tt::d2m::ViewOpInterface>(op);
  auto resultMemref = mlir::cast<mlir::MemRefType>(op->getResult(0).getType());
  if (!viewOp) {
    return std::make_pair(
        resultMemref, mlir::AffineMap::getMultiDimIdentityMap(
                          resultMemref.getRank(), resultMemref.getContext()));
  }

  mlir::AffineMap map;
  if (auto viewLayoutOp = mlir::dyn_cast<mlir::tt::d2m::ViewLayoutOp>(op)) {
    map = viewLayoutOp.getRemapping();
  } else if (auto streamLayoutOp =
                 mlir::dyn_cast<mlir::tt::d2m::StreamLayoutOp>(op)) {
    map = streamLayoutOp.getRemapping();
  } else {
    // Fallback to identity if no remapping is present.
    map = mlir::AffineMap::getMultiDimIdentityMap(resultMemref.getRank(),
                                                  resultMemref.getContext());
  }

  Value input = viewOp.getInput();
  auto inputMemref = mlir::cast<mlir::MemRefType>(input.getType());

  // Recursively apply view composition if the input is also a view. This
  // handles nested view chains by composing affine maps.
  if (auto *inputOp = input.getDefiningOp()) {
    if (mlir::isa<ViewOpInterface>(inputOp)) {
      auto [baseMemref, inputMap] = applyViews(inputOp);
      return std::make_pair(baseMemref, inputMap.compose(map));
    }
  }

  auto devLayout = mlir::dyn_cast_or_null<ttcore::DeviceLayoutInterface>(
      inputMemref.getLayout());
  assert(devLayout && devLayout.isPhysical() &&
         "Expected physical layout attr");

  return std::make_pair(inputMemref, map);
}

// Include generated interface method definitions
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.cpp.inc"
