// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
#define TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir {
namespace tt {
namespace ttir {

// Common utility for folding constant ops, e.g. in folders or canonicalization
// patterns.
template <typename OpTy, typename TransformFn>
static mlir::OpFoldResult foldConstantOpHelper(OpTy op, TransformFn transform) {
  auto constantOp =
      op.getInput().template getDefiningOp<mlir::tt::ttir::ConstantOp>();
  if (!constantOp) {
    return nullptr;
  }

  mlir::Attribute constAttr = constantOp.getValue();
  // Handle DenseElementsAttr.
  if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constAttr)) {
    return transform(denseAttr);
  }

  // Handle DenseResourceElementsAttr by materializing to DenseElementsAttr.
  if (auto resourceAttr =
          mlir::dyn_cast<mlir::DenseResourceElementsAttr>(constAttr)) {
    auto originalType =
        mlir::cast<mlir::RankedTensorType>(resourceAttr.getType());
    mlir::ArrayRef<char> rawData = resourceAttr.getData();
    mlir::DenseElementsAttr tempDenseAttr =
        mlir::DenseElementsAttr::getFromRawBuffer(originalType, rawData);
    return transform(tempDenseAttr);
  }

  return nullptr;
}

} // namespace ttir
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
