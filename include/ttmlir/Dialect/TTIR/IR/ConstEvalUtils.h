// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
#define TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

namespace mlir::tt::ttir {

// Common utilities for folding constant ops, e.g. in folders or
// canonicalization patterns.

// Process a constant op directly.
inline mlir::OpFoldResult foldConstantOpHelper(
    mlir::tt::ttir::ConstantOp constantOp,
    llvm::function_ref<mlir::Attribute(mlir::DenseElementsAttr)> transform) {
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

// Process a value that might be a constant.
inline mlir::OpFoldResult foldConstantOpHelper(
    mlir::Value value,
    llvm::function_ref<mlir::Attribute(mlir::DenseElementsAttr)> transform) {
  auto constantOp = value.getDefiningOp<mlir::tt::ttir::ConstantOp>();
  if (!constantOp) {
    return nullptr;
  }
  return foldConstantOpHelper(constantOp, transform);
}

// Computes the permutation of a constant tensor according to the permutation
// array. Returns a new ElementsAttr containing the permuted values.
inline mlir::DenseElementsAttr
computePermutation(mlir::DenseElementsAttr inputTensor,
                   llvm::ArrayRef<int64_t> permutation) {
  mlir::ShapedType inputType =
      mlir::cast<mlir::ShapedType>(inputTensor.getType());
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  mlir::Type elementType = inputType.getElementType();

  // Compute output shape based on permutation.
  llvm::SmallVector<int64_t> outputShape(inputShape.size());
  for (size_t i = 0; i < inputShape.size(); ++i) {
    outputShape[i] = inputShape[permutation[i]];
  }

  mlir::RankedTensorType outputType =
      mlir::RankedTensorType::get(outputShape, elementType);
  std::vector<mlir::Attribute> newValues;
  newValues.reserve(inputTensor.getNumElements());

  llvm::SmallVector<uint64_t> indices(inputShape.size(), 0);
  llvm::SmallVector<uint64_t> inputIndices(inputShape.size(), 0);
  llvm::SmallVector<uint64_t> limits;
  for (auto dim : outputShape) {
    limits.push_back(dim);
  }

  // Iterate through all elements in output order.
  bool done = false;
  while (!done) {
    for (size_t i = 0; i < indices.size(); ++i) {
      inputIndices[permutation[i]] = indices[i];
    }

    newValues.push_back(inputTensor.getValues<mlir::Attribute>()[inputIndices]);

    // Increment indices (row-major order).
    for (int i = indices.size() - 1; i >= 0; --i) {
      indices[i]++;
      if (indices[i] < limits[i]) {
        break;
      }
      indices[i] = 0;
      if (i == 0) {
        done = true;
      }
    }
  }

  return mlir::DenseElementsAttr::get(outputType, newValues);
}

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
