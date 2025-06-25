// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
#define TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cfenv>

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
      if (indices[i] < static_cast<uint64_t>(outputShape[i])) {
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

// Helper function to convert a vector of quantized float
// values into a DenseElementsAttr with the specified integer storage type.
template <typename T>
mlir::DenseElementsAttr
createQuantizedDenseAttr(mlir::ShapedType outputType,
                         llvm::SmallVector<double> &quantizedVals) {
  auto data = llvm::to_vector<T>(quantizedVals);
  return mlir::DenseElementsAttr::get(outputType, data);
}

// Computes the quantization of a constant tensor according to the
// scale/zero point info embedded in the output type. Returns a
// new ElementsAttr containing the quantized values.
inline mlir::DenseElementsAttr
computeQuantization(mlir::DenseElementsAttr inputTensor,
                    mlir::RankedTensorType outputType) {
  mlir::quant::QuantizedType quantType =
      mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());

  if (!quantType || !mlir::isa<mlir::FloatType>(quantType.getExpressedType())) {
    return nullptr;
  }
  mlir::ShapedType inputType = inputTensor.getType();

  auto inputValues = inputTensor.getValues<mlir::APFloat>();
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t numElements = inputTensor.getNumElements();

  llvm::SmallVector<double> quantizedVals;
  quantizedVals.reserve(numElements);

  int64_t axis = -1;
  llvm::SmallVector<double> scales;
  llvm::SmallVector<int64_t> zeroPoints;

  if (mlir::quant::UniformQuantizedType perTensor =
          mlir::dyn_cast<mlir::quant::UniformQuantizedType>(quantType)) {
    scales = {perTensor.getScale()};
    zeroPoints = {perTensor.getZeroPoint()};
  } else if (mlir::quant::UniformQuantizedPerAxisType perAxis =
                 mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
                     quantType)) {
    axis = perAxis.getQuantizedDimension();
    scales = llvm::to_vector(perAxis.getScales());
    zeroPoints = llvm::to_vector(perAxis.getZeroPoints());
  } else {
    return nullptr;
  }

  // Scales and zero points must be the same size.
  if (scales.size() != zeroPoints.size()) {
    return nullptr;
  }

  // Define rounding function to round to nearest even.
  int oldMode = std::fegetround();
  std::fesetround(FE_TONEAREST);

  int64_t quantMin = quantType.getStorageTypeMin();
  int64_t quantMax = quantType.getStorageTypeMax();

  // Compute how often the quant param changes (only applies to per-axis).
  int64_t stride = 1;
  if (axis >= 0) {
    stride = std::accumulate(inputShape.begin() + axis + 1, inputShape.end(),
                             1LL, std::multiplies<int64_t>());
  }

  // Perform the quantization of each element.
  for (int64_t i = 0; i < numElements; ++i) {
    double val = inputValues[i].convertToDouble();

    int64_t idx = (axis == -1) ? 0 : (i / stride) % scales.size();
    double scale = scales[idx];
    int64_t zp = zeroPoints[idx];

    double quant = val / scale + zp;

    // Apply rounding mode (to nearest even).
    quant = std::nearbyint(quant);
    quant = std::clamp(quant, static_cast<double>(quantMin),
                       static_cast<double>(quantMax));
    quantizedVals.push_back(quant);
  }

  // Restore rounding mode.
  std::fesetround(oldMode);

  mlir::Type storageType = quantType.getStorageType();
  mlir::RankedTensorType newOutputType =
      mlir::RankedTensorType::get(outputType.getShape(), storageType);
  int bitWidth = storageType.getIntOrFloatBitWidth();
  bool isSigned = storageType.isSignedInteger();

  if (bitWidth == 8) {
    return isSigned
               ? createQuantizedDenseAttr<int8_t>(newOutputType, quantizedVals)
               : createQuantizedDenseAttr<uint8_t>(newOutputType,
                                                   quantizedVals);
  }

  if (bitWidth == 16) {
    return isSigned
               ? createQuantizedDenseAttr<int16_t>(newOutputType, quantizedVals)
               : createQuantizedDenseAttr<uint16_t>(newOutputType,
                                                    quantizedVals);
  }

  if (bitWidth == 32) {
    return isSigned
               ? createQuantizedDenseAttr<int32_t>(newOutputType, quantizedVals)
               : createQuantizedDenseAttr<uint32_t>(newOutputType,
                                                    quantizedVals);
  }

  return nullptr;
}

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_IR_CONSTEVALUTILS_H
