// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "llvm/ADT/APInt.h"

namespace mlir::tt::ttir::utils {

// Returns the min/max representable values for the given integer type, or emits
// an error if the bitwidth is invalid.
inline mlir::FailureOr<std::pair<int64_t, int64_t>>
getStorageTypeMinMax(IntegerType intType, mlir::Location loc) {
  unsigned bitWidth = intType.getWidth();
  bool isSigned = intType.isSigned();
  if (bitWidth == 0 || bitWidth > 64) {
    emitError(loc, "Quantized min/max bitwidth must be in (0, 64].");
    return mlir::failure();
  }
  int64_t min, max;
  if (isSigned) {
    min = llvm::APInt::getSignedMinValue(bitWidth).getSExtValue();
    max = llvm::APInt::getSignedMaxValue(bitWidth).getSExtValue();
  } else {
    min = 0;
    max = llvm::APInt::getMaxValue(bitWidth).getZExtValue();
  }
  return std::make_pair(min, max);
}

// Returns true if all values that are quantized are ranked tensors with the
// same quantized element type.
inline bool
areQuantizationParamsAligned(mlir::ArrayRef<mlir::Value> quantizedValues) {
  if (quantizedValues.empty()) {
    return true;
  }

  mlir::quant::QuantizedType referenceQType = nullptr;
  return llvm::all_of(quantizedValues, [&](mlir::Value val) {
    if (!val) {
      return true;
    }
    auto rankedType = mlir::cast<mlir::RankedTensorType>(val.getType());

    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(rankedType.getElementType());
    // Ignore non-quantized values.
    if (!qType) {
      return true;
    }

    if (!referenceQType) {
      referenceQType = qType;
    } else if (qType != referenceQType) {
      return false;
    }

    return true;
  });
}

// Computes effective output scale given input and weight quantized types.
// Applicable for ops that compute a weighted sum over input elements (e.g.,
// matmul, conv, linear) requiring composed output scales. Handles:
// 1. Per-tensor input + per-tensor weight
// 2. Per-tensor input + per-channel weight
// TODO(anuragsingh): 3. Per-channel input + per-channel weight (must match axis
// length) Returns scale[i] = input_scale * weight_scale[i] for each channel.
inline mlir::quant::QuantizedType computeOutputScalesAndZeroPoint(
    quant::QuantizedType quantInputType, quant::QuantizedType quantWeightType,
    mlir::IntegerType storageType, mlir::Location loc,
    std::optional<int64_t> requiredOutAxis = std::nullopt,
    std::optional<int64_t> requiredWeightAxis = std::nullopt,
    std::optional<int64_t> requiredAxisSize = std::nullopt) {
  // Find the min/max of the output storage type.
  mlir::FailureOr<std::pair<int64_t, int64_t>> storageTypeMinMax =
      getStorageTypeMinMax(storageType, loc);
  if (mlir::failed(storageTypeMinMax)) {
    emitError(loc, "Invalid storage type for quantized output.");
    return nullptr;
  }
  const auto [storageTypeMin, storageTypeMax] = *storageTypeMinMax;

  // Now determine the output scales/zero points.
  std::pair<SmallVector<double>, SmallVector<int64_t>> scalesAndZeroPoints;
  SmallVector<double> scales;
  SmallVector<int64_t> zeroPoints;
  double inputScale;
  int64_t inputZeroPoint;
  if (quant::UniformQuantizedType uniformType =
          dyn_cast<quant::UniformQuantizedType>(quantInputType)) {
    inputScale = uniformType.getScale();
    inputZeroPoint = uniformType.getZeroPoint();
  } else if (quant::UniformQuantizedPerAxisType perAxisType =
                 dyn_cast<quant::UniformQuantizedPerAxisType>(quantInputType)) {
    emitError(loc,
              "Per-axis quantization is not supported for quantized input.");
    return nullptr;
  }
  // Per-tensor weights (inputScale * weightScale for output scale).
  if (quant::UniformQuantizedType uniformType =
          dyn_cast<quant::UniformQuantizedType>(quantWeightType)) {
    if (uniformType.getZeroPoint() != inputZeroPoint) {
      emitError(
          loc,
          "Zero points must match for per-tensor quantized input and weight.");
      return nullptr;
    }
    double weightScale = uniformType.getScale();
    scales.push_back(inputScale * weightScale);
    zeroPoints.push_back(inputZeroPoint);
    scalesAndZeroPoints = std::make_pair(scales, zeroPoints);
  }
  // Per-axis weights (find the right axis and perform inputScale *
  // weightScale).
  else if (quant::UniformQuantizedPerAxisType perAxisType =
               dyn_cast<quant::UniformQuantizedPerAxisType>(quantWeightType)) {
    ArrayRef<int64_t> weightZeroPoints = perAxisType.getZeroPoints();
    if (!llvm::all_of(weightZeroPoints,
                      [&](int64_t zp) { return zp == inputZeroPoint; })) {
      emitError(loc, "Zero points must match across all channels for per-axis "
                     "weight and per-tensor input.");
      return nullptr;
    }
    zeroPoints.assign(weightZeroPoints.begin(), weightZeroPoints.end());

    // Check if the weight axis is the same as the required weight axis.
    if (requiredWeightAxis &&
        perAxisType.getQuantizedDimension() != *requiredWeightAxis) {
      emitError(loc, "Per-axis weight axis must be kernel_output_feature.");
      return nullptr;
    }
    // Check if the number of per-axis weight scales is the same as the required
    // axis size.
    if (requiredAxisSize &&
        static_cast<int64_t>(perAxisType.getScales().size()) !=
            *requiredAxisSize) {
      emitError(loc,
                "Number of per-axis weight scales must equal output channels.");
      return nullptr;
    }

    ArrayRef<double> weightScales = perAxisType.getScales();
    scales.assign(weightScales.begin(), weightScales.end());
    for (double &scale : scales) {
      scale *= inputScale;
    }
    scalesAndZeroPoints = std::make_pair(scales, zeroPoints);
  }
  // Some unrecognized type.
  else {
    emitError(loc, "Invalid quantized input and weight types.");
    return nullptr;
  }
  // Construct the new output type (scales/zero points/storage type min & max).
  mlir::quant::QuantizedType quantOutputType;
  if (scalesAndZeroPoints.first.size() == 1) {
    quantOutputType = quant::UniformQuantizedType::get(
        quantInputType.getFlags(), storageType,
        quantInputType.getExpressedType(), scalesAndZeroPoints.first[0],
        scalesAndZeroPoints.second[0], storageTypeMin, storageTypeMax);
  } else {
    // Per-axis output type (same as the axis for the weight).
    quant::UniformQuantizedPerAxisType perAxisWeightType =
        dyn_cast<quant::UniformQuantizedPerAxisType>(quantWeightType);
    // The output axis is the same as the weight axis if not specified.
    const int64_t outAxis =
        requiredOutAxis.value_or(perAxisWeightType.getQuantizedDimension());
    quantOutputType = quant::UniformQuantizedPerAxisType::get(
        perAxisWeightType.getFlags(), storageType,
        perAxisWeightType.getExpressedType(), scalesAndZeroPoints.first,
        scalesAndZeroPoints.second, outAxis, storageTypeMin, storageTypeMax);
  }
  return quantOutputType;
}

} // namespace mlir::tt::ttir::utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H
