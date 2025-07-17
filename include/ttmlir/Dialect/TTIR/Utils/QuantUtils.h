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
getStorageTypeMinMax(IntegerType intType, Location loc) {
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

// Returns true if all values are ranked tensors with the same quantized element
// type.
inline bool
areQuantizationParamsAligned(mlir::ArrayRef<mlir::Value> quantizedValues) {
  if (quantizedValues.empty()) {
    return true;
  }

  auto firstType =
      mlir::dyn_cast<mlir::RankedTensorType>(quantizedValues[0].getType());
  if (!firstType) {
    return false;
  }

  auto firstQuantType =
      mlir::dyn_cast<mlir::quant::QuantizedType>(firstType.getElementType());
  if (!firstQuantType) {
    return false;
  }

  return llvm::all_of(quantizedValues.drop_front(), [&](mlir::Value val) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
    if (!type) {
      return false;
    }
    auto qType =
        mlir::dyn_cast<mlir::quant::QuantizedType>(type.getElementType());
    return qType && qType == firstQuantType;
  });
}

} // namespace mlir::tt::ttir::utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H
