// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H
#define TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H

#include "llvm/ADT/APInt.h"

namespace mlir::tt::ttir::utils {

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

} // namespace mlir::tt::ttir::utils

#endif // TTMLIR_DIALECT_TTIR_UTILS_QUANTUTILS_H
