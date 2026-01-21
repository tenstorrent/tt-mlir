// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_UTILS_H

#define TTMLIR_DIALECT_STABLEHLO_UTILS_UTILS_H

#ifdef TTMLIR_ENABLE_STABLEHLO

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo::utils {
// Helper to check if this convolution is a transposed convolution.
// Determine if the stablehlo.convolution op represents a regular or
// transposed convolution, based on Torch-MLIR lowering patterns:
// https://github.com/llvm/torch-mlir/blob/main/lib/Conversion/TorchToStablehlo/Linear.cpp
// Only transposed convolutions can have input dilation greater than 1.
// Transposed convolutions always have a window stride of 1.
inline bool isTransposedConv(::mlir::stablehlo::ConvolutionOp convolutionOp) {
  // If lhs_dilation is not set, it defaults to 1 (not transposed).
  // Transposed convolutions have input dilation > 1 for at least one dimension.
  auto lhsDilationAttr = convolutionOp.getLhsDilationAttr();
  if (!lhsDilationAttr) {
    return false;
  }

  bool isTransposed = llvm::any_of(lhsDilationAttr.asArrayRef(),
                                   [](int64_t d) { return d > 1; });

  // Transposed convolutions always have window stride of 1.
  // If window_strides is not set, it defaults to 1, so isTransposed remains the
  // same.
  auto windowStridesAttr = convolutionOp.getWindowStridesAttr();
  if (windowStridesAttr) {
    isTransposed &= llvm::all_of(windowStridesAttr.asArrayRef(),
                                 [](int64_t s) { return s == 1; });
  }

  return isTransposed;
}

} // namespace mlir::tt::stablehlo::utils

#endif // TTMLIR_ENABLE_STABLEHLO

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_UTILS_H
