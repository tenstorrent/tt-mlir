// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_PASSOVERRIDES_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_PASSOVERRIDES_H

#include "llvm/ADT/StringRef.h"

namespace mlir::tt::stablehlo {

#ifdef TTMLIR_ENABLE_STABLEHLO
struct OptionNames {
  static constexpr llvm::StringRef meshShape = "mesh-shape";
  static constexpr llvm::StringRef resultPresharded = "result-presharded";
  static constexpr llvm::StringRef automaticArgAnalysis =
      "automatic-arg-analysis";
};

#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_PASSOVERRIDES_H
