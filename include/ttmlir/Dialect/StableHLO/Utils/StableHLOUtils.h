// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_STABLEHLOUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_STABLEHLOUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::stablehlo::utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Create a new private function with the provided ops within the module.
// - Captures become function arguments (in declared order).
// - Escapes become function results (in declared order).
// Returns the new callee symbol.
mlir::func::FuncOp createPrivateFunction(mlir::ModuleOp module,
                                         mlir::StringRef namePrefix,
                                         mlir::StringRef baseName,
                                         mlir::ArrayRef<mlir::Value> captures,
                                         mlir::ArrayRef<mlir::Value> escapes,
                                         mlir::ArrayRef<mlir::Operation *> ops);

#endif // TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::stablehlo::utils

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_STABLEHLOUTILS_H
