// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_STABLEHLOUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_STABLEHLOUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tt::stablehlo::utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Composite op flattening/re-outlining related string definitions.
inline constexpr llvm::StringLiteral kReoutlineGroupAttr("reoutline.group");
inline constexpr llvm::StringLiteral kReoutlineSeedAttr("reoutline.seed");
inline constexpr llvm::StringLiteral
    kReoutlineOrigNameAttr("reoutline.orig_name");
inline constexpr llvm::StringLiteral
    kReoutlineCompAttrsAttr("reoutline.comp_attrs");
inline constexpr llvm::StringLiteral
    kReoutlineArgOperandIndicesAttr("reoutline.arg_operand_indices");
inline constexpr llvm::StringLiteral
    kReoutlineResultPosAttr("reoutline.result_pos");

// Composite op related string definitions.
inline constexpr llvm::StringLiteral kCompDecompositionKey("decomposition");
inline constexpr llvm::StringLiteral kCompAttrsKey("composite_attributes");
inline constexpr llvm::StringLiteral kCompNameKey("name");

// Key for carrying composite attributes on custom_call ops converted from
// composites with custom sharding rules.
inline constexpr llvm::StringLiteral
    kCompositeAttributesKey("tt.composite_attributes");

// UnitAttr marker on custom_call ops that were converted from composites
// with custom sharding rules. Useful for debugging.
inline constexpr llvm::StringLiteral
    kHasCustomShardingAttr("tt.has_custom_sharding");

// Composite names that have custom sharding rules. These composites are
// converted to stablehlo.custom_call ops (instead of being flattened) so
// that Shardy can propagate shardings through them using the CustomCall
// sharding model. This array is the single source of truth used by both
// FlattenOrConvertCompositesPass and RegisterCustomShardingRulePass.
inline constexpr llvm::StringLiteral kCompositesWithCustomSharding[] = {
    "tenstorrent.rms_norm",
};

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
