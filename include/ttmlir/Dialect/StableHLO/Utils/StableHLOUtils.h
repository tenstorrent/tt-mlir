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

// Discardable attribute key on custom_call ops converted from composites
// with custom sharding rules. Carries the original composite attributes.
// Not to be confused with kCompAttrsKey which is the stablehlo.composite
// internal attribute key.
inline constexpr llvm::StringLiteral
    kCustomCallCompositeAttrsKey("tt.composite_attributes");

// UnitAttr tag on custom_call ops that were converted from composites
// with custom sharding rules.
inline constexpr llvm::StringLiteral
    kHasCustomShardingAttr("tt.has_custom_sharding");

// Target name for the RMS norm custom_call op.
inline constexpr llvm::StringLiteral
    kTTRMSNormCustomCallTargetName("tenstorrent.rms_norm");

// Composite names that have custom sharding rules. These composites are
// converted to stablehlo.custom_call ops so that Shardy can propagate shardings
// defined by the custom sharding rule for that composite as if the composite
// is one op. This array is the source of truth used by any pass that deals
// with ops and sharding (currently these passes are
// FlattenOrConvertCompositesPass and RegisterCustomShardingRulePass).
inline constexpr llvm::StringLiteral kCompositesWithCustomSharding[] = {
    kTTRMSNormCustomCallTargetName,
};

// Target name for the distributed RMS norm custom_call op.
inline constexpr llvm::StringLiteral
    kDistributedRmsNormTargetName("tenstorrent.distributed_rms_norm");

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
