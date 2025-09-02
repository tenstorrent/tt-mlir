// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Function declarations for custom passes
#ifdef TTMLIR_ENABLE_STABLEHLO
std::unique_ptr<Pass> createCombinedRoundTripAndInlinePass();
#endif // TTMLIR_ENABLE_STABLEHLO

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"
} // namespace mlir::tt::stablehlo

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_PASSES_H
