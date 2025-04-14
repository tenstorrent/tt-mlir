// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef StableHLOMLIR_DIALECT_TRANSFORMS_PASSES_H
#define StableHLOMLIR_DIALECT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"
} // namespace mlir::tt::stablehlo

#endif
