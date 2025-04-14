// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef SHLOMLIR_DIALECT_TRANSFORMS_PASSES_H
#define SHLOMLIR_DIALECT_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::shlo {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/SHLO/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/SHLO/Transforms/Passes.h.inc"
} // namespace mlir::tt::shlo

#endif
