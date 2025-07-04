// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTCORE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttcore {

#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

} // namespace mlir::tt::ttcore

#endif // TTMLIR_DIALECT_TTCORE_TRANSFORMS_PASSES_H
