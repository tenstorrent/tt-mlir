// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTMETAL_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tt {
namespace ttmetal {

std::unique_ptr<mlir::Pass> createTTMetalBufferizePass();

#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

} // namespace ttmetal
} // namespace tt
} // namespace mlir

#endif // TTMLIR_DIALECT_TTMETAL_TRANSFORMS_PASSES_H
