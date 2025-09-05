// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTMETAL_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTMETAL_TRANSFORMS_PASSES_H

namespace mlir::tt::ttmetal {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttmetal

#endif
