// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_SFPU_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_SFPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::sfpu {

#define GEN_PASS_DECL
#include "ttmlir/Dialect/SFPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/SFPU/Transforms/Passes.h.inc"

} // namespace mlir::tt::sfpu

#endif
