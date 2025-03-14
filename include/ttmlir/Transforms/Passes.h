// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DECL
#include "ttmlir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Transforms/Passes.h.inc"

// Special creater func which takes in variadic template list of ops to not
// hoist
template <typename... OpTypes>
std::unique_ptr<Pass> createConstEvalHoistTransformWithIgnoreTypes();

std::unique_ptr<Pass> createConstEvalHoistTransformNoIgnoreTypes();
} // namespace mlir::tt::transforms

#endif
