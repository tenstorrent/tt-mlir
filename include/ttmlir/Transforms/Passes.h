// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TRANSFORMS_PASSES_H
#define TTMLIR_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::tt::transforms {
#define GEN_PASS_DECL
#include "ttmlir/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Transforms/Passes.h.inc"
} // namespace mlir::tt::transforms

#endif
