// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_LLVM_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_LLVM_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::llvm_util {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"
} // namespace mlir::tt::llvm_util

#endif
