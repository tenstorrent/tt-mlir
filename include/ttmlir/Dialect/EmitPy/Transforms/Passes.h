// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_EMITPY_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_EMITPY_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::emitpy {

#define GEN_PASS_DECL
#include "ttmlir/Dialect/EmitPy/Transforms/Passes.h.inc"

/// Create a pass to split distinct EmitPy logic into separate files.
std::unique_ptr<OperationPass<ModuleOp>> createEmitPySplitFiles();

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/EmitPy/Transforms/Passes.h.inc"

} // namespace mlir::tt::emitpy

#endif // TTMLIR_DIALECT_EMITPY_TRANSFORMS_PASSES_H
