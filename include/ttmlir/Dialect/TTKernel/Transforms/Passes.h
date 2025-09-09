// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTKERNEL_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTKERNEL_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttkernel {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h.inc"
} // namespace mlir::tt::ttkernel

#endif
