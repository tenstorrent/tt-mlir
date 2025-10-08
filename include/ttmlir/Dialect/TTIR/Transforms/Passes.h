// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::bufferization {
struct OneShotBufferizationOptions;
} // namespace mlir::bufferization

namespace mlir::tt::ttir {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

template <typename... Dialects>
std::unique_ptr<Pass> createTTIRHoistTransformForDialects();

} // namespace mlir::tt::ttir

#endif
