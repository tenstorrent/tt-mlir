// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_H
#define TTMLIR_DIALECT_D2M_TRANSFORMS_PASSES_H

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::bufferization {
struct OneShotBufferizationOptions;
} // namespace mlir::bufferization

namespace mlir::tt::d2m {
#define GEN_PASS_DECL
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

template <typename... Dialects>
std::unique_ptr<Pass> createD2MHoistTransformForDialects();

} // namespace mlir::tt::d2m

#endif
