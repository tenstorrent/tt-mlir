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

// Creates a CPU hoist transform pass which hoists ops manually tagged
// with ttir.should_hoist attribute.
std::unique_ptr<Pass> createCPUHoistManuallyTaggedOpsTransform();

// Creates a pass that infers KV cache argument types from cache operations.
std::unique_ptr<Pass> createTTIRInferKVCacheArgumentTypes();

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// Creates a CPU hoist transform pass which hoists all ops whose
// dialect matches any of the provided dialects.
template <typename... Dialects>
std::unique_ptr<Pass> createCPUHoistForDialectsTransform();

// Creates a CPU hoist transform pass which hoists all ops whose
// type matches any of the provided ops.
template <typename... Ops>
std::unique_ptr<Pass> createCPUHoistForOpsTransform();

// Creates a CPU hoist transform pass which hoists const-eval functions
// as a whole.
std::unique_ptr<Pass> createCPUHoistConstEvalTransform();

} // namespace mlir::tt::ttir

#endif
