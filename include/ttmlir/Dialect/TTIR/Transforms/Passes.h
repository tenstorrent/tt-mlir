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

using OpsVectorType = llvm::SmallVector<mlir::Operation *, 4>;
using ValuesVectorType = llvm::SmallVector<mlir::Value, 4>;
using TypesVectorType = llvm::SmallVector<mlir::Type, 4>;

// Descriptor for the set of operations which are to be CPU-hoisted.
struct CPUHoistedOpsDescriptor {
  // Vector of operations to be hoisted.
  OpsVectorType operations;
  // Values representing the outputs of the hoisted operations.
  ValuesVectorType outputValues;
  // Name of the generated hoisted function.
  std::string funcName;

  CPUHoistedOpsDescriptor(const OpsVectorType &ops,
                          const ValuesVectorType &outputs, std::string name)
      : operations(ops), outputValues(outputs), funcName(name) {}
};

// Predicate type for determining sets of ops to hoist in the provided module.
using CPUHoistAnalyzerType =
    std::function<llvm::SmallVector<CPUHoistedOpsDescriptor>(mlir::ModuleOp)>;

// Creates a CPU hoist transform pass based on the provided
// analyzer implementation.
std::unique_ptr<Pass>
createCustomCPUHoistTransform(CPUHoistAnalyzerType analyzer);

// Predicate type for determining whether an op should be hoisted in an op-by-op
// CPU hoisting analyzer.
using ShouldHoistOpType = std::function<bool(mlir::Operation *)>;

// Creates a CPU hoist transform pass which hoists individual ops based on
// the provided predicate.
std::unique_ptr<Pass>
createSingleOpCPUHoistTransform(ShouldHoistOpType predicate);

// Creates a CPU hoist transform pass which hoists ops manually tagged
// with ttir.should_hoist attribute.
std::unique_ptr<Pass> createCPUHoistManuallyTagedOpsTransform();

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

#define GEN_PASS_REGISTRATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

} // namespace mlir::tt::ttir

#endif
