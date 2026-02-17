// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_TRANSFORMS_HOISTCPUOPS_HOISTCPUOPS_H
#define TTMLIR_DIALECT_TTIR_TRANSFORMS_HOISTCPUOPS_HOISTCPUOPS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {

using OpsVectorType = llvm::SmallVector<mlir::Operation *>;
using ValuesVectorType = llvm::SmallVector<mlir::Value>;
using TypesVectorType = llvm::SmallVector<mlir::Type>;

// Descriptor for the set of operations which are to be CPU-hoisted.
struct CPUHoistedOpsDescriptor {
  // Vector of operations to be hoisted.
  OpsVectorType operations;
  // Values representing the outputs of the hoisted operations.
  ValuesVectorType outputValues;
  // Suffix for the hoisted function name (appears after "cpu_hoisted_",
  // and before the implementation hash).
  llvm::SmallString<64> funcNameSuffix;

  CPUHoistedOpsDescriptor(const OpsVectorType &ops,
                          const ValuesVectorType &outputs,
                          llvm::SmallString<64> suffix)
      : operations(ops), outputValues(outputs),
        funcNameSuffix(std::move(suffix)) {}
};

// Hoists the ops from the provided descriptors into the CPU module.
// Creates the CPU module if it doesn't already exist.
void runCPUHoist(mlir::ModuleOp rootModule,
                 llvm::SmallVector<CPUHoistedOpsDescriptor> descriptors);

// Common entry point for all CPU hoisting passes which hoist ops based on a
// simple boolean predicate. The predicate determines whether an op should be
// hoisted. Returns a vector of descriptors, where each descriptor
// corresponds to an op or a group of ops to be hoisted together.
llvm::SmallVector<CPUHoistedOpsDescriptor> createDescriptorsWithPredicate(
    func::FuncOp funcOp, llvm::function_ref<bool(mlir::Operation *)> predicate);

// Returns the inner ModuleOp of the DeviceModuleOp inside the provided root
// module. Returns nullptr if not found.
mlir::ModuleOp getDeviceInnerModule(mlir::ModuleOp rootModule);

} // namespace mlir::tt::ttir

#endif
