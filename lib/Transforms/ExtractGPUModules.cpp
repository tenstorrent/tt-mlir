// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_EXTRACTGPUMODULES
#include "ttmlir/Transforms/Passes.h.inc"

namespace {

/// Extracts GPU modules from LLVM by separating GPU kernels from host code.
class ExtractGPUModulesPass
    : public impl::ExtractGPUModulesBase<ExtractGPUModulesPass> {

public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    SmallVector<LLVM::LLVMFuncOp> gpuFunctions;
    for (auto GPUModuleOp : moduleOp.getOps<mlir::gpu::GPUModuleOp>()) {
      for (auto func : GPUModuleOp.getOps<LLVM::LLVMFuncOp>()) {
        std::string newFuncName =
            GPUModuleOp.getName().str() + "_" + func.getName().str();
        func.setSymName(newFuncName);
        gpuFunctions.push_back(func);
      }
    }

    // If no GPU functions found, nothing to extract.
    if (gpuFunctions.empty()) {
      return;
    }

    // Remove all operations that are NOT GPU modules.
    for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {
      if (!isa<mlir::gpu::GPUModuleOp>(op)) {
        op.erase();
      }
    }

    // Copy all GPU functions directly to the main module.
    builder.setInsertionPointToStart(moduleOp.getBody());
    moduleOp.removeSymNameAttr();
    moduleOp->removeAttr("gpu.container_module");

    for (auto gpuFunc : gpuFunctions) {
      auto clonedFunc = gpuFunc.clone();
      builder.insert(clonedFunc);
    }

    // Remove remaining GPU modules.
    SmallVector<Operation *> gpusToErase;
    for (auto &op : llvm::make_early_inc_range(moduleOp.getOps())) {
      if (isa<mlir::gpu::GPUModuleOp>(op)) {
        op.erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::transforms
