// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"

namespace mlir::tt::llvm_util {
#define GEN_PASS_DEF_PROMOTETOSTATICALLOCATIONPASS
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h.inc"

class PromoteToStaticAllocation
    : public impl::PromoteToStaticAllocationPassBase<
          PromoteToStaticAllocation> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto cpuModule = moduleOp->getParentOfType<ttcore::CPUModuleOp>();
    if (!cpuModule || cpuModule.getRole() != ttcore::CPURole::Device) {
      return;
    }

    OpPassManager pm("builtin.module");
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

    mlir::bufferization::PromoteBuffersToStackPassOptions promoteToStackOptions;
    promoteToStackOptions.maxAllocSizeInBytes = 1024 * 1024;
    pm.nest<func::FuncOp>().addPass(
        mlir::bufferization::createPromoteBuffersToStackPass(
            promoteToStackOptions));

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::tt::llvm_util
