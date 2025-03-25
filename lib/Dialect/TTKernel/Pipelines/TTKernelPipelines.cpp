// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Pipelines/TTKernelPipelines.h"

#include "ttmlir/Conversion/Passes.h"

#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttkernel {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createPyKernelCompilePipeline(
    OpPassManager &pm, const PyKernelCompilePipelineOptions &options) {
  // Use Options to Enable/Disable certain Optimizations
  if (options.enableCanonicalizer) {
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (options.enableFormExpressions) {
    pm.addPass(mlir::emitc::createFormExpressionsPass());
  }
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTKernelPipelines() {
  mlir::PassPipelineRegistration<
      mlir::tt::ttkernel::PyKernelCompilePipelineOptions>(
      "pykernel-compile-pipeline",
      "Pipeline applying optimizations to MLIR Module from PyKernel",
      mlir::tt::ttkernel::createPyKernelCompilePipeline);
}
} // namespace mlir::tt::ttkernel
