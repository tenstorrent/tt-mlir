// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTKernel/Pipelines/TTKernelPipelines.h"

#include "mlir/Pass/PassManager.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttkernel {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTKernelToEmitCPipeline(OpPassManager &pm) {
  pm.addPass(createConvertTTKernelToEmitC());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTKernelPipelines() {
  mlir::PassPipelineRegistration<>(
      "ttkernel-to-emitc-pipeline", "Pipeline lowering ttkernel to emitc.",
      mlir::tt::ttkernel::createTTKernelToEmitCPipeline);
}
} // namespace mlir::tt::ttkernel
