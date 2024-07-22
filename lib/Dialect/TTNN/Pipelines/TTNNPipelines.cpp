// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Pass/PassManager.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice());
  pm.addPass(mlir::tt::ttir::createTTIRLayout());

  if (options.gridSetPassEnabled) {
    ttir::TTIRGridSetOptions gridSetOptions;
    gridSetOptions.overrideGridSizes = options.overrideGridSizes;
    pm.addPass(mlir::tt::ttir::createTTIRGridSet(gridSetOptions));
  }

  pm.addPass(createTTNNOpenDevice());
  pm.addPass(createConvertTTIRToTTNN());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTNNPipelines() {
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions>(
      "ttir-to-ttnn-backend-pipeline",
      "Pipeline lowering ttir to ttmetal backend.",
      mlir::tt::ttnn::createTTIRToTTNNBackendPipeline);
}
} // namespace mlir::tt::ttnn
