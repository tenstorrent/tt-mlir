// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "mlir/Pass/PassManager.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTIRPipelineFirst(OpPassManager &pm,
                             const TTIRToTTNNBackendPipelineOptions &options) {
  ttir::TTIRLoadSystemDescOptions systemDescOptions;
  systemDescOptions.path = options.systemDescPath;
  pm.addPass(mlir::tt::ttir::createTTIRSlidingWindow2dFixShapes());
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));

  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = options.meshShape;
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
  mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
  layoutOptions.initMemorySpace = mlir::tt::MemorySpace::System;
  layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceDRAM;
  layoutOptions.defaultDeviceMemoryLayout =
      mlir::tt::TensorMemoryLayout::Interleaved;
  pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));

  if (options.optimizerPassEnabled) {
    ttir::TTIROptimizerOptions optimizerOptions;
    optimizerOptions.overrideGridSizes = options.overrideGridSizes;
    optimizerOptions.shardingPassEnabled = options.shardingPassEnabled;
    pm.addPass(mlir::tt::ttir::createTTIROptimizer(optimizerOptions));
  }
}

void createTTNNPipelineSecond(OpPassManager &pm,
                              const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createConvertTTIRToTTNNPass());
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTIRPipelineFirst(pm, options);
  createTTNNPipelineSecond(pm, options);
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
