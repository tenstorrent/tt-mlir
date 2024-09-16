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

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

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

  if (not options.skipTTNNConversion) {
    // Passes to convert TTIR -> TTNN.
    pm.addPass(createConvertTTIRToTTNNPass());
  }
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
