// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "mlir/Pass/PassManager.h"

#include "mlir/Transforms/Passes.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  ttir::TTIRLoadSystemDescOptions systemDescOptions;
  systemDescOptions.path = options.systemDescPath;

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());

  pm.addPass(mlir::tt::ttir::createTTIRSlidingWindow2dFixShapes());
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));

  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = ::llvm::SmallVector<int64_t>(
      options.meshShape.begin(), options.meshShape.end());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
  mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
  layoutOptions.initMemorySpace = mlir::tt::MemorySpace::System;
  layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceDRAM;
  layoutOptions.defaultDeviceMemoryLayout =
      mlir::tt::TensorMemoryLayout::Interleaved;
  pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
}

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  if (options.optimizerPassEnabled) {
    ttnn::TTNNOptimizerOptions optimizerOptions;
    optimizerOptions.overrideInputLayout = options.overrideInputLayout;
    optimizerOptions.overrideOutputLayout = options.overrideOutputLayout;
    optimizerOptions.memoryLayoutAnalysisEnabled =
        options.memoryLayoutAnalysisEnabled;
    optimizerOptions.memReconfigEnabled = options.memReconfigEnabled;
    optimizerOptions.maxLegalLayouts = options.maxLegalLayouts;
    pm.addPass(mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
  }
}

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Add pass to convert TTIR to TTNN.
  pm.addPass(createConvertTTIRToTTNNPass());
  // Add pass to remove unused values.
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeCompositeLayouts());
}

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDeallocate());
}

void createTTNNPipelineTTIRPassesFromString(OpPassManager &pm,
                                            std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineTTIRPasses(pm, *optionsStruct);
}

void createTTNNPipelineAnalysisPassesFromString(OpPassManager &pm,
                                                std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineAnalysisPasses(pm, *optionsStruct);
}

void createTTNNPipelineLoweringPassesFromString(OpPassManager &pm,
                                                std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineLoweringPasses(pm, *optionsStruct);
}

void createTTNNPipelineLayoutDecompositionPassFromString(OpPassManager &pm,
                                                         std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineLayoutDecompositionPass(pm, *optionsStruct);
}

void createTTNNPipelineDeallocPassFromString(OpPassManager &pm,
                                             std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineDeallocPass(pm, *optionsStruct);
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNPipelineTTIRPasses(pm, options);
  createTTNNPipelineLoweringPasses(pm, options);
  createTTNNPipelineAnalysisPasses(pm, options);
  createTTNNPipelineLayoutDecompositionPass(pm, options);
  createTTNNPipelineDeallocPass(pm, options);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTNNPipelines() {
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions>(
      "ttir-to-ttnn-backend-pipeline",
      "Pipeline lowering ttir to ttnn backend.",
      mlir::tt::ttnn::createTTIRToTTNNBackendPipeline);
}
} // namespace mlir::tt::ttnn
