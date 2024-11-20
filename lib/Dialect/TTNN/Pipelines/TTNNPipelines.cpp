// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  ttir::TTIRLoadSystemDescOptions systemDescOptions;
  systemDescOptions.path = options.systemDescPath;

  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());

  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));

  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = ::llvm::SmallVector<int64_t>(
      options.meshShape.begin(), options.meshShape.end());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
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
    optimizerOptions.memoryLayoutAnalysisPolicy =
        options.memoryLayoutAnalysisPolicy;
    optimizerOptions.maxLegalLayouts = options.maxLegalLayouts;
    pm.addPass(mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
  }
}

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Add pass to add layout information.
  pm.addPass(createTTNNLayout());
  // Add pass to convert TTIR to TTNN.
  pm.addPass(createConvertTTIRToTTNNPass());
  // Add pass to remove unused values.
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

void createTTNNPipelineLayoutDecompositionPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
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

void createTTNNPipelineTTIRBroadcastFoldPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::tt::ttir::createTTIRBroadcastFold());
}

void createTTNNPipelineTTIRBroadcastFoldPassFromString(OpPassManager &pm,
                                                       std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineDeallocPass(pm, *optionsStruct);
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNPipelineTTIRBroadcastFoldPass(pm, options);
  createTTNNPipelineTTIRPasses(pm, options);
  createTTNNPipelineLoweringPasses(pm, options);
  createTTNNPipelineAnalysisPasses(pm, options);
  createTTNNPipelineLayoutDecompositionPass(pm, options);
  createTTNNPipelineDeallocPass(pm, options);
}

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  createTTIRToTTNNBackendPipeline(pm, options);
  pm.addPass(createConvertTTNNToEmitCPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTNNPipelines() {
  // TTIR to TTNN backend pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTIRToTTNNBackendPipelineOptions>(
      "ttir-to-ttnn-backend-pipeline",
      "Pipeline lowering TTIR to TTNN backend.",
      mlir::tt::ttnn::createTTIRToTTNNBackendPipeline);

  // TTIR to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitCPipelineOptions>(
      "ttir-to-emitc-pipeline",
      "Pipeline lowering TTIR to EmitC. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and then converts the resulting TTNN "
      "dialect to EmitC.",
      mlir::tt::ttnn::createTTIRToEmitCPipeline);
}
} // namespace mlir::tt::ttnn
