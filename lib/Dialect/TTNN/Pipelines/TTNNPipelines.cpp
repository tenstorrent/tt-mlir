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

void createTTNNOptimizerHelper(
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
    optimizerOptions.rowMajorEnabled = options.rowMajorEnabled;
    pm.addPass(mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
  }
}

void createTTNNWorkaroundsHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  TTNNWorkaroundsOptions workaroundOptions{
      options.layouotWorkaroundsEnabled,
      options.decompositionWorkaroundsEnabled};
  pm.addPass(createTTNNWorkarounds(workaroundOptions));
}

void createTTNNDeallocateHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDeallocate());
}

void createTTIRImplicitDeviceHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = ::llvm::SmallVector<int64_t>(
      options.meshShape.begin(), options.meshShape.end());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
}

void createTTIRToTTIRDecompositionPassHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());
}

void createInlinerPassHelper(OpPassManager &pm,
                             const TTIRToTTNNBackendPipelineOptions &options) {
  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());
}

void createTTIRLoadSystemDescHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  ttir::TTIRLoadSystemDescOptions systemDescOptions;
  systemDescOptions.path = options.systemDescPath;
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));
}

void createTTNNLayoutHelper(OpPassManager &pm,
                            const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNLayout());
}

void createConvertTTIRToTTNNPassHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createConvertTTIRToTTNNPass());
}

void createRemoveDeadValuesPassHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::createRemoveDeadValuesPass());
}

void createCanonicalizerPassHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::createCanonicalizerPass());
}

void createConvertTTNNToEmitCPassHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createConvertTTNNToEmitCPass());
}

void createTTNNDecomposeLayoutsHelper(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
}

void createTTNNPipelineLoweringPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNLayoutHelper(pm, options);
  createConvertTTIRToTTNNPassHelper(pm, options);
  createRemoveDeadValuesPassHelper(pm, options);
}

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTIRToTTIRDecompositionPassHelper(pm, options);
  createInlinerPassHelper(pm, options);
  createTTIRLoadSystemDescHelper(pm, options);
  createTTIRImplicitDeviceHelper(pm, options);
}

// Create a pass to workaround issues in the TTNN dialect.
void createTTNNPipelineWorkaroundPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNWorkaroundsHelper(pm, options);
  createCanonicalizerPassHelper(pm, options);
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  createTTNNPipelineTTIRPasses(pm, options);
  createTTNNPipelineLoweringPasses(pm, options);
  createTTNNPipelineWorkaroundPass(pm, options);
  createTTNNOptimizerHelper(pm, options);
  createTTNNDecomposeLayoutsHelper(pm, options);
  createTTNNDeallocateHelper(pm, options);
}

void createTTNNDecomposeLayoutsHelperFromString(OpPassManager &pm,
                                                std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNDecomposeLayoutsHelper(pm, *optionsStruct);
}

void createTTNNDeallocateHelperFromString(OpPassManager &pm,
                                          std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNDeallocateHelper(pm, *optionsStruct);
}

void createTTIRToTTIRDecompositionPassFromString(OpPassManager &pm,
                                                 std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTIRToTTIRDecompositionPassHelper(pm, *optionsStruct);
}

void createInlinerPassFromString(OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createInlinerPassHelper(pm, *optionsStruct);
}

void createTTIRLoadSystemDescFromString(OpPassManager &pm,
                                        std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTIRLoadSystemDescHelper(pm, *optionsStruct);
}

void createTTIRImplicitDeviceFromString(OpPassManager &pm,
                                        std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTIRImplicitDeviceHelper(pm, *optionsStruct);
}

void createTTNNLayoutFromString(OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNLayoutHelper(pm, *optionsStruct);
}

void createConvertTTIRToTTNNPassFromString(OpPassManager &pm,
                                           std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createConvertTTIRToTTNNPassHelper(pm, *optionsStruct);
}

void createRemoveDeadValuesPassFromString(OpPassManager &pm,
                                          std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createRemoveDeadValuesPassHelper(pm, *optionsStruct);
}

void createTTNNPipelineLoweringPassesFromString(OpPassManager &pm,
                                                std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineLoweringPasses(pm, *optionsStruct);
}

void createTTNNPipelineTTIRPassesFromString(OpPassManager &pm,
                                            std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineTTIRPasses(pm, *optionsStruct);
}

void createTTNNWorkaroundsFromString(OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNWorkaroundsHelper(pm, *optionsStruct);
}

void createCanonicalizerPassFromString(OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createCanonicalizerPassHelper(pm, *optionsStruct);
}

void createTTNNPipelineWorkaroundPassFromString(OpPassManager &pm,
                                                std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineWorkaroundPass(pm, *optionsStruct);
}

void createTTNNOptimizerFromString(OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNOptimizerHelper(pm, *optionsStruct);
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
