// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
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

  tt::TTRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(mlir::tt::createTTRegisterDevicePass(registerDeviceOptions));

  pm.addPass(mlir::tt::createTTPopulateArgumentTypes(options.argumentTypeMap));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());
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
    optimizerOptions.rowMajorEnabled = options.rowMajorEnabled;
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
  if (options.removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}

// Create a pass to workaround issues in the TTNN dialect.
void createTTNNPipelineWorkaroundPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  TTNNWorkaroundsOptions workaroundOptions{
      options.layoutWorkaroundsEnabled, options.decompositionWorkaroundsEnabled,
      options.repeatFoldingWorkaroundEnabled};
  pm.addPass(createTTNNWorkarounds(workaroundOptions));
  pm.addPass(mlir::createCanonicalizerPass());
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

void createTTNNPipelineTTIRImplicitBroadcastFoldPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  if (options.implicitBroadcastFoldingEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRImplicitBroadcastFold());
  }
}

void createTTNNPipelineTTIRImplicitBroadcastFoldPassFromString(
    OpPassManager &pm, std::string options) {
  auto optionsStruct =
      TTIRToTTNNBackendPipelineOptions::createFromString(options);
  createTTNNPipelineTTIRImplicitBroadcastFoldPass(pm, *optionsStruct);
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Create DeviceModule to wrap all ops.
  pm.addPass(tt::createTTWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular TTIR to TTNN pipeline on DeviceModule.
  OpPassManager &devicePm =
      pm.nest<tt::DeviceModuleOp>().nest<mlir::ModuleOp>();
  createTTNNPipelineTTIRPasses(devicePm, options);
  createTTNNPipelineTTIRImplicitBroadcastFoldPass(devicePm, options);
  createTTNNPipelineLoweringPasses(devicePm, options);
  createTTNNPipelineWorkaroundPass(devicePm, options);
  createTTNNPipelineAnalysisPasses(devicePm, options);
  createTTNNPipelineLayoutDecompositionPass(devicePm, options);
  createTTNNPipelineDeallocPass(devicePm, options);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  OpPassManager &cpuPm = pm.nest<tt::CPUModuleOp>().nest<mlir::ModuleOp>();
  cpuPm.addPass(createConvertTTIRToLinalgPass());
  ttir::LinalgToLLVMPipelineOptions linalgToLLLVMOptions;
  ttir::createLinalgToLLVMPipeline(cpuPm, linalgToLLLVMOptions);
  cpuPm.addPass(llvm_util::createLLVMEmitCallingConventionWrapperFuncs());
}

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  createTTIRToTTNNBackendPipeline(pm, options);
  pm.addPass(tt::createTTUnwrapDeviceModulePass());
  pm.addPass(createTTNNCreateInputGenerators());
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
