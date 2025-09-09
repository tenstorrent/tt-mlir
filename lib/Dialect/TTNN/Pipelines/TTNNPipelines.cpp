// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <llvm/Support/raw_ostream.h>

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

  ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(
      mlir::tt::ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions));

  pm.addPass(
      mlir::tt::ttcore::createTTPopulateArgumentTypes(options.argumentTypeMap));
  pm.addPass(mlir::createCanonicalizerPass());
  ttir::TTIRFusingOptions fusingOptions{
      options.enableFusingConv2dWithMultiplyPattern};
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
  if (options.enableQuantDequantConversion) {
    pm.addPass(mlir::tt::ttir::createTTIRQuantDequantConversion());
  }
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());

  // Flattening sliding window ops for compatibility with conversion to TTNN
  pm.addPass(mlir::tt::ttir::createTTIRFlattenSlidingWindow());

  // Add pass to erase inverse ops. We will explicate TMs so that
  // erase inverse ops can commute TMs through otherwise implicit
  // broadcasts, and handle rank-changing reshape ops which are
  // also otherwise implicit.
  if (options.eraseInverseOpsEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRExplicateTMs());
    pm.addPass(mlir::tt::ttir::createTTIREraseInverseOps());
  }
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
}

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Add pass to check for unique operation locations if enabled
  if (options.checkUniqueLocations) {
    pm.addPass(mlir::tt::ttnn::createTTNNUniqueLocations());
  }
  if (options.optimizerPassEnabled) {
    ttnn::TTNNOptimizerOptions optimizerOptions;
    optimizerOptions.insertMemReconfig = options.insertMemReconfig;
    optimizerOptions.overrideOutputLayout = options.overrideOutputLayout;
    optimizerOptions.overrideConv2dConfig = options.overrideConv2dConfig;
    optimizerOptions.memoryLayoutAnalysisEnabled =
        options.memoryLayoutAnalysisEnabled;
    optimizerOptions.l1InterleavedFallbackAnalysisEnabled =
        options.l1InterleavedFallbackAnalysisEnabled;
    optimizerOptions.memReconfigEnabled = options.memReconfigEnabled;
    optimizerOptions.memoryLayoutAnalysisPolicy =
        options.memoryLayoutAnalysisPolicy;
    optimizerOptions.maxLegalLayouts = options.maxLegalLayouts;
    optimizerOptions.rowMajorEnabled = options.rowMajorEnabled;
    optimizerOptions.tensorL1UsageCap = options.tensorL1UsageCap;
    optimizerOptions.devicePtr = options.devicePtr;
    pm.addPass(mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
    pm.addPass(mlir::createCanonicalizerPass());
#ifdef TTMLIR_ENABLE_OPMODEL
    pm.addPass(mlir::tt::ttnn::createTTNNOperationValidationAndFallback());
    pm.addPass(mlir::tt::ttnn::createTTNNPrepareConv2dWeightsAndBias());
#endif
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
      options.layoutWorkaroundsEnabled,
      options.decompositionWorkaroundsEnabled};

  if (options.optimizerPassEnabled) {
    workaroundOptions.optimizerEnabled = true;
  }

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

void createTTNNPipelineTTIRImplicitBroadcastFoldPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  if (options.implicitBroadcastFoldingEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRImplicitBroadcastFold());
  }
}

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  pm.addPass(mlir::createCanonicalizerPass());
  // Element type normalization should be the first pass in the pipeline.
  ttir::ElementTypeNormalizationOptions elementTypeNormalizationOptions;
  elementTypeNormalizationOptions.enableBfp8Conversion =
      options.enableBfp8Conversion;
  pm.addPass(
      ttir::createElementTypeNormalization(elementTypeNormalizationOptions));
  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular TTIR to TTNN pipeline on DeviceModule.
  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();
  createTTNNPipelineTTIRPasses(devicePm, options);
  createTTNNPipelineTTIRImplicitBroadcastFoldPass(devicePm, options);

  ttir::TTIRQuantDataTypeConversionPassOptions quantOptions;
  quantOptions.targetBitWidth = options.quantBitWidth;
  devicePm.addPass(ttir::createTTIRQuantDataTypeConversionPass(quantOptions));

  createTTNNPipelineLoweringPasses(devicePm, options);
  if (options.enableFusing) {
    devicePm.addPass(tt::ttnn::createTTNNFusing());
  }
  createTTNNPipelineWorkaroundPass(devicePm, options);
  if (options.enableConstEval) {
    devicePm.addPass(transforms::createConstEvalHoistTransform());
  }
  createTTNNPipelineAnalysisPasses(devicePm, options);
  // We need to re-run const-eval to pick up const prepare conv2d weight ops
  // split during the analysis passes.
  if (options.enableConstEval) {
    devicePm.addPass(transforms::createConstEvalHoistTransform());
  }
  createTTNNPipelineLayoutDecompositionPass(devicePm, options);
  if (options.enableTrace) {
    devicePm.addPass(tt::ttnn::createTTNNTraceHoistTransform());
  }
  createTTNNPipelineDeallocPass(devicePm, options);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  ttir::LinalgToLLVMPipelineOptions linalgToLLVMOptions;
  ttir::createTTIRToCPUPipeline(pm, linalgToLLVMOptions);
}

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitCPipeline");
  }
  createTTIRToTTNNBackendPipeline(pm, options);
  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());
  pm.addPass(createTTNNTuplifyTensors());
  pm.addPass(createTTNNCreateInputGenerators());
  pm.addPass(createConvertTTNNToEmitCPass());
}

void createTTIRToEmitCSOPipeline(OpPassManager &pm,
                                 const TTIRToEmitCSOPipelineOptions &options) {
  // Pass specific options.
  //
  // Always set input tuplification to true - dylib signatures are contractual
  // and required to have tuples/vectors on the input signature (and output).
  //
  TTNNTuplifyTensorsOptions tuplifyOptions;
  tuplifyOptions.tuplifyInputIfEmpty = true;

  // Construct pipeline from other pipelines/passes.
  //
  createTTIRToTTNNBackendPipeline(pm, options);
  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());
  pm.addPass(createTTNNTuplifyTensors(tuplifyOptions));
  pm.addPass(createConvertTTNNToEmitCPass());
}

void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options) {
  createTTIRToTTNNBackendPipeline(pm, options);
  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());
  pm.addPass(createTTNNTuplifyTensors());
  pm.addPass(createTTNNCreateInputGenerators());
  pm.addPass(createConvertTTNNToEmitPyPass());
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

  // TTIR to EmitC SO pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitCSOPipelineOptions>(
      "ttir-to-emitc-so-pipeline",
      "Pipeline lowering TTIR to EmitC, similar to TTIRToEmitCPipeline, but "
      "with emitted C++ code packaged so that it's suitable for compiling into "
      "a shared object.",
      mlir::tt::ttnn::createTTIRToEmitCSOPipeline);

  // TTIR to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitPyPipelineOptions>(
      "ttir-to-emitpy-pipeline",
      "Pipeline lowering TTIR to EmitPy. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and then converts the resulting TTNN "
      "dialect to EmitPy.",
      mlir::tt::ttnn::createTTIRToEmitPyPipeline);
}
} // namespace mlir::tt::ttnn
