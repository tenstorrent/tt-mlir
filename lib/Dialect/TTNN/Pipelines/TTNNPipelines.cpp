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
#include "ttmlir/Dialect/TTNN/Transforms/OptimizerPassesWrapper.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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
  if (options.implicitBroadcastFoldingEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRImplicitBroadcastFold());
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
#ifdef TTMLIR_ENABLE_OPMODEL
    ttnn::TTNNOptimizerOptions optimizerOptions(options);
    // Wrap all Optimizer passes with device lifecycle management.
    OptimizerPassesWrapperOptions wrapperOptions;
    wrapperOptions.devicePtr = options.devicePtr;
    wrapperOptions.tensorL1UsageCap = options.tensorL1UsageCap;

    ttnn::TTNNOperationValidationAndFallbackOptions validationOptions{
        options.tensorL1UsageCap};

    pm.addPass(createOptimizerPassesWrapper(
        [optimizerOptions, validationOptions](OpPassManager &innerPm) {
          // All Optimizer passes will be run inside the wrapper.
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNOptimizer(optimizerOptions));
          innerPm.addPass(mlir::createCanonicalizerPass());
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNOperationValidationAndFallback(
                  validationOptions));
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNPrepareConv2dWeightsAndBias());
        },
        wrapperOptions));
#else
    llvm::llvm_unreachable_internal(
        "TTNNOptimizer passes require OpModel support to be enabled.");
#endif
  }
}

void createTTNNPipelineLoweringPasses(OpPassManager &pm,
                                      bool removeDeadValuesEnabled) {
  // Add pass to add layout information.
  pm.addPass(createTTNNLayout());
  // Add pass to convert TTIR to TTNN.
  pm.addPass(createConvertTTIRToTTNNPass());
  // Add pass to remove unused values.
  if (removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}

// Create a pass to workaround issues in the TTNN dialect.
void createTTNNPipelineWorkaroundPass(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

  // If the workaround pass is disabled, skip adding it.
  if (options.disableWorkarounds) {
    return;
  }

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

void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {
  // Resolve options controlled by optimization_level.
  options.resolveOptimizationLevelOptions();

  pm.addPass(mlir::createCanonicalizerPass());

  // Create device module, if not already present.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());

  // Hoist manually tagged ops to CPU module.
  pm.addPass(ttir::createCPUHoistManuallyTagedOpsTransform());

  auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

  // Add Decomposition pass here to ensure it runs before hoisting.
  TTIRToTTIRDecompositionOptions decompOptions;
  decompOptions.decompConfig = DecompMode::CPUFallback;
  devicePm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass(decompOptions));

  // Element type normalization should be applied only to the ops in the
  // Device Module, since we aren't restricted with element types on CPU.
  ttir::ElementTypeNormalizationOptions elementTypeNormalizationOptions;
  elementTypeNormalizationOptions.enableBfp8Conversion =
      options.enableBfp8Conversion;
  devicePm.addPass(
      ttir::createElementTypeNormalization(elementTypeNormalizationOptions));

  createTTNNPipelineTTIRPasses(devicePm, options);

  // Quant data type conversion pass
  ttir::TTIRQuantDataTypeConversionPassOptions quantOptions;
  quantOptions.targetBitWidth = options.quantBitWidth;
  devicePm.addPass(ttir::createTTIRQuantDataTypeConversionPass(quantOptions));

  // Const-eval hoisting passes
  if (options.enableConstEval) {
    // Hoist const-eval subgraphs into separate functions in Device module.
    devicePm.addPass(transforms::createConstEvalHoistTransform());

    // Then, hoist const-eval subgraphs to CPU module.
    devicePm.addPass(ttir::createCPUHoistConstEvalTransform());
  }

  // Enable DPS semantics for hoisted functions in Device module
  // if the lowering is enabled.
  if (options.enableCPUModuleLowering) {
    devicePm.addPass(transforms::createEnableDPSForHoistedFuncs());
  }

  // Run TTNN lowering passes on Device module.
  createTTNNPipelineLoweringPasses(devicePm, options.removeDeadValuesEnabled);
  if (options.enableFusing) {
    devicePm.addPass(tt::ttnn::createTTNNFusing());
  }
  createTTNNPipelineWorkaroundPass(devicePm, options);
  // Add BFP8 weight conversion pass before analysis passes.
  // Analysis passes need to know data formats to decide on shardings.
  if (options.experimentalBfp8Weights) {
    devicePm.addPass(createTTNNWeightBFP8Conversion());
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
  // Fold ttcore.optimization_barrier ops before deallocation
  devicePm.addPass(ttcore::createTTCoreOptimizationBarrierFold());

  createTTNNPipelineDeallocPass(devicePm, options);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  if (options.enableCPUModuleLowering) {
    auto &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

    ttir::LinalgToLLVMPipelineOptions linalgToLLVMOptions;
    ttir::createTTIRToCPUPipeline(cpuPm, linalgToLLVMOptions);
  }
}

void createTTNNBackendToEmitCPipeline(
    OpPassManager &pm, const TTNNBackendToEmitCPipelineOptions &options) {

  pm.addPass(createTTNNAdjustDeallocs());

  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());

  if (options.targetDylib) {
    // In dylib path, only run tuplification with forced settings.
    // This ensures tensor inputs are always tuplified even when the input is
    // empty, which is necessary for proper dylib interface generation.
    //
    TTNNTuplifyTensorsOptions tuplifyOptions;
    tuplifyOptions.tuplifyInputIfEmpty = true;
    pm.addPass(createTTNNTuplifyTensors(tuplifyOptions));
  } else {
    // In canonical path, run tuplification + input generation/loading.
    //
    TTNNTuplifyTensorsOptions tuplifyOptions;
    tuplifyOptions.tuplifyInputIfEmpty = options.tuplifyInputIfEmpty;
    pm.addPass(createTTNNTuplifyTensors(tuplifyOptions));

    if (options.loadInputTensorsFromDisk) {
      TTNNLoadInputTensorsOptions loadOptions;
      loadOptions.tensorLoadDirectory = options.tensorLoadDirectory;
      loadOptions.tensorLoadFilePrefix = options.tensorLoadFilePrefix;
      pm.addPass(createTTNNLoadInputTensors(loadOptions));
    } else {
      pm.addPass(createTTNNCreateInputGenerators());
    }
  }

  pm.addPass(createConvertTTNNToEmitCPass());
}

void createTTNNBackendToEmitPyPipeline(
    OpPassManager &pm, const TTNNBackendToEmitPyPipelineOptions &options) {
  // Device module passes
  //
  {
    auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

    devicePm.addPass(createTTNNEmitPyWorkarounds());
  }

  // CPU module passes
  //
  {
    auto &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

    // Lower CPU module to TTNN
    //
    cpuPm.addPass(ttir::createTTIRFlattenSlidingWindow());
    createTTNNPipelineLoweringPasses(cpuPm, true);
  }

  // Merge CPU and Device modules back into a single ModuleOp by
  // replacing CPU-hosited function stubs with their definitions.
  //
  pm.addPass(ttcore::createTTCoreMergeCPUAndDeviceModulesPass());

  // Tuplify tensors + input generation/loading in the merged module.
  //
  {
    pm.addPass(createTTNNTuplifyTensors());

    if (options.loadInputTensorsFromDisk) {
      TTNNLoadInputTensorsOptions loadOptions;
      loadOptions.tensorLoadDirectory = options.tensorLoadDirectory;
      loadOptions.tensorLoadFilePrefix = options.tensorLoadFilePrefix;
      pm.addPass(createTTNNLoadInputTensors(loadOptions));
    } else {
      pm.addPass(createTTNNCreateInputGenerators());
    }
  }

  // Finally, perform TTNN to EmitPy conversion.
  //
  pm.addPass(createConvertTTNNToEmitPyPass());
}

void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitCPipeline");
  }

  // TTIR -> TTNN Backend -> EmitC.
  //
  createTTIRToTTNNBackendPipeline(pm, options);
  createTTNNBackendToEmitCPipeline(pm, options);
}

void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitPyPipeline");
  }

  // TTIR -> TTNN Backend -> EmitPy.
  //
  createTTIRToTTNNBackendPipeline(pm, options);
  createTTNNBackendToEmitPyPipeline(pm, options);
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

  // TTNN backend to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNBackendToEmitCPipelineOptions>(
      "ttnn-backend-to-emitc-pipeline",
      "Pipeline lowering TTNN backend to EmitC.",
      mlir::tt::ttnn::createTTNNBackendToEmitCPipeline);

  // TTNN backend to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNBackendToEmitPyPipelineOptions>(
      "ttnn-backend-to-emitpy-pipeline",
      "Pipeline lowering TTNN backend to EmitPy.",
      mlir::tt::ttnn::createTTNNBackendToEmitPyPipeline);

  // TTIR to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitCPipelineOptions>(
      "ttir-to-emitc-pipeline",
      "Pipeline lowering TTIR to EmitC. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and --ttnn-backend-to-emitc-pipeline.",
      mlir::tt::ttnn::createTTIRToEmitCPipeline);

  // TTIR to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitPyPipelineOptions>(
      "ttir-to-emitpy-pipeline",
      "Pipeline lowering TTIR to EmitPy. Under the hood, it runs "
      "--ttir-to-ttnn-backend-pipeline and --ttnn-backend-to-emitpy-pipeline.",
      mlir::tt::ttnn::createTTIRToEmitPyPipeline);
}
} // namespace mlir::tt::ttnn
