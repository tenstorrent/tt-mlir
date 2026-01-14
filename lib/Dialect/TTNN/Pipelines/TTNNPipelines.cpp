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
#include "ttmlir/Dialect/TTNN/Transforms/DevicePassesWrapper.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttnn {
//===----------------------------------------------------------------------===//
// Helper functions which combine multiple passes into logical groupings.
//===----------------------------------------------------------------------===//

void createTTNNPipelineTTIRPasses(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {

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
      options.enableFusingConv2dWithMultiplyPattern,
      options.enablePermuteMatmulFusion};
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
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {

  // Add pass to check for unique operation locations if enabled
  if (options.checkUniqueLocations) {
    pm.addPass(mlir::tt::ttnn::createTTNNUniqueLocations());
  }
  if (options.optimizerPassEnabled) {
#ifdef TTMLIR_ENABLE_OPMODEL
    ttnn::TTNNOptimizerOptions optimizerOptions(options);
    // Wrap all Optimizer passes with device lifecycle management.
    DevicePassesWrapperOptions wrapperOptions;
    wrapperOptions.devicePtr = options.devicePtr;
    wrapperOptions.tensorL1UsageCap = options.tensorL1UsageCap;

    ttnn::TTNNOperationValidationAndFallbackOptions validationOptions;
    validationOptions.tensorL1UsageCap = options.tensorL1UsageCap;
    validationOptions.maxFallbackAttempts = options.maxFallbackAttempts;

    pm.addPass(createDevicePassesWrapper(
        [optimizerOptions, validationOptions](OpPassManager &innerPm) {
          // All Optimizer passes will be run inside the wrapper.
          innerPm.addPass(
              mlir::tt::ttnn::createTTNNRowMajorLayoutPropagation());
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
                                      bool removeDeadValuesEnabled = false) {
  // Add pass to add layout information.
  pm.addPass(createTTNNLayout());
  // Add pass to convert TTIR to TTNN.
  pm.addPass(createConvertTTIRToTTNNPass());
  // Add pass to remove unused values.
  if (removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
}

// Create TTNN fusing pass.
// If optimizer is enabled we wrap fusing pass inside optimizer wrapper
// to ensure device is properly initialized. This is required for op constraint
// validation for certain fusing patterns.
// If optimizer is not enabled we just add fusing pass directly and we don't
// do op constraint validation.
void createTTNNFusingPass(OpPassManager &pm,
                          const TTIRToTTNNDevicePipelineOptions &options) {
  if (options.enableFusing) {
    if (options.optimizerPassEnabled) {
#ifdef TTMLIR_ENABLE_OPMODEL
      DevicePassesWrapperOptions wrapperOptions;
      wrapperOptions.devicePtr = options.devicePtr;
      wrapperOptions.tensorL1UsageCap = options.tensorL1UsageCap;

      pm.addPass(createDevicePassesWrapper(
          [](OpPassManager &innerPm) {
            TTNNFusingOptions fusingOptions;
            fusingOptions.enableOpConstraints = true;
            innerPm.addPass(mlir::tt::ttnn::createTTNNFusing(fusingOptions));
          },
          wrapperOptions));
#else
      llvm::llvm_unreachable_internal(
          "TTNNOptimizer passes require OpModel support to be enabled.");
#endif
    } else {
      pm.addPass(mlir::tt::ttnn::createTTNNFusing());
    }
  }
}

// Create a pass to workaround issues in the TTNN dialect.
void createTTNNPipelineWorkaroundPass(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {

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
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {
  pm.addPass(createTTNNDecomposeLayouts());
}

void createTTNNPipelineDeallocPass(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {
  pm.addPass(createTTNNDeallocate());
}

//===----------------------------------------------------------------------===//
// Intermediate pipelines used to build end-to-end pipelines.
// Each of these pipelines lowers either the Device or CPU module, which is
// encoded in the pipeline name.
//
// The OpPassManager argument is expected to correspond to the top-level
// (root) ModuleOp.
//===----------------------------------------------------------------------===//

// Pipeline which prepares the TTIR ops in the Device module for TTNN
// lowering, and then lowers them to TTNN dialect.
//
// CPU module does get modified in this pipeline, but only by
// adding more TTIR ops to it (CPUHoistTransform passes).
//
void createTTIRToTTNNDevicePipeline(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {
  // Resolve options controlled by optimization_level.
  options.resolveOptimizationLevelOptions();

  pm.addPass(mlir::createCanonicalizerPass());

  // Add Decomposition pass here to ensure it runs before hoisting.
  TTIRToTTIRDecompositionOptions decompOptions;
  decompOptions.decompConfig = DecompMode::CPUFallback;
  pm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass(decompOptions));

  // Create device module, if not already present.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());

  // Hoist manually tagged ops to CPU module.
  pm.addPass(ttir::createCPUHoistManuallyTaggedOpsTransform());

  // Device module passes before const-eval CPU hoisting.
  {
    auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

    // Element type normalization should be applied only to the ops in the
    // Device Module, since we aren't restricted with element types on CPU.
    ttir::ElementTypeNormalizationOptions elementTypeNormalizationOptions;
    elementTypeNormalizationOptions.enableBfp8Conversion =
        options.enableBfp8Conversion;
    devicePm.addPass(
        ttir::createElementTypeNormalization(elementTypeNormalizationOptions));

    createTTNNPipelineTTIRPasses(devicePm, options);

    ttir::TTIRQuantDataTypeConversionPassOptions quantOptions;
    quantOptions.targetBitWidth = options.quantBitWidth;
    devicePm.addPass(ttir::createTTIRQuantDataTypeConversionPass(quantOptions));

    // Const-eval hoisting pass.
    if (options.enableConstEval) {
      // Hoist const-eval subgraphs into separate functions in Device module.
      devicePm.addPass(transforms::createConstEvalHoistTransform());
    }
  }

  // CPU-hoisting pass for const-eval subgraphs.
  if (options.enableCPUHoistedConstEval) {
    pm.addPass(ttir::createCPUHoistConstEvalTransform());
  }

  // Device module passes after const-eval CPU hoisting.
  {
    auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

    // Run TTNN lowering passes on Device module.
    createTTNNPipelineLoweringPasses(devicePm, options.removeDeadValuesEnabled);
    createTTNNFusingPass(devicePm, options);

    createTTNNPipelineWorkaroundPass(devicePm, options);
    // Add BFP8 weight conversion pass before analysis passes.
    // Analysis passes need to know data formats to decide on shardings.
    if (options.experimentalBfp8Weights) {
      devicePm.addPass(createTTNNWeightBFP8Conversion());
    }

    // Apply ComputeKernelConfig settings before analysis passes.
    // This ensures that analysis passes see the configured values.
    // Check if math fidelity is explicitly set (CLI or programmatic).
    bool mathFidelityExplicitlySet =
        options.computeCfgMathFidelitySet ||
        (options.computeCfgMathFidelity.getNumOccurrences() > 0);

    // Run pass only when at least one compute config option is explicitly set.
    if (mathFidelityExplicitlySet || options.computeCfgFp32DestAccEn) {
      // Create options struct and forward pipeline options.
      TTNNSetComputeKernelConfigOptions setConfigOptions;

      // Forward math fidelity only if explicitly set.
      if (mathFidelityExplicitlySet) {
        setConfigOptions.mathFidelity = options.computeCfgMathFidelity;
      }

      // Forward fp32DestAccEn value (defaults to true).
      setConfigOptions.fp32DestAccEn = options.computeCfgFp32DestAccEn;

      devicePm.addPass(createTTNNSetComputeKernelConfig(setConfigOptions));
    }

    createTTNNPipelineAnalysisPasses(devicePm, options);
    // We need to re-run const-eval to pick up const prepare conv2d weight ops
    // split during the analysis passes.
    if (options.enableConstEval) {
      devicePm.addPass(transforms::createConstEvalHoistTransform());

      // Now that all const-eval passes have run, we can force the const-eval
      // function inputs to system memory.
      if (options.enableConstEvalInputsToSystemMemory) {
        devicePm.addPass(createTTNNConstEvalInputsToSystemMemory());

        // Clean up any redundant to_layout ops that may have been introduced
        // previously.
        devicePm.addPass(mlir::createCanonicalizerPass());
      }
    }
    createTTNNPipelineLayoutDecompositionPass(devicePm, options);
    if (options.enableTrace) {
      devicePm.addPass(tt::ttnn::createTTNNTraceHoistTransform());
    }
    // Fold ttcore.optimization_barrier ops before deallocation.
    devicePm.addPass(ttcore::createTTCoreOptimizationBarrierFold());

    createTTNNPipelineDeallocPass(devicePm, options);

    if (options.ttnnPerfMetricsEnabled) {
      ttnn::TTNNCollectPerfMetricsOptions metricsOptions{
          options.ttnnPerfMetricsOutputFile,
          options.ttnnPerfMetricsVerboseOutputEnabled, options.enableTrace};

      devicePm.addPass(
          mlir::tt::ttnn::createTTNNCollectPerfMetrics(metricsOptions));
    }
  }
}

// Pipeline which lowers the Device module from TTNN to EmitC dialect.
//
void createTTNNToEmitCDevicePipeline(
    OpPassManager &pm, const TTNNToEmitCDevicePipelineOptions &options) {
  pm.addPass(createTTNNAdjustDeallocs());

  // Unwrapping the device module.
  //
  // TODO(dmilinkovic): Should be removed after support for generating
  // a dynamic library from CPU module is implemented inside EmitC translation
  // pipeline - issue #6100.
  //
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

// Pipeline which lowers the Device module from TTNN to EmitPy dialect.
//
void createTTNNToEmitPyDevicePipeline(
    OpPassManager &pm, const TTNNToEmitPyDevicePipelineOptions &options) {
  auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

  devicePm.addPass(createTTNNAdjustDeallocs());

  // Apply EmitPy-specific workarounds before conversion
  devicePm.addPass(createTTNNEmitPyWorkarounds());

  if (options.targetModule) {
    // In module path, run tuplification with forced settings and add device
    // argument. This ensures tensor inputs are always tuplified even when the
    // input is empty, which is necessary for proper module interface
    // generation.
    //
    TTNNTuplifyTensorsOptions tuplifyOptions;
    tuplifyOptions.tuplifyInputIfEmpty = true;
    devicePm.addPass(createTTNNTuplifyTensors(tuplifyOptions));
    devicePm.addPass(createTTNNPrepareModuleForExport());
  } else {
    // In canonical path, run tuplification + input generation/loading.
    //
    devicePm.addPass(createTTNNTuplifyTensors());

    if (options.loadInputTensorsFromDisk) {
      TTNNLoadInputTensorsOptions loadOptions;
      loadOptions.tensorLoadDirectory = options.tensorLoadDirectory;
      loadOptions.tensorLoadFilePrefix = options.tensorLoadFilePrefix;
      devicePm.addPass(createTTNNLoadInputTensors(loadOptions));
    } else {
      devicePm.addPass(createTTNNCreateInputGenerators());
    }
  }

  devicePm.addPass(createConvertTTNNToEmitPyPass());

  devicePm.addPass(createEmitPyNameVarsPass());
}

// Pipeline which lowers the CPU module from TTIR to EmitPy using
// TTNN golden functions.
//
void createTTIRToEmitPyCPUPipeline(OpPassManager &pm) {
  auto &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

  // Prepare CPU module TTIR ops for TTNN lowering.
  //
  cpuPm.addPass(mlir::tt::ttir::createTTIRFusing());
  cpuPm.addPass(mlir::tt::createTTIRToTTIRDecompositionPass());
  cpuPm.addPass(mlir::tt::ttir::createTTIRFusing());
  cpuPm.addPass(ttir::createTTIRFlattenSlidingWindow());

  // Lower CPU module to TTNN.
  //
  createTTNNPipelineLoweringPasses(cpuPm);

  // Lower CPU module to EmitPy.
  //
  ConvertTTNNToEmitPyOptions options;
  options.enableGoldenMode = true;
  cpuPm.addPass(createConvertTTNNToEmitPyPass(options));

  cpuPm.addPass(createEmitPyNameVarsPass());
}

//===----------------------------------------------------------------------===//
// End-to-end pipelines.
//===----------------------------------------------------------------------===//

// Complete pipeline for lowering TTIR to TTNN backend.
//
// Device module: TTIR -> TTNN.
// CPU module: TTIR (+ StableHLO) -> LLVM.
//
void createTTIRToTTNNBackendPipeline(
    OpPassManager &pm, const TTIRToTTNNBackendPipelineOptions &options) {

  createTTIRToTTNNDevicePipeline(pm, options);

  ttir::createTTIRToLLVMCPUPipeline(pm, options);
}

// Complete pipeline for lowering TTIR to EmitC.
//
// Device module: TTIR -> TTNN -> EmitC.
// CPU module: TTIR (+ StableHLO) -> LLVM.
//
void createTTIRToEmitCPipeline(OpPassManager &pm,
                               const TTIRToEmitCPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitCPipeline");
  }

  createTTIRToTTNNDevicePipeline(pm, options);
  createTTNNToEmitCDevicePipeline(pm, options);

  // TODO(dmilinkovic): Lower CPU module to LLVM - issue #6100.
}

// Complete pipeline for lowering TTIR to EmitPy.
//
// Device module: TTIR -> TTNN -> EmitPy.
// CPU module: TTIR -> TTNN -> EmitPy (with golden functions).
//
void createTTIRToEmitPyPipeline(OpPassManager &pm,
                                const TTIRToEmitPyPipelineOptions &options) {
  if (options.enableTrace) {
    llvm::report_fatal_error(
        "Trace currently not supported in createTTIRToEmitPyPipeline");
  }

  createTTIRToTTNNDevicePipeline(pm, options);
  createTTNNToEmitPyDevicePipeline(pm, options);

  // Lower CPU module to EmitPy using TTNN golden functions.
  //
  createTTIRToEmitPyCPUPipeline(pm);

  // Link Device and CPU modules into the root module.
  //
  pm.addPass(createEmitPyLinkModulesPass());
}

// Workaround pipeline for lowering mixed TTIR/TTNN to EmitPy.
//
// This is a temporary pipeline used when the input is already in TTNN dialect
// (e.g., from tt-alchemist with TTNN input). It handles the case where the
// device module contains TTNN ops but the CPU module may still need TTIR
// processing.
//
// Device module: TTNN -> EmitPy.
// CPU module: TTIR -> TTNN -> EmitPy (with golden functions).
//
void createWorkaroundMixedTTIRTTNNToEmitPyPipeline(
    OpPassManager &pm, const TTNNToEmitPyDevicePipelineOptions &options) {
  createTTNNToEmitPyDevicePipeline(pm, options);
  createTTIRToEmitPyCPUPipeline(pm);
  pm.addPass(createEmitPyLinkModulesPass());
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

  // TTNN to EmitC Device pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNToEmitCDevicePipelineOptions>(
      "ttnn-to-emitc-device-pipeline",
      "Pipeline lowering TTNN to EmitC in the Device module.",
      mlir::tt::ttnn::createTTNNToEmitCDevicePipeline);

  // TTNN to EmitPy Device pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNToEmitPyDevicePipelineOptions>(
      "ttnn-to-emitpy-device-pipeline",
      "Pipeline lowering TTNN to EmitPy in the Device module.",
      mlir::tt::ttnn::createTTNNToEmitPyDevicePipeline);

  // TTIR to EmitC pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitCPipelineOptions>(
      "ttir-to-emitc-pipeline", "Pipeline lowering TTIR to EmitC.",
      mlir::tt::ttnn::createTTIRToEmitCPipeline);

  // TTIR to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<mlir::tt::ttnn::TTIRToEmitPyPipelineOptions>(
      "ttir-to-emitpy-pipeline", "Pipeline lowering TTIR to EmitPy.",
      mlir::tt::ttnn::createTTIRToEmitPyPipeline);

  // Workaround pipeline for mixed TTIR/TTNN to EmitPy.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNToEmitPyDevicePipelineOptions>(
      "mixed-ttnn-ttir-to-emitpy-pipeline",
      "Workaround pipeline lowering mixed TTIR/TTNN to EmitPy.",
      mlir::tt::ttnn::createWorkaroundMixedTTIRTTNNToEmitPyPipeline);
}
} // namespace mlir::tt::ttnn
