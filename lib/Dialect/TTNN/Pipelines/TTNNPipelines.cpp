// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"

#include "ttmlir/Conversion/TTIRToEmitPy/TTIRToEmitPy.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
#include "ttmlir/Conversion/TTNNToTTIR/TTNNToTTIR.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/DevicePassesWrapper.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    registerDeviceOptions.meshTopology = llvm::to_vector(options.meshTopology);
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
  pm.addPass(mlir::createCanonicalizerPass());
  if (options.implicitBroadcastFoldingEnabled) {
    pm.addPass(mlir::tt::ttir::createTTIRImplicitBroadcastFold());
  }
  if (options.enableFusing) {
    pm.addPass(mlir::tt::ttir::createTTIRFusing(fusingOptions));
  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Inlines all private functions. I.e flattens the program into the main
  // function. Removes all private functions.
  pm.addPass(mlir::createInlinerPass());

  // Infer kv_cache argument types from cache operations.
  pm.addPass(mlir::tt::ttir::createTTIRInferKVCacheArgumentTypes());

  // Propagate per-arg weight_dtype annotations through TM ops to consumers.
  pm.addPass(mlir::tt::ttir::createTTIRPropagateWeightDtype());

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
  pm.addPass(mlir::tt::ttir::createTTIRFoldFullToScalar());
}

void createTTNNPipelineAnalysisPasses(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {

  pm.addPass(mlir::tt::ttnn::createTTNNConfigureCCLOps());

  // Add pass to check for unique operation locations if enabled
  if (options.checkUniqueLocations) {
    pm.addPass(mlir::tt::ttnn::createTTNNUniqueLocations());
  }
  if (options.optimizerPassEnabled) {
#ifdef TTMLIR_ENABLE_OPMODEL
    // Wrap all Optimizer passes with device lifecycle management.
    DevicePassesWrapperOptions wrapperOptions;
    wrapperOptions.devicePtr = options.devicePtr;
    wrapperOptions.tensorL1UsageCap = options.tensorL1UsageCap;

    ttnn::TTNNOperationValidationAndFallbackOptions validationOptions;
    validationOptions.maxFallbackAttempts = options.maxFallbackAttempts;

    if (!options.enableGreedyOptimizer) {
      // Default: chain-based TTNNOptimizer.
      ttnn::TTNNOptimizerOptions optimizerOptions(options);
      pm.addPass(createDevicePassesWrapper(
          [optimizerOptions, validationOptions](OpPassManager &innerPm) {
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
    } else {
      // Greedy optimizer: two new passes replace TTNNOptimizer.
      TTNNGreedyMemoryLayoutPropagationPipelineOptions propagationOptions;
      propagationOptions.maxLegalLayouts = options.maxLegalLayouts;
      propagationOptions.rowMajorEnabled = options.rowMajorEnabled;
      propagationOptions.beamWidth = 8;
      propagationOptions.enableL1ShardingLayouts =
          options.memoryLayoutAnalysisEnabled;
      propagationOptions.overrideOutputLayout = options.overrideOutputLayout;
      propagationOptions.overrideConv2dConfig = options.overrideConv2dConfig;
      propagationOptions.enableDecisionTrace = options.enableDecisionTrace;
      propagationOptions.decisionTraceDir = options.decisionTraceDir;
      propagationOptions.enableCompileTimeStats =
          options.enableCompileTimeStats;

      TTNNGreedyL1SpillManagementOptions spillOptions;
      spillOptions.enableDecisionTrace = options.enableDecisionTrace;
      spillOptions.decisionTraceDir = options.decisionTraceDir;

      bool memLayoutEnabled = options.memoryLayoutAnalysisEnabled;
      pm.addPass(createDevicePassesWrapper(
          [propagationOptions, spillOptions, validationOptions,
           memLayoutEnabled](OpPassManager &innerPm) {
            innerPm.addPass(
                mlir::tt::ttnn::createTTNNRowMajorLayoutPropagation());
            innerPm.addPass(
                mlir::tt::ttnn::createTTNNGreedyMemoryLayoutPropagation(
                    propagationOptions));
            if (memLayoutEnabled) {
              innerPm.addPass(mlir::tt::ttnn::createTTNNGreedyL1SpillManagement(
                  spillOptions));
            }
            innerPm.addPass(mlir::createCanonicalizerPass());
            innerPm.addPass(
                mlir::tt::ttnn::createTTNNOperationValidationAndFallback(
                    validationOptions));
            innerPm.addPass(
                mlir::tt::ttnn::createTTNNPrepareConv2dWeightsAndBias());
          },
          wrapperOptions));
    }
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

      uint32_t fallbackAttempts = options.maxFallbackAttempts;
      pm.addPass(createDevicePassesWrapper(
          [fallbackAttempts](OpPassManager &innerPm) {
            TTNNFusingOptions fusingOptions;
            fusingOptions.enableOpConstraints = true;
            fusingOptions.maxFallbackAttempts = fallbackAttempts;
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
  pm.addPass(mlir::createCSEPass());
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
// adding more TTIR ops to it (CPU hoisting passes).
//
void createTTIRToTTNNDevicePipeline(
    OpPassManager &pm, const TTIRToTTNNDevicePipelineOptions &options) {
  // Resolve options controlled by optimization_level.
  options.resolveOptimizationLevelOptions();

  // TODO(dmilinkovic): Remove this once multithreading issues in MetalContext
  // are resolved - tt-metal issue #31041.
  if (options.optimizerPassEnabled) {
    static_cast<PassManager &>(pm).getContext()->disableMultithreading();
  }

  // Mark all public functions without a type assigned to them as Device Forward
  // functions before any other. This provides a consistent mechanism for
  // identifying Device Forward functions downstream.
  pm.addPass(ttcore::createTTCoreMarkFunctionsAsForwardPass());

  // Create device module, if not already present.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());

  // Hoist manually tagged ops to CPU module.
  pm.addPass(ttir::createCPUHoistManuallyTaggedOpsTransform());

  // Device module passes before const-eval CPU hoisting.
  {
    auto &devicePm = pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

    // Element type normalization must run before canonicalization and other
    // transformative passes. Canonicalization patterns assume normalized types
    // (e.g., no i64/f64) and may produce incorrect results otherwise.
    // Element type normalization should be applied only to the ops in the
    // Device Module, since we aren't restricted with element types on CPU.
    devicePm.addPass(ttir::createElementTypeNormalization());

    createTTNNPipelineTTIRPasses(devicePm, options);

    ttir::TTIRQuantDataTypeConversionPassOptions quantOptions;
    quantOptions.targetBitWidth = options.quantBitWidth;
    devicePm.addPass(ttir::createTTIRQuantDataTypeConversionPass(quantOptions));

    devicePm.addPass(mlir::createCSEPass());

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
    devicePm.addPass(createTTNNDecomposition());

    if (options.dramSpaceSavingOptimizationEnabled) {
      devicePm.addPass(createTTNNMemoryManagement());
    }
    createTTNNPipelineWorkaroundPass(devicePm, options);
    // Add weight dtype conversion pass before analysis passes.
    // Analysis passes need to know data formats to decide on shardings.
    // Always added: per-arg "ttcore.weight_dtype" annotations may exist even
    // without a global dtype. The pass is a no-op when no annotations exist
    // and no global dtype is set.
    {
      TTNNWeightDtypeConversionOptions convOpts;
      convOpts.targetDtype = options.experimentalWeightDtype;
      devicePm.addPass(createTTNNWeightDtypeConversion(convOpts));
    }

    // Apply ComputeKernelConfig settings before analysis passes.
    // Create options struct and forward pipeline options.
    TTNNSetComputeKernelConfigOptions setConfigOptions;

    // Forward the OptionalMathFidelity value directly
    setConfigOptions.mathFidelity = options.computeCfgMathFidelity.getValue();
    setConfigOptions.fp32DestAccEn = options.computeCfgFp32DestAccEn.getValue();
    setConfigOptions.maxAccuracy =
        (options.accuracyMode == AccuracyMode::Accuracy);

    if (setConfigOptions.fp32DestAccEn ||
        setConfigOptions.mathFidelity != OptionalMathFidelity::Undefined ||
        setConfigOptions.maxAccuracy) {
      devicePm.addPass(createTTNNSetComputeKernelConfig(setConfigOptions));
    }

    if (options.enableD2MFusing) {
      if (!options.optimizerPassEnabled) {
        llvm::errs()
            << "WARNING: D2M fusing pass only supported with Optimizer "
               "enabled. Automatically enabling Optimizer as a dependency.\n";
      }
      options.optimizerPassEnabled = true;
      devicePm.addPass(tt::ttnn::createTTNND2MFusing());
    }

    // Const-eval pass which should pick up any const-evalable ops created in
    // TTNN workarounds, weight dtype conversion, or any TTNN pass after the
    // first const-eval pass.
    //
    // Without this pass, optimizer might L1-shard certain ops which would get
    // const-evaled in the later const-eval pass.
    if (options.enableConstEval) {
      devicePm.addPass(transforms::createConstEvalHoistTransform());
    }

    createTTNNPipelineAnalysisPasses(devicePm, options);

    if (options.enableD2MFusing) {
      createTTNNPipelineD2MPass(devicePm);
      devicePm.addPass(createTTNNCollaspeD2M());
      devicePm.addPass(createCanonicalizerPass());
    }

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

    // Trace hoisting must run before layout decomposition because it adjusts
    // layouts of function arguments (e.g. moving inputs to system_memory). It
    // is much easier to work at the layout abstraction level than on individual
    // ops after they have been decomposed.
    if (options.enableTrace) {
      devicePm.addPass(tt::ttnn::createTTNNTraceHoistTransform());
    }

    createTTNNPipelineLayoutDecompositionPass(devicePm, options);

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

void createRecoverStructureXLATorchPipeline(
    OpPassManager &pm, const RecoverStructureXLATorchPipelineOptions &options) {
  // Simplify locations to remove nested location information
  //
  // The nested locations appear for parameters, and describe original parameter
  // names, among other info, and will be useful for naming variables in the
  // generated code.
  //
  pm.addPass(createTTNNSimplifyLocsForCodegen());

  // Recover program structure by splitting IR into functions based on source
  // locations
  //
  pm.addPass(createTTNNRecoverStructure());

  // TODO (#6297): This is a temporary workaround - deallocs aren't properly
  // placed in structure recovery pass yet. They are often called before a
  // tensor is last used. A good approach today is to leave deallocs in the IR
  // and (re)move them with an LLM later, by asking it to move deallocs to after
  // last use.
  //
  pm.addPass(createTTNNRemoveDeallocs());
}

// Pipeline which lowers the Device module from TTNN to EmitC dialect.
//
void createTTNNToEmitCDevicePipeline(
    OpPassManager &pm, const TTNNToEmitCDevicePipelineOptions &options) {
  // Unwrapping the device module.
  //
  // TODO(dmilinkovic): Should be removed after support for generating
  // a dynamic library from CPU module is implemented inside EmitC translation
  // pipeline - issue #6100.
  //
  pm.addPass(ttcore::createTTCoreUnwrapDeviceModulePass());

  // These passes operate on TTNN IR inside the (now unwrapped) top-level
  // module.
  //
  pm.addPass(createTTNNAdjustDeallocs());
  if (options.tryRecoverStructure) {
    createRecoverStructureXLATorchPipeline(
        pm, RecoverStructureXLATorchPipelineOptions());
  }

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
  if (options.tryRecoverStructure) {
    createRecoverStructureXLATorchPipeline(
        devicePm, RecoverStructureXLATorchPipelineOptions());
  }

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

    // Optionally create main_for_test wrapper for frontend-driven execution
    // (e.g. PythonModelRunner). This must run after the input generator/loader
    // pass so that _main already exists.
    //
    if (options.createMainForTest) {
      devicePm.addPass(createTTNNCreateMainForTest());
    }
  }

  devicePm.addPass(createTTNNPrepareConstEvalCaching());
  // Optionally run TTNNFileSplit pass.
  if (options.splitFiles) {
    TTNNFileSplitOptions fileSplitOptions;
    fileSplitOptions.target = FileSplitTarget::EmitPy;
    devicePm.addPass(createTTNNFileSplit(fileSplitOptions));
  }

  // Both paths (targetModule and TTNNCreateMainForTest) inject device as an
  // explicit argument into the forward function. Const-eval functions also
  // need device injected, but this can't be done as a separate MLIR pass
  // because load_cached ops verify callee argument count between passes
  // (issue #6746). Setting targetModule=true on the EmitPy pass tells it to
  // handle const-eval device injection inside its runOnOperation, before
  // applyFullConversion.
  //
  ConvertTTNNToEmitPyOptions emitpyOptions;
  emitpyOptions.targetModule =
      options.targetModule || options.createMainForTest;
  devicePm.addPass(createConvertTTNNToEmitPyPass(emitpyOptions));

  devicePm.addPass(createEmitPyConstEvalCachingPass());

  devicePm.addPass(createEmitPyFormExpressionsPass());

  devicePm.addPass(createEmitPyNameVarsPass());
}

// Pipeline which lowers the CPU module from TTIR to EmitPy.
//
// TTIR ops are converted directly to ttir_cpu.<op> calls, bypassing
// TTNN lowering entirely.
//
void createTTIRToEmitPyCPUPipeline(OpPassManager &pm) {
  auto &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

  // Lower TTIR directly to EmitPy (ttir_cpu).
  cpuPm.addPass(createConvertTTIRCPUToEmitPyPass());

  cpuPm.addPass(createEmitPyNameVarsPass());
}

void createTTNNPipelineD2MPass(OpPassManager &pm) {
  // TODO(vtang): pass to strip intermediate layouts.
  pm.addPass(tt::createConvertTTNNToTTIRPass());
  // pm.addPass(strip layouts pass)

  // Can't use createTTIRToTTMetalPipeline because TTCoreWrapDeviceModulePass
  // only works on top-level modules (doesn't run module has a parent op).
  ttmetal::TTIRToTTMetalPipelineOptions ttmetalOptions;
  ttmetalOptions.ttnnMode = true;
  ttmetal::createTTIRToTTMetalFrontendPipeline(pm, ttmetalOptions);
  ttmetal::createTTIRToTTMetalMiddleendPipeline(pm, ttmetalOptions);
  ttmetal::createTTIRToTTMetalBackendPipeline(pm, ttmetalOptions);
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

  // Add module-level Python import statements.
  //
  pm.addPass(createEmitPyAddImportsPass());
}

// Complete pipeline for lowering TTNN to EmitPy.
//
// This pipeline is used when the input is already in TTNN dialect, and assumes
// the CPU module is still in TTIR.
//
// Device module: TTNN -> EmitPy.
// CPU module: TTIR -> TTNN -> EmitPy (with golden functions).
//
void createTTNNToEmitPyPipeline(
    OpPassManager &pm, const TTNNToEmitPyDevicePipelineOptions &options) {
  createTTNNToEmitPyDevicePipeline(pm, options);
  createTTIRToEmitPyCPUPipeline(pm);
  pm.addPass(createEmitPyLinkModulesPass());
  pm.addPass(createEmitPyAddImportsPass());
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

  // TTNN to EmitPy pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::TTNNToEmitPyDevicePipelineOptions>(
      "ttnn-to-emitpy-pipeline", "Pipeline lowering TTNN to EmitPy.",
      mlir::tt::ttnn::createTTNNToEmitPyPipeline);

  // Recover Structure XLA/Torch pipeline.
  //
  mlir::PassPipelineRegistration<
      mlir::tt::ttnn::RecoverStructureXLATorchPipelineOptions>(
      "recover-structure-xla-torch-pipeline",
      "Pipeline to recover structure from TTNN IR for code generation from "
      "XLA/Torch. ",
      mlir::tt::ttnn::createRecoverStructureXLATorchPipeline);

  // TTNN D2M pipeline - runs D2M compilation on TTNN d2m_subgraph ops.
  //
  mlir::PassPipelineRegistration<>(
      "ttnn-through-d2m-pipeline",
      "Pipeline to compile D2M subgraphs inside ttnn.d2m_subgraph ops.",
      [](OpPassManager &pm) {
        auto &devicePm =
            pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();
        mlir::tt::ttnn::createTTNNPipelineD2MPass(devicePm);
      });
}
} // namespace mlir::tt::ttnn
