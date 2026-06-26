// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Pipelines/D2MPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

// translates top level flags into specific disable/enable patterns for
// canonicalizer pass
std::unique_ptr<Pass>
createCanonicalizerPassWithOptions(const D2MPipelineOptions &options) {
  llvm::SmallVector<std::string, 2> disabledPatterns;
  if (options.disableToLayoutFolding) {
    disabledPatterns.push_back("ttir.ToLayoutFoldRedundantPattern");
    disabledPatterns.push_back("d2m.ToLayoutFoldRedundantPattern");
  }
  return mlir::createCanonicalizerPass({}, disabledPatterns);
}

void createTTIRBufferizationPipeline(OpPassManager &pm,
                                     const D2MPipelineOptions &options) {
  if (options.ttnnMode) {
    bufferization::OneShotBufferizePassOptions bufferizePassOptions;
    bufferizePassOptions.allowUnknownOps = true;
    bufferizePassOptions.bufferizeFunctionBoundaries = false;
    bufferizePassOptions.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    bufferizePassOptions.unknownTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizePassOptions));
  } else {
    // Use custom bufferization pass with MetalLayoutAttr support
    pm.addPass(ttcore::createTTCoreOneShotBufferizePass());
  }
  // TODO(#2246)
  // bufferization::BufferDeallocationPipelineOptions
  // bufferDeallocationOptions;
  // mlir::bufferization::buildBufferDeallocationPipeline(
  //    pm, bufferDeallocationOptions);
}

void addFunctionOptimizationPasses(OpPassManager &funcPm) {
  funcPm.addPass(mlir::createLoopInvariantCodeMotionPass());
  funcPm.addPass(mlir::createSCCPPass());
  funcPm.addPass(mlir::createCSEPass());
  funcPm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  funcPm.addPass(mlir::createLoopInvariantCodeMotionPass());
}

void createD2MFrontendPipeline(OpPassManager &pm,
                               const D2MPipelineOptions &options) {
  // Create multi-device tensor annotation for graph with mesh.
  pm.addPass(ttir::createTTIRMultiDeviceTensorAnnotation());
  ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
    registerDeviceOptions.meshTopology = llvm::to_vector(options.meshTopology);
  }
  pm.addPass(ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions));
  pm.addPass(ttir::createPredicateTypeAlignment());
  pm.addPass(ttir::createElementTypeNormalization());
  pm.addPass(ttir::createTTIRDecomposeComposites());
  pm.addPass(tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(ttir::createTTIRExplicateTMs());
  pm.addPass(ttir::createTTIREraseInverseOps());
  pm.addPass(ttir::createTTIRMoveReshapeToConstant());
  pm.addPass(ttir::createTTIRFoldConstantReshapeBroadcast());
  pm.addPass(ttir::createTTIRReductionForceKeepDim());
  pm.addPass(ttir::createTTIRDecomposeComplexReshape());
  pm.addPass(ttir::createTTIRImplicitBroadcastFold());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  if (!options.globalDataFormatTarget.empty()) {
    ttir::TTIRGlobalDataFormatConversionOptions globalFormatOptions;
    { globalFormatOptions.targetFormat = options.globalDataFormatTarget; }
    pm.addPass(ttir::createTTIRGlobalDataFormatConversion(globalFormatOptions));
  }
  pm.addPass(ttir::createTTIRDecomposeComplexPermute());
  tt::TTIRToD2MOptions toD2MOptions;
  {
    toD2MOptions.defaultInputMemSpace = options.defaultInputMemSpace;
    toD2MOptions.defaultOutputMemSpace = options.defaultOutputMemSpace;
    toD2MOptions.ttnnMode = options.ttnnMode;
    toD2MOptions.collapseTensorsTo2D = options.collapseTensors;
    toD2MOptions.enableMulticastInference = options.enableMulticastInference;
  }
  pm.addPass(tt::createTTIRToD2MPass(toD2MOptions));
  pm.addPass(d2m::createD2MScalarizeConstTensors());
  d2m::D2MGridSelectionOptions gridOptOptions;
  {
    gridOptOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
    gridOptOptions.ttnnMode = options.ttnnMode;
  }
  pm.addPass(d2m::createD2MMaterializeViewReturns());
  pm.addPass(d2m::createD2MGridSelection(gridOptOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(d2m::createD2MOptimizeMasks());
  pm.addPass(d2m::createD2MLowerToLayout());
  pm.addPass(d2m::createD2MMaterializeViewReturns());

  if (options.enableElementwiseFusion || options.enableEltwiseReductionFusion) {
    d2m::D2MGenericFusionOptions fusionOptions;
    fusionOptions.enableEltwiseReductionFusion =
        options.enableEltwiseReductionFusion;
    pm.addPass(d2m::createD2MGenericFusion(fusionOptions));
  }
  pm.addPass(mlir::createCanonicalizerPass());
  createTTIRBufferizationPipeline(pm, options);
  pm.addPass(d2m::createD2MInsertScratchBuffers());

  d2m::D2MGenericApplyInterchangeOptions applyInterchangeOptions;
  {
    applyInterchangeOptions.matmulInterchange =
        llvm::to_vector(options.matmulInterchange);
  }

  pm.addPass(d2m::createD2MGenericApplyInterchange(applyInterchangeOptions));

  // After GenerateOuterLoops, all generic ops are in Affine Blocked form.
  pm.addPass(d2m::createD2MGenerateOuterLoops());
  d2m::D2MDecomposeMaskingOptions decomposeMaskingOptions;
  { decomposeMaskingOptions.numStreamBuffers = options.numStreamBuffers; }
  pm.addPass(d2m::createD2MDecomposeMasking(decomposeMaskingOptions));

  d2m::D2MReblockGenericsOptions reblockGenericsOptions;
  {
    reblockGenericsOptions.numStreamBuffers = options.numStreamBuffers;
    reblockGenericsOptions.testBufferSizePolicy = options.testBufferSizePolicy;
  }
  pm.addPass(d2m::createD2MReblockGenerics(reblockGenericsOptions));
  pm.addPass(d2m::createD2MMaterializeViewReturns());

  // Run right before allocate to mark synchronized buffers
  d2m::D2MMarkSynchronizedBuffersOptions markSyncBuffersOptions;
  { markSyncBuffersOptions.numStreamBuffers = options.numStreamBuffers; }
  pm.addPass(d2m::createD2MMarkSynchronizedBuffers(markSyncBuffersOptions));

  d2m::D2MAllocateOptions allocateOptions;
  {
    allocateOptions.numStreamBuffers = options.numStreamBuffers;
    allocateOptions.allowL1OutputSpilling = options.allowL1OutputSpilling;
    allocateOptions.streamInsertPolicy = options.streamInsertPolicy;
    allocateOptions.availableL1AddrRange.assign(
        options.availableL1AddrRange.begin(),
        options.availableL1AddrRange.end());
    allocateOptions.forceSpillToDramIfLegal = options.forceSpillToDramIfLegal;
    allocateOptions.testAssumeL1Capacity = options.testAssumel1Capacity;
  }
  pm.addPass(d2m::createD2MAllocate(allocateOptions));
  pm.addPass(d2m::createD2MLowerMulticastLoads());

  // After LowerToExplicitForm, all generic op are in Explicit Datamovement
  // form.
  pm.addPass(d2m::createD2MLowerToExplicitForm());
  pm.addPass(createCanonicalizerPassWithOptions(options));
}

void createD2MBackendPipeline(OpPassManager &pm,
                              const D2MPipelineOptions &options) {
  pm.addPass(d2m::createD2MDecomposeArange());

  d2m::D2MGenericTileComputeLoopsOptions tileComputeLoopsOptions;
  {
    tileComputeLoopsOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MGenericTileComputeLoops(tileComputeLoopsOptions));
  d2m::D2MLinalgToAffineOptions linalgToAffineOptions;
  { linalgToAffineOptions.markRootLoops = true; }
  pm.addPass(d2m::createD2MLinalgToAffine(linalgToAffineOptions));

  d2m::D2MOpSchedulerOptions opSchedulerOptions;
  {
    // TODO(mbagherbeikTT)
    // Has to be hard enabled for now until DST allocation is made fully
    // consistent with elementwise fusion
    opSchedulerOptions.enableOpScheduler = true; /* options.enableOpScheduler */
    ;
  }
  pm.addPass(d2m::createD2MOpScheduler(opSchedulerOptions));
  pm.addPass(d2m::createD2MInsertSpillAndScratch());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(d2m::createD2MLowerScratchAllocate());
  pm.addPass(mlir::createCanonicalizerPass());
  d2m::D2MInsertDstRegisterAccessUnscheduledOptions unschedDstOpts;
  {
    unschedDstOpts.maxDstPhysicalSizeTiles = options.maxDstPhysicalSizeTiles;
    unschedDstOpts.disableL1Acc = options.disableL1Acc;
  }
  pm.addPass(d2m::createD2MInsertDstRegisterAccessUnscheduled(unschedDstOpts));
  d2m::D2MInsertDstRegisterAccessScheduledOptions schedDstOpts;
  {
    schedDstOpts.maxDstPhysicalSizeTiles = options.maxDstPhysicalSizeTiles;
    schedDstOpts.disableL1Acc = options.disableL1Acc;
  }
  pm.addPass(d2m::createD2MInsertDstRegisterAccessScheduled(schedDstOpts));
  d2m::D2MInsertTileMatmulBlockOptions insertTileMatmulBlockOptions;
  { insertTileMatmulBlockOptions.useTileMatmul = options.useTileMatmul; }
  pm.addPass(d2m::createD2MInsertTileMatmulBlock(insertTileMatmulBlockOptions));

  pm.addPass(d2m::createD2MSFPUTileLoopFission());
  pm.addPass(mlir::createCanonicalizerPass());

  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(affine::createAffineLoopInvariantCodeMotionPass());

  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(d2m::createD2MGenericLinearizeMemref());
  // GenericLinearizeMemref generates affine apply ops that must be lowered here
  pm.addPass(mlir::createLowerAffinePass());

  // Frontend of DMA lowering pipeline; insert compute-side CB
  // sync ops and split the unified thread into separate compute
  // and datamovement threads.
  pm.addPass(d2m::createD2MHoistCBAllocs());
  pm.addPass(d2m::createD2MSplitUnifiedThread());

  // Backend of DMA lowering pipeline; generic ops are now
  // in split compute-dma form. All remote loads and stores
  // are lowered to concrete dma ops (dma_read, dma_write,
  // dma_wait, etc.) implementable by D2MToTTKernel lowering
  // pass.
  pm.addPass(d2m::createD2MPreallocateMcastSemaphores());
  pm.addPass(d2m::createD2MScheduleDMA());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(d2m::createD2MLowerLoadStoreOpsToDMA());
  pm.addPass(d2m::createD2MOptimizeDMA());
  pm.addPass(d2m::createD2MExpandDMAReadCompositeView());
  pm.addPass(d2m::createD2MLowerDMAToFullyIndexedForm());

  // Normalize thread argument access by inserting d2m.get_arg ops for any
  // remaining additional arguments and setting resolution_stage on
  // d2m.get_cb/d2m.get_arg so the D2MToTTKernel lowering pass can uniformly
  // treat all arguments.
  pm.addPass(d2m::createD2MNormalizeThreadArgs());

  pm.addPass(d2m::createD2MGenericRegionsToFuncs());
  OpPassManager &postGenericRegionsFuncPm = pm.nest<func::FuncOp>();
  postGenericRegionsFuncPm.addPass(createCanonicalizerPassWithOptions(options));
  postGenericRegionsFuncPm.addPass(mlir::createLowerAffinePass());
  addFunctionOptimizationPasses(postGenericRegionsFuncPm);
}

void createD2MToTTMetalPipeline(OpPassManager &pm,
                                const D2MPipelineOptions &options) {
  d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
  { d2mToTTMetalOptions.mathFidelity = options.mathFidelity; }
  pm.addPass(tt::createConvertD2MToTTMetalPass(d2mToTTMetalOptions));
}

void createD2MToTTNNPipeline(OpPassManager &pm,
                             const D2MPipelineOptions &options) {
  d2m::ConvertD2MToTTNNOptions d2mToTTNNOptions;
  { d2mToTTNNOptions.mathFidelity = options.mathFidelity; }
  pm.addPass(tt::createConvertD2MToTTNNPass(d2mToTTNNOptions));
}

// Adds the D2M→TTKernel conversion and TTKernel optimisation passes, but
// intentionally stops short of EmitC lowering. Callers that need dispatch-level
// D2M passes (e.g. ConvertD2MToTTMetalPass) to inspect TTKernel ops must run
// those passes between here and createD2MEmitCPipeline(). TTKernelHoistInits
// and TTKernelInsertDeviceZoneScopes are intentionally excluded: they must run
// AFTER dispatch-level conversion passes so those passes see the TTKernel op
// structure intact (e.g. TypecastTileOp locality for BFP8 unpack-mode
// selection). Callers are responsible for adding them at the right point.
void createD2MToTTKernelPreEmitCPipeline(OpPassManager &pm,
                                         const D2MPipelineOptions &options) {
  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
  {
    D2MToTTKernelOptions.ttnnMode = options.ttnnMode;
    D2MToTTKernelOptions.forceCompileTimeArgs = options.forceCompileTimeArgs;
  }
  funcPm.addPass(tt::createConvertD2MToTTKernelPass(D2MToTTKernelOptions));
  funcPm.addPass(createCanonicalizerPassWithOptions(options));
  funcPm.addPass(ttkernel::createTTKernelControlDstSection());
  funcPm.addPass(createCanonicalizerPassWithOptions(options));
  addFunctionOptimizationPasses(funcPm);
}

void createD2MEmitCPipeline(OpPassManager &pm,
                            const D2MPipelineOptions &options) {
  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(createConvertTTKernelToEmitC());
  funcPm.addPass(createCanonicalizerPassWithOptions(options));
  funcPm.addPass(createRemoveDeadEmitCExpressionsPass());
  funcPm.addPass(mlir::emitc::createFormExpressionsPass());
}

void createD2MToTTKernelPipeline(OpPassManager &pm,
                                 const D2MPipelineOptions &options) {
  createD2MToTTKernelPreEmitCPipeline(pm, options);
  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(ttkernel::createTTKernelHoistInits());
  funcPm.addPass(ttkernel::createTTKernelDedupInits());
  if (options.insertProfilerTraces) {
    ttkernel::TTKernelInsertDeviceZoneScopesOptions passOpts;
    if (options.profilerTraits.empty()) {
      passOpts.traitNames.push_back("device-zone");
    } else {
      for (const std::string &n : options.profilerTraits) {
        passOpts.traitNames.push_back(n);
      }
    }
    funcPm.addPass(ttkernel::createTTKernelInsertDeviceZoneScopes(passOpts));
  }
  createD2MEmitCPipeline(pm, options);
}

void createTTIRToTTMetalPipeline(OpPassManager &pm,
                                 const D2MPipelineOptions &options) {
  // Mark all public functions without a type assigned to them as Device Forward
  // functions before any other. This provides a consistent mechanism for
  // identifying Device Forward functions downstream.
  pm.addPass(ttcore::createTTCoreMarkFunctionsAsForwardPass());

  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());

  // Hoist manually-tagged ops to CPU module.
  pm.addPass(ttir::createCPUHoistManuallyTaggedOpsTransform());

  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

  // Run D2M pipelines on IR in DeviceModule.
  createD2MFrontendPipeline(devicePm, options);
  createD2MBackendPipeline(devicePm, options);
  // Stop before EmitC: ConvertD2MToTTMetalPass inspects TTKernel ops (e.g.
  // TypecastTileOp) to configure hardware unpack modes, so the dispatch-level
  // D2M→TTMetal/TTNN conversion must see the TTKernel ops before they are
  // lowered away by EmitC.
  createD2MToTTKernelPreEmitCPipeline(devicePm, options);
  if (options.ttnnMode) {
    createD2MToTTNNPipeline(devicePm, options);
  } else {
    createD2MToTTMetalPipeline(devicePm, options);
  }
  // Hoist TTKernel init ops and insert profiler traces after the dispatch-level
  // conversion so ConvertD2MToTTMetalPass sees TTKernel ops in their original
  // loop structure (e.g. TypecastTileOp locality for BFP8 unpack-mode
  // selection).
  OpPassManager &funcPm = devicePm.nest<func::FuncOp>();
  funcPm.addPass(ttkernel::createTTKernelHoistInits());
  funcPm.addPass(ttkernel::createTTKernelDedupInits());
  if (options.insertProfilerTraces) {
    ttkernel::TTKernelInsertDeviceZoneScopesOptions passOpts;
    if (options.profilerTraits.empty()) {
      passOpts.traitNames.push_back("device-zone");
    } else {
      for (const std::string &n : options.profilerTraits) {
        passOpts.traitNames.push_back(n);
      }
    }
    funcPm.addPass(ttkernel::createTTKernelInsertDeviceZoneScopes(passOpts));
  }
  createD2MEmitCPipeline(devicePm, options);

  // Run pipeline for lowering the CPU module to LLVM.
  OpPassManager &cpuPm = pm.nest<ttcore::CPUModuleOp>().nest<mlir::ModuleOp>();

  ttir::SHLOAndTTIRToLLVMPipelineOptions cpuOptions;
  ttir::createSHLOAndTTIRToLLVMPipeline(cpuPm, cpuOptions);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerD2MPipelines() {
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "ttir-to-ttmetal-pipeline", "Pipeline lowering ttir to ttmetal.",
      tt::ttmetal::createTTIRToTTMetalPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-fe-pipeline", "D2M frontend: TTIR to D2M explicit form.",
      tt::ttmetal::createD2MFrontendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-be-pipeline", "D2M backend: D2M explicit form to fully lowered.",
      tt::ttmetal::createD2MBackendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-to-ttkernel-pipeline", "Convert D2M to TTKernel + EmitC.",
      tt::ttmetal::createD2MToTTKernelPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-to-ttkernel-pre-emitc-pipeline",
      "D2M -> TTKernel passes, stopping short of EmitC so dispatch-level "
      "conversion passes (e.g. ConvertD2MToTTMetalPass) can still inspect "
      "TTKernel ops (e.g. TypecastTileOp for fp32 unpack-mode selection).",
      tt::ttmetal::createD2MToTTKernelPreEmitCPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-emitc-pipeline",
      "Lower TTKernel ops to EmitC. Pair with d2m-to-ttkernel-pre-emitc-"
      "pipeline (plus dispatch-level conversion + ttkernel-hoist-inits "
      "in between) to reproduce the full d2m-to-ttkernel-pipeline.",
      tt::ttmetal::createD2MEmitCPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-to-ttmetal-pipeline", "Convert D2M to TTMetal.",
      tt::ttmetal::createD2MToTTMetalPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "d2m-to-ttnn-pipeline", "Convert D2M to TTNN.",
      tt::ttmetal::createD2MToTTNNPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::D2MPipelineOptions>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
