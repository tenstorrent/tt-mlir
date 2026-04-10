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

void createOptimizationPasses(OpPassManager &pm,
                              const D2MPipelineOptions &options) {
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
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
  pm.addPass(tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(ttir::createTTIRDecomposeMinReduction());
  pm.addPass(ttir::createTTIRRMSNormDecomposition());
  pm.addPass(ttir::createTTIRExplicateTMs());
  pm.addPass(ttir::createTTIREraseInverseOps());
  pm.addPass(ttir::createTTIRMoveReshapeToConstant());
  pm.addPass(ttir::createTTIRFoldConstantReshapeBroadcast());
  pm.addPass(ttir::createTTIRReductionForceKeepDim());
  pm.addPass(ttir::createTTIRRankNormalization());
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
  pm.addPass(d2m::createD2MLowerToLayout());
  pm.addPass(d2m::createD2MMaterializeViewReturns());

  if (options.enableElementwiseFusion) {
    d2m::D2MElementwiseFusionOptions elementwiseFusionOptions;
    elementwiseFusionOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
    pm.addPass(d2m::createD2MElementwiseFusion(elementwiseFusionOptions));
  }
  pm.addPass(mlir::createCanonicalizerPass());
  createTTIRBufferizationPipeline(pm, options);
  pm.addPass(d2m::createD2MAddScratchInputs());

  d2m::D2MGenericApplyInterchangeOptions applyInterchangeOptions;
  {
    applyInterchangeOptions.matmulInterchange =
        llvm::to_vector(options.matmulInterchange);
  }

  pm.addPass(d2m::createD2MGenericApplyInterchange(applyInterchangeOptions));

  // After GenerateOuterLoops, all generic ops are in Affine Blocked form.
  pm.addPass(d2m::createD2MGenerateOuterLoops());

  d2m::D2MAllocateOptions allocateOptions;
  {
    allocateOptions.numStreamBuffers = options.numStreamBuffers;
    allocateOptions.allowL1OutputSpilling = options.allowL1OutputSpilling;
    allocateOptions.streamInsertPolicy = options.streamInsertPolicy;
    allocateOptions.availableL1AddrRange.assign(
        options.availableL1AddrRange.begin(),
        options.availableL1AddrRange.end());
    allocateOptions.testAssumeL1Capacity = options.testAssumel1Capacity;
    allocateOptions.testBufferSizePolicy = options.testBufferSizePolicy;
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
  pm.addPass(d2m::createD2MDecomposeMasking());
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
  d2m::D2MInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  {
    insertDstRegisterAccessOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
    insertDstRegisterAccessOptions.enableL1Acc = options.enableL1Acc;
  }
  pm.addPass(
      d2m::createD2MInsertDstRegisterAccess(insertDstRegisterAccessOptions));
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

  // Frontend of DMA lowering pipeline; lower abstract
  // remote loads and stores to explicit CB form split the
  // unified thread into separate compute and datamovement
  // threads.
  pm.addPass(d2m::createD2MHoistCBAllocs());
  pm.addPass(d2m::createD2MConvertLocalLoadStoreOpsToAliasedCBs());
  pm.addPass(d2m::createD2MLowerLoadStoreOpsToExplicitCBForm());
  pm.addPass(d2m::createD2MSplitUnifiedThread());

  // Backend of DMA lowering pipeline; generic ops are now
  // in split compute-dma form. All remote loads and stores
  // are lowered to concrete dma ops (dma_read, dma_write,
  // dma_wait, etc.) implementable by D2MToTTKernel lowering
  // pass.
  pm.addPass(d2m::createD2MPreallocateMcastSemaphores());
  pm.addPass(d2m::createD2MScheduleDMA());
  pm.addPass(d2m::createD2MLowerLoadStoreOpsToDMA());
  pm.addPass(d2m::createD2MOptimizeDMA());
  pm.addPass(d2m::createD2MExpandDMAReadCompositeView());
  pm.addPass(d2m::createD2MLowerDMAToFullyIndexedForm());

  createOptimizationPasses(pm, options);

  pm.addPass(d2m::createD2MGenericRegionsToFuncs());
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

void createD2MToTTKernelPipeline(OpPassManager &pm,
                                 const D2MPipelineOptions &options) {
  d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
  { D2MToTTKernelOptions.ttnnMode = options.ttnnMode; }
  pm.addPass(tt::createConvertD2MToTTKernelPass(D2MToTTKernelOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm, options);
  pm.addPass(ttkernel::createTTKernelHoistInits());
  // Insert DeviceZone scopes around selected ttkernel ops before EmitC
  // lowering.
  if (options.insertProfilerTraces) {
    pm.addPass(ttkernel::createTTKernelInsertDeviceZoneScopes());
  }
  pm.addPass(createConvertTTKernelToEmitC());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::emitc::createFormExpressionsPass());
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
  createD2MToTTKernelPipeline(devicePm, options);
  if (options.ttnnMode) {
    createD2MToTTNNPipeline(devicePm, options);
  } else {
    createD2MToTTMetalPipeline(devicePm, options);
  }

  // Run lowering to LLVM pass.
  ttir::TTIRToLLVMCPUPipelineOptions ttirToCPUOptions;
  ttir::createTTIRToLLVMCPUPipeline(pm, ttirToCPUOptions);
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
