// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

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
std::unique_ptr<Pass> createCanonicalizerPassWithOptions(
    const TTIRToTTMetalPipelineOptions &options) {
  llvm::SmallVector<std::string, 2> disabledPatterns;
  if (options.disableToLayoutFolding) {
    disabledPatterns.push_back("ttir.ToLayoutFoldRedundantPattern");
    disabledPatterns.push_back("d2m.ToLayoutFoldRedundantPattern");
  }
  return mlir::createCanonicalizerPass({}, disabledPatterns);
}

void createTTIRBufferizationPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
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
                              const TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
}

void createTTIRToTTMetalFrontendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  // Create multi-device tensor annotation for graph with mesh.
  pm.addPass(ttir::createTTIRMultiDeviceTensorAnnotation());
  ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions));
  pm.addPass(tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(ttir::createTTIRMoveReshapeToConstant());
  pm.addPass(ttir::createTTIRFoldConstantReshapeBroadcast());
  pm.addPass(d2m::createD2MRankNormalization());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  if (!options.globalDataFormatTarget.empty()) {
    d2m::D2MGlobalDataFormatConversionOptions globalFormatOptions;
    { globalFormatOptions.targetFormat = options.globalDataFormatTarget; }
    pm.addPass(d2m::createD2MGlobalDataFormatConversion(globalFormatOptions));
  }
  pm.addPass(d2m::createD2MDecomposeComplexPermute());
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
  }
  pm.addPass(d2m::createD2MMaterializeViewReturns());
  pm.addPass(d2m::createD2MGridSelection(gridOptOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(d2m::createD2MLowerToLayout());
  pm.addPass(d2m::createD2MMaterializeViewReturns());
}

void createTTIRToTTMetalMiddleendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  d2m::D2MElementwiseFusionOptions elementwiseFusionOptions;
  {
    elementwiseFusionOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MElementwiseFusion(elementwiseFusionOptions));

  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  createTTIRBufferizationPipeline(pm, options);

  pm.addPass(d2m::createD2MAddScratchInputs());

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
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(d2m::createD2MDecomposeMasking());
  pm.addPass(d2m::createD2MDecomposeArange());

  d2m::D2MGenericApplyInterchangeOptions applyInterchangeOptions;
  {
    applyInterchangeOptions.matmulInterchange =
        llvm::to_vector(options.matmulInterchange);
  }
  pm.addPass(d2m::createD2MGenericApplyInterchange(applyInterchangeOptions));
  d2m::D2MGenericTileComputeLoopsOptions tileComputeLoopsOptions;
  {
    tileComputeLoopsOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MGenericTileComputeLoops(tileComputeLoopsOptions));
  d2m::D2MLinalgToAffineOptions linalgToAffineOptions;
  {
    linalgToAffineOptions.useTileMatmul = options.useTileMatmul;
    linalgToAffineOptions.markRootLoops = true;
  }
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

  pm.addPass(d2m::createD2MLowerScratchAllocate());

  d2m::D2MInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  {
    insertDstRegisterAccessOptions.useTileMatmul = options.useTileMatmul;
    insertDstRegisterAccessOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
    insertDstRegisterAccessOptions.enableL1Acc =
        options.enableL1Acc;
  }
  pm.addPass(
      d2m::createD2MInsertDstRegisterAccess(insertDstRegisterAccessOptions));

  pm.addPass(d2m::createD2MLowerMulticastLoads());
  pm.addPass(d2m::createD2MGenerateOuterLoops());

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

  pm.addPass(createCanonicalizerPassWithOptions(options));
  createOptimizationPasses(pm, options);

  pm.addPass(d2m::createD2MGenericRegionsToFuncs());
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  d2m::ConvertD2MToTTKernelOptions D2MToTTKernelOptions;
  { D2MToTTKernelOptions.ttnnMode = options.ttnnMode; }
  pm.addPass(tt::createConvertD2MToTTKernelPass(D2MToTTKernelOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm, options);
  if (options.ttnnMode) {
    d2m::ConvertD2MToTTNNOptions d2mToTTNNOptions;
    { d2mToTTNNOptions.mathFidelity = options.mathFidelity; }
    pm.addPass(tt::createConvertD2MToTTNNPass(d2mToTTNNOptions));
  } else {
    d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
    { d2mToTTMetalOptions.mathFidelity = options.mathFidelity; }
    pm.addPass(tt::createConvertD2MToTTMetalPass(d2mToTTMetalOptions));
  }
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
                                 const TTIRToTTMetalPipelineOptions &options) {
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

  // Run regular ttir to ttmetal pipelines on IR in DeviceModule.
  createTTIRToTTMetalFrontendPipeline(devicePm, options);
  createTTIRToTTMetalMiddleendPipeline(devicePm, options);
  createTTIRToTTMetalBackendPipeline(devicePm, options);

  // Run lowering to LLVM pass.
  ttir::TTIRToLLVMCPUPipelineOptions ttirToCPUOptions;
  ttir::createTTIRToLLVMCPUPipeline(pm, ttirToCPUOptions);
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTMetalPipelines() {
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-pipeline", "Pipeline lowering ttir to ttmetal.",
      tt::ttmetal::createTTIRToTTMetalPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-fe-pipeline", "Frontend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalFrontendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-me-pipeline", "Middleend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalMiddleendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-to-ttmetal-be-pipeline", "Backend lowering passes.",
      tt::ttmetal::createTTIRToTTMetalBackendPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
