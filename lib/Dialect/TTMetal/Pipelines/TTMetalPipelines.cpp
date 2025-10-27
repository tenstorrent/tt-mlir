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

void createD2MBufferizationPipeline(OpPassManager &pm) {
  bufferization::OneShotBufferizePassOptions bufferizePassOptions;
  bufferizePassOptions.allowUnknownOps = false;
  bufferizePassOptions.bufferizeFunctionBoundaries = true;
  bufferizePassOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizePassOptions.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizePassOptions));
  // TODO(#2246)
  // bufferization::BufferDeallocationPipelineOptions
  // bufferDeallocationOptions;
  // mlir::bufferization::buildBufferDeallocationPipeline(
  //    pm, bufferDeallocationOptions);
}

void createTTNND2MBufferizationPipeline(OpPassManager &pm) {
  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.allowUnknownOps = true;
  bufferizeOptions.bufferizeFunctionBoundaries = false;
  bufferizeOptions.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  bufferizeOptions.unknownTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizeOptions));
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
  pm.addPass(createCanonicalizerPassWithOptions(options));
  tt::TTIRToD2MOptions toD2MOptions;
  {
    toD2MOptions.defaultInputMemSpace = options.defaultInputMemSpace;
    toD2MOptions.defaultOutputMemSpace = options.defaultOutputMemSpace;
    toD2MOptions.collapseTensorsTo2D = options.collapseTensors;
  }
  pm.addPass(tt::createTTIRToD2MPass(toD2MOptions));
  d2m::D2MGridSelectionOptions gridOptOptions;
  {
    gridOptOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
  }
  pm.addPass(d2m::createD2MGridSelection(gridOptOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(d2m::createD2MLowerToLayout());
}

void createTTIRToTTMetalMiddleendPipelineImpl(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options,
    bool ttnnMode = false) {
  d2m::D2MElementwiseFusionOptions elementwiseFusionOptions;
  {
    elementwiseFusionOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(d2m::createD2MElementwiseFusion(elementwiseFusionOptions));
  pm.addPass(createLinalgElementwiseOpFusionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (ttnnMode) {
    createTTNND2MBufferizationPipeline(pm);
    d2m::D2MInsertStreamsOptions insertStreamsOptions;
    {
      insertStreamsOptions.numStreamBuffers = options.numStreamBuffers;
      insertStreamsOptions.allowL1OutputSpilling =
          options.allowL1OutputSpilling;
    }
    pm.addPass(d2m::createD2MInsertStreams(insertStreamsOptions));
  } else {
    createD2MBufferizationPipeline(pm);
    d2m::D2MAllocateOptions allocateOptions;
    {
      allocateOptions.numStreamBuffers = options.numStreamBuffers;
      allocateOptions.allowL1OutputSpilling = options.allowL1OutputSpilling;
    }
    pm.addPass(d2m::createD2MAllocate(allocateOptions));
  }

  pm.addPass(createCanonicalizerPassWithOptions(options));
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
  d2m::D2MInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  {
    insertDstRegisterAccessOptions.useTileMatmul = options.useTileMatmul;
    insertDstRegisterAccessOptions.maxDstPhysicalSizeTiles =
        options.maxDstPhysicalSizeTiles;
  }
  pm.addPass(
      d2m::createD2MInsertDstRegisterAccess(insertDstRegisterAccessOptions));

  pm.addPass(d2m::createD2MSFPUTileLoopFission());
  pm.addPass(mlir::createCanonicalizerPass());

  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(affine::createAffineLoopInvariantCodeMotionPass());

  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(d2m::createD2MGenericLinearizeMemref());
  pm.addPass(d2m::createD2MGenericGenerateDatamovement());
  pm.addPass(d2m::createD2MGenericLowerDMAs());
  pm.addPass(d2m::createD2MGenericHWThreadSelection());
  pm.addPass(d2m::createD2MGenericGenerateLoops());
  createOptimizationPasses(pm, options);
  pm.addPass(d2m::createD2MGenericRegionsToFuncs());
}

void createTTIRToTTMetalMiddleendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  createTTIRToTTMetalMiddleendPipelineImpl(pm, options);
}

void createTTIRToTTMetalBackendPipelineImpl(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options,
    bool ttnnMode = false) {
  pm.addPass(tt::createConvertD2MToTTKernelPass());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm, options);
  if (ttnnMode) {
    // TODO(#5075): set MathFidelity of ttnn generic op.
    pm.addPass(tt::createConvertD2MToTTNNPass());
  } else {
    d2m::ConvertD2MToTTMetalOptions d2mToTTMetalOptions;
    { d2mToTTMetalOptions.mathFidelity = options.mathFidelity; }
    pm.addPass(tt::createConvertD2MToTTMetalPass(d2mToTTMetalOptions));
  }
  // Insert DeviceZone scopes around selected ttkernel ops before EmitC
  // lowering.
  if (options.insertProfilerTraces) {
    pm.addPass(ttkernel::createTTKernelInsertDeviceZoneScopes());
  }
  pm.addPass(createConvertTTKernelToEmitC());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(mlir::emitc::createFormExpressionsPass());
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  createTTIRToTTMetalBackendPipelineImpl(pm, options);
}

void createTTIRToTTMetalPipeline(OpPassManager &pm,
                                 const TTIRToTTMetalPipelineOptions &options) {
  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular ttir to ttmetal pipelines on IR in DeviceModule.
  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();
  createTTIRToTTMetalFrontendPipeline(devicePm, options);
  createTTIRToTTMetalMiddleendPipeline(devicePm, options);
  createTTIRToTTMetalBackendPipeline(devicePm, options);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  ttir::LinalgToLLVMPipelineOptions linalgToLLVMOptions;
  ttir::createTTIRToCPUPipeline(pm, linalgToLLVMOptions);
}

void createTTIRToTTNNGenericPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  // Create DeviceModule to wrap all ops.
  pm.addPass(ttcore::createTTCoreWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular ttir to ttmetal pipelines on IR in DeviceModule.
  OpPassManager &devicePm =
      pm.nest<ttcore::DeviceModuleOp>().nest<mlir::ModuleOp>();

  createTTIRToTTMetalFrontendPipeline(devicePm, options);
  createTTIRToTTMetalMiddleendPipelineImpl(devicePm, options, true);
  createTTIRToTTMetalBackendPipelineImpl(devicePm, options, true);

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  ttir::LinalgToLLVMPipelineOptions linalgToLLVMOptions;
  ttir::createTTIRToCPUPipeline(pm, linalgToLLVMOptions);
}

void createTTNNToTTMetalBackendPipeline(
    OpPassManager &pm,
    const tt::ttmetal::TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(tt::createConvertTTNNToTTIRPass());
  createTTIRToTTNNGenericPipeline(pm, options);
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
      "ttir-to-ttnn-generic-pipeline",
      "Pipeline lowering TTIR to ttnn.generic + kernels.",
      tt::ttmetal::createTTIRToTTNNGenericPipeline);
  mlir::PassPipelineRegistration<tt::ttmetal::TTIRToTTMetalPipelineOptions>(
      "ttnn-to-ttmetal-backend-pipeline",
      "Pipeline lowering ttnn to ttnn.generic + kernels.",
      tt::ttmetal::createTTNNToTTMetalBackendPipeline);
  mlir::PassPipelineRegistration<>(
      "d2m-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createD2MBufferizationPipeline);
  mlir::PassPipelineRegistration<>(
      "ttnn-d2m-bufferization-pipeline",
      "Pipeline bufferizing ttnn ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTNND2MBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
