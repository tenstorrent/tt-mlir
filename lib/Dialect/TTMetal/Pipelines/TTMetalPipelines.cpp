// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
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
  }
  return mlir::createCanonicalizerPass({}, disabledPatterns);
}

void createTTIRBufferizationPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  bufferization::OneShotBufferizePassOptions bufferizePassOptions;
  if (options.ttnnMode) {
    bufferizePassOptions.allowUnknownOps = true;
    bufferizePassOptions.bufferizeFunctionBoundaries = false;
  } else {
    bufferizePassOptions.allowUnknownOps = false;
    bufferizePassOptions.bufferizeFunctionBoundaries = true;
  }
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
  ttcore::TTCoreRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(ttcore::createTTCoreRegisterDevicePass(registerDeviceOptions));
  pm.addPass(tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  if (options.ttnnMode) {
    pm.addPass(ttir::createTTNNLayoutTranslation());
  }

  ttir::TTIRToTTIRGenericOptions toTTIRGenericOptions;
  {
    toTTIRGenericOptions.defaultInputMemSpace = options.defaultInputMemSpace;
    toTTIRGenericOptions.defaultOutputMemSpace = options.defaultOutputMemSpace;
    toTTIRGenericOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
  }
  pm.addPass(tt::createTTIRToTTIRGenericPass(toTTIRGenericOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttir::createTTIRLowerToLayout());
}

void createTTIRToTTMetalMiddleendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  createTTIRBufferizationPipeline(pm, options);
  ttir::TTIRAllocateOptions allocateOptions;
  {
    allocateOptions.numStreamBuffers = options.numStreamBuffers;
    allocateOptions.alwaysStream = options.ttnnMode;
  }
  pm.addPass(ttir::createTTIRAllocate(allocateOptions));
  pm.addPass(createCanonicalizerPassWithOptions(options));
  ttir::TTIRGenericApplyInterchangeOptions applyInterchangeOptions;
  {
    applyInterchangeOptions.matmulInterchange =
        llvm::to_vector(options.matmulInterchange);
  }
  pm.addPass(ttir::createTTIRGenericApplyInterchange(applyInterchangeOptions));
  ttir::TTIRGenericTileComputeLoopsOptions tileComputeLoopsOptions;
  {
    tileComputeLoopsOptions.maxDstRegisterSizeTiles =
        options.maxDstRegisterSizeTiles;
  }
  pm.addPass(ttir::createTTIRGenericTileComputeLoops(tileComputeLoopsOptions));
  ttir::TTIRInsertDstRegisterAccessOptions insertDstRegisterAccessOptions;
  { insertDstRegisterAccessOptions.useTileMatmul = options.useTileMatmul; }
  pm.addPass(
      ttir::createTTIRInsertDstRegisterAccess(insertDstRegisterAccessOptions));

  OpPassManager &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(affine::createAffineLoopInvariantCodeMotionPass());

  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(ttir::createTTIRGenericLinearizeMemref());
  pm.addPass(ttir::createTTIRGenericGenerateDatamovement());
  pm.addPass(ttir::createTTIRGenericHWThreadSelection());
  pm.addPass(ttir::createTTIRGenericLowerDMAs());
  pm.addPass(ttir::createTTIRGenericGenerateLoops());
  createOptimizationPasses(pm, options);
  pm.addPass(ttir::createTTIRGenericRegionsToFuncs());
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(tt::createConvertTTIRToTTKernelPass());
  pm.addPass(createCanonicalizerPassWithOptions(options));
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm, options);
  if (options.ttnnMode) {
    pm.addPass(tt::createConvertD2MToTTNNPass());
  } else {
    ttir::ConvertTTIRToTTMetalOptions ttirToTTMetalOptions;
    { ttirToTTMetalOptions.mathFidelity = options.mathFidelity; }
    pm.addPass(tt::createConvertTTIRToTTMetalPass(ttirToTTMetalOptions));
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
