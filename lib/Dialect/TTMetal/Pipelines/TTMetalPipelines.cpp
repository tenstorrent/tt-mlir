// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.h"
#include "ttmlir/Dialect/LLVM/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTIRBufferizationPipeline(OpPassManager &pm) {
  pm.addPass(ttir::createTTIRPrepareTensorsForBufferization());
  bufferization::OneShotBufferizePassOptions bufferizePassOptions;
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

void createOptimizationPasses(OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::arith::createIntRangeOptimizationsPass());
}

void createTTIRToTTMetalFrontendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  tt::TTRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.mockSystemDescArch = options.mockSystemDescArch;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(tt::createTTRegisterDevicePass(registerDeviceOptions));
  pm.addPass(tt::createTTIRToTTIRDecompositionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(tt::createTTIRToTTIRGenericPass());
  pm.addPass(mlir::createCanonicalizerPass());
  ttir::TTIROptimizeTensorLayoutOptions optimizeTensorLayoutOptions;
  {
    optimizeTensorLayoutOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
  }
  pm.addPass(ttir::createTTIROptimizeTensorLayout(optimizeTensorLayoutOptions));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(ttir::createTTIRLowerToLayout());
}

void createTTIRToTTMetalMiddleendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  createTTIRBufferizationPipeline(pm);
  pm.addPass(ttir::createTTIRAllocate());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(ttir::createTTIRInsertDstRegisterAccess());
  pm.addPass(ttir::createTTIRGenericLinearizeMemref());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(ttir::createTTIRGenericGenerateDatamovement());
  pm.addPass(ttir::createTTIRGenericLowerDMAs());
  pm.addPass(ttir::createTTIRGenericHWThreadSelection());
  pm.addPass(ttir::createTTIRGenericGenerateLoops());
  createOptimizationPasses(pm);
  pm.addPass(ttir::createTTIRGenericRegionsToFuncs());
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalPipelineOptions &options) {
  pm.addPass(tt::createConvertTTIRToTTKernelPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(pm);
  pm.addPass(createConvertTTIRToTTMetalPass());
  pm.addPass(createConvertTTKernelToEmitC());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::emitc::createFormExpressionsPass());
}

void createTTIRToTTMetalPipeline(OpPassManager &pm,
                                 const TTIRToTTMetalPipelineOptions &options) {
  // Create DeviceModule to wrap all ops.
  pm.addPass(tt::createTTWrapDeviceModulePass());
  // Create CPUModuleOp to wrap hoisted ops (if any).
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular ttir to ttmetal pipelines on IR in DeviceModule.
  OpPassManager &devicePm =
      pm.nest<tt::DeviceModuleOp>().nest<mlir::ModuleOp>();
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
  mlir::PassPipelineRegistration<>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
