// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "ttmlir/Conversion/Passes.h"
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
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  ttir::initializeOneShotBufferizationOptions(bufferizationOptions);
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
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

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalBackendPipelineOptions &options) {
  tt::TTRegisterDevicePassOptions registerDeviceOptions;
  {
    registerDeviceOptions.systemDescPath = options.systemDescPath;
    registerDeviceOptions.meshShape = llvm::to_vector(options.meshShape);
  }
  pm.addPass(tt::createTTWrapDeviceModulePass());
  pm.addPass(ttir::createTTIRHoistTransform());

  // Run regular TTIR to TT_Metal pipeline on DeviceModule.
  OpPassManager &devicePm =
      pm.nest<tt::DeviceModuleOp>().nest<mlir::ModuleOp>();

  devicePm.addPass(tt::createTTRegisterDevicePass(registerDeviceOptions));
  devicePm.addPass(tt::createTTIRToTTIRGenericPass());
  devicePm.addPass(mlir::createCanonicalizerPass());
  ttir::TTIROptimizeTensorLayoutOptions optimizeTensorLayoutOptions;
  {
    optimizeTensorLayoutOptions.overrideDeviceShape =
        llvm::to_vector(options.overrideDeviceShape);
  }
  devicePm.addPass(
      ttir::createTTIROptimizeTensorLayout(optimizeTensorLayoutOptions));
  devicePm.addPass(mlir::createCanonicalizerPass());
  devicePm.addPass(ttir::createTTIRLowerToLayout());
  createTTIRBufferizationPipeline(devicePm);
  devicePm.addPass(ttir::createTTIRAllocate());
  devicePm.addPass(mlir::createCanonicalizerPass());
  devicePm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  devicePm.addPass(ttir::createTTIRGenericLinearizeMemref());
  devicePm.addPass(mlir::createLowerAffinePass());
  devicePm.addPass(ttir::createTTIRGenericGenerateDatamovement());
  devicePm.addPass(ttir::createTTIRGenericLowerDMAs());
  devicePm.addPass(ttir::createTTIRGenericHWThreadSelection());
  devicePm.addPass(ttir::createTTIRGenericGenerateLoops());
  createOptimizationPasses(devicePm);
  devicePm.addPass(ttir::createTTIRGenericRegionsToFuncs());
  devicePm.addPass(tt::createConvertTTIRToTTKernelPass());
  devicePm.addPass(mlir::createCanonicalizerPass());
  devicePm.addPass(ttkernel::createTTKernelControlDstSection());
  createOptimizationPasses(devicePm);
  devicePm.addPass(createConvertTTIRToTTMetalPass());
  devicePm.addPass(createConvertTTKernelToEmitC());
  devicePm.addPass(mlir::createCanonicalizerPass());
  devicePm.addPass(mlir::emitc::createFormExpressionsPass());

  // Run lowering to LLVM pass on hoisted funcs in CPUModule.
  OpPassManager &cpuPm = pm.nest<tt::CPUModuleOp>().nest<mlir::ModuleOp>();
  cpuPm.addPass(createConvertTTIRToLinalgPass());
  ttir::LinalgToLLVMPipelineOptions linalgToLLLVMOptions;
  ttir::createLinalgToLLVMPipeline(cpuPm, linalgToLLLVMOptions);
  cpuPm.addPass(llvm_util::createLLVMEmitCallingConventionWrapperFuncs());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTMetalPipelines() {
  mlir::PassPipelineRegistration<
      tt::ttmetal::TTIRToTTMetalBackendPipelineOptions>(
      "ttir-to-ttmetal-backend-pipeline",
      "Pipeline lowering ttir to ttmetal backend.",
      tt::ttmetal::createTTIRToTTMetalBackendPipeline);
  mlir::PassPipelineRegistration<>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
