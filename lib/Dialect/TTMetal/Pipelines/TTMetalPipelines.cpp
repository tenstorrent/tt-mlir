// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
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
  pm.addPass(tt::createTTRegisterDevicePass(registerDeviceOptions));
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
  createTTIRBufferizationPipeline(pm);
  pm.addPass(ttir::createTTIRAllocate());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(ttir::createTTIRGenericLinearizeMemref());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(ttir::createTTIRGenericGenerateDatamovement());
  pm.addPass(ttir::createTTIRGenericLowerDMAs());
  pm.addPass(ttir::createTTIRGenericHWThreadSelection());
  pm.addPass(ttir::createTTIRGenericGenerateLoops());
  createOptimizationPasses(pm);
  pm.addPass(ttir::createTTIRGenericRegionsToFuncs());
  pm.addPass(createConvertTTIRToTTMetalPass());
  pm.addPass(mlir::createCanonicalizerPass());
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
