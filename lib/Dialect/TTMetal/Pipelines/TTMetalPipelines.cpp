// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTIRBufferizationPipeline(OpPassManager &pm) {
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  {
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    bufferizationOptions.functionArgTypeConverterFn =
        [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
           func::FuncOp funcOp,
           const bufferization::BufferizationOptions &bufferizationOptions)
        -> ::mlir::BaseMemRefType {
      auto rankedTensorType = mlir::cast<::mlir::RankedTensorType>(tensorType);
      if (!rankedTensorType.getEncoding()) {
        return bufferization::getMemRefTypeWithStaticIdentityLayout(
            tensorType, memorySpace);
      }
      return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
          .getMemref();
    };
    bufferizationOptions.defaultMemorySpaceFn =
        [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
      auto rankedTensorType = mlir::cast<::mlir::RankedTensorType>(tensorType);
      if (!rankedTensorType.getEncoding()) {
        return mlir::tt::MemorySpaceAttr::get(tensorType.getContext(),
                                              mlir::tt::MemorySpace::DeviceL1);
      }
      return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
          .getMemref()
          .getMemorySpace();
    };
  }
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  // TODO(#2246)
  // bufferization::BufferDeallocationPipelineOptions
  // bufferDeallocationOptions;
  // mlir::bufferization::buildBufferDeallocationPipeline(
  //    pm, bufferDeallocationOptions);
}

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalBackendPipelineOptions &options) {
  ttir::TTIRLoadSystemDescOptions systemDescOptions;
  { systemDescOptions.path = options.systemDescPath; }
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));
  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  {
    implicitDeviceOptions.meshShape = ::llvm::SmallVector<int64_t>(
        options.meshShape.begin(), options.meshShape.end());
  }
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
  pm.addPass(mlir::tt::ttir::createTTIRConstantAsFill());
  ttir::TTIRAttachMetalLayoutOptions attachMetalLayoutOptions;
  {
    // TODO(vroubtsovTT): 'options.version' is WIP until StreamLayout is ok
    // to use end-to-end
    attachMetalLayoutOptions.useStreamLayout = options.version > 0;
  }
  pm.addPass(
      mlir::tt::ttir::createTTIRAttachMetalLayout(attachMetalLayoutOptions));
  // TODO(#1951): replace with TTIRToGeneric implemented as a converter:
  // pm.addPass(mlir::tt::ttir::createTTIRGenericRegion());
  if (options.version > 0) {
    createTTIRBufferizationPipeline(pm);
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::tt::ttir::createTTIRGenericLinearizeMemref());
    pm.addPass(mlir::createLowerAffinePass());
  } else {
    mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
    {
      layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
      layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceL1;
    }
    pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
    pm.addPass(mlir::tt::ttir::createTTIRAllocate());
    pm.addPass(createConvertTTIRToTTMetalPass());
  }
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTMetalPipelines() {
  mlir::PassPipelineRegistration<
      mlir::tt::ttmetal::TTIRToTTMetalBackendPipelineOptions>(
      "ttir-to-ttmetal-backend-pipeline",
      "Pipeline lowering ttir to ttmetal backend.",
      mlir::tt::ttmetal::createTTIRToTTMetalBackendPipeline);
  mlir::PassPipelineRegistration<>(
      "ttir-bufferization-pipeline",
      "Pipeline bufferizing ttir ops on tensors to ops on buffers (memrefs).",
      mlir::tt::ttmetal::createTTIRBufferizationPipeline);
}
} // namespace mlir::tt::ttmetal
