// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

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
    // TODO(vroubtsovTT): 'options.version' is WIP until once StreamLayout is ok
    // to use end-to-end
    attachMetalLayoutOptions.useStreamLayout = options.version > 0;
  }
  pm.addPass(
      mlir::tt::ttir::createTTIRAttachMetalLayout(attachMetalLayoutOptions));
  ttir::TTIRGenericRegionOptions genericRegionOptions;
  { genericRegionOptions.newLowering = options.version > 0; }
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegion(genericRegionOptions));
  if (options.version > 0) {
    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    bufferizationOptions.functionArgTypeConverterFn =
        [](mlir::TensorType tensorType, mlir::Attribute memorySpace,
           mlir::FunctionOpInterface functionOp,
           const bufferization::BufferizationOptions &bufferizationOptions) {
          auto rankedTensorType =
              mlir::cast<::mlir::RankedTensorType>(tensorType);
          mlir::Type memrefResultType =
              mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
                  .getMemref();
          return mlir::cast<::mlir::BaseMemRefType>(memrefResultType);
        };
    bufferizationOptions.defaultMemorySpaceFn =
        [](mlir::TensorType tensorType) -> std::optional<mlir::Attribute> {
      auto rankedTensorType = mlir::cast<::mlir::RankedTensorType>(tensorType);
      return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
          .getMemref()
          .getMemorySpace();
    };
    // bufferizationOptions.bufferAlignment = TODO;
    pm.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::tt::ttir::createTTIRGenericLinearizeMemref());
    pm.addPass(mlir::createLowerAffinePass());
  } else {
    mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
    {
      layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
      layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceL1;
      layoutOptions.defaultDeviceMemoryLayout =
          mlir::tt::TensorMemoryLayout::None;
    }
    pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
    pm.addPass(mlir::tt::ttir::createTTIRGenericOpCBs());
    pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
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
}
} // namespace mlir::tt::ttmetal
