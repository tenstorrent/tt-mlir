// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "mlir/Pass/PassManager.h"

#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttmetal {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createTTIRToTTMetalBackendPipeline(
    OpPassManager &pm, const TTIRToTTMetalBackendPipelineOptions &options) {
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc());
  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = options.meshShape;
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegion());
  mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
  layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
  layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceL1;
  layoutOptions.defaultDeviceMemoryLayout = mlir::tt::TensorMemoryLayout::None;
  pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  pm.addPass(mlir::tt::ttir::createTTIRAllocate());
  pm.addPass(createConvertTTIRToTTMetalPass());
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
