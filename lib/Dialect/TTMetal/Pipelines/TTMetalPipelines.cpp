// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
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
  systemDescOptions.path = options.systemDescPath;
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc(systemDescOptions));
  ttir::TTIRImplicitDeviceOptions implicitDeviceOptions;
  implicitDeviceOptions.meshShape = ::llvm::SmallVector<int64_t>(
      options.meshShape.begin(), options.meshShape.end());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice(implicitDeviceOptions));
  pm.addPass(mlir::tt::ttir::createTTIRConstantAsFill());
  pm.addPass(mlir::tt::ttir::createTTIRAttachMetalLayout());
  ttir::TTIRGenericRegionOptions genericRegionOptions;
  genericRegionOptions.newLowering = options.newLowering;
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegion(genericRegionOptions));
  if (options.newLowering) {
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
  } else {
    mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
    layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
    layoutOptions.defaultMemorySpace = mlir::tt::MemorySpace::DeviceL1;
    layoutOptions.defaultDeviceMemoryLayout = mlir::tt::TensorMemoryLayout::None;
    pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
    pm.addPass(mlir::tt::ttir::createTTIRGenericOpCBs());
    pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  }
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
