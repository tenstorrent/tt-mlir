// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Pipelines/TTIRPipelines.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "ttmlir/Conversion/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/transforms/Passes.h"
#endif

namespace mlir::tt::ttir {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

#ifdef TTMLIR_ENABLE_STABLEHLO
void createStableHLOToTTIRPipeline(
    OpPassManager &pm, const StableHLOToTTIRPipelineOptions &options) {
  if (options.arithDialectConversionsEnabled) {
    pm.addPass(createConvertArithToStableHLOPass());
  }
  if (options.legalizeCompositeToCallEnabled) {
    pm.addPass(stablehlo::createStablehloLegalizeCompositeToCallPass());
  }
  pm.addPass(createConvertStableHLOToTTIRPass());
  if (options.removeDeadValuesEnabled) {
    pm.addPass(mlir::createRemoveDeadValuesPass());
  }
  if (options.sparseConstantPropagationEnabled) {
    pm.addPass(mlir::createSCCPPass());
  }
}
#endif

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTTIRPipelines() {
#ifdef TTMLIR_ENABLE_STABLEHLO
  mlir::PassPipelineRegistration<StableHLOToTTIRPipelineOptions>(
      "stablehlo-to-ttir-pipeline",
      "Pipeline lowering stablehlo to ttir dialect.",
      mlir::tt::ttir::createStableHLOToTTIRPipeline);
#endif
}
} // namespace mlir::tt::ttir
