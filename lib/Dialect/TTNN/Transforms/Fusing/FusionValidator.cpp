// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/FusionValidator.h"

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::tt::ttnn {

FusionValidationResult FusionValidator::runValidationPipeline(ModuleOp module) {
  // Suppress diagnostics from the validation sub-pipeline. Passes like
  // OperationValidationAndFallback call emitError() on failure, which would
  // propagate through the shared MLIRContext and poison the outer pipeline.
  // We detect failure via the PassManager return value instead.
  ScopedDiagnosticHandler diagHandler(context,
                                      [](Diagnostic &) { return success(); });

  // Run workaround passes first.
  {
    PassManager pm(context);
    TTNNWorkaroundsOptions workaroundOptions;
    workaroundOptions.decompositionWorkaroundsEnabled =
        config.applyDecompositionWorkarounds;
    workaroundOptions.layoutWorkaroundsEnabled = config.applyLayoutWorkarounds;
    workaroundOptions.optimizerEnabled = true;
    pm.addPass(mlir::tt::ttnn::createTTNNWorkarounds(workaroundOptions));

    if (failed(pm.run(module))) {
      return FusionValidationResult::failure(
          FusionValidationResult::WorkaroundFailed,
          "Workaround passes failed on fused operation");
    }
  }

  // Run operation validation and fallback.
  {
    PassManager pm(context);
    TTNNOperationValidationAndFallbackOptions validationOptions;
    validationOptions.maxFallbackAttempts = config.maxFallbackAttempts;
    pm.addPass(mlir::tt::ttnn::createTTNNOperationValidationAndFallback(
        validationOptions));

    if (failed(pm.run(module))) {
      return FusionValidationResult::failure(
          FusionValidationResult::ValidationFailed,
          "Op validation/fallback failed on fused operation");
    }
  }

  return FusionValidationResult::success();
}

} // namespace mlir::tt::ttnn
