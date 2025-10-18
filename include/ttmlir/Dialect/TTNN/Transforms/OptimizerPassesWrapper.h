// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZERPASSESWRAPPER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZERPASSESWRAPPER_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn {

#ifdef TTMLIR_ENABLE_OPMODEL
// Options for OptimizerPassesWrapper pass
struct OptimizerPassesWrapperOptions {
  // External device pointer (if provided by frontend)
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> devicePtr = nullptr;
};

// Creates a pass that wraps Optimizer passes with device lifecycle management.
//
// This pass:
// 1. Makes sure device singleton is initialized before running nested passes.
// 2. Runs the given pipeline population function, and then runs the nested
//    pipeline.
// 3. Closes device singleton instance always, even on failures within pipeline
//    (regardless if we opened it or not).
//
// Usage in pipeline:
//   OpPassManager &nestedPm = pm.nest<SomeOp>();
//   OptimizerPassesWrapperOptions options;
//   options.devicePtr = ...; // optional
//   nestedPm.addPass(createOptimizerPassesWrapper(
//       [](OpPassManager &innerPm) {
//         innerPm.addPass(createTTNNOptimizer(...));
//         innerPm.addPass(createTTNNOperationValidationAndFallback(...));
//       },
//       options));
std::unique_ptr<Pass> createOptimizerPassesWrapper(
    std::function<void(OpPassManager &)> populatePipeline,
    const OptimizerPassesWrapperOptions &options = {});
#endif

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_OPTIMIZERPASSESWRAPPER_H
