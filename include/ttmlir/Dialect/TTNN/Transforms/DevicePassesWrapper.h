// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_DEVICEPASSESWRAPPER_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_DEVICEPASSESWRAPPER_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace tt::tt_metal::distributed {
class MeshDevice;
} // namespace tt::tt_metal::distributed

namespace mlir::tt::ttnn {

#ifdef TTMLIR_ENABLE_OPMODEL
// Options for DevicePassesWrapper pass
struct DevicePassesWrapperOptions {
  // External device pointer (if provided by frontend)
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> devicePtr = nullptr;
  // Tensor L1 usage cap (fraction of available L1 memory)
  // This is set as a module attribute by DevicePassesWrapper, making it
  // accessible to all nested passes via utils::getTensorL1UsageCap()
  float tensorL1UsageCap = 0.95f;
};

// Creates a pass that wraps device-dependent passes with device lifecycle
// management.
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
//   DevicePassesWrapperOptions options;
//   options.devicePtr = ...; // optional
//   nestedPm.addPass(createDevicePassesWrapper(
//       [](OpPassManager &innerPm) {
//         innerPm.addPass(createTTNNOptimizer(...));
//         innerPm.addPass(createTTNNOperationValidationAndFallback(...));
//       },
//       options));
std::unique_ptr<Pass>
createDevicePassesWrapper(std::function<void(OpPassManager &)> populatePipeline,
                          const DevicePassesWrapperOptions &options = {});
#endif

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_DEVICEPASSESWRAPPER_H
