// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTNN/Transforms/DevicePassesWrapper.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"

#include <cstdlib>

namespace mlir::tt::ttnn {

namespace {

class DevicePassesWrapper
    : public PassWrapper<DevicePassesWrapper, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DevicePassesWrapper)
  DevicePassesWrapper(std::function<void(OpPassManager &)> populatePipeline,
                      const DevicePassesWrapperOptions &options)
      : populatePipeline(std::move(populatePipeline)),
        externalDevice(options.devicePtr),
        tensorL1UsageCap(options.tensorL1UsageCap) {}

  StringRef getArgument() const override { return "device-passes-wrapper"; }

  StringRef getDescription() const override {
    return "Wraps device-dependent passes for device lifecycle management";
  }

  std::unique_ptr<Pass> clonePass() const override {
    auto copy = std::make_unique<DevicePassesWrapper>(
        *static_cast<const DevicePassesWrapper *>(this));
    copy->externalDevice = externalDevice;
    copy->populatePipeline = populatePipeline;
    copy->tensorL1UsageCap = tensorL1UsageCap;
    return copy;
  }

  void runOnOperation() override {
    // Disable tt-metal backtrace generation.
    setenv("TT_METAL_DISABLE_BACKTRACE", "1", 1);

    // Open device if not externally provided.
    if (externalDevice) {
      op_model::SingletonDeviceContext::setExternalDevice(externalDevice);
    } else {
      op_model::SingletonDeviceContext::setSystemDesc(
          ttcore::getCurrentScopeSystemDesc(getOperation()));
      // Read mesh shape from the DeviceAttr registered by TTCoreRegisterDevicePass
      // (which always runs before this wrapper). Multi-chip shapes (rows*cols > 1)
      // trigger fabric initialization required for CCL op constraint queries.
      std::optional<std::pair<size_t, size_t>> devMeshShape = std::nullopt;
      if (auto deviceOp = ttcore::lookupDeviceOp(getOperation())) {
        auto shape = deviceOp.getDeviceAttr().getMeshShape();
        if (shape.size() >= 2) {
          devMeshShape = {static_cast<size_t>(shape[0]),
                          static_cast<size_t>(shape[1])};
        }
      }
      op_model::SingletonDeviceContext::getInstance().openMockDevice(
          ::tt::constants::opModelDefaultTraceRegionSize, devMeshShape);
    }

    // Set tensorL1UsageCap as a module attribute so it's accessible to nested
    // passes without parameter threading.
    Operation *op = getOperation();
    OpBuilder builder(op->getContext());
    op->setAttr(utils::g_TensorL1UsageCapAttrName,
                builder.getF32FloatAttr(tensorL1UsageCap));

    // Create nested pass manager and populate it.
    OpPassManager nestedPm(getOperation()->getName());
    populatePipeline(nestedPm);

    // Ensure closeInstance() gets called and backtrace env is unset.
    auto guard = llvm::make_scope_exit([]() noexcept {
      op_model::SingletonDeviceContext::closeInstance();
      unsetenv("TT_METAL_DISABLE_BACKTRACE");
    });

    // Install a scoped diagnostic handler to detect error diagnostics from
    // nested passes. MLIR's OpToOpPassAdaptor runs passes on nested operations
    // matching the pipeline's anchor type. When passes call signalPassFailure()
    // inside these nested invocations, the failure may not propagate through
    // runPipeline(). Catching error diagnostics provides a reliable fallback.
    bool nestedPassEmittedError = false;
    ScopedDiagnosticHandler diagHandler(
        op->getContext(), [&](Diagnostic &diag) {
          if (diag.getSeverity() == DiagnosticSeverity::Error) {
            nestedPassEmittedError = true;
          }
          return failure();
        });

    // Run the nested pipeline.
    auto pipelineResult = runPipeline(nestedPm, getOperation());
    if (failed(pipelineResult) || nestedPassEmittedError) {
      signalPassFailure();
      return;
    }

    // Clean up the attribute after the nested passes complete.
    op->removeAttr(utils::g_TensorL1UsageCapAttrName);
  }

private:
  std::function<void(OpPassManager &)> populatePipeline;
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> externalDevice;
  float tensorL1UsageCap;
};
} // namespace

std::unique_ptr<Pass>
createDevicePassesWrapper(std::function<void(OpPassManager &)> populatePipeline,
                          const DevicePassesWrapperOptions &options) {
  return std::make_unique<DevicePassesWrapper>(std::move(populatePipeline),
                                               options);
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_ENABLE_OPMODEL
