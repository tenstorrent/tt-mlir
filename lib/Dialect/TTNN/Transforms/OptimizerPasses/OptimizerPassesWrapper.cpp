// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/Logger.h"
#ifdef TTMLIR_ENABLE_OPMODEL

#include "ttmlir/Dialect/TTNN/Transforms/OptimizerPassesWrapper.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"

#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::tt::ttnn {

namespace {

class OptimizerPassesWrapper
    : public PassWrapper<OptimizerPassesWrapper, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizerPassesWrapper)
  OptimizerPassesWrapper(std::function<void(OpPassManager &)> populatePipeline,
                         const OptimizerPassesWrapperOptions &options)
      : populatePipeline(std::move(populatePipeline)),
        externalDevice(options.devicePtr) {}

  StringRef getArgument() const override { return "optimizer-passes-wrapper"; }

  StringRef getDescription() const override {
    return "Wraps passes withing Optimizer group for better device lifecycle "
           "management";
  }

  std::unique_ptr<Pass> clonePass() const override {
    auto copy = std::make_unique<OptimizerPassesWrapper>(
        *static_cast<const OptimizerPassesWrapper *>(this));
    copy->externalDevice = externalDevice;
    copy->populatePipeline = populatePipeline;
    return copy;
  }

  void runOnOperation() override {
    // Open device if not externally provided.
    if (externalDevice) {
      op_model::SingletonDeviceContext::setExternalDevice(externalDevice);
    } else {
      op_model::SingletonDeviceContext::getInstance().openDevice();
    }

    // Create nested pass manager and populate it.
    OpPassManager nestedPm(getOperation()->getName());
    populatePipeline(nestedPm);

    // Ensure closeInstance() gets called always.
    auto guard = llvm::make_scope_exit(
        []() noexcept { op_model::SingletonDeviceContext::closeInstance(); });

    // Run the nested pipeline
    if (failed(runPipeline(nestedPm, getOperation()))) {
      signalPassFailure();
      return;
    }
  }

private:
  std::function<void(OpPassManager &)> populatePipeline;
  std::shared_ptr<::tt::tt_metal::distributed::MeshDevice> externalDevice;
};
} // namespace

std::unique_ptr<Pass> createOptimizerPassesWrapper(
    std::function<void(OpPassManager &)> populatePipeline,
    const OptimizerPassesWrapperOptions &options) {
  return std::make_unique<OptimizerPassesWrapper>(std::move(populatePipeline),
                                                  options);
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_ENABLE_OPMODEL
