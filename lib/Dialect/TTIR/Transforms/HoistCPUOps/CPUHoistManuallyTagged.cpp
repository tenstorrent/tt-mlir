// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CPUHOISTMANUALLYTAGGEDOPSTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Transform pass to hoist ops manually tagged with ttir.should_hoist.
class CPUHoistManuallyTaggedOpsTransform
    : public impl::CPUHoistManuallyTaggedOpsTransformBase<
          CPUHoistManuallyTaggedOpsTransform> {
public:
  using CPUHoistManuallyTaggedOpsTransformBase::
      CPUHoistManuallyTaggedOpsTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    mlir::ModuleOp deviceInnerModule = getDeviceInnerModule(rootModule);
    if (!deviceInnerModule) {
      return;
    }

    llvm::SmallVector<CPUHoistedOpsDescriptor> descriptors;
    deviceInnerModule.walk([&](func::FuncOp funcOp) {
      auto result =
          createDescriptorsWithPredicate(funcOp, [](mlir::Operation *op) {
            return op->hasAttr(ttir::ShouldHoistAttr::name);
          });
      descriptors.append(std::make_move_iterator(result.begin()),
                         std::make_move_iterator(result.end()));
    });

    runCPUHoist(rootModule, std::move(descriptors));
  }
};

} // namespace
} // namespace mlir::tt::ttir
