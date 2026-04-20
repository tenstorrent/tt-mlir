// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/HoistCPUOps/HoistCPUOps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#ifdef TTMLIR_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CPUHOISTNONLOWERABLESHLOOPSTRANSFORM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Allow-list of StableHLO ops to hoist to CPU.
// Before adding an op here, make sure it is supported in
// StableHLO to TOSA/Linalg conversion.
static bool isNonLowerableSHLOOp([[maybe_unused]] mlir::Operation *op) {
#ifdef TTMLIR_ENABLE_STABLEHLO
  return llvm::isa<stablehlo::DynamicUpdateSliceOp, stablehlo::EinsumOp>(op);
#else
  return false;
#endif
}

// Transform pass to hoist non-lowerable StableHLO ops to CPU module.
class CPUHoistNonLowerableSHLOOpsTransform
    : public impl::CPUHoistNonLowerableSHLOOpsTransformBase<
          CPUHoistNonLowerableSHLOOpsTransform> {
public:
  using CPUHoistNonLowerableSHLOOpsTransformBase::
      CPUHoistNonLowerableSHLOOpsTransformBase;

  void runOnOperation() final {
    mlir::ModuleOp rootModule = getOperation();
    mlir::ModuleOp deviceInnerModule = getDeviceInnerModule(rootModule);
    if (!deviceInnerModule) {
      return;
    }

    llvm::SmallVector<CPUHoistedOpsDescriptor> descriptors;
    deviceInnerModule.walk([&](func::FuncOp funcOp) {
      auto result =
          createDescriptorsWithPredicate(funcOp, isNonLowerableSHLOOp);
      descriptors.append(std::make_move_iterator(result.begin()),
                         std::make_move_iterator(result.end()));
    });

    runCPUHoist(rootModule, std::move(descriptors));
  }
};

} // namespace
} // namespace mlir::tt::ttir
