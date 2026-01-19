// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSETCOMPUTEKERNELCONFIG
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNSetComputeKernelConfig
    : public impl::TTNNSetComputeKernelConfigBase<TTNNSetComputeKernelConfig> {

public:
  using impl::TTNNSetComputeKernelConfigBase<
      TTNNSetComputeKernelConfig>::TTNNSetComputeKernelConfigBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    // Convert OptionalMathFidelity to std::optional<MathFidelity>
    // If Undefined, leave as nullopt (don't override math fidelity)
    // Otherwise convert to corresponding MathFidelity value
    std::optional<MathFidelity> mathFidelityOverride;
    if (mathFidelity != OptionalMathFidelity::Undefined) {
      mathFidelityOverride =
          static_cast<MathFidelity>(static_cast<int>(mathFidelity.getValue()));
    }

    // Walk through all operations in the moduleOp
    moduleOp->walk([&](Operation *op) {
      // Check if operation implements ComputeKernelConfigOpInterface
      auto computeConfigOp = dyn_cast<TTNNComputeKernelConfigOpInterface>(op);
      if (!computeConfigOp) {
        return;
      }

      // Get existing compute config attribute (may be nullptr)
      DeviceComputeKernelConfigAttr config =
          computeConfigOp.getComputeConfigAttr();

      // Log operation info and config before setting overrides
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "TTNNSetComputeKernelConfig - Operation: {0}",
                   op->getName().getStringRef());
      if (config) {
        TTMLIR_DEBUG(ttmlir::LogComponent::General,
                     "  Existing config before override: {0}", config);
      } else {
        TTMLIR_DEBUG(
            ttmlir::LogComponent::General,
            "  Existing config before override: nullptr (no config set)");
        config = DeviceComputeKernelConfigAttr::get(context);
      }

      // Apply overrides only for parameters that are not already set.
      // Each withX() method returns a new attribute, so we chain them.

      // Math fidelity: only override if not already set and we have a value.
      if (!config.getMathFidelity().has_value() &&
          mathFidelityOverride.has_value()) {
        config = config.withMathFidelity(*mathFidelityOverride);
      }

      // Bool options: only override if not already set and option is true.
      if (!config.getMathApproxMode() && mathApproxMode) {
        config = config.withMathApproxMode(mathApproxMode);
      }

      if (!config.getFp32DestAccEn() && fp32DestAccEn) {
        config = config.withFp32DestAccEn(fp32DestAccEn);
      }

      if (!config.getPackerL1Acc() && packerL1Acc) {
        config = config.withPackerL1Acc(packerL1Acc);
      }

      if (!config.getDstFullSyncEn() && dstFullSyncEn) {
        config = config.withDstFullSyncEn(dstFullSyncEn);
      }

      // Log config after applying overrides
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "  Config after override: {0}\n", config);

      // Set the updated config back to the operation
      computeConfigOp.setComputeConfigAttr(config);
    });
  }
};

} // namespace mlir::tt::ttnn
