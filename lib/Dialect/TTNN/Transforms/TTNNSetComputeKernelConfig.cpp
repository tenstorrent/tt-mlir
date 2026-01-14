// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/MathFidelityParser.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNSETCOMPUTEKERNELCONFIG
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNSetComputeKernelConfig
    : public impl::TTNNSetComputeKernelConfigBase<TTNNSetComputeKernelConfig> {

public:
  using impl::TTNNSetComputeKernelConfigBase<
      TTNNSetComputeKernelConfig>::TTNNSetComputeKernelConfigBase;

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    TTNNSetComputeKernelConfigOptions opts;
    llvm::errs() << "[DEBUG CLONE] src=" << (void *)this
                 << " hasValue=" << (mathFidelity.hasValue() ? "true" : "false")
                 << " value="
                 << (mathFidelity.getValue()
                         ? stringifyMathFidelity(*mathFidelity.getValue())
                         : "nullopt")
                 << "\n";
    opts.mathFidelity = this->mathFidelity.getValue();
    opts.mathApproxMode = this->mathApproxMode.getValue();
    opts.fp32DestAccEn = this->fp32DestAccEn.getValue();
    opts.packerL1Acc = this->packerL1Acc.getValue();
    opts.dstFullSyncEn = this->dstFullSyncEn.getValue();
    auto pass = std::make_unique<TTNNSetComputeKernelConfig>(std::move(opts));
    llvm::errs() << "[DEBUG CLONE] Created pass ptr=" << (void *)pass.get()
                 << "\n";
    llvm::errs() << "[DEBUG CLONE] config_pass textual pipeline: ";
    pass->printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "-------\n";
    return std::move(pass);
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    llvm::errs() << "[DEBUG PASS] this ptr=" << (void *)this << "\n";
    llvm::errs() << "[DEBUG PASS] option mathFidelity:\n";
    llvm::errs() << "  hasValue() = "
                 << (mathFidelity.hasValue() ? "true" : "false") << "\n";

    auto v = mathFidelity.getValue();
    llvm::errs() << "  getValue() = ";
    if (v) {
      llvm::errs() << stringifyMathFidelity(*v) << "\n";
    } else {
      llvm::errs() << "nullopt\n";
    }

    llvm::errs() << "  textual pipeline for *this* pass: ";
    this->printAsTextualPipeline(llvm::errs());
    llvm::errs() << "\n";

    std::optional<MathFidelity> mathFidelityOverride = mathFidelity;
    llvm::errs()
        << "[DEBUG PASS] mathFidelityOverride after implicit conversion: ";
    if (mathFidelityOverride.has_value()) {
      llvm::errs() << "has value: "
                   << stringifyMathFidelity(*mathFidelityOverride) << "\n";
      std::cout << "*** mathFidelityOverride.has_value() is TRUE" << std::endl;
    } else {
      llvm::errs() << "nullopt\n";
      std::cout << "*** mathFidelityOverride.has_value() is FALSE" << std::endl;
    }

    // Debug: Check fp32 option for comparison
    llvm::errs() << "[DEBUG PASS] fp32DestAccEn Option member:\n";
    llvm::errs() << "  getValue() -> " << fp32DestAccEn.getValue() << "\n";
    llvm::errs() << "  implicit bool conversion -> "
                 << (fp32DestAccEn ? "true" : "false") << "\n";

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
