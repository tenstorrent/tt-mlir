// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {
namespace {

// During training backward run we have observed matmuls with inner dimension
// equal to vocab size fail due to precision issues. Upon inspection, we have
// not found a model that has vocab size smaller than this threshold. Fix is
// only applied to these types of matmuls, as we do not expect other matmuls to
// have this type of inner dimension.
constexpr int64_t kLargeInnerDimThreshold = 50000;

static bool hasSetComputeConfigFields(DeviceComputeKernelConfigAttr config) {
  return config.getMathFidelity().has_value() || config.getMathApproxMode() ||
         config.getFp32DestAccEn() || config.getPackerL1Acc() ||
         config.getDstFullSyncEn();
}

template <typename OpTy>
void applyLargeInnerDimBf16MatmulConfig(OpTy op,
                                        DeviceComputeKernelConfigAttr &config) {
  std::optional<int64_t> innerDim =
      getMatmulInnerDim(op.getA().getType(), op.getB().getType(),
                        op.getTransposeA(), op.getTransposeB());
  if (!innerDim || *innerDim <= kLargeInnerDimThreshold) {
    return;
  }

  auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
  if (ttcore::elementTypeToDataType(outputType.getElementType()) !=
      ttcore::DataType::BFloat16) {
    return;
  }

  config = config.withFp32DestAccEn(true);
  config = config.withPackerL1Acc(true);
}

} // namespace

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

      // Get existing compute config attribute (may be nullptr).
      DeviceComputeKernelConfigAttr originalConfig =
          computeConfigOp.getComputeConfigAttr();
      DeviceComputeKernelConfigAttr config = originalConfig;

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

      // Apply overrides only for parameters that are not already set on the op.
      // Each withX() method returns a new attribute, so we chain them.
      // A pipeline option that is unset (std::nullopt for the bool options,
      // Undefined for math fidelity) is never applied, so the op parameter is
      // left for TTNN to decide. A bool option that is explicitly set - even to
      // false - is applied, giving a genuine true / false / unset tri-state.
      //
      // When a global override is available but the op already sets that knob,
      // we keep the op's value and emit a debug log so the user is aware the
      // global compute-kernel-config was not applied for that parameter.
      auto logSkippedOverride = [&](llvm::StringRef knob) {
        TTMLIR_DEBUG(ttmlir::LogComponent::General,
                     "  Skipping global {0} override on {1}: op already sets "
                     "this knob, keeping its value",
                     knob, op->getName().getStringRef());
      };

      // Math fidelity: only override if not already set and we have a value.
      if (mathFidelityOverride.has_value()) {
        if (!config.getMathFidelity().has_value()) {
          config = config.withMathFidelity(*mathFidelityOverride);
        } else {
          logSkippedOverride("math_fidelity");
        }
      }

      // Bool options: only override if not already set on the op and the
      // pipeline option carries a value (true or false).
      if (mathApproxMode.has_value()) {
        if (!config.getMathApproxMode()) {
          config = config.withMathApproxMode(*mathApproxMode);
        } else {
          logSkippedOverride("math_approx_mode");
        }
      }

      if (fp32DestAccEn.has_value()) {
        if (!config.getFp32DestAccEn()) {
          config = config.withFp32DestAccEn(*fp32DestAccEn);
        } else {
          logSkippedOverride("fp32_dest_acc_en");
        }
      }

      if (packerL1Acc.has_value()) {
        if (!config.getPackerL1Acc()) {
          config = config.withPackerL1Acc(*packerL1Acc);
        } else {
          logSkippedOverride("packer_l1_acc");
        }
      }

      if (dstFullSyncEn.has_value()) {
        if (!config.getDstFullSyncEn()) {
          config = config.withDstFullSyncEn(*dstFullSyncEn);
        } else {
          logSkippedOverride("dst_full_sync_en");
        }
      }

      // This fix is required for correctness of large matmuls/linears.
      if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
        applyLargeInnerDimBf16MatmulConfig(matmulOp, config);
      } else if (auto linearOp = dyn_cast<LinearOp>(op)) {
        applyLargeInnerDimBf16MatmulConfig(linearOp, config);
      }

      // Log config after applying overrides
      TTMLIR_DEBUG(ttmlir::LogComponent::General,
                   "  Config after override: {0}\n", config);

      // Avoid materializing an empty compute_config when nothing changed.
      if (config != originalConfig && hasSetComputeConfigFields(config)) {
        computeConfigOp.setComputeConfigAttr(config);
      }
    });
  }
};

} // namespace mlir::tt::ttnn
