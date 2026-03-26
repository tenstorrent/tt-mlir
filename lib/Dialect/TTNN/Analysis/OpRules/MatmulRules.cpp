// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

static bool isL1Interleaved(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  auto memLayout = config.outputLayout.getMemLayout();
  return config.outputLayout.getBufferType() == BufferType::L1 && memLayout &&
         memLayout.getValue() == TensorMemoryLayout::Interleaved;
}

OutputHints MatmulRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // Use partial configs: deduplicate by (bufferType, memLayout),
  // set ignorePhysicalLayout=true. Backend decides physical layout.
  auto partialConfigs =
      optimizer_utils::getUniqueTestConfigsForMatmulLinear(legalConfigs);

  // Remove L1-interleaved hints for matmul/linear output.
  //
  // L1-interleaved output is "worst of both worlds" for matmul:
  //  - generateMatmulProgramConfig() returns nullopt for non-sharded
  //    output, so no program config is emitted by the compiler.
  //  - tt-metal runtime falls back to MatmulMultiCoreProgramConfig{}
  //    with hardcoded HiFi4 math fidelity (the slowest kernel path).
  //  - Same NOC write overhead as DRAM interleaved (not eliminated
  //    like sharded), loses bias fusion, optimized mcast program
  //    configs, and narrow-shape 1D optimization.
  //  - Subject to L1 capacity constraints unlike DRAM interleaved.
  //
  // DRAM-interleaved is the safe default: full bias fusion, optimized
  // 1D/2D mcast configs, and no L1 pressure from the output tensor.
  // L1-sharded is best when applicable (eliminates NOC writes entirely).
  std::vector<OpConfig> filtered;
  for (const auto &cfg : partialConfigs) {
    if (isL1Interleaved(cfg)) {
      continue;
    }
    filtered.push_back(cfg);
  }

  return OutputHints{filtered, {}, /*attemptL1Sharding=*/true};
}

void MatmulRuleBook::applyOpSpecificAttrs(
    Operation *op, const BeamCandidate &candidate) const {
  auto matmulOp = dyn_cast<MatmulOp>(op);
  auto linearOp = dyn_cast<LinearOp>(op);
  if (!matmulOp && !linearOp) {
    return;
  }

  if (!std::holds_alternative<MatmulAttrs>(
          candidate.configHint.opSpecificAttrs)) {
    return;
  }
  MatmulAttrs matmulAttrs =
      std::get<MatmulAttrs>(candidate.configHint.opSpecificAttrs);
  if (!matmulAttrs.matmulProgramConfig.has_value()) {
    return;
  }
  auto programConfig = matmulAttrs.matmulProgramConfig.value();

  auto setConfigAndFixup = [&](auto concreteOp) {
    concreteOp.setMatmulProgramConfigAttr(programConfig);
    // Workaround for tt-metal issue #35060: if the program config carries a
    // fused activation, remove the op-level activation attr to prevent
    // double application.
    bool hasFusedActivation =
        llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
            .template Case<
                MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                MatmulMultiCoreReuseMultiCast1DProgramConfigAttr,
                MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
                [](auto config) {
                  return config.getFusedActivation() != nullptr;
                })
            .Default([](mlir::Attribute) { return false; });
    if (hasFusedActivation) {
      concreteOp.removeActivationAttr();
    }
  };

  if (matmulOp) {
    setConfigAndFixup(matmulOp);
  } else {
    setConfigAndFixup(linearOp);
  }
}

} // namespace mlir::tt::ttnn
