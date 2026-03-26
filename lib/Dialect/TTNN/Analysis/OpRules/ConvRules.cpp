// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {

OutputHints Conv2dRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // Conv2d configs carry Conv2dConfig tied to output hint.
  return OutputHints{legalConfigs, {}, /*attemptL1Sharding=*/true};
}

void Conv2dRuleBook::applyOpSpecificAttrs(
    Operation *op, const BeamCandidate &candidate) const {
  auto conv2d = dyn_cast<Conv2dOp>(op);
  auto convT = dyn_cast<ConvTranspose2dOp>(op);
  if (!conv2d && !convT) {
    return;
  }

  if (!std::holds_alternative<Conv2dAttrs>(
          candidate.configHint.opSpecificAttrs)) {
    return;
  }
  Conv2dAttrs conv2dAttrs =
      std::get<Conv2dAttrs>(candidate.configHint.opSpecificAttrs);

  auto setAttrs = [&](auto convOp) {
    if (conv2dAttrs.conv2dConfig.has_value()) {
      convOp.setConv2dConfigAttr(conv2dAttrs.conv2dConfig.value());
    }
    if (conv2dAttrs.deviceComputeKernelConfig.has_value()) {
      convOp.setComputeConfigAttr(
          conv2dAttrs.deviceComputeKernelConfig.value());
    }
  };

  if (conv2d) {
    setAttrs(conv2d);
  } else {
    setAttrs(convT);
  }
}

/// Extract act_block_h_override from a BeamCandidate's Conv2dAttrs.
/// Returns UINT32_MAX if not a Conv2d config (sorts last).
static uint32_t getActBlockHOverride(const BeamCandidate &c) {
  if (auto *conv2d = std::get_if<Conv2dAttrs>(&c.configHint.opSpecificAttrs)) {
    if (conv2d->conv2dConfig.has_value() && conv2d->conv2dConfig.value()) {
      auto abh = conv2d->conv2dConfig.value().getActBlockHOverride();
      return abh.has_value() ? abh.value() : 0;
    }
  }
  return UINT32_MAX;
}

bool Conv2dRuleBook::preferCandidate(Operation * /*op*/, const BeamCandidate &a,
                                     const BeamCandidate &b) const {
  // Prefer act_block_h_override=0 (auto, best), then higher over lower.
  // Ordering: 0 > 64 > 32 > ...
  uint32_t abhA = getActBlockHOverride(a);
  uint32_t abhB = getActBlockHOverride(b);
  if (abhA != abhB) {
    if (abhA == 0) {
      return true;
    }
    if (abhB == 0) {
      return false;
    }
    return abhA > abhB;
  }
  return false;
}

void applyConvSliceConfig(ModuleOp moduleOp) {
  moduleOp->walk([](Conv2dOp conv2dOp) {
    conv2dOp.setConv2dSliceConfigAttr(Conv2dSliceConfigAttr::get(
        conv2dOp.getContext(), Conv2dSliceType::L1Full, 0));
  });
}

void fixupConvDeallocate(func::FuncOp func) {
  func->walk([&](Operation *op) {
    auto disableDeallocIfMultiUser = [](auto convOp) {
      auto config = convOp.getConv2dConfigAttr();
      if (!config || !config.getDeallocateActivation() ||
          !config.getDeallocateActivation().getValue()) {
        return;
      }
      Value input = convOp.getInput();
      if (!input.hasOneUse()) {
        convOp.setConv2dConfigAttr(config.withDeallocateActivation(false));
        TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                     "Disabled deallocate_activation for conv2d with "
                     "multi-use input: {}",
                     ttmlir::opToString(convOp));
      }
    };

    if (auto conv2d = dyn_cast<Conv2dOp>(op)) {
      disableDeallocIfMultiUser(conv2d);
    } else if (auto convT = dyn_cast<ConvTranspose2dOp>(op)) {
      disableDeallocIfMultiUser(convT);
    }
  });
}

} // namespace mlir::tt::ttnn
