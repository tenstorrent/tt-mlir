// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/ConvRules.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Conv2dConfigParams.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

OutputHints Conv2dRuleBook::getOutputHints(
    Operation * /*op*/, const std::vector<OpConfig> &legalConfigs) const {
  // Conv2d configs carry Conv2dConfig tied to output hint.
  // Embed the output shard layout into the conv2d config so that the tt-metal
  // runtime respects it when choosing a kernel variant. Only propagate for
  // sharded outputs; shard_layout is not applicable (and rejected by the
  // backend) for interleaved DRAM/L1 outputs.
  std::vector<OpConfig> configs = legalConfigs;
  for (auto &config : configs) {
    assert(config.outputLayout &&
           "Conv2d legal config must have output layout");
    auto ml = config.outputLayout.getMemLayout();
    assert(ml &&
           "Conv2d legal config must have memory layout in output layout");
    if (!isShardedMemoryLayout(ml.getValue())) {
      continue;
    }

    auto *attrs = std::get_if<Conv2dAttrs>(&config.opSpecificAttrs);
    assert(attrs && "Conv2d legal config must have Conv2dAttrs");
    assert(attrs->conv2dConfig.has_value() &&
           "Conv2d legal config must have conv2d config");

    Conv2dConfigParams params(attrs->conv2dConfig.value());
    params.shardLayout = ml.getValue();
    attrs->conv2dConfig =
        params.buildConv2dConfigAttr(config.outputLayout.getContext());
  }
  return OutputHints{configs, {}};
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
    // Note: outputDtype (if selected) is applied by the outer applyOpConfig
    // loop via TTNNDtypeOpInterface::setDtypeAttr and result.setType(), using
    // candidate.outputLayouts[0] which is the metal-backend-validated layout
    // with the correct shard shape. Do NOT set/rebuild the result type here;
    // doing so recomputes shard shape with a simplified formula and creates a
    // mismatch with the memory config shard spec in downstream to_memory_config ops.
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

static bool getBothDoubleBuffersEnabled(const BeamCandidate &c) {
  if (auto *conv2d =
          std::get_if<Conv2dAttrs>(&c.configHint.opSpecificAttrs)) {
    if (conv2d->conv2dConfig.has_value() && conv2d->conv2dConfig.value()) {
      auto cfg = conv2d->conv2dConfig.value();
      auto weightsDB = cfg.getEnableWeightsDoubleBuffer();
      auto actDB = cfg.getEnableActDoubleBuffer();
      return weightsDB && weightsDB.getValue() && actDB && actDB.getValue();
    }
  }
  return false;
}

bool Conv2dRuleBook::preferCandidate(Operation *op, const BeamCandidate &a,
                                     const BeamCandidate &b) const {
  // Prefer act_block_h_override=0 (auto, best), then higher over lower.
  // Ordering: 0 > 384 > 64 > 32 > ...
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
  // Among same act_block_h, prefer both double-buffers enabled.
  bool dbA = getBothDoubleBuffersEnabled(a);
  bool dbB = getBothDoubleBuffersEnabled(b);
  if (dbA != dbB) {
    return dbA;
  }
  return OpRuleBook::preferCandidate(op, a, b);
}

bool Conv2dRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  if (inputLayouts.empty() || !hint.outputLayout) {
    return true;
  }

  auto inputML = inputLayouts[0].getMemLayout();
  auto outputML = hint.outputLayout.getMemLayout();
  if (!inputML || !outputML) {
    return true;
  }

  // If both input and output are sharded, require matching sharding type.
  // Mixing sharding types (e.g. HS input -> BS output) forces costly reshards
  // and can cascade into DRAM fallbacks at downstream concat ops.
  if (isShardedMemoryLayout(inputML.getValue()) &&
      isShardedMemoryLayout(outputML.getValue())) {
    return inputML.getValue() == outputML.getValue();
  }

  return true;
}

// Compute the activation tensor size in bytes when stored as TILE-format bf16
// and height-sharded across all cores. Used to decide whether L1Full is safe.
static uint64_t computeActBytesPerCore(Conv2dOp op) {
  static constexpr uint64_t kTileWidth     = 32;
  static constexpr uint64_t kBytesPerElem  = 2;  // bf16
  static constexpr uint64_t kDefaultCores  = 64;  // 8×8 Wormhole grid

  uint64_t inH     = static_cast<uint64_t>(op.getInputHeight());
  uint64_t inW     = static_cast<uint64_t>(op.getInputWidth());
  uint64_t inC     = static_cast<uint64_t>(op.getInChannels());
  uint64_t cPadded = ((inC + kTileWidth - 1) / kTileWidth) * kTileWidth;

  uint64_t totalBytes = inH * inW * cPadded * kBytesPerElem;
  return totalBytes / kDefaultCores;
}

// Set the conv2d_slice_config attribute on every Conv2dOp before the
// OperationValidationAndFallback pass runs.
//
// Default: L1Full — the optimizer attempts to fit the entire activation in L1
// (best performance when it fits).
//
// Guard: switch to DramHeight when the activation, if height-sharded across
// all 64 cores in TILE-format bf16, would exceed the per-core usable L1.
// Without this guard the mock-device validation in
// OperationValidationAndFallback may incorrectly mark L1Full as valid, only
// for the real device to TT_FATAL at runtime.
//
// Example that triggers the guard:
//   UV downsampling depthwise conv2d: in_channels=2, H=1280, W=2304
//   Activation in TILE: 1280 × 2304 × 32 × 2 B = 188 MB total
//   Per core (64):      2,949,120 B  >  1,329,888 B (usable L1)  → OOM
void applyConvSliceConfig(ModuleOp moduleOp) {
  moduleOp->walk([](Conv2dOp conv2dOp) {
    Conv2dSliceType sliceType = Conv2dSliceType::L1Full;

    if (auto chipDesc = ttcore::getOpChipDescAttr(conv2dOp)) {
      uint64_t l1PerCore    = chipDesc.getUsableL1Size();
      uint64_t perCoreBytes = computeActBytesPerCore(conv2dOp);

      if (perCoreBytes > l1PerCore) {
        sliceType = Conv2dSliceType::DramHeight;
        TTMLIR_DEBUG(
            ttmlir::LogComponent::GreedyOptimizer,
            "applyConvSliceConfig: Conv2d (H={}, W={}, C={}) activation "
            "{}B/core exceeds L1 {}B — using DramHeight",
            conv2dOp.getInputHeight(), conv2dOp.getInputWidth(),
            conv2dOp.getInChannels(), perCoreBytes, l1PerCore);
      }
    }

    conv2dOp.setConv2dSliceConfigAttr(
        Conv2dSliceConfigAttr::get(conv2dOp.getContext(), sliceType, 0));
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
