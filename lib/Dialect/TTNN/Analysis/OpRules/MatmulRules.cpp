// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>

namespace mlir::tt::ttnn {

// ============================================================================
// DRAM-sharded matmul policy constants
// ============================================================================
//
// The shard geometry, layout, and config *generation* lives in
// MatmulProgramConfig.{h,cpp} (computeShardParams, buildDRAMSharded*). These
// constants are the rule book's policy inputs: they drive eligibility and are
// passed into computeShardParams as numBanks / numIn0Cores.

static constexpr int64_t kTileSize = 32;
static constexpr int64_t kNumDRAMBanks = 12;
// Single source of truth for how many cores the DS-matmul activation (in0) is
// width-sharded across. Drives the in0 shard width, the K-divisibility
// eligibility gate, and the in0 L1 tensor-buffer reservation in
// computeShardParams. Keep these uses consistent.
static constexpr int64_t kNumIn0Cores = 8;

// ============================================================================
// Eligibility helpers
// ============================================================================

static bool isBfpDRAMInterleaved(Value weight) {
  auto rtt = mlir::dyn_cast<RankedTensorType>(weight.getType());
  if (!rtt) {
    return false;
  }
  auto elType = rtt.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elType)) {
    auto dt = tileType.getDataType();
    if (dt != ttcore::DataType::BFP_BFloat8 &&
        dt != ttcore::DataType::BFP_BFloat4) {
      return false;
    }
  } else {
    return false;
  }
  auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rtt.getEncoding());
  if (!layoutAttr) {
    return false;
  }
  return layoutAttr.hasInterleavedDRAMTensorMemoryLayout();
}

static ttcore::DataType getWeightDataType(Value weight) {
  auto rtt = mlir::cast<RankedTensorType>(weight.getType());
  auto tileType = mlir::cast<ttcore::TileType>(rtt.getElementType());
  return tileType.getDataType();
}

static std::pair<int64_t, int64_t> getWeightKN(RankedTensorType rtt) {
  auto shape = rtt.getShape();
  assert(shape.size() == 2 && "Expected 2D weight tensor");
  return {shape[0], shape[1]};
}

static int64_t getActivationM(RankedTensorType rtt) {
  int64_t M = 1;
  for (int64_t dim : rtt.getShape().drop_back()) {
    M *= dim;
  }
  return M;
}

static bool isDRAMShardEligible(MatmulOp matmulOp) {
  Value weight = matmulOp.getB();

  if (!isBfpDRAMInterleaved(weight)) {
    return false;
  }
  if (!ttcore::valueTracesToConstantArgs(weight)) {
    return false;
  }
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  if (weightType.getRank() != 2) {
    return false;
  }

  auto in0Type = mlir::cast<RankedTensorType>(matmulOp.getA().getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);

  if (M % kTileSize != 0 || K % kTileSize != 0 || N % kTileSize != 0) {
    return false;
  }
  // K is the contraction dim, width-sharded across the in0 cores, so it must
  // divide evenly by the in0 core count (same requirement computeShardParams
  // enforces via kTiles % numIn0Cores). Gate on it here so an ineligible op is
  // rejected up front rather than deep in shard-param computation.
  if ((K / kTileSize) % kNumIn0Cores != 0) {
    return false;
  }
  // Decode-only: factory asserts per_core_M == 1 when num_blocks_per_shard > 1.
  if (M / kTileSize > 1) {
    return false;
  }

  return true;
}

// ============================================================================
// MatmulRuleBook — existing helpers
// ============================================================================

static bool isL1Interleaved(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  auto memLayout = config.outputLayout.getMemLayout();
  return config.outputLayout.getBufferType() == BufferType::L1 && memLayout &&
         memLayout.getValue() == TensorMemoryLayout::Interleaved;
}

static bool isSharded(const OpConfig &config) {
  if (!config.outputLayout) {
    return false;
  }
  auto memLayout = config.outputLayout.getMemLayout();
  return memLayout && isShardedMemoryLayout(memLayout.getValue());
}

static bool hasMatmulProgramConfig(const OpConfig &config) {
  if (const auto *attrs = std::get_if<MatmulAttrs>(&config.opSpecificAttrs)) {
    return attrs->matmulProgramConfig.has_value() &&
           attrs->matmulProgramConfig.value();
  }
  return false;
}

// ============================================================================
// MatmulRuleBook — private DRAM-sharding helpers
// ============================================================================
//
// buildDRAMShardingHint produces the DS output hint consumed by getOutputHints;
// applyDRAMShardedTransformation rewrites the op at apply time and is called by
// applyOpSpecificAttrs. Both are defined ahead of their callers below.

std::optional<OpConfig>
MatmulRuleBook::buildDRAMShardingHint(Operation *op) const {
  auto matmulOp = dyn_cast<MatmulOp>(op);
  if (!matmulOp) {
    return std::nullopt;
  }

  if (!isDRAMShardEligible(matmulOp)) {
    return std::nullopt;
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return std::nullopt;
  }
  auto systemDescAttr = moduleOp->getAttr(ttcore::SystemDescAttr::name);
  if (!systemDescAttr) {
    return std::nullopt;
  }
  auto systemDesc = mlir::cast<ttcore::SystemDescAttr>(systemDescAttr);
  int64_t l1Available =
      static_cast<int64_t>(ttnn::utils::getTensorL1UsageCap(moduleOp) *
                           systemDesc.getChipDescs()[0].getUsableL1Size());

  auto in0Type = mlir::cast<RankedTensorType>(matmulOp.getA().getType());
  auto weightType = mlir::cast<RankedTensorType>(matmulOp.getB().getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  auto weightDataType = getWeightDataType(matmulOp.getB());

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  int64_t numAvailableCores =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());

  auto pOpt =
      computeShardParams(M, K, N, kNumDRAMBanks, kNumIn0Cores,
                         numAvailableCores, weightDataType, l1Available);
  if (!pOpt) {
    return std::nullopt;
  }
  const auto &p = *pOpt;

  auto *ctx = op->getContext();
  auto outLayout = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());
  auto resultType = mlir::cast<RankedTensorType>(op->getResult(0).getType());

  // numOutputCores = div_up(N_tiles, per_core_N_storage): exactly how many
  // output cores compute_output_specs will allocate, ensuring no assertion
  // fire.
  int64_t numOutputCores = (N / kTileSize + p.perCoreN - 1) / p.perCoreN;

  llvm::SmallVector<int64_t, 2> outputGrid = {1, numOutputCores};
  TTNNLayoutAttr l1OutLayout =
      TTNNLayoutAttr::Builder(outLayout, resultType.getShape())
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayoutAttr::get(
              ctx, TensorMemoryLayout::WidthSharded))
          .setGridShape(outputGrid)
          .buildWithCanonicalCorePlacement(deviceAttr);

  // Activation is handled as a separate elementwise op after the DS matmul
  // (see applyDRAMShardedTransformation). Fusing it into the DS kernel is
  // significantly slower. The op model is told no activation so it validates
  // the DS config cleanly; the activation attribute on the op is stripped and
  // a separate op is inserted at apply time.
  UnaryWithParamAttr fusedAct;
  auto progConfig = buildDRAMShardedProgramConfig(ctx, p, fusedAct);
  auto computeConfig = buildComputeConfig(ctx, weightDataType);

  return OpConfig(l1OutLayout, MatmulAttrs{progConfig, computeConfig});
}

void MatmulRuleBook::applyDRAMShardedTransformation(
    MatmulOp matmulOp, const MatmulAttrs &matmulAttrs) const {
  auto *ctx = matmulOp.getContext();
  // Input reshards (activation → L1 1×8, weight → DRAM 1×12) are handled by
  // pass-2 in applyToIR via reshardLayouts populated from the input candidates
  // injected by getExtraInputReshardCandidates.

  OpBuilder builder(matmulOp);

  // --- 1. Set program config and compute config ---
  matmulOp.setMatmulProgramConfigAttr(
      mlir::cast<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          matmulAttrs.matmulProgramConfig.value()));

  if (matmulAttrs.computeKernelConfig.has_value()) {
    matmulOp.setComputeConfigAttr(*matmulAttrs.computeKernelConfig);
  }

  // --- 2. Strip activation, insert separate elementwise op after ---
  // The DS kernel with fused activation is significantly slower than a
  // separate elementwise op running across all cores with full parallelism.
  auto activationAttr = matmulOp.getActivationAttr();
  if (activationAttr) {
    matmulOp.removeActivationAttr();

    auto actStr = activationAttr.getValue();
    StringRef opName;
    if (actStr == "silu") {
      opName = "ttnn.silu";
    } else if (actStr == "relu") {
      opName = "ttnn.relu";
    } else if (actStr == "gelu") {
      opName = "ttnn.gelu";
    }
    if (!opName.empty()) {
      builder.setInsertionPointAfter(matmulOp);
      Value matmulResult = matmulOp.getResult();
      auto *activationOp = builder.create(
          matmulOp.getLoc(), StringAttr::get(ctx, opName),
          ValueRange{matmulResult}, TypeRange{matmulResult.getType()});
      matmulResult.replaceAllUsesExcept(activationOp->getResult(0),
                                        activationOp);
    }
  }
}

// ============================================================================
// MatmulRuleBook::getOutputHints
// ============================================================================

OutputHints MatmulRuleBook::getOutputHints(
    Operation *op, const std::vector<OpConfig> &legalConfigs) const {

  auto partialConfigs =
      optimizer_utils::getUniqueTestConfigsForMatmulLinear(legalConfigs);

  // Filter out L1-interleaved and sharded configs without a program config.
  std::vector<OpConfig> filtered;
  for (const auto &cfg : partialConfigs) {
    if (isL1Interleaved(cfg)) {
      continue;
    }
    // Skip sharded outputs when no MatmulProgramConfig is available.
    //
    // Without a program config, tt-metal's runtime auto-picker
    // (create_simple_matmul_program_config) is non-idempotent due to allocator
    // dependency. At compile time, validation invokes the autopicker which may
    // emit a captured output spec on grid G1 (e.g. 5x6), which we adopt into
    // the IR. At runtime, the matmul is re-invoked with that adopted G1 spec,
    // the autopicker re-runs against G1 and can pick a different mcast path and
    // per_core_M/N pair, producing a new grid G2.
    if (isSharded(cfg) && !hasMatmulProgramConfig(cfg)) {
      continue;
    }
    filtered.push_back(cfg);
  }

  // Prepend the DS hint when eligible. adjustScore gives it priority over the
  // normal hints via isDRAMShardedCandidate; normal hints remain as fallback
  // in case DS validation fails for a given input combination.
  if (auto dramHint = buildDRAMShardingHint(op)) {
    filtered.insert(filtered.begin(), *dramHint);
  }

  return OutputHints{filtered, {}};
}

// ============================================================================
// MatmulRuleBook::getInputLayoutFilter
// ============================================================================

LayoutFilterFn MatmulRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  // Weight (operand 1): reject L1.
  // DRAM WIDTH_SHARDED is allowed for the DRAM-sharded matmul path.
  // DRAM interleaved is always allowed.
  if (operandIdx == 1) {
    return layout_filter_utils::rejectAllL1;
  }
  return nullptr;
}

// ============================================================================
// MatmulRuleBook::applyOpSpecificAttrs
// ============================================================================

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

  // DRAM-sharded path: weight reshard, program/compute config, activation
  // split.
  bool isDRAMSharded =
      mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          programConfig);
  if (isDRAMSharded && matmulOp) {
    applyDRAMShardedTransformation(matmulOp, matmulAttrs);
    return;
  }

  // Non-DRAM-sharded path: set program config, handle fused activation dedup.
  auto setConfigAndFixup = [&](auto concreteOp) {
    concreteOp.setMatmulProgramConfigAttr(programConfig);
    // Workaround for tt-metal issue #35060: if the program config carries a
    // fused activation, remove the op-level activation attr to prevent
    // double application.
    bool hasFusedActivation =
        llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
            .template Case<MatmulMultiCoreReuseMultiCastProgramConfigAttr,
                           MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
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

// ============================================================================
// MatmulRuleBook::isValidOutputHintForInputs
// ============================================================================

// Reject in0 whose shard width is incompatible with the config's in0_block_w.
// tt-metal needs (tiles): K % per_core_K == 0 and per_core_K % in0_block_w ==
// 0. Guards all in0 candidates the cross-product pairs with the DS hint; though
// our injected in0 is valid by construction. tt-metal should be patched to
// reject a bad combo catchably — until then it TT_FATALs (uncatchable abort),
// so we must gate here. per_core_K = in0 shard width (tiles); K (tiles) = in1
// shard height.
static bool dsIn0CompatibleWithConfig(
    TTNNLayoutAttr in0, TTNNLayoutAttr in1,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr dsCfg) {
  auto in0Shard = in0.getShardShape();
  auto in1Shard = in1.getShardShape();
  if (in0Shard.size() != 2 || in1Shard.size() != 2) {
    // Cannot read the shard width, so cannot verify the combo is legal. A
    // width-sharded in0/in1 for these matmuls is always 2-D (and our injected
    // in0 always is), so reject rather than risk a tt-metal abort.
    return false;
  }
  int64_t perCoreK = in0Shard[1];
  int64_t kTiles = in1Shard[0];
  int64_t in0BlockW = static_cast<int64_t>(dsCfg.getIn0BlockW());
  return perCoreK != 0 && in0BlockW != 0 && kTiles % perCoreK == 0 &&
         perCoreK % in0BlockW == 0;
}

bool MatmulRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  const auto *attrs = std::get_if<MatmulAttrs>(&hint.opSpecificAttrs);
  if (!attrs || !attrs->matmulProgramConfig.has_value() ||
      !mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          attrs->matmulProgramConfig.value())) {
    return true;
  }
  // DS hint: only the canonical DS input combination is valid — L1
  // width-sharded in0, DRAM width-sharded in1, with an in0 shard width
  // compatible with the config's in0_block_w (see dsIn0CompatibleWithConfig).
  // This runs for every in0 the cross-product pairs with the DS hint, not just
  // the one we inject in getExtraInputReshardCandidates (that one is valid by
  // construction).
  if (inputLayouts.size() < 2) {
    return false;
  }
  auto in0 = inputLayouts[0];
  auto in1 = inputLayouts[1];
  if (!in0 || !in1) {
    return false;
  }
  auto ml0 = in0.getMemLayoutOpt();
  if (!in0.hasL1BufferType() || !ml0 ||
      *ml0 != TensorMemoryLayout::WidthSharded) {
    return false;
  }
  auto ml1 = in1.getMemLayoutOpt();
  if (in1.hasL1BufferType() || !ml1 ||
      *ml1 != TensorMemoryLayout::WidthSharded) {
    return false;
  }
  auto dsCfg =
      mlir::cast<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          attrs->matmulProgramConfig.value());
  return dsIn0CompatibleWithConfig(in0, in1, dsCfg);
}

// ============================================================================
// MatmulRuleBook::adjustScore
// ============================================================================

LayoutScore
MatmulRuleBook::adjustScore(Operation * /*op*/, LayoutScore base,
                            const OpConfig &config,
                            llvm::ArrayRef<TTNNLayoutAttr> inputLayouts,
                            bool /*requiresReshard*/) const {
  const auto *attrs = std::get_if<MatmulAttrs>(&config.opSpecificAttrs);
  if (!attrs || !attrs->matmulProgramConfig.has_value() ||
      !attrs->matmulProgramConfig.value()) {
    return base;
  }
  if (!mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          attrs->matmulProgramConfig.value())) {
    return base;
  }
  base.isDRAMShardedCandidate = true;
  if (!inputLayouts.empty()) {
    auto in0 = inputLayouts[0];
    if (in0 && in0.hasL1BufferType()) {
      auto ml = in0.getMemLayoutOpt();
      if (ml && *ml == TensorMemoryLayout::WidthSharded) {
        auto shape = in0.getGridShape();
        if (shape.size() == 2 && shape[0] == 1 &&
            shape[1] == kNumIn0Cores) {
          base.hasCanonicalDSIn0 = true;
        }
      }
    }
  }
  return base;
}

// ============================================================================
// MatmulRuleBook::getExtraInputReshardCandidates
// ============================================================================

std::vector<TTNNLayoutAttr>
MatmulRuleBook::getExtraInputReshardCandidates(Operation *op,
                                               unsigned operandIdx) const {
  auto matmulOp = dyn_cast<MatmulOp>(op);
  if (!matmulOp || !isDRAMShardEligible(matmulOp)) {
    return {};
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    return {};
  }
  auto systemDescAttr = moduleOp->getAttr(ttcore::SystemDescAttr::name);
  if (!systemDescAttr) {
    return {};
  }
  auto systemDesc = mlir::cast<ttcore::SystemDescAttr>(systemDescAttr);
  int64_t l1Available =
      static_cast<int64_t>(ttnn::utils::getTensorL1UsageCap(moduleOp) *
                           systemDesc.getChipDescs()[0].getUsableL1Size());

  auto in0Type = mlir::cast<RankedTensorType>(matmulOp.getA().getType());
  auto weightType = mlir::cast<RankedTensorType>(matmulOp.getB().getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  auto weightDataType = getWeightDataType(matmulOp.getB());

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  int64_t numAvailCores =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());
  auto pOpt = computeShardParams(M, K, N, kNumDRAMBanks, kNumIn0Cores,
                                 numAvailCores, weightDataType, l1Available);
  if (!pOpt) {
    return {};
  }
  const auto &p = *pOpt;

  auto *ctx = op->getContext();
  if (operandIdx == 0) {
    auto in0Layout = mlir::cast<TTNNLayoutAttr>(in0Type.getEncoding());
    return {buildL1ShardedLayout(ctx, in0Layout, in0Type.getShape(),
                                 kNumIn0Cores, deviceAttr)};
  }
  if (operandIdx == 1) {
    auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
    return {buildDRAMShardedWeightLayout(ctx, weightLayout, p)};
  }
  return {};
}

} // namespace mlir::tt::ttnn
