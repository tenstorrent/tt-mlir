// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>
#include <optional>

namespace mlir::tt::ttnn {

// ============================================================================
// DRAM-sharded matmul policy constants
// ============================================================================

// Cores the activation (in0) is width-sharded across. Experimental: tt-metal
// only requires K tiles / cores to be a multiple of in0_block_w, and 8 divides
// every targeted hidden size while leaving room for a wide in0_block_w.
static constexpr int64_t kNumIn0Cores = 8;

static bool isWidthSharded(TTNNLayoutAttr layout) {
  auto ml = layout.getMemLayoutOpt();
  return ml && *ml == TensorMemoryLayout::WidthSharded;
}

// L1 width-sharded over a single row of `numCores`.
static bool isL1WidthShardedOn(TTNNLayoutAttr layout, int64_t numCores) {
  if (!layout.hasL1BufferType() || !isWidthSharded(layout)) {
    return false;
  }
  auto grid = layout.getGridShape();
  return grid.size() == 2 && grid[0] == 1 && grid[1] == numCores;
}

// The DS program config `config` carries, or null.
static MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
getDSProgramConfig(const OpConfig &config) {
  const auto *attrs = std::get_if<MatmulAttrs>(&config.opSpecificAttrs);
  if (!attrs || !attrs->matmulProgramConfig.has_value()) {
    return nullptr;
  }
  return mlir::dyn_cast_or_null<
      MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
      attrs->matmulProgramConfig.value());
}

// Weight layouts the matmul accepts: interleaved anywhere, as for any matmul,
// plus the DRAM width-sharded layout the DS reshard candidate injects.
static bool acceptWeightLayout(TTNNLayoutAttr layout) {
  return layout_filter_utils::rejectAllSharded(layout) ||
         (!layout.hasL1BufferType() && isWidthSharded(layout));
}

// DRAM banks the weight is width-sharded across: every bank the device exposes,
// read from the same grid canonical DRAM placement uses. Nullopt when that grid
// is not a single row, which canonical placement cannot express.
static std::optional<int64_t> getNumDRAMBanks(ttcore::DeviceAttr deviceAttr) {
  if (!deviceAttr) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> dramGrid = deviceAttr.getDramGrid().getShape();
  if (dramGrid.size() != 2 || dramGrid[0] != 1 || dramGrid[1] < 1) {
    return std::nullopt;
  }
  return dramGrid[1];
}

// ============================================================================
// Eligibility helpers
// ============================================================================

static TTNNLayoutAttr getWeightLayout(Value weight) {
  auto rtt = mlir::dyn_cast<RankedTensorType>(weight.getType());
  if (!rtt) {
    return {};
  }
  return mlir::dyn_cast_or_null<TTNNLayoutAttr>(rtt.getEncoding());
}

// bfp4/bfp8 only: a heuristic against DRAM OOM, not a tt-metal constraint.
static bool isBfpDRAMInterleaved(Value weight) {
  TTNNLayoutAttr layout = getWeightLayout(weight);
  if (!layout || !layout.isTiled()) {
    return false;
  }
  ttcore::DataType dt = layout.getDataType();
  if (dt != ttcore::DataType::BFP_BFloat8 &&
      dt != ttcore::DataType::BFP_BFloat4) {
    return false;
  }
  return layout.hasInterleavedDRAMTensorMemoryLayout();
}

// A 2-D matrix, optionally behind unit batch dims. A non-unit batch dim needs
// the batched DS config, which this path does not emit.
static bool isDSWeightShaped(RankedTensorType rtt) {
  llvm::ArrayRef<int64_t> shape = rtt.getShape();
  if (shape.size() < 2) {
    return false;
  }
  return llvm::all_of(shape.drop_back(2), [](int64_t dim) { return dim == 1; });
}

static std::pair<int64_t, int64_t> getWeightKN(RankedTensorType rtt) {
  llvm::ArrayRef<int64_t> shape = rtt.getShape();
  assert(isDSWeightShaped(rtt) && "Expected a 2-D (or unit-batched) weight");
  return {shape[shape.size() - 2], shape[shape.size() - 1]};
}

static int64_t getActivationM(RankedTensorType rtt) {
  int64_t M = 1;
  for (int64_t dim : rtt.getShape().drop_back()) {
    M *= dim;
  }
  return M;
}

// (activation, weight) of a bias-free, non-transposed matmul or linear. A bias
// is declined: the DS kernel reads it per DRAM bank, so it needs a DRAM
// width-sharded layout nothing produces yet, and an interleaved one is read
// wrong silently.
static std::optional<std::pair<Value, Value>> getMatmulOperands(Operation *op) {
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    if (matmulOp.getTransposeA() || matmulOp.getTransposeB()) {
      return std::nullopt;
    }
    return std::make_pair(matmulOp.getA(), matmulOp.getB());
  }
  if (auto linearOp = dyn_cast<LinearOp>(op)) {
    if (linearOp.getBias() || linearOp.getTransposeA() ||
        linearOp.getTransposeB()) {
      return std::nullopt;
    }
    return std::make_pair(linearOp.getA(), linearOp.getB());
  }
  return std::nullopt;
}

// Logs the reason on every decline.
static bool isDSEligible(Operation *op, Value activation, Value weight) {
  [[maybe_unused]] StringRef opName = op->getName().getStringRef();

  if (!isBfpDRAMInterleaved(weight)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight is not a tiled bfp4/bfp8 "
                 "DRAM-interleaved tensor",
                 opName);
    return false;
  }
  if (!ttcore::valueTracesToConstantArgs(weight)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight does not trace to constant args",
                 opName);
    return false;
  }
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  if (!isDSWeightShaped(weightType)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight rank {1} has a non-unit batch dim, "
                 "so it is not a 2-D (optionally unit-batched) matrix",
                 opName, weightType.getRank());
    return false;
  }

  auto in0Type = mlir::cast<RankedTensorType>(activation.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);

  if (K % TILE_WIDTH != 0 || N % TILE_WIDTH != 0) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): K={1} / N={2} not tile-aligned", opName, K,
                 N);
    return false;
  }
  // K is width-sharded across the in0 cores.
  if ((K / TILE_WIDTH) % kNumIn0Cores != 0) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): K tiles {1} not divisible by the in0 core "
                 "count {2}",
                 opName, K / TILE_WIDTH, kNumIn0Cores);
    return false;
  }
  // tt-metal asserts M == 1 tile row, uncatchably, so decline taller here. A
  // sub-tile batch pads up to one row and is accepted.
  if (llvm::divideCeil(M, TILE_HEIGHT) != 1) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::GreedyOptimizer,
        "DS declined ({0}): activation M={1} is more than one tile row", opName,
        M);
    return false;
  }

  return true;
}

// What the shard geometry needs from the device and the system descriptor.
struct DSDeviceContext {
  ttcore::DeviceAttr deviceAttr;
  int64_t numDRAMBanks;
  int64_t numWorkerCores;
  int64_t l1Available;
};

static std::optional<DSDeviceContext> getDSDeviceContext(Operation *op) {
  [[maybe_unused]] StringRef opName = op->getName().getStringRef();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): op has no parent module", opName);
    return std::nullopt;
  }
  auto systemDescAttr = moduleOp->getAttr(ttcore::SystemDescAttr::name);
  if (!systemDescAttr) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): module has no system descriptor", opName);
    return std::nullopt;
  }

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  std::optional<int64_t> numDRAMBanks = getNumDRAMBanks(deviceAttr);
  if (!numDRAMBanks) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): device DRAM grid is not a single row that "
                 "canonical DRAM placement can express",
                 opName);
    return std::nullopt;
  }

  int64_t l1Available =
      static_cast<int64_t>(ttnn::utils::getUsableL1PerCore(op));

  return DSDeviceContext{
      deviceAttr, *numDRAMBanks,
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape()),
      l1Available};
}

// Operand types and shard geometry. The output hint and the input reshard
// candidates are built from the same plan, so they cannot disagree.
struct DSPlan {
  RankedTensorType in0Type;
  RankedTensorType weightType;
  DRAMShardParams params;
  ttcore::DeviceAttr deviceAttr;
};

// Rebuilt on every query: the optimizer rewrites operand layouts between calls,
// so a cache would go stale.
static std::optional<DSPlan> buildDSPlan(Operation *op) {
  [[maybe_unused]] StringRef opName = op->getName().getStringRef();

  // The single choke point of the DS path: every entry needs a plan.
  if (!ttnn::utils::isDRAMShardedMatmulEnabled(op)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): enable-dram-sharded-matmul is off",
                 opName);
    return std::nullopt;
  }

  // A DS program config carries no fused_activation, so an op-level one would
  // be validated without and run with.
  StringAttr fusedActivation =
      llvm::TypeSwitch<Operation *, StringAttr>(op)
          .Case<MatmulOp, LinearOp>(
              [](auto concreteOp) { return concreteOp.getActivationAttr(); })
          .Default([](Operation *) { return StringAttr(); });
  if (fusedActivation) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): op carries a fused activation '{1}'",
                 opName, fusedActivation.getValue());
    return std::nullopt;
  }

  std::optional<std::pair<Value, Value>> operands = getMatmulOperands(op);
  if (!operands) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): not a bias-free, non-transposed "
                 "matmul/linear",
                 opName);
    return std::nullopt;
  }
  auto [activation, weight] = *operands;

  if (!isDSEligible(op, activation, weight)) {
    return std::nullopt;
  }
  std::optional<DSDeviceContext> device = getDSDeviceContext(op);
  if (!device) {
    return std::nullopt;
  }

  auto in0Type = mlir::cast<RankedTensorType>(activation.getType());
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);

  std::optional<DRAMShardParams> params = computeShardParams(
      M, K, N, device->numDRAMBanks, kNumIn0Cores, device->numWorkerCores,
      getWeightLayout(weight).getDataType(), device->l1Available);
  if (!params) {
    // No in0_block_w both fits L1 and clears the floor; see computeShardParams.
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): no in0_block_w both fits L1 and avoids a "
                 "degenerate block count (M={1} K={2} "
                 "N={3} banks={4} in0Cores={5} cores={6} l1Available={7})",
                 opName, M, K, N, device->numDRAMBanks, kNumIn0Cores,
                 device->numWorkerCores, device->l1Available);
    return std::nullopt;
  }

  return DSPlan{in0Type, weightType, *params, device->deviceAttr};
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

std::optional<OpConfig>
MatmulRuleBook::buildDRAMShardingHint(Operation *op) const {
  std::optional<DSPlan> plan = buildDSPlan(op);
  if (!plan) {
    return std::nullopt;
  }
  const DRAMShardParams &p = plan->params;
  ttcore::DeviceAttr deviceAttr = plan->deviceAttr;

  auto *ctx = op->getContext();
  auto outLayout = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());
  auto resultType = mlir::cast<RankedTensorType>(op->getResult(0).getType());

  // The storage grid tt-metal allocates for the output.
  int64_t numOutputCores = llvm::divideCeil(p.nTiles, p.perCoreNStorage);

  llvm::SmallVector<int64_t, 2> outputGrid = {1, numOutputCores};
  TTNNLayoutAttr l1OutLayout =
      TTNNLayoutAttr::Builder(outLayout, resultType.getShape())
          .setBufferType(BufferType::L1)
          .setMemoryLayout(TensorMemoryLayoutAttr::get(
              ctx, TensorMemoryLayout::WidthSharded))
          .setGridShape(outputGrid)
          .buildWithCanonicalCorePlacement(deviceAttr);

  // buildDSPlan declines a fused activation, so there is none to pass.
  UnaryWithParamAttr fusedAct;
  auto progConfig = buildDRAMShardedProgramConfig(ctx, p, fusedAct);
  auto computeConfig = buildComputeConfig(ctx, p.weightDataType);

  return OpConfig(l1OutLayout, MatmulAttrs{progConfig, computeConfig});
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

  // DS hint first; the others stay as fallback.
  if (auto dramHint = buildDRAMShardingHint(op)) {
    filtered.insert(filtered.begin(), *dramHint);
  }

  return OutputHints{filtered, {}};
}

// ============================================================================
// MatmulRuleBook::getInputLayoutFilter
// ============================================================================

LayoutFilterFn MatmulRuleBook::getInputLayoutFilter(unsigned operandIdx) const {
  if (operandIdx == 1) {
    return acceptWeightLayout;
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

  // DS: program/compute config only. The operand reshards come from the input
  // candidates getExtraInputReshardCandidates injects.
  bool isDRAMSharded =
      mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          programConfig);
  if (isDRAMSharded) {
    auto setDSConfig = [&](auto concreteOp) {
      concreteOp.setMatmulProgramConfigAttr(programConfig);
      if (matmulAttrs.computeKernelConfig.has_value()) {
        concreteOp.setComputeConfigAttr(*matmulAttrs.computeKernelConfig);
      }
    };
    if (matmulOp) {
      setDSConfig(matmulOp);
    } else {
      setDSConfig(linearOp);
    }
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

// tt-metal needs, in tiles, K % per_core_K == 0 and per_core_K % in0_block_w
// == 0, with per_core_K the in0 shard width and K the in1 shard height.
static bool dsIn0CompatibleWithConfig(
    TTNNLayoutAttr in0, TTNNLayoutAttr in1,
    MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr dsCfg) {
  auto in0Shard = in0.getShardShape();
  auto in1Shard = in1.getShardShape();
  if (in0Shard.size() != 2 || in1Shard.size() != 2) {
    // Not 2-D: the check cannot be made, so reject rather than risk an abort.
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
  auto dsCfg = getDSProgramConfig(hint);
  if (!dsCfg) {
    return true;
  }
  // Every in0 the cross-product pairs with the hint passes through here.
  if (inputLayouts.size() < 2 || !inputLayouts[0] || !inputLayouts[1]) {
    return false;
  }
  auto in0 = inputLayouts[0];
  auto in1 = inputLayouts[1];
  if (!in0.hasL1BufferType() || !isWidthSharded(in0)) {
    return false;
  }
  if (in1.hasL1BufferType() || !isWidthSharded(in1)) {
    return false;
  }
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
  if (!getDSProgramConfig(config)) {
    return base;
  }
  // DS above every other L1-sharded candidate; within DS, an in0 already on
  // the canonical grid above one that needs a reshard.
  bool canonicalIn0 = !inputLayouts.empty() && inputLayouts[0] &&
                      isL1WidthShardedOn(inputLayouts[0], kNumIn0Cores);
  base.rulePreference = canonicalIn0 ? 2 : 1;
  return base;
}

// ============================================================================
// MatmulRuleBook::getExtraInputReshardCandidates
// ============================================================================

std::vector<TTNNLayoutAttr>
MatmulRuleBook::getExtraInputReshardCandidates(Operation *op,
                                               unsigned operandIdx) const {
  std::optional<DSPlan> plan = buildDSPlan(op);
  if (!plan) {
    return {};
  }

  auto *ctx = op->getContext();
  if (operandIdx == 0) {
    auto in0Layout = mlir::cast<TTNNLayoutAttr>(plan->in0Type.getEncoding());
    return {buildWidthShardedLayout(ctx, in0Layout, plan->in0Type.getShape(),
                                    BufferType::L1, kNumIn0Cores,
                                    plan->deviceAttr)};
  }
  if (operandIdx == 1) {
    auto weightLayout =
        mlir::cast<TTNNLayoutAttr>(plan->weightType.getEncoding());
    return {buildWidthShardedLayout(
        ctx, weightLayout, plan->weightType.getShape(), BufferType::DRAM,
        plan->params.numBanks, plan->deviceAttr)};
  }
  return {};
}

} // namespace mlir::tt::ttnn
