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
//
// The shard geometry, layout, and config *generation* lives in
// MatmulProgramConfig.{h,cpp} (computeShardParams, buildDRAMSharded*). These
// are the rule book's policy inputs: they drive eligibility and are passed into
// computeShardParams as numBanks / numIn0Cores.

static constexpr int64_t kTileSize = 32;
// Single source of truth for how many cores the DS-matmul activation (in0) is
// width-sharded across. Drives the in0 shard width, the K-divisibility
// eligibility gate, and the in0 L1 tensor-buffer reservation in
// computeShardParams. Keep these uses consistent.
static constexpr int64_t kNumIn0Cores = 8;

// Number of DRAM banks the weight is width-sharded across: every bank the
// device exposes. Deliberately the same source deriveCanonicalDramCoreRangeSet
// reads to build the layout's core range set, so the bank count and the
// placement cannot disagree.
//
// Returns nullopt when the grid is not a shape canonical DRAM placement can
// express (it requires a single row), so the DS path declines instead of
// tripping the assert inside deriveCanonicalDramCoreRangeSet.
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

// The weight's TTNN layout, or null when its type does not carry one. The
// layout is where the on-device tiling and dtype live, so it is the single
// thing the checks below need.
static TTNNLayoutAttr getWeightLayout(Value weight) {
  auto rtt = mlir::dyn_cast<RankedTensorType>(weight.getType());
  if (!rtt) {
    return {};
  }
  return mlir::dyn_cast_or_null<TTNNLayoutAttr>(rtt.getEncoding());
}

// Whether the weight is a tiled, DRAM-interleaved tensor of a data type the DS
// kernel is offered for.
//
// bfp4/bfp8 only. tt-metal imposes no dtype constraint on the DS config (there
// is no dtype TT_FATAL in the DRAM-sharded validation block), so this is a
// heuristic trying to prevent DRAM OOM, not a legality check.
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

// A weight is DS-shaped when it is a plain 2-D matrix, possibly carrying
// leading unit batch dims: ttnn models routinely hold projection weights as [1,
// 1, K, N], which is the same matrix as [K, N] — same element count, same tile
// grid, and TTNN already collapses both to a 2-D memref in the layout.
//
// A non-unit leading dim is a genuinely batched matmul, which this path does
// not emit: tt-metal serves that case with a different config,
// MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig.
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

// The (activation, weight) pair for a DS-capable matmul-like op, or nullopt if
// `op` is not one.
//
// The DS path covers ttnn.matmul and ttnn.linear alike: a bias-free linear IS
// the matmul the DS kernel implements, and ttnn decoders write their
// projections as ttnn.linear, so both have to be recognized here.
//
// A bias is rejected even though tt-metal's DS kernel supports one. The kernel
// reads the bias per DRAM bank at a bank-local offset
// (reader_bmm_tile_layout_in1_sender_dram_sharded.cpp, in3) and adds it on the
// DRAM-bank compute cores, so the bias must be DRAM width-sharded across the
// same bank grid as the weight. Nothing in this rule book produces such a
// layout for operand 2 yet, and a DRAM-interleaved bias would be read as a
// strided subset of itself — wrong results, not a crash, and tt-metal validates
// no bias memory config for DS. Allow it only together with a bias layout
// builder. Should check if bias would be bringing perf regression.
static std::optional<std::pair<Value, Value>> getDSOperands(Operation *op) {
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

static std::optional<std::pair<Value, Value>>
getDSEligibleOperands(Operation *op) {
  // Respect the disable-dram-sharded-matmul pipeline option (set as a module
  // attribute by DevicePassesWrapper). This is the single choke point for the
  // DS path: buildDRAMShardingHint and getExtraInputReshardCandidates both gate
  // on it, and getOutputHints reaches DS only through buildDRAMShardingHint.
  if (ttnn::utils::isDRAMShardedMatmulDisabled(op)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): disabled by disable-dram-sharded-matmul",
                 op->getName().getStringRef());
    return std::nullopt;
  }

  std::optional<std::pair<Value, Value>> operands = getDSOperands(op);
  if (!operands) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): not a bias-free, non-transposed "
                 "matmul/linear",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  auto [activation, weight] = *operands;

  if (!isBfpDRAMInterleaved(weight)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight is not a tiled bfp4/bfp8 "
                 "DRAM-interleaved tensor",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  if (!ttcore::valueTracesToConstantArgs(weight)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight does not trace to constant args",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  if (!isDSWeightShaped(weightType)) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): weight rank {1} has a non-unit batch dim, "
                 "so it is not a 2-D (optionally unit-batched) matrix",
                 op->getName().getStringRef(), weightType.getRank());
    return std::nullopt;
  }

  auto in0Type = mlir::cast<RankedTensorType>(activation.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);

  if (K % kTileSize != 0 || N % kTileSize != 0) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): K={1} / N={2} not tile-aligned",
                 op->getName().getStringRef(), K, N);
    return std::nullopt;
  }
  // K is the contraction dim, width-sharded across the in0 cores, so it must
  // divide evenly by the in0 core count (same requirement computeShardParams
  // enforces via kTiles % numIn0Cores). Gate on it here so an ineligible op is
  // rejected up front rather than deep in shard-param computation — and so that
  // computeShardParams' assert holds by construction rather than by contract.
  if ((K / kTileSize) % kNumIn0Cores != 0) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): K tiles {1} not divisible by the in0 core "
                 "count {2}",
                 op->getName().getStringRef(), K / kTileSize, kNumIn0Cores);
    return std::nullopt;
  }
  // Decode-only, and this is tt-metal's constraint rather than a proxy for it:
  // the DS validation asserts TT_FATAL(M == 1) on the activation height in
  // tiles, i.e. exactly one tile row. Note M here is the logical row count with
  // leading dims collapsed, so this accepts a sub-tile batch (1..31, padded up
  // to one tile row by tt-metal) and rejects anything taller. The assert is
  // uncatchable, so the rejection has to happen here rather than in the op
  // model.
  if (llvm::divideCeil(M, kTileSize) != 1) {
    TTMLIR_DEBUG(
        ttmlir::LogComponent::GreedyOptimizer,
        "DS declined ({0}): activation M={1} is more than one tile row",
        op->getName().getStringRef(), M);
    return std::nullopt;
  }

  return operands;
}

// Everything the DS path derives from an op: the operand types and the shard
// geometry.
//
// The output hint and the input reshard candidates are both built from one of
// these, which is what keeps them consistent: the in0 layout's shard width has
// to match the config's in0_block_w, and the weight layout's bank count has to
// match the one the geometry was computed for.
struct DSPlan {
  RankedTensorType in0Type;
  RankedTensorType weightType;
  DRAMShardParams params;
  ttcore::DeviceAttr deviceAttr;
};

// Returns nullopt, having logged the reason, when the op is not DS-eligible or
// the shard geometry does not fit L1.
static std::optional<DSPlan> buildDSPlan(Operation *op) {
  std::optional<std::pair<Value, Value>> operands = getDSEligibleOperands(op);
  if (!operands) {
    return std::nullopt;
  }
  auto [dsActivation, dsWeight] = *operands;

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): op has no parent module",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  auto systemDescAttr = moduleOp->getAttr(ttcore::SystemDescAttr::name);
  if (!systemDescAttr) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): module has no system descriptor",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  auto systemDesc = mlir::cast<ttcore::SystemDescAttr>(systemDescAttr);
  int64_t l1Available =
      static_cast<int64_t>(ttnn::utils::getTensorL1UsageCap(moduleOp) *
                           systemDesc.getChipDescs()[0].getUsableL1Size());

  auto in0Type = mlir::cast<RankedTensorType>(dsActivation.getType());
  auto weightType = mlir::cast<RankedTensorType>(dsWeight.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  ttcore::DataType weightDataType = getWeightLayout(dsWeight).getDataType();

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  std::optional<int64_t> numDRAMBanks = getNumDRAMBanks(deviceAttr);
  if (!numDRAMBanks) {
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): device DRAM grid is not a single row that "
                 "canonical DRAM placement can express",
                 op->getName().getStringRef());
    return std::nullopt;
  }
  int64_t numAvailableCores =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());

  std::optional<DRAMShardParams> params =
      computeShardParams(M, K, N, *numDRAMBanks, kNumIn0Cores,
                         numAvailableCores, weightDataType, l1Available);
  if (!params) {
    // Either no in0_block_w dividing K-per-core leaves room for the circular
    // buffers, or the largest one that does is a degenerate fraction of
    // K-per-core (see kMinBlockWidthFraction in MatmulProgramConfig.cpp).
    TTMLIR_DEBUG(ttmlir::LogComponent::GreedyOptimizer,
                 "DS declined ({0}): no in0_block_w both fits L1 and avoids a "
                 "degenerate block count (M={1} K={2} "
                 "N={3} banks={4} in0Cores={5} cores={6} l1Available={7})",
                 op->getName().getStringRef(), M, K, N, *numDRAMBanks,
                 kNumIn0Cores, numAvailableCores, l1Available);
    return std::nullopt;
  }

  return DSPlan{in0Type, weightType, *params, deviceAttr};
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

  // numOutputCores = div_up(N_tiles, per_core_N_storage): exactly how many
  // output cores compute_output_specs will allocate, ensuring no assertion
  // fire.
  int64_t numOutputCores = llvm::divideCeil(p.N / kTileSize, p.perCoreN);

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
  auto computeConfig = buildComputeConfig(ctx, p.weightDataType);

  return OpConfig(l1OutLayout, MatmulAttrs{progConfig, computeConfig});
}

void MatmulRuleBook::applyDRAMShardedTransformation(
    Operation *op, const MatmulAttrs &matmulAttrs) const {
  auto *ctx = op->getContext();
  // Input reshards (activation → L1 1×kNumIn0Cores, weight → DRAM 1×numBanks)
  // handled by pass-2 in applyToIR via reshardLayouts populated from the input
  // candidates injected by getExtraInputReshardCandidates.

  OpBuilder builder(op);

  // --- 1. Set program config and compute config ---
  // ttnn.matmul and ttnn.linear both carry matmul_program_config,
  // compute_config and activation, so the DS rewrite is identical for either
  // (see getDSOperands).
  auto dsProgConfig =
      mlir::cast<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          matmulAttrs.matmulProgramConfig.value());
  StringAttr activationAttr =
      llvm::TypeSwitch<Operation *, StringAttr>(op)
          .Case<MatmulOp, LinearOp>([&](auto concreteOp) {
            concreteOp.setMatmulProgramConfigAttr(dsProgConfig);
            if (matmulAttrs.computeKernelConfig.has_value()) {
              concreteOp.setComputeConfigAttr(*matmulAttrs.computeKernelConfig);
            }
            // Strip the activation here and hand it back; step 2 re-homes it.
            StringAttr act = concreteOp.getActivationAttr();
            if (act) {
              concreteOp.removeActivationAttr();
            }
            return act;
          })
          .Default([](Operation *) { return StringAttr(); });

  // --- 2. Handle the fused activation ---
  // Fusing the activation into the DS matmul kernel is significantly slower
  // (measured: the 12-core DS matmul applies silu much slower than an op on all
  // 64 cores). So strip it off the matmul and either (a) fold it into a
  // consuming multiply as its operand-A activation — SwiGLU:
  // multiply(silu(gate), up) — which runs on the full grid (cheapest), or (b)
  // fall back to a separate elementwise op.
  if (activationAttr) {
    StringRef actStr = activationAttr.getValue();
    Value matmulResult = op->getResult(0);

    // (a) Try to fold silu into a consuming ttnn.multiply's lhs_activation.
    // Only when the matmul result's sole non-dealloc consumer is a multiply
    // (either operand — multiply is commutative, so we normalize the silu'd
    // value to operand A). Currently silu only (runtime plumbs lhs_activation).
    MultiplyOp fuseInto = nullptr;
    if (actStr == "silu") {
      Operation *soleUser = nullptr;
      bool multiUse = false;
      for (Operation *u : matmulResult.getUsers()) {
        if (isa<DeallocateOp>(u)) {
          continue;
        }
        if (soleUser) {
          multiUse = true;
          break;
        }
        soleUser = u;
      }
      if (!multiUse && soleUser) {
        fuseInto = dyn_cast<MultiplyOp>(soleUser);
      }
    }

    if (fuseInto && !fuseInto.getLhsActivation() &&
        (fuseInto.getLhs() == matmulResult ||
         fuseInto.getRhs() == matmulResult)) {
      // Normalize the silu'd (matmul) value to operand A, then tag
      // lhs_activation.
      Value other = fuseInto.getLhs() == matmulResult ? fuseInto.getRhs()
                                                      : fuseInto.getLhs();
      fuseInto.getLhsMutable().assign(matmulResult);
      fuseInto.getRhsMutable().assign(other);
      fuseInto.setLhsActivationAttr(StringAttr::get(ctx, "silu"));
    } else {
      // (b) Fallback: separate elementwise op running across all cores.
      StringRef opName;
      if (actStr == "silu") {
        opName = "ttnn.silu";
      } else if (actStr == "relu") {
        opName = "ttnn.relu";
      } else if (actStr == "gelu") {
        opName = "ttnn.gelu";
      }
      if (!opName.empty()) {
        builder.setInsertionPointAfter(op);
        auto *activationOp = builder.create(
            op->getLoc(), StringAttr::get(ctx, opName),
            ValueRange{matmulResult}, TypeRange{matmulResult.getType()});
        matmulResult.replaceAllUsesExcept(activationOp->getResult(0),
                                          activationOp);
      }
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
  if (isDRAMSharded) {
    applyDRAMShardedTransformation(op, matmulAttrs);
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
        if (shape.size() == 2 && shape[0] == 1 && shape[1] == kNumIn0Cores) {
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
  std::optional<DSPlan> plan = buildDSPlan(op);
  if (!plan) {
    return {};
  }

  auto *ctx = op->getContext();
  if (operandIdx == 0) {
    auto in0Layout = mlir::cast<TTNNLayoutAttr>(plan->in0Type.getEncoding());
    return {buildL1ShardedLayout(ctx, in0Layout, plan->in0Type.getShape(),
                                 kNumIn0Cores, plan->deviceAttr)};
  }
  if (operandIdx == 1) {
    auto weightLayout =
        mlir::cast<TTNNLayoutAttr>(plan->weightType.getEncoding());
    return {buildDRAMShardedWeightLayout(ctx, weightLayout, plan->params)};
  }
  return {};
}

} // namespace mlir::tt::ttnn
