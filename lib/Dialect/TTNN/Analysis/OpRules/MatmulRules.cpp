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
#include <limits>

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

// How many DRAM banks the weight is width-sharded across. This MUST come from
// the system descriptor, not from a constant: Wormhole has 12 DRAM channels but
// Blackhole has 8, and a hardcoded 12 produces a weight layout that cannot be
// allocated on Blackhole at all -- tensor creation aborts with
// "logical_coord.x < num_dram_views" in metal_soc_descriptor.cpp. It slips past
// the op model because validateTensorSpec deliberately skips the shard
// bounding-box check for DRAM buffers, so the bad layout is only caught on
// silicon.
static int64_t getNumDRAMBanks(ttcore::SystemDescAttr systemDesc) {
  return static_cast<int64_t>(
      systemDesc.getChipDescs().front().getNumDramChannels());
}

// Single source of truth for how many cores the DS-matmul activation (in0) is
// width-sharded across. Drives the in0 shard width, the K-divisibility
// eligibility gate, and the in0 L1 tensor-buffer reservation in
// computeShardParams. Keep these uses consistent.
static constexpr int64_t kNumIn0Cores = 8;

// How many cores the DS activation (in0) is width-sharded across. 8 is the
// empirically preferred value, but it MUST divide K in tiles -- the contraction
// dim is split evenly across these cores. Hardcoding 8 silently excluded every
// model whose K/32 is not a multiple of 8: gpt-oss (hidden 2880 -> 90 K-tiles)
// lost DRAM sharding entirely on that arithmetic alone. Verified on silicon
// (ds-runs/probe_ds_sweep.py) that 90 K-tiles runs at PCC 1.0000 with 10, 9, 6
// or 5 in0 cores, so this is a search parameter, not a kernel constraint.
// Returns 0 when no usable split exists.
static int64_t chooseNumIn0Cores(int64_t kTiles) {
  if (kTiles <= 0) {
    return 0;
  }
  if (kTiles % kNumIn0Cores == 0) {
    return kNumIn0Cores;
  }
  // Nearest usable divisor to the preferred count, searching outwards so the
  // in0 shard stays as close to the tuned width as the shape allows.
  for (int64_t delta = 1; delta < kNumIn0Cores; ++delta) {
    for (int64_t candidate : {kNumIn0Cores + delta, kNumIn0Cores - delta}) {
      if (candidate >= 1 && kTiles % candidate == 0) {
        return candidate;
      }
    }
  }
  return 1;
}

// ============================================================================
// Eligibility helpers
// ============================================================================

// Whether the weight is a tiled, DRAM-interleaved tensor of a data type the DS
// kernel should be offered for.
//
// The TTNNLayoutAttr is the source of truth, not the tensor's element type: a
// bfp4/bfp8 tensor carries a ttcore::TileType element, but a bf16 tensor is
// spelled with a scalar bf16 element and records its tiling in the layout. An
// element-type-only check therefore rejected every bf16 weight before the dtype
// question was even reached.
static bool isBfpDRAMInterleaved(Value weight, bool allowBf16) {
  auto rtt = mlir::dyn_cast<RankedTensorType>(weight.getType());
  if (!rtt) {
    return false;
  }
  auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rtt.getEncoding());
  if (!layoutAttr || !layoutAttr.isTiled()) {
    return false;
  }
  auto dt = layoutAttr.getDataType();
  // bf16 is buildable -- verified on silicon at PCC 1.0000, and the gpt-oss
  // bring-up swept DS attention in BF16 too. It is off by default purely on
  // bandwidth: DS streams the weights, so bf16 moves 2x the bytes of bfp8 and
  // 4x bfp4, which is the regime where 1D-mcast wins. Opt in with
  // allow-bf16-dram-sharded-matmul when you intend to measure it.
  bool isBfp = dt == ttcore::DataType::BFP_BFloat8 ||
               dt == ttcore::DataType::BFP_BFloat4;
  if (!isBfp && !(allowBf16 && dt == ttcore::DataType::BFloat16)) {
    return false;
  }
  return layoutAttr.hasInterleavedDRAMTensorMemoryLayout();
}

static ttcore::DataType getWeightDataType(Value weight) {
  // From the layout, for the same reason isBfpDRAMInterleaved uses it: a bf16
  // weight has a scalar element type and casting it to TileType would abort.
  auto rtt = mlir::cast<RankedTensorType>(weight.getType());
  return mlir::cast<TTNNLayoutAttr>(rtt.getEncoding()).getDataType();
}

// A weight is DS-shaped when it is a plain 2-D matrix, possibly carrying
// leading unit batch dims: ttnn models routinely hold projection weights as
// [1, 1, K, N] (the autoport Llama decoders do), which is the same matrix as
// [K, N] -- same element count, same tile grid, and TTNN already collapses both
// to a 2-D memref in the layout. Anything with a non-unit batch dim is a real
// batched matmul and is not DS-shaped.
static bool isDSWeightShaped(RankedTensorType rtt) {
  auto shape = rtt.getShape();
  if (shape.size() < 2) {
    return false;
  }
  return llvm::all_of(shape.drop_back(2), [](int64_t dim) { return dim == 1; });
}

static std::pair<int64_t, int64_t> getWeightKN(RankedTensorType rtt) {
  auto shape = rtt.getShape();
  assert(isDSWeightShaped(rtt) && "Expected a 2D (or unit-batched) weight");
  return {shape[shape.size() - 2], shape[shape.size() - 1]};
}

static int64_t getActivationM(RankedTensorType rtt) {
  int64_t M = 1;
  for (int64_t dim : rtt.getShape().drop_back()) {
    M *= dim;
  }
  return M;
}

// The DS path covers ttnn.matmul and ttnn.linear alike: a bias-free linear IS
// the matmul the DS kernel implements, and most of the ttnn decoders in the
// wild write their projections as ttnn.linear (both Llamas, and the packed QKV
// of Qwen2.5-Coder), so restricting DS to MatmulOp made it invisible for them.
// Either transpose flag is outside the DS contract and is rejected here rather
// than silently mis-modelled.
//
// Returns the (activation, weight) operand pair, or nullopt if `op` is not a
// DS-capable matmul-like op.
static std::optional<std::pair<Value, Value>> getDSOperands(Operation *op) {
  if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
    if (matmulOp.getTransposeA() || matmulOp.getTransposeB()) {
      return std::nullopt;
    }
    return std::make_pair(matmulOp.getA(), matmulOp.getB());
  }
  if (auto linearOp = dyn_cast<LinearOp>(op)) {
    // A bias is fine: verified on silicon that ttnn.linear with a bias and a DS
    // program config runs (gpt-oss QKV shape, PCC 0.9996 -- the delta is bfp8
    // quantization). The kernel takes it; the earlier exclusion was policy.
    if (linearOp.getTransposeA() || linearOp.getTransposeB()) {
      return std::nullopt;
    }
    return std::make_pair(linearOp.getA(), linearOp.getB());
  }
  return std::nullopt;
}

static bool isDRAMShardEligible(Operation *op) {
  // Respect the disable-dram-sharded-matmul pipeline option (set as a module
  // attribute by DevicePassesWrapper). This is the single choke point for the
  // DS path: buildDRAMShardingHint and getExtraInputReshardCandidates both gate
  // on it, and getOutputHints reaches DS only through buildDRAMShardingHint.
  if (ttnn::utils::isDRAMShardedMatmulDisabled(op)) {
    return false;
  }

  auto operands = getDSOperands(op);
  if (!operands) {
    return false;
  }
  auto [activation, weight] = *operands;

  if (!isBfpDRAMInterleaved(weight,
                            ttnn::utils::isBf16DRAMShardedMatmulAllowed(op))) {
    return false;
  }
  if (!ttcore::valueTracesToConstantArgs(weight)) {
    return false;
  }
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  if (!isDSWeightShaped(weightType)) {
    return false;
  }

  auto in0Type = mlir::cast<RankedTensorType>(activation.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);

  if (K % kTileSize != 0 || N % kTileSize != 0) {
    return false;
  }
  // K is the contraction dim, width-sharded across the in0 cores, so it must
  // divide evenly by the in0 core count (same requirement computeShardParams
  // enforces via kTiles % numIn0Cores). Gate on it here so an ineligible op is
  // rejected up front rather than deep in shard-param computation.
  if (chooseNumIn0Cores(K / kTileSize) == 0) {
    return false;
  }
  // No M gate. M is the LOGICAL row count, so any batch -- 1, 32, or a prefill
  // sequence -- is offered as a DS candidate and the op model decides. The
  // previous rule (M % 32 == 0 && M / 32 == 1) admitted batch 32 and nothing
  // else, which excluded both ends of the useful range: batch 1, where the QB2
  // bring-up measured DRAM-sharded as the FASTEST option and shipped it as the
  // small-batch default (Qwen2.5-Coder DS-40c, 2.1503 ms), and every batch in
  // between. Whether a multi-tile-row M is buildable is a tt-metal question
  // (the factory constrains per_core_M against num_blocks_per_shard), so let
  // the constraint query answer it per shape instead of hardcoding one.
  (void)M;

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

static bool directConsumerRejectsLayout(Operation *producer,
                                        TTNNLayoutAttr layout) {
  if (!layout || !layout.hasShardedTensorMemoryLayout()) {
    return false;
  }

  for (OpResult result : producer->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *consumer = use.getOwner();
      if (isa<DeallocateOp>(consumer)) {
        continue;
      }
      LayoutFilterFn filter =
          getRuleBook(consumer).getInputLayoutFilter(use.getOperandNumber());
      if (filter && !filter(layout)) {
        return true;
      }
    }
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
  if (!isDRAMShardEligible(op)) {
    return std::nullopt;
  }
  auto [dsActivation, dsWeight] = *getDSOperands(op);

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

  auto in0Type = mlir::cast<RankedTensorType>(dsActivation.getType());
  auto weightType = mlir::cast<RankedTensorType>(dsWeight.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  auto weightDataType = getWeightDataType(dsWeight);

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  int64_t numAvailableCores =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());

  auto pOpt = computeShardParams(
      M, K, N, getNumDRAMBanks(systemDesc), chooseNumIn0Cores(K / kTileSize),
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
    Operation *matmulLikeOp, const MatmulAttrs &matmulAttrs) const {
  auto *ctx = matmulLikeOp->getContext();
  // Input reshards (activation → L1 1×kNumIn0Cores, weight → DRAM 1×numBanks)
  // handled by pass-2 in applyToIR via reshardLayouts populated from the input
  // candidates injected by getExtraInputReshardCandidates.

  OpBuilder builder(matmulLikeOp);

  // --- 1. Set program config and compute config ---
  // ttnn.matmul and ttnn.linear both carry matmul_program_config /
  // compute_config / activation, so the DS rewrite is identical for either (see
  // getDSOperands).
  auto dsProgConfig =
      mlir::cast<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          matmulAttrs.matmulProgramConfig.value());
  llvm::TypeSwitch<Operation *>(matmulLikeOp)
      .Case<MatmulOp, LinearOp>([&](auto concreteOp) {
        concreteOp.setMatmulProgramConfigAttr(dsProgConfig);
        if (matmulAttrs.computeKernelConfig.has_value()) {
          concreteOp.setComputeConfigAttr(*matmulAttrs.computeKernelConfig);
        }
      });

  // --- 2. Handle the fused activation ---
  // Fusing the activation into the DS matmul kernel is significantly slower
  // (measured: the 12-core DS matmul applies silu much slower than an op on all
  // 64 cores). So strip it off the matmul and either (a) fold it into a
  // consuming multiply as its operand-A activation — SwiGLU:
  // multiply(silu(gate), up) — which runs on the full grid (cheapest), or (b)
  // fall back to a separate elementwise op.
  auto activationAttr = matmulLikeOp->getAttrOfType<StringAttr>("activation");
  if (activationAttr) {
    matmulLikeOp->removeAttr("activation");
    StringRef actStr = activationAttr.getValue();
    Value matmulResult = matmulLikeOp->getResult(0);

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
        builder.setInsertionPointAfter(matmulLikeOp);
        auto *activationOp = builder.create(
            matmulLikeOp->getLoc(), StringAttr::get(ctx, opName),
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

  if (ttnn::utils::shouldAvoidGuaranteedOutputReshards(op)) {
    filtered.erase(std::remove_if(filtered.begin(), filtered.end(),
                                  [&](const OpConfig &config) {
                                    return directConsumerRejectsLayout(
                                        op, config.outputLayout);
                                  }),
                   filtered.end());
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

static bool shardedInputOutputMemoryConfigsMatch(TTNNLayoutAttr input,
                                                 TTNNLayoutAttr output) {
  if (!output) {
    return true;
  }
  auto outputMemLayout = output.getMemLayoutOpt();
  if (!outputMemLayout || !isShardedMemoryLayout(*outputMemLayout)) {
    return true;
  }
  return input.getBufferType() == output.getBufferType() &&
         input.getMemLayoutOpt() == outputMemLayout;
}

static uint64_t getPhysicalCoreCount(TTNNLayoutAttr layout) {
  CoreRangeSetAttr ranges = layout.getCoreRangeSet();
  if (!ranges) {
    return 0;
  }

  uint64_t count = 0;
  for (CoreRangeAttr range : ranges.getCoreRanges()) {
    uint64_t width =
        range.getEndCoord().getX() - range.getStartCoord().getX() + 1;
    uint64_t height =
        range.getEndCoord().getY() - range.getStartCoord().getY() + 1;
    count += width * height;
  }
  return count;
}

static bool shardedInputFitsProgramGrid(TTNNLayoutAttr input,
                                        CoreCoordAttr programGrid) {
  auto memLayout = input.getMemLayoutOpt();
  if (!memLayout || !isShardedMemoryLayout(*memLayout)) {
    return true;
  }

  uint64_t programCores =
      static_cast<uint64_t>(programGrid.getX()) * programGrid.getY();
  uint64_t inputCores = getPhysicalCoreCount(input);
  return inputCores != 0 && inputCores <= programCores;
}

static bool
isMcast2DIn0Compatible(const OpConfig &hint, TTNNLayoutAttr in0,
                       MatmulMultiCoreReuseMultiCastProgramConfigAttr config) {
  auto memLayout = in0.getMemLayoutOpt();
  if (!memLayout || !isShardedMemoryLayout(*memLayout)) {
    return true;
  }

  if (!shardedInputFitsProgramGrid(in0,
                                   config.getComputeWithStorageGridSize())) {
    return false;
  }

  if (!config.getFuseBatch() ||
      (*memLayout != TensorMemoryLayout::BlockSharded &&
       *memLayout != TensorMemoryLayout::HeightSharded)) {
    return false;
  }

  auto shardShape = in0.getShardShape();
  int64_t in0BlockW = static_cast<int64_t>(config.getIn0BlockW());
  if (shardShape.size() != 2 || in0BlockW == 0 ||
      static_cast<int64_t>(config.getPerCoreM()) != shardShape[0] ||
      shardShape[1] % in0BlockW != 0) {
    return false;
  }

  if (*memLayout == TensorMemoryLayout::HeightSharded) {
    // The 2D mcast height-sharded path requires the whole K dimension in one
    // shard and does not support transpose multicast.
    return !config.getTransposeMcast() && shardShape[1] == in0BlockW;
  }

  // For a block-sharded activation, tt-metal requires a sharded output to use
  // the same buffer type and memory-layout type.
  return shardedInputOutputMemoryConfigsMatch(in0, hint.outputLayout);
}

static bool isMcast1DIn0Compatible(
    const OpConfig &hint, TTNNLayoutAttr in0,
    MatmulMultiCoreReuseMultiCast1DProgramConfigAttr config) {
  auto outputMemLayout = hint.outputLayout.getMemLayoutOpt();
  bool movesIn0 = config.getMcastIn0() || config.getGatherIn0();
  TensorMemoryLayout directionalOutputLayout =
      movesIn0 ? TensorMemoryLayout::WidthSharded
               : TensorMemoryLayout::HeightSharded;
  if (!outputMemLayout ||
      (*outputMemLayout != directionalOutputLayout &&
       *outputMemLayout != TensorMemoryLayout::BlockSharded)) {
    return false;
  }
  auto outputRanges = hint.outputLayout.getCoreRangeSet();
  auto outputBBox = outputRanges ? outputRanges.getBoundingBox() : std::nullopt;
  if (!outputBBox) {
    return false;
  }
  if (movesIn0) {
    if (outputBBox->getStartCoord().getY() !=
        outputBBox->getEndCoord().getY()) {
      return false;
    }
  } else if (outputBBox->getStartCoord().getX() !=
             outputBBox->getEndCoord().getX()) {
    return false;
  }

  auto memLayout = in0.getMemLayoutOpt();
  if (!memLayout || !isShardedMemoryLayout(*memLayout)) {
    return true;
  }

  if (!shardedInputFitsProgramGrid(in0,
                                   config.getComputeWithStorageGridSize())) {
    return false;
  }

  TensorMemoryLayout requiredLayout = movesIn0
                                          ? TensorMemoryLayout::WidthSharded
                                          : TensorMemoryLayout::HeightSharded;
  if (!config.getFuseBatch() || *memLayout != requiredLayout) {
    return false;
  }

  auto shardShape = in0.getShardShape();
  int64_t in0BlockW = static_cast<int64_t>(config.getIn0BlockW());
  if (shardShape.size() != 2 || in0BlockW == 0 ||
      static_cast<int64_t>(config.getPerCoreM()) != shardShape[0] ||
      shardShape[1] % in0BlockW != 0) {
    return false;
  }

  return shardedInputOutputMemoryConfigsMatch(in0, hint.outputLayout);
}

static bool shardedOutputCBFits(TTNNLayoutAttr output, uint64_t perCoreM,
                                uint64_t perCoreN) {
  if (!output) {
    return true;
  }
  auto memLayout = output.getMemLayoutOpt();
  if (!memLayout || !isShardedMemoryLayout(*memLayout)) {
    return true;
  }

  uint64_t elementSize = output.getElementSizeBytes();
  if (perCoreM != 0 &&
      perCoreN > std::numeric_limits<uint64_t>::max() / perCoreM) {
    return false;
  }
  uint64_t requiredTiles = perCoreM * perCoreN;
  if (elementSize != 0 &&
      requiredTiles > std::numeric_limits<uint64_t>::max() / elementSize) {
    return false;
  }
  return requiredTiles * elementSize <= output.getShardSizeInBytes();
}

bool MatmulRuleBook::isValidOutputHintForInputs(
    const OpConfig &hint, llvm::ArrayRef<TTNNLayoutAttr> inputLayouts) const {
  const auto *attrs = std::get_if<MatmulAttrs>(&hint.opSpecificAttrs);
  if (!attrs || !attrs->matmulProgramConfig.has_value()) {
    return true;
  }
  if (hint.outputLayout && hint.outputLayout.hasShardedTensorMemoryLayout() &&
      hint.outputLayout.getIgnorePhysicalLayout()) {
    return false;
  }
  mlir::Attribute programConfig = attrs->matmulProgramConfig.value();

  if (auto config =
          mlir::dyn_cast<MatmulMultiCoreReuseMultiCastProgramConfigAttr>(
              programConfig)) {
    if (!shardedOutputCBFits(hint.outputLayout, config.getPerCoreM(),
                             config.getPerCoreN())) {
      return false;
    }
  } else if (auto config = mlir::dyn_cast<
                 MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>(
                 programConfig)) {
    if (!shardedOutputCBFits(hint.outputLayout, config.getPerCoreM(),
                             config.getPerCoreN())) {
      return false;
    }
  }

  if (auto dsConfig = mlir::dyn_cast<
          MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          programConfig)) {
    // DS hint: only the canonical DS input combination is valid — L1
    // width-sharded in0, DRAM width-sharded in1, with an in0 shard width
    // compatible with the config's in0_block_w. Preserve this existing
    // behavior unchanged.
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
    return dsIn0CompatibleWithConfig(in0, in1, dsConfig);
  }

  if (inputLayouts.empty() || !inputLayouts[0]) {
    return true;
  }
  TTNNLayoutAttr in0 = inputLayouts[0];
  return llvm::TypeSwitch<mlir::Attribute, bool>(programConfig)
      .Case<MatmulMultiCoreReuseMultiCastProgramConfigAttr>([&](auto config) {
        return isMcast2DIn0Compatible(hint, in0, config);
      })
      .Case<MatmulMultiCoreReuseMultiCast1DProgramConfigAttr>([&](auto config) {
        return isMcast1DIn0Compatible(hint, in0, config);
      })
      .Default([](mlir::Attribute) { return true; });
}

// ============================================================================
// MatmulRuleBook::adjustScore
// ============================================================================

LayoutScore
MatmulRuleBook::adjustScore(Operation *op, LayoutScore base,
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
        // "Canonical" is the in0 split this op's K actually uses, which is 8
        // only when 8 divides K in tiles (see chooseNumIn0Cores).
        int64_t wantCores = kNumIn0Cores;
        if (auto operands = getDSOperands(op)) {
          auto weightType =
              mlir::dyn_cast<RankedTensorType>(operands->second.getType());
          if (weightType && isDSWeightShaped(weightType)) {
            wantCores =
                chooseNumIn0Cores(getWeightKN(weightType).first / kTileSize);
          }
        }
        auto shape = in0.getGridShape();
        if (shape.size() == 2 && shape[0] == 1 && shape[1] == wantCores) {
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
  if (!isDRAMShardEligible(op)) {
    return {};
  }
  auto [dsActivation, dsWeight] = *getDSOperands(op);

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

  auto in0Type = mlir::cast<RankedTensorType>(dsActivation.getType());
  auto weightType = mlir::cast<RankedTensorType>(dsWeight.getType());
  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  auto weightDataType = getWeightDataType(dsWeight);

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);
  int64_t numAvailCores =
      ttmlir::utils::volume(deviceAttr.getWorkerGrid().getShape());
  auto pOpt = computeShardParams(M, K, N, getNumDRAMBanks(systemDesc),
                                 chooseNumIn0Cores(K / kTileSize),
                                 numAvailCores, weightDataType, l1Available);
  if (!pOpt) {
    return {};
  }
  const auto &p = *pOpt;

  auto *ctx = op->getContext();
  if (operandIdx == 0) {
    auto in0Layout = mlir::cast<TTNNLayoutAttr>(in0Type.getEncoding());
    return {buildL1ShardedLayout(ctx, in0Layout, in0Type.getShape(),
                                 chooseNumIn0Cores(K / kTileSize), deviceAttr)};
  }
  if (operandIdx == 1) {
    auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
    return {buildDRAMShardedWeightLayout(ctx, weightLayout, p)};
  }
  return {};
}

} // namespace mlir::tt::ttnn
