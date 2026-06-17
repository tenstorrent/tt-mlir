// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
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
// Constants and shard-parameter computation for DRAM-sharded matmul
// ============================================================================

static constexpr int64_t kTileSize = 32;
static constexpr int64_t kNumDRAMBanks = 12;
// Single source of truth for how many cores the DS-matmul activation (in0) is
// width-sharded across. Drives the in0 shard width, the K-divisibility
// eligibility gate, and the in0 L1 tensor-buffer reservation in
// computeShardParams. Keep these uses consistent.
static constexpr int64_t kNumIn0Cores = 8;

struct DRAMShardParams {
  int64_t K;
  int64_t N;
  int64_t M;
  int64_t numBanks;
  int64_t numIn0Cores;
  int64_t numOutCores;
  int64_t nPadded;
  int64_t shardH;
  int64_t shardW;
  int64_t kTiles;
  int64_t shardWTiles;
  int64_t in0BlockW;
  int64_t perCoreM;
  int64_t perCoreN;
  int64_t in0ShardW;
  ttcore::DataType weightDataType;
};

static int64_t padToDRAMBanks(int64_t n, int64_t numBanks) {
  int64_t lcm = kTileSize * numBanks;
  return ((n + lcm - 1) / lcm) * lcm;
}

static std::optional<DRAMShardParams>
computeShardParams(int64_t M, int64_t K, int64_t N, int64_t numBanks,
                   int64_t numIn0Cores, int64_t numOutCores,
                   ttcore::DataType weightDataType, int64_t l1Available) {
  DRAMShardParams p;
  p.K = K;
  p.N = N;
  p.M = M;
  p.numBanks = numBanks;
  p.numIn0Cores = numIn0Cores;
  p.numOutCores = numOutCores;
  p.nPadded = padToDRAMBanks(N, numBanks);
  p.shardH = K;
  p.shardW = p.nPadded / numBanks;
  p.kTiles = K / kTileSize;
  p.shardWTiles = p.shardW / kTileSize;
  p.perCoreM = M / kTileSize;
  p.perCoreN = (N / kTileSize + numOutCores - 1) / numOutCores; // div_up
  p.in0ShardW = K / numIn0Cores;
  p.weightDataType = weightDataType;

  static constexpr int64_t kBf16Tile = 2048; // 32×32 × 2 B
  static constexpr int64_t kBfp8Tile = 1088; // 32×32 × 1 B + 64 B row exponents
  static constexpr int64_t kBfp4Tile =
      576; // 32×32 × 0.5 B + 64 B row exponents
  static constexpr int64_t kFp32Tile = 4096; // 32×32 × 4 B

  int64_t kWeightTile =
      (weightDataType == ttcore::DataType::BFP_BFloat4) ? kBfp4Tile : kBfp8Tile;

  assert(p.kTiles % numIn0Cores == 0 &&
         "kTiles must be divisible by numIn0Cores before the per-core divide");
  int64_t kPerCore = p.kTiles / numIn0Cores;
  // perCoreNCompute: tiles computed per DRAM-bank/compute core (= weight shard
  // width per bank). Used for CB sizing — this is what the compute kernel
  // actually accumulates per core before scattering to output storage cores.
  int64_t perCoreNCompute = p.shardWTiles;

  // Use numIn0Cores for the output tensor buffer estimate to keep the budget
  // conservative and avoid inflating in0BlockW (which doubles in1CB per step).
  // perCoreNStorage is only used for the output layout grid, not CB sizing.
  int64_t outTensorBufPerCore =
      p.perCoreM * ((N / kTileSize) / numIn0Cores) * kBf16Tile;
  int64_t in0TensorBuf = p.perCoreM * kPerCore * kBf16Tile;
  int64_t cbBudget = l1Available - in0TensorBuf - outTensorBufPerCore;

  // Fixed CBs (independent of in0BlockW).
  int64_t outCB = p.perCoreM * perCoreNCompute * kBf16Tile;
  int64_t interm0CB = p.perCoreM * perCoreNCompute * kFp32Tile;
  int64_t fixedCost = outCB + interm0CB;

  if (fixedCost > cbBudget) {
    return std::nullopt;
  }

  p.in0BlockW = kPerCore;
  bool found = false;
  while (p.in0BlockW >= 1) {
    int64_t numBlocks = p.kTiles / p.in0BlockW;
    bool doubleBuf = numBlocks > 1;

    int64_t in0CB = p.in0BlockW * p.perCoreM * kBf16Tile * (doubleBuf ? 2 : 1);
    int64_t in1CB = p.in0BlockW * perCoreNCompute * kWeightTile *
                    (doubleBuf ? 3 : 1); // weight shard per DRAM bank

    if (fixedCost + in0CB + in1CB <= cbBudget && kPerCore % p.in0BlockW == 0) {
      found = true;
      break;
    }
    if (p.in0BlockW == 1) {
      break;
    }
    p.in0BlockW--;
    while (p.in0BlockW > 1 && kPerCore % p.in0BlockW != 0) {
      p.in0BlockW--;
    }
  }

  if (!found) {
    return std::nullopt;
  }

  return p;
}

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
// Layout and config builders
// ============================================================================

static TTNNLayoutAttr buildDRAMShardedWeightLayout(MLIRContext *ctx,
                                                   TTNNLayoutAttr origLayout,
                                                   const DRAMShardParams &p) {
  auto startCoord = CoreCoordAttr::get(ctx, 0, 0);
  auto endCoord = CoreCoordAttr::get(ctx, p.numBanks - 1, 0);
  auto coreRange = CoreRangeAttr::get(ctx, startCoord, endCoord);
  auto crs = CoreRangeSetAttr::get(ctx, {coreRange});
  auto tileType =
      ttcore::TileType::get(ctx, {kTileSize, kTileSize}, p.weightDataType);
  auto dramSpace = BufferTypeAttr::get(ctx, BufferType::DRAM);
  auto memrefType = MemRefType::get({p.kTiles, p.shardWTiles}, tileType,
                                    MemRefLayoutAttrInterface{}, dramSpace);
  auto memLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);
  return TTNNLayoutAttr::get(ctx, origLayout.getLinear(),
                             llvm::ArrayRef<int64_t>{1, p.numBanks}, memrefType,
                             memLayout, /*tensorMesh=*/nullptr,
                             /*ignorePhysicalLayout=*/false, crs);
}

// Build an L1 width-sharded layout for `tensorShape` over `numCores`, using
// canonical core placement that wraps across the worker grid. A single-row
// placement (0,0)-(numCores-1,0) would be invalid once numCores exceeds the
// grid width (e.g. 16 cores on an 8x8 grid) — validateTensorSpec rejects it —
// so canonical placement fills the grid row-by-row (16 cores -> 8x2), mirroring
// the DS output-layout construction.
static TTNNLayoutAttr buildL1ShardedLayout(MLIRContext *ctx,
                                           TTNNLayoutAttr origLayout,
                                           llvm::ArrayRef<int64_t> tensorShape,
                                           int64_t numCores,
                                           ttcore::DeviceAttr deviceAttr) {
  return TTNNLayoutAttr::Builder(origLayout, tensorShape)
      .setBufferType(BufferType::L1)
      .setMemoryLayout(
          TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded))
      .setGridShape({1, numCores})
      .buildWithCanonicalCorePlacement(deviceAttr);
}

static MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
buildDRAMShardedProgramConfig(MLIRContext *ctx, const DRAMShardParams &p,
                              UnaryWithParamAttr fusedAct) {
  return MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
      ctx, p.in0BlockW, p.perCoreM, p.perCoreN, fusedAct);
}

static DeviceComputeKernelConfigAttr
buildComputeConfig(MLIRContext *ctx, ttcore::DataType weightDataType) {
  MathFidelity fidelity = (weightDataType == ttcore::DataType::BFP_BFloat4)
                              ? MathFidelity::LoFi
                              : MathFidelity::HiFi2;
  return DeviceComputeKernelConfigAttr::get(
      ctx,
      /*mathFidelity=*/fidelity,
      /*mathApproxMode=*/mlir::BoolAttr{},
      /*fp32DestAccEn=*/mlir::BoolAttr::get(ctx, true),
      /*packerL1Acc=*/mlir::BoolAttr::get(ctx, true),
      /*dstFullSyncEn=*/mlir::BoolAttr{});
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
// MatmulRuleBook::buildDRAMShardingHint
// ============================================================================

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

// ============================================================================
// MatmulRuleBook — public interface
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
// MatmulRuleBook::applyDRAMShardedTransformation
// ============================================================================

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
            shape[1] == kNumStorageCores) {
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
    int64_t in0ShardHTiles = M / kTileSize;
    int64_t in0ShardWTiles = (K / kTileSize) / kNumIn0Cores;
    return {buildL1ShardedLayout(ctx, in0Layout, in0Type.getShape(),
                                 kNumIn0Cores, deviceAttr)};
  }
  if (operandIdx == 1) {
    auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
    return {buildDRAMShardedWeightLayout(ctx, weightLayout, p)};
  }
  return {};
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

} // namespace mlir::tt::ttnn
