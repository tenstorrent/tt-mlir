// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/OpRules/MatmulRules.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpRules/LayoutFilterUtils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"

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
static constexpr int64_t kNumStorageCores = 8;

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
  p.perCoreN = (N / kTileSize) / numOutCores;
  p.in0ShardW = K / numIn0Cores;
  p.weightDataType = weightDataType;

  static constexpr int64_t kBf16Tile = 2048;
  static constexpr int64_t kBfp8Tile = 1088;
  static constexpr int64_t kBfp4Tile = 576;
  static constexpr int64_t kFp32Tile = 4096;

  int64_t kWeightTile =
      (weightDataType == ttcore::DataType::BFP_BFloat4) ? kBfp4Tile : kBfp8Tile;

  int64_t kPerCore = p.kTiles / numIn0Cores;
  int64_t perCoreNCompute = p.shardWTiles;

  // Tensor buffers placed in L1 (conservative estimate — see §4f of
  // unified_ds_analysis.md: use numIn0Cores for both to keep budget identical
  // to the previous pass and avoid increasing in0BlockW).
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
    int64_t in1CB =
        p.in0BlockW * perCoreNCompute * kWeightTile * (doubleBuf ? 3 : 1);

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
  auto shape = rtt.getShape();
  int64_t M = 1;
  for (size_t i = 0; i < shape.size() - 1; i++) {
    M *= shape[i];
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
  if ((K / kTileSize) % kNumStorageCores != 0) {
    return false;
  }
  if ((N / kTileSize) % kNumStorageCores != 0) {
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

static RankedTensorType withLayout(RankedTensorType origType,
                                   TTNNLayoutAttr newLayout) {
  return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                               newLayout);
}

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

static TTNNLayoutAttr buildL1ShardedLayout(MLIRContext *ctx,
                                           TTNNLayoutAttr origLayout,
                                           int64_t shardHTiles,
                                           int64_t shardWTiles,
                                           int64_t numCores) {
  auto startCoord = CoreCoordAttr::get(ctx, 0, 0);
  auto endCoord = CoreCoordAttr::get(ctx, numCores - 1, 0);
  auto coreRange = CoreRangeAttr::get(ctx, startCoord, endCoord);
  auto crs = CoreRangeSetAttr::get(ctx, {coreRange});
  auto tileType = ttcore::TileType::get(ctx, {kTileSize, kTileSize},
                                        ttcore::DataType::BFloat16);
  auto l1Space = BufferTypeAttr::get(ctx, BufferType::L1);
  auto memrefType = MemRefType::get({shardHTiles, shardWTiles}, tileType,
                                    MemRefLayoutAttrInterface{}, l1Space);
  auto memLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);
  return TTNNLayoutAttr::get(ctx, origLayout.getLinear(),
                             llvm::ArrayRef<int64_t>{1, numCores}, memrefType,
                             memLayout, /*tensorMesh=*/nullptr,
                             /*ignorePhysicalLayout=*/false, crs);
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
// Helper: check if a layout is already the right DS in0 layout
// ============================================================================

static bool isAlreadyDSIn0Layout(TTNNLayoutAttr layout, int64_t shardWTiles) {
  if (!layout.hasL1BufferType()) {
    return false;
  }
  auto ml = layout.getMemLayoutOpt();
  if (!ml || *ml != TensorMemoryLayout::WidthSharded) {
    return false;
  }
  auto shape = layout.getGridShape();
  if (shape.size() != 2 || shape[0] != 1 || shape[1] != kNumStorageCores) {
    return false;
  }
  // Check shard width matches expected K/kNumStorageCores tiles.
  auto memref = layout.getMemref();
  auto memrefShape = memref.getShape();
  if (memrefShape.size() < 2) {
    return false;
  }
  return memrefShape.back() == shardWTiles;
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

  // Temporary diagnostic — remove after debugging.
  {
    Value weight = matmulOp.getB();
    auto in0Type = mlir::cast<RankedTensorType>(matmulOp.getA().getType());
    auto weightType = mlir::dyn_cast<RankedTensorType>(weight.getType());
    int64_t M = weightType ? getActivationM(in0Type) : -1;
    llvm::errs() << "[DS] MatmulOp at " << op->getLoc() << " M=" << M
                 << " bfpDRAM="
                 << (weightType ? isBfpDRAMInterleaved(weight) : false)
                 << " constArg=" << ttcore::valueTracesToConstantArgs(weight)
                 << "\n";
  }

  if (!isDRAMShardEligible(matmulOp)) {
    llvm::errs() << "[DS]   -> ineligible\n";
    return std::nullopt;
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    llvm::errs() << "[DS]   -> no moduleOp\n";
    return std::nullopt;
  }
  auto systemDescAttr = moduleOp->getAttr(ttcore::SystemDescAttr::name);
  if (!systemDescAttr) {
    llvm::errs() << "[DS]   -> no systemDesc\n";
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

  auto pOpt = computeShardParams(M, K, N, kNumDRAMBanks, kNumStorageCores,
                                 kNumStorageCores, weightDataType, l1Available);
  if (!pOpt) {
    llvm::errs() << "[DS]   -> computeShardParams failed M=" << M << " K=" << K
                 << " N=" << N << " l1=" << l1Available << "\n";
    return std::nullopt;
  }
  llvm::errs() << "[DS]   -> hint built M=" << M << " K=" << K << " N=" << N
               << " blkw=" << pOpt->in0BlockW << "\n";
  const auto &p = *pOpt;

  auto *ctx = op->getContext();
  auto outLayout = mlir::cast<TTNNLayoutAttr>(
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());

  int64_t outShardHTiles = M / kTileSize;
  int64_t outShardWTiles = (N / kTileSize) / kNumStorageCores;
  auto l1OutLayout = buildL1ShardedLayout(ctx, outLayout, outShardHTiles,
                                          outShardWTiles, kNumStorageCores);

  UnaryWithParamAttr fusedAct; // null — activation is split into a separate op
  auto progConfig = buildDRAMShardedProgramConfig(ctx, p, fusedAct);
  auto computeConfig = buildComputeConfig(ctx, weightDataType);

  OpConfig hint(l1OutLayout, MatmulAttrs{progConfig, computeConfig});
  hint.prevalidated = true;
  return hint;
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

  if (auto dramHint = buildDRAMShardingHint(op)) {
    llvm::errs() << "[DS] getOutputHints returning DS hint prevalidated="
                 << dramHint->prevalidated << "\n";
    return OutputHints{{*dramHint}, {}};
  }

  auto partialConfigs =
      optimizer_utils::getUniqueTestConfigsForMatmulLinear(legalConfigs);

  // Filter out L1-interleaved (worst of both worlds for matmul — see comment
  // in the original getOutputHints).
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

  return OutputHints{filtered, {}};
}

// ============================================================================
// MatmulRuleBook::applyDRAMShardedTransformation
// ============================================================================

void MatmulRuleBook::applyDRAMShardedTransformation(
    MatmulOp matmulOp, const MatmulAttrs &matmulAttrs) const {
  auto *ctx = matmulOp.getContext();
  OpBuilder builder(matmulOp); // inserts before matmulOp

  Value in0 = matmulOp.getA();
  Value weight = matmulOp.getB();
  auto in0Type = mlir::cast<RankedTensorType>(in0.getType());
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());

  auto moduleOp = matmulOp->getParentOfType<ModuleOp>();
  auto systemDesc = mlir::cast<ttcore::SystemDescAttr>(
      moduleOp->getAttr(ttcore::SystemDescAttr::name));
  int64_t l1Available =
      static_cast<int64_t>(ttnn::utils::getTensorL1UsageCap(moduleOp) *
                           systemDesc.getChipDescs()[0].getUsableL1Size());

  int64_t M = getActivationM(in0Type);
  auto [K, N] = getWeightKN(weightType);
  auto weightDataType = getWeightDataType(weight);

  auto pOpt = computeShardParams(M, K, N, kNumDRAMBanks, kNumStorageCores,
                                 kNumStorageCores, weightDataType, l1Available);
  if (!pOpt) {
    return;
  }
  const auto &p = *pOpt;

  // --- 1. Reshard weight → DRAM WIDTH_SHARDED ---
  auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
  auto dramShardedWeightLayout =
      buildDRAMShardedWeightLayout(ctx, weightLayout, p);
  auto dramShardedWeightType = withLayout(weightType, dramShardedWeightLayout);
  auto weightReshard = builder.create<ToMemoryConfigOp>(
      matmulOp.getLoc(), dramShardedWeightType, weight);
  matmulOp.setOperand(1, weightReshard.getResult());

  // --- 2. Shard in0 → L1 WIDTH_SHARDED (skip if already correct) ---
  // At this point in the first pass of applyToIR, in0's type may already be
  // L1 WIDTH_SHARDED if a preceding DS matmul in the chain set it
  // (applyOpConfig processes ops in order, so A's output type is set before we
  // process B).
  auto in0Layout = mlir::cast<TTNNLayoutAttr>(in0Type.getEncoding());
  int64_t in0ShardWTiles = (K / kTileSize) / kNumStorageCores;

  if (!isAlreadyDSIn0Layout(in0Layout, in0ShardWTiles)) {
    int64_t in0ShardHTiles = M / kTileSize;
    auto l1In0Layout = buildL1ShardedLayout(ctx, in0Layout, in0ShardHTiles,
                                            in0ShardWTiles, kNumStorageCores);
    auto l1In0Type = withLayout(in0Type, l1In0Layout);

    // Reuse an existing reshard if a prior DS matmul already produced the
    // same {1, kNumStorageCores} L1 width-sharded value from this in0
    // (common when gate and up projections share a fork-point producer).
    Value reshardedIn0;
    for (Operation *user : in0.getUsers()) {
      auto existingReshard = dyn_cast<ToMemoryConfigOp>(user);
      if (!existingReshard) {
        continue;
      }
      auto resultType = mlir::dyn_cast<RankedTensorType>(
          existingReshard.getResult().getType());
      if (!resultType) {
        continue;
      }
      auto resultLayout =
          mlir::dyn_cast<TTNNLayoutAttr>(resultType.getEncoding());
      if (resultLayout && resultLayout == l1In0Layout) {
        reshardedIn0 = existingReshard.getResult();
        break;
      }
    }

    if (!reshardedIn0) {
      reshardedIn0 = builder
                         .create<ToMemoryConfigOp>(matmulOp.getLoc(), l1In0Type,
                                                   in0)
                         .getResult();
    }
    matmulOp.setOperand(0, reshardedIn0);
  }

  // --- 3. Set program config and compute config ---
  llvm::errs() << "[DS] applyDRAMShardedTransformation called for matmul at "
               << matmulOp.getLoc() << "\n";
  matmulOp.setMatmulProgramConfigAttr(
      mlir::cast<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          matmulAttrs.matmulProgramConfig.value()));

  if (matmulAttrs.computeKernelConfig.has_value()) {
    matmulOp.setComputeConfigAttr(*matmulAttrs.computeKernelConfig);
  }

  // --- 4. Strip activation, insert separate activation op after ---
  // Benchmarking showed fusing activation into the DRAM-sharded program config
  // is ~38% slower (see unified_ds_analysis.md §4 — stalls the matmul pipeline
  // on per-core activation of many output tiles). A separate elementwise op
  // runs across all cores with full parallelism at negligible extra cost.
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
      // Replace all downstream uses of the matmul result with the activation
      // result, keeping the activation op's own use of the matmul result.
      matmulResult.replaceAllUsesExcept(activationOp->getResult(0),
                                        activationOp);
    }
  }
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

  // DRAM-sharded path: full IR transformation (input reshards, config,
  // activation split).
  bool isDRAMSharded =
      mlir::isa<MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr>(
          programConfig);
  llvm::errs() << "[DS] applyOpSpecificAttrs: op=" << op->getName()
               << " isDRAMSharded=" << isDRAMSharded
               << " isMatmul=" << (matmulOp != nullptr)
               << " score.isPrevalidated=" << candidate.score.isPrevalidated
               << "\n";
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
