// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cmath>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDRAMSHARDEDMATMUL
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

static constexpr int64_t kTileSize = 32;

/// Parameters computed for a single DRAM-sharded matmul.
struct DRAMShardParams {
  int64_t K;           // inner dimension
  int64_t N;           // output width
  int64_t M;           // output height (batch * seq_len rows)
  int64_t numBanks;    // DRAM banks for weight sharding (compute cores)
  int64_t numIn0Cores; // storage cores for in0 (activation) sharding.
                       // NOT compute cores — the DRAM-sharded matmul
                       // factory always uses numBanks (12) DRAM bank
                       // cores for compute, regardless of this value.
  int64_t numOutCores; // storage cores for output sharding. Independent
                       // of numIn0Cores: the factory's out_reshard CB
                       // scatters compute output to any output grid for
                       // free (see unified_ds_analysis.md §14a).
  int64_t nPadded;     // N padded to lcm(tileSize, tileSize * numBanks)
  int64_t shardH;      // shard height in elements = K
  int64_t shardW;      // shard width in elements = nPadded / numBanks
  int64_t kTiles;      // K / tileSize
  int64_t shardWTiles; // shardW / tileSize
  int64_t in0BlockW;   // tiles per K-loop iteration per core
  int64_t perCoreM;    // output tile rows per output storage core
  int64_t perCoreN;    // output tile cols per output storage core
                       // = N_tiles / numOutCores
  int64_t in0ShardW;   // L1 shard width for in0 = K / numIn0Cores
  ttcore::DataType weightDataType; // BFP_BFloat8 or BFP_BFloat4
};

/// Pad N up to a multiple of (tileSize * numBanks) so it divides evenly
/// across DRAM banks at tile granularity.
static int64_t padToDRAMBanks(int64_t n, int64_t numBanks) {
  int64_t lcm = kTileSize * numBanks;
  return ((n + lcm - 1) / lcm) * lcm;
}

/// Compute DRAM shard parameters for a matmul with shape M x K times K x N.
/// Returns std::nullopt if the matmul cannot fit in L1 even with the smallest
/// in0BlockW.
///
/// `numIn0Cores` controls the in0 (activation) shard grid; `numOutCores`
/// controls the output shard grid. They are independent inside the factory
/// (see unified_ds_analysis.md §4f / §14a). When the original output layout
/// is already L1 width-sharded with a compatible grid, callers can pass
/// `numOutCores != numIn0Cores` to have the matmul write directly onto that
/// grid and skip a downstream reshard.
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

  // Choose in0BlockW to fit in L1.  All CBs are created on every core in the
  // bounding box (all_cores_in_rect_grid) by the factory.
  //
  // CB                New L1?  Size
  // in0 (c_0)         yes      in0BlockW * perCoreM * bf16Tile  (2x double-buf)
  // in1 (c_1)         yes      in0BlockW * shardWTiles * weightTile  (3x)
  // out (c_4)         yes      perCoreM * shardWTiles * bf16Tile
  // interm0 (c_5)     yes      perCoreM * shardWTiles * fp32Tile
  // in2 (c_2)         NO       globally allocated to in0 tensor buffer
  // out_reshard (c_6) NO       globally allocated to out tensor buffer
  //
  // in2 and out_reshard use set_globally_allocated_address() — they alias the
  // pre-existing in0/out L1 tensor buffers and add no new CB memory.  However,
  // those tensor buffers themselves occupy L1 and must be subtracted from the
  // budget before sizing CBs.
  //
  static constexpr int64_t kBf16Tile = 2048;
  static constexpr int64_t kBfp8Tile = 1088;
  static constexpr int64_t kBfp4Tile = 576;
  static constexpr int64_t kFp32Tile = 4096;

  int64_t kWeightTile =
      (weightDataType == ttcore::DataType::BFP_BFloat4) ? kBfp4Tile : kBfp8Tile;

  int64_t kPerCore = p.kTiles / numIn0Cores;
  // per_core_N on compute cores = ceil(N_tiles / numBanks) = shardWTiles.
  int64_t perCoreNCompute = p.shardWTiles;

  // Tensor buffers placed in L1 (in0 shard + out shard per storage core).
  //
  // For outTensorBuf we deliberately use numIn0Cores (not numOutCores) — i.e.
  // pretend the output is sharded on numIn0Cores even when we inherited a
  // wider output grid. The actual per-core output footprint is smaller with
  // a wider grid, so this is conservative. The reason: the pass has no global
  // L1 visibility (see unified_ds_analysis.md §4f), so it cannot see L1
  // buffers from co-running ops. Using the larger (older) outTensorBuf here
  // keeps the L1 budget identical to what the old pass computed, so the
  // in0_block_w selection is unchanged. Without this, decoupling the output
  // grid would free up budget, the pass would pick a larger in0_block_w, and
  // the resulting larger in1 CB pushes the CB region into address ranges
  // already occupied by other ops' tensor buffers — turning the prior
  // non-fatal CB clash warning into a hard validate_circular_buffer_region
  // failure.
  int64_t outTensorBufPerCore =
      p.perCoreM * ((N / kTileSize) / numIn0Cores) * kBf16Tile;
  int64_t in0TensorBuf = p.perCoreM * kPerCore * kBf16Tile;
  int64_t cbBudget = l1Available - in0TensorBuf - outTensorBufPerCore;

  // Fixed CBs (independent of in0BlockW, no double/triple buffering).
  int64_t outCB = p.perCoreM * perCoreNCompute * kBf16Tile;
  int64_t interm0CB = p.perCoreM * perCoreNCompute * kFp32Tile;
  int64_t fixedCost = outCB + interm0CB;

  // If the fixed CBs alone don't fit (e.g. per_core_n too large on matmuls
  // with very wide outputs like LM head), the matmul cannot be DRAM sharded
  // on this core/bank config. Skip.
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

/// Check if a weight tensor is bfp8 or bfp4 and in DRAM interleaved layout.
static bool isBfpDRAMInterleaved(Value weight) {
  auto rtt = mlir::dyn_cast<RankedTensorType>(weight.getType());
  if (!rtt) {
    return false;
  }

  // Check element type is bfp_bf8 or bfp_bf4 tile.
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

  // Check it has a TTNN layout with DRAM interleaved.
  auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(rtt.getEncoding());
  if (!layoutAttr) {
    return false;
  }
  if (!layoutAttr.hasInterleavedDRAMTensorMemoryLayout()) {
    return false;
  }

  return true;
}

/// Get the weight tile's data type (BFP_BFloat8 or BFP_BFloat4).
static ttcore::DataType getWeightDataType(Value weight) {
  auto rtt = mlir::cast<RankedTensorType>(weight.getType());
  auto tileType = mlir::cast<ttcore::TileType>(rtt.getElementType());
  return tileType.getDataType();
}

/// Get the 2D shape (K, N) from a weight tensor type.
static std::pair<int64_t, int64_t> getWeightKN(RankedTensorType rtt) {
  auto shape = rtt.getShape();
  assert(shape.size() == 2 && "Expected 2D weight tensor");
  return {shape[0], shape[1]};
}

/// Get M (first dim) from an activation tensor.
static int64_t getActivationM(RankedTensorType rtt) {
  auto shape = rtt.getShape();
  // Could be 2D (M x K) or 3D (batch x seq x K). Use product of all but last.
  int64_t M = 1;
  for (size_t i = 0; i < shape.size() - 1; i++) {
    M *= shape[i];
  }
  return M;
}

/// If `outLayout` is L1 width-sharded with a `<1 x C>` grid where `C` evenly
/// divides `nTiles`, return `C`. Otherwise return nullopt.
///
/// When this returns a value, the DRAM-sharded matmul can produce its output
/// directly on that grid, reusing `outLayout` as the matmul's result type and
/// avoiding the reshard the pass would otherwise emit. The matmul kernel cost
/// is unchanged — the factory's `out_reshard` CB scatters compute output to
/// any output grid for free (see unified_ds_analysis.md §14a).
static std::optional<int64_t> tryInheritOutGrid(TTNNLayoutAttr outLayout,
                                                int64_t nTiles) {
  if (!outLayout.hasShardedL1TensorMemoryLayout()) {
    return std::nullopt;
  }
  auto memLayoutOpt = outLayout.getMemLayoutOpt();
  if (!memLayoutOpt || *memLayoutOpt != TensorMemoryLayout::WidthSharded) {
    return std::nullopt;
  }
  auto gridShape = outLayout.getGrid().getShape();
  if (gridShape.size() != 2 || gridShape[0] != 1) {
    return std::nullopt;
  }
  int64_t outCores = gridShape[1];
  if (outCores <= 0 || nTiles % outCores != 0) {
    return std::nullopt;
  }
  return outCores;
}

/// Build a DRAM WIDTH_SHARDED TTNNLayoutAttr for the weight tensor.
static TTNNLayoutAttr buildDRAMShardedWeightLayout(MLIRContext *ctx,
                                                   TTNNLayoutAttr origLayout,
                                                   const DRAMShardParams &p) {
  // Grid: 1 x numBanks (DRAM bank logical coords).
  auto grid = ttcore::GridAttr::get(ctx, {1, p.numBanks});

  // Memref: kTiles x shardWTiles tiles of the weight's dtype in DRAM.
  auto tileType =
      ttcore::TileType::get(ctx, {kTileSize, kTileSize}, p.weightDataType);
  auto dramSpace = BufferTypeAttr::get(ctx, BufferType::DRAM);
  auto memrefType = MemRefType::get({p.kTiles, p.shardWTiles}, tileType,
                                    MemRefLayoutAttrInterface{}, dramSpace);

  auto memLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);

  return TTNNLayoutAttr::get(ctx, origLayout.getLinear(), grid, memrefType,
                             memLayout, /*tensorMesh=*/nullptr,
                             /*ignorePhysicalLayout=*/false,
                             /*exactGrid=*/true);
}

/// Build an L1 WIDTH_SHARDED TTNNLayoutAttr for the activation or output.
static TTNNLayoutAttr buildL1ShardedLayout(MLIRContext *ctx,
                                           TTNNLayoutAttr origLayout,
                                           int64_t shardHTiles,
                                           int64_t shardWTiles,
                                           int64_t numCores) {
  auto grid = ttcore::GridAttr::get(ctx, {1, numCores});
  auto tileType = ttcore::TileType::get(ctx, {kTileSize, kTileSize},
                                        ttcore::DataType::BFloat16);
  auto l1Space = BufferTypeAttr::get(ctx, BufferType::L1);
  auto memrefType = MemRefType::get({shardHTiles, shardWTiles}, tileType,
                                    MemRefLayoutAttrInterface{}, l1Space);
  auto memLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);

  return TTNNLayoutAttr::get(ctx, origLayout.getLinear(), grid, memrefType,
                             memLayout, /*tensorMesh=*/nullptr,
                             /*ignorePhysicalLayout=*/false,
                             /*exactGrid=*/false);
}

/// Build a MemoryConfigAttr for DRAM WIDTH_SHARDED with explicit core range
/// (0,0)-(numBanks-1, 0).
static MemoryConfigAttr buildDRAMShardedMemoryConfig(MLIRContext *ctx,
                                                     const DRAMShardParams &p) {
  auto tensorMemLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);
  auto bufferType = BufferTypeAttr::get(ctx, BufferType::DRAM);

  // Core range: (0,0) to (numBanks-1, 0) — DRAM bank logical coords.
  auto startCoord = CoreCoordAttr::get(ctx, 0, 0);
  auto endCoord = CoreCoordAttr::get(ctx, p.numBanks - 1, 0);
  auto coreRange = CoreRangeAttr::get(ctx, startCoord, endCoord);
  auto coreRangeSet = CoreRangeSetAttr::get(ctx, {coreRange});

  auto shardShape = ShapeAttr::get(ctx, {p.shardH, p.shardW});
  auto orientation = ShardOrientationAttr::get(ctx, ShardOrientation::RowMajor);
  auto shardSpec =
      ShardSpecAttr::get(ctx, coreRangeSet, shardShape, orientation);

  return MemoryConfigAttr::get(ctx, tensorMemLayout, bufferType, shardSpec);
}

/// Build a MemoryConfigAttr for L1 WIDTH_SHARDED on compute cores.
static MemoryConfigAttr buildL1ShardedMemoryConfig(MLIRContext *ctx,
                                                   int64_t shardH,
                                                   int64_t shardW,
                                                   int64_t numCores) {
  auto tensorMemLayout =
      TensorMemoryLayoutAttr::get(ctx, TensorMemoryLayout::WidthSharded);
  auto bufferType = BufferTypeAttr::get(ctx, BufferType::L1);

  auto startCoord = CoreCoordAttr::get(ctx, 0, 0);
  auto endCoord = CoreCoordAttr::get(ctx, numCores - 1, 0);
  auto coreRange = CoreRangeAttr::get(ctx, startCoord, endCoord);
  auto coreRangeSet = CoreRangeSetAttr::get(ctx, {coreRange});

  auto shardShape = ShapeAttr::get(ctx, {shardH, shardW});
  auto orientation = ShardOrientationAttr::get(ctx, ShardOrientation::RowMajor);
  auto shardSpec =
      ShardSpecAttr::get(ctx, coreRangeSet, shardShape, orientation);

  return MemoryConfigAttr::get(ctx, tensorMemLayout, bufferType, shardSpec);
}

/// Build the DRAM sharded matmul program config attribute.
static MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
buildDRAMShardedProgramConfig(MLIRContext *ctx, const DRAMShardParams &p,
                              UnaryWithParamAttr fusedAct) {
  return MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
      ctx, p.in0BlockW, p.perCoreM, p.perCoreN, fusedAct);
}

/// Build compute kernel config: math fidelity + packer_l1_acc + fp32_dest_acc.
///
/// Fidelity is dtype-dependent because "HiFi2 is free" only holds while the
/// kernel is DRAM-bandwidth bound:
///
///   BFP8 weights → HiFi2.
///     The kernel is DRAM-bound at decode (M=32). Math (2 cycles/MAC at HiFi2
///     vs 1 at LoFi) finishes well before DRAM and the kernel waits on DRAM
///     either way — measured ~1 μs cost on the Gate/Up shape (BFP8 sweep, see
///     unified_ds_analysis.md §14d). HiFi2 reads the full BF16 activation
///     mantissa (vs LoFi's truncated 5 bits) for better numerical accuracy at
///     no perf cost.
///
///   BFP4 weights → LoFi.
///     BFP4 halves DRAM bytes, pushing the kernel into FLOP-bound territory.
///     HiFi2 then directly slows the matmul: measured +45% at the production
///     blkw=8 (142 → 206 μs) and +11% at blkw=4 (193 → 214 μs) on the Gate/Up
///     shape — see up_matmul/results_summary.txt BFP4 sweep. Switching to
///     HiFi2 globally cost ~4 ms/token on llama 3.1 8B decode (32 layers ×
///     2 BFP4 matmuls × ~64 μs).
///
/// HiFi4 is never used: ~2× LoFi cost when FLOP-bound and reads no extra
/// precision from BFP{4,8} weights (7- and 3-bit shared mantissa respectively).
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

/// Build a RankedTensorType with a new layout attribute.
static RankedTensorType withLayout(RankedTensorType origType,
                                   TTNNLayoutAttr newLayout) {
  return RankedTensorType::get(origType.getShape(), origType.getElementType(),
                               newLayout);
}

// ============================================================================
// The pass implementation.
// ============================================================================

class TTNNDRAMShardedMatmulPass
    : public impl::TTNNDRAMShardedMatmulBase<TTNNDRAMShardedMatmulPass> {
public:
  using impl::TTNNDRAMShardedMatmulBase<
      TTNNDRAMShardedMatmulPass>::TTNNDRAMShardedMatmulBase;

  void runOnOperation() final {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    auto deviceGrid = ttcore::lookupDevice(moduleOp).getWorkerGrid();

    // Compute usable L1 the same way GreedyOptimizer does:
    //   chipDesc.getUsableL1Size() = l1Size - l1UnreservedBase
    //   getTensorL1UsageCap()      = 0.95 by default (or module attribute)
    ttcore::SystemDescAttr systemDesc = mlir::cast<ttcore::SystemDescAttr>(
        moduleOp->getAttr(ttcore::SystemDescAttr::name));
    int64_t l1Available =
        static_cast<int64_t>(ttnn::utils::getTensorL1UsageCap(moduleOp) *
                             systemDesc.getChipDescs()[0].getUsableL1Size());

    // Collect matmul ops that are eligible for DRAM sharding.
    // We collect first to avoid modifying while iterating.
    SmallVector<MatmulOp> eligibleMatmuls;

    moduleOp->walk([&](MatmulOp matmulOp) {
      Value weight = matmulOp.getB();

      // Must be bfp8 or bfp4 in DRAM interleaved.
      if (!isBfpDRAMInterleaved(weight)) {
        return;
      }

      // Must trace to constant/parameter args (i.e., a weight, not an
      // activation).
      if (!ttcore::valueTracesToConstantArgs(weight)) {
        return;
      }

      // Must be 2D weight tensor.
      auto weightType = mlir::cast<RankedTensorType>(weight.getType());
      if (weightType.getRank() != 2) {
        return;
      }

      // Get activation shape — must have M dimension.
      auto in0Type = mlir::cast<RankedTensorType>(matmulOp.getA().getType());
      int64_t M = getActivationM(in0Type);
      auto [K, N] = getWeightKN(weightType);

      // Sanity: dimensions must be tile-aligned and divisible by cores.
      if (M % kTileSize != 0 || K % kTileSize != 0 || N % kTileSize != 0) {
        return;
      }
      if ((K / kTileSize) % numStorageCores != 0) {
        return;
      }
      if ((N / kTileSize) % numStorageCores != 0) {
        return;
      }
      // Decode-only: the factory asserts num_blocks_per_shard == 1 when
      // per_core_M > 1, but our in0_block_w search may pick a smaller value
      // under L1 pressure. DS is also DRAM-bound only at small M; at prefill
      // the shape becomes compute-bound and a non-DS multi-cast factory wins
      // anyway. Filter out anything with more than one M-tile.
      if (M / kTileSize > 1) {
        return;
      }

      eligibleMatmuls.push_back(matmulOp);
    });

    if (eligibleMatmuls.empty()) {
      return;
    }

    // Process each eligible matmul.
    for (auto matmulOp : eligibleMatmuls) {
      OpBuilder builder(matmulOp);

      auto in0 = matmulOp.getA();
      auto weight = matmulOp.getB();
      auto in0Type = mlir::cast<RankedTensorType>(in0.getType());
      auto weightType = mlir::cast<RankedTensorType>(weight.getType());
      auto outType =
          mlir::cast<RankedTensorType>(matmulOp.getResult().getType());

      int64_t M = getActivationM(in0Type);
      auto [K, N] = getWeightKN(weightType);
      auto weightDataType = getWeightDataType(weight);

      auto outLayout = mlir::cast<TTNNLayoutAttr>(outType.getEncoding());
      // If the original output layout is already L1 width-sharded with a
      // grid that divides N_tiles, produce the matmul output directly on
      // that grid — preserves the original CoreRangeSet / virt_to_physical_map
      // exactly and skips the redundant <1xnumStorageCores> -> <1xC> reshard.
      auto inheritedOutCores = tryInheritOutGrid(outLayout, N / kTileSize);
      int64_t numOutCores = inheritedOutCores.value_or(numStorageCores);
      bool inheritOutLayout = inheritedOutCores.has_value();

      auto pOpt = computeShardParams(M, K, N, numDRAMBanks, numStorageCores,
                                     numOutCores, weightDataType, l1Available);
      if (!pOpt) {
        // Matmul cannot be DRAM sharded on this config (e.g. output too wide
        // for L1). Leave the original matmul in place.
        continue;
      }
      auto &p = *pOpt;

      auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
      auto in0Layout = mlir::cast<TTNNLayoutAttr>(in0Type.getEncoding());

      // --- 1. Reshard weight to DRAM WIDTH_SHARDED ---
      auto dramShardedWeightLayout =
          buildDRAMShardedWeightLayout(ctx, weightLayout, p);
      auto dramShardedWeightType =
          withLayout(weightType, dramShardedWeightLayout);
      auto dramShardedMemConfig = buildDRAMShardedMemoryConfig(ctx, p);

      auto weightReshard = builder.create<ToMemoryConfigOp>(
          matmulOp.getLoc(), dramShardedWeightType, weight,
          dramShardedMemConfig);

      // --- 2. Shard in0 to L1 WIDTH_SHARDED ---
      int64_t in0ShardHTiles = M / kTileSize;
      int64_t in0ShardWTiles = (K / kTileSize) / numStorageCores;
      auto l1In0Layout = buildL1ShardedLayout(ctx, in0Layout, in0ShardHTiles,
                                              in0ShardWTiles, numStorageCores);
      auto l1In0Type = withLayout(in0Type, l1In0Layout);
      auto l1In0MemConfig = buildL1ShardedMemoryConfig(
          ctx, M, K / numStorageCores, numStorageCores);

      auto in0Reshard = builder.create<ToMemoryConfigOp>(
          matmulOp.getLoc(), l1In0Type, in0, l1In0MemConfig);

      // --- 3. Build output type (L1 WIDTH_SHARDED) ---
      RankedTensorType l1OutType;
      if (inheritOutLayout) {
        // Reuse the original output layout exactly — preserves any
        // virt_to_physical_map / CoreRangeSet the optimizer assigned upstream.
        l1OutType = outType;
      } else {
        int64_t outShardHTiles = M / kTileSize;
        int64_t outShardWTiles = (N / kTileSize) / numStorageCores;
        auto l1OutLayout = buildL1ShardedLayout(
            ctx, outLayout, outShardHTiles, outShardWTiles, numStorageCores);
        l1OutType = withLayout(outType, l1OutLayout);
      }

      // --- 4. Do NOT fuse activation into the DRAM sharded program config ---
      // Benchmarking on llama 3 8B (gate_proj, 4096x14336 with silu) showed
      // that fusing activation into the DRAM sharded matmul kernel is ~38%
      // slower than running it as a separate op afterwards.  With per_core_n=56
      // tiles, each core applies the activation inline to many output tiles,
      // stalling the matmul pipeline.  A separate elementwise op runs across
      // all cores on interleaved data with full parallelism and only costs
      // ~14K ns vs ~165K ns overhead when fused.
      //
      // Instead, we strip the activation from the matmul and insert a separate
      // unary op (e.g. ttnn.silu) after the output deshard.  The runtime
      // requires sharded matmuls to have no user_fused_activation.
      UnaryWithParamAttr fusedAct; // null — intentionally not fused
      auto activationAttr = matmulOp.getActivationAttr();

      // --- 5. Build program config and compute config ---
      auto progConfig = buildDRAMShardedProgramConfig(ctx, p, fusedAct);
      auto computeConfig = buildComputeConfig(ctx, weightDataType);

      // --- 6. Create the new matmul (without activation) ---
      auto newMatmul = builder.create<MatmulOp>(
          matmulOp.getLoc(), l1OutType, in0Reshard.getResult(),
          weightReshard.getResult(), matmulOp.getTransposeA(),
          matmulOp.getTransposeB(),
          /*matmulProgramConfig=*/progConfig,
          /*activation=*/nullptr,
          /*computeConfig=*/computeConfig);

      // --- 7. Reshard output to the original matmul's output layout ---
      // When inheritOutLayout is true the matmul already produces outType, so
      // no deshard is needed. Otherwise build a memory config that matches
      // the original output layout (DRAM interleaved, L1 width_sharded with a
      // different grid, etc.) and reshard.
      Value matmulOutput = newMatmul.getResult();
      Value desharded;
      if (inheritOutLayout) {
        desharded = matmulOutput;
      } else {
        auto outMemConfig = MemoryConfigAttr::get(outLayout, deviceGrid);
        auto outputDeshard = builder.create<ToMemoryConfigOp>(
            matmulOp.getLoc(), outType, matmulOutput, outMemConfig);
        desharded = outputDeshard.getResult();
      }

      // --- 8. Insert separate activation op if needed ---
      Value finalResult = desharded;
      if (activationAttr) {
        auto actStr = activationAttr.getValue();
        std::string opName;
        if (actStr == "silu") {
          opName = "ttnn.silu";
        } else if (actStr == "relu") {
          opName = "ttnn.relu";
        } else if (actStr == "gelu") {
          opName = "ttnn.gelu";
        }
        if (!opName.empty()) {
          auto *activationOp =
              builder.create(matmulOp.getLoc(), StringAttr::get(ctx, opName),
                             /*operands=*/ValueRange{finalResult},
                             /*types=*/TypeRange{outType});
          finalResult = activationOp->getResult(0);
        }
      }

      // Replace all uses and erase.
      matmulOp.getResult().replaceAllUsesWith(finalResult);
      matmulOp.erase();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
