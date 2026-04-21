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
  int64_t numBanks;    // DRAM banks for weight sharding
  int64_t numCores;    // compute cores for activation sharding
  int64_t nPadded;     // N padded to lcm(tileSize, tileSize * numBanks)
  int64_t shardH;      // shard height in elements = K
  int64_t shardW;      // shard width in elements = nPadded / numBanks
  int64_t kTiles;      // K / tileSize
  int64_t shardWTiles; // shardW / tileSize
  int64_t in0BlockW;   // tiles per K-loop iteration per core
  int64_t perCoreM;    // output tile rows per core
  int64_t perCoreN;    // output tile cols per core
  int64_t in0ShardW;   // L1 shard width for in0 = K / numCores
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
static std::optional<DRAMShardParams>
computeShardParams(int64_t M, int64_t K, int64_t N, int64_t numBanks,
                   int64_t numCores, ttcore::DataType weightDataType) {
  DRAMShardParams p;
  p.K = K;
  p.N = N;
  p.M = M;
  p.numBanks = numBanks;
  p.numCores = numCores;
  p.nPadded = padToDRAMBanks(N, numBanks);
  p.shardH = K;
  p.shardW = p.nPadded / numBanks;
  p.kTiles = K / kTileSize;
  p.shardWTiles = p.shardW / kTileSize;
  p.perCoreM = M / kTileSize;
  p.perCoreN = (N / kTileSize) / numCores;
  p.in0ShardW = K / numCores;
  p.weightDataType = weightDataType;

  // Choose in0BlockW to fit in L1.  The budget model matches the CB
  // allocations in tt-metal's DRAM-sharded matmul program factory
  // (matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp):
  //
  //   in0 CB : in0BlockW * perCoreM * bf16Tile  (2x when numBlocks > 1)
  //   in1 CB : in0BlockW * perCoreNSender * bfp8Tile  (3x when numBlocks > 1)
  //   out CB : perCoreM * perCoreN * bf16Tile   (1x, no double-buffer)
  //   interm0: perCoreM * perCoreN * fp32Tile   (fp32 dest accumulator)
  //   in2 CB : perCoreM * kPerCore * bf16Tile   (full in0 shard)
  //   out_reshard: perCoreM * perCoreN * bf16Tile
  //
  // L1 available after base reservation: 1,499,136 - 103,712 = 1,395,424.
  // We subtract an additional ~10% for kernel binary sizes on compute cores
  // and runtime scratch buffers that are not modeled here.
  static constexpr int64_t kL1Available = 1250000;
  static constexpr int64_t kBf16Tile = 2048;
  static constexpr int64_t kBfp8Tile = 1088;
  static constexpr int64_t kBfp4Tile = 576;
  static constexpr int64_t kFp32Tile = 4096;

  int64_t kWeightTile =
      (weightDataType == ttcore::DataType::BFP_BFloat4) ? kBfp4Tile : kBfp8Tile;

  int64_t kPerCore = p.kTiles / numCores;
  // per_core_N_in1_sender = ceil(N_tiles / numBanks), same as shardWTiles.
  int64_t perCoreNSender = p.shardWTiles;

  // Fixed CBs (independent of in0BlockW):
  int64_t outCB = p.perCoreM * p.perCoreN * kBf16Tile;
  int64_t interm0CB = p.perCoreM * p.perCoreN * kFp32Tile;
  int64_t in2CB = p.perCoreM * kPerCore * kBf16Tile;
  int64_t outReshardCB = p.perCoreM * p.perCoreN * kBf16Tile;
  int64_t fixedCost = outCB + interm0CB + in2CB + outReshardCB;

  // If the fixed CBs alone don't fit (e.g. per_core_n too large on matmuls
  // with very wide outputs like LM head), the matmul cannot be DRAM sharded
  // on this core/bank config. Skip.
  if (fixedCost > kL1Available) {
    return std::nullopt;
  }

  p.in0BlockW = kPerCore;
  bool found = false;
  while (p.in0BlockW >= 1) {
    int64_t numBlocks = p.kTiles / p.in0BlockW;
    bool doubleBuf = numBlocks > 1;

    int64_t in0CB = p.in0BlockW * p.perCoreM * kBf16Tile * (doubleBuf ? 2 : 1);
    int64_t in1CB =
        p.in0BlockW * perCoreNSender * kWeightTile * (doubleBuf ? 3 : 1);

    if (fixedCost + in0CB + in1CB <= kL1Available &&
        kPerCore % p.in0BlockW == 0) {
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

/// Build compute kernel config: LoFi + packer_l1_acc + fp32_dest_acc.
static DeviceComputeKernelConfigAttr buildComputeConfig(MLIRContext *ctx) {
  return DeviceComputeKernelConfigAttr::get(
      ctx,
      /*mathFidelity=*/MathFidelity::LoFi,
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
      if ((K / kTileSize) % numComputeCores != 0) {
        return;
      }
      if ((N / kTileSize) % numComputeCores != 0) {
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
      auto pOpt = computeShardParams(M, K, N, numDRAMBanks, numComputeCores,
                                     weightDataType);
      if (!pOpt) {
        // Matmul cannot be DRAM sharded on this config (e.g. output too wide
        // for L1). Leave the original matmul in place.
        continue;
      }
      auto &p = *pOpt;

      auto weightLayout = mlir::cast<TTNNLayoutAttr>(weightType.getEncoding());
      auto in0Layout = mlir::cast<TTNNLayoutAttr>(in0Type.getEncoding());
      auto outLayout = mlir::cast<TTNNLayoutAttr>(outType.getEncoding());

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
      int64_t in0ShardWTiles = (K / kTileSize) / numComputeCores;
      auto l1In0Layout = buildL1ShardedLayout(ctx, in0Layout, in0ShardHTiles,
                                              in0ShardWTiles, numComputeCores);
      auto l1In0Type = withLayout(in0Type, l1In0Layout);
      auto l1In0MemConfig = buildL1ShardedMemoryConfig(
          ctx, M, K / numComputeCores, numComputeCores);

      auto in0Reshard = builder.create<ToMemoryConfigOp>(
          matmulOp.getLoc(), l1In0Type, in0, l1In0MemConfig);

      // --- 3. Build output type (L1 WIDTH_SHARDED) ---
      int64_t outShardHTiles = M / kTileSize;
      int64_t outShardWTiles = (N / kTileSize) / numComputeCores;
      auto l1OutLayout = buildL1ShardedLayout(ctx, outLayout, outShardHTiles,
                                              outShardWTiles, numComputeCores);
      auto l1OutType = withLayout(outType, l1OutLayout);

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
      auto computeConfig = buildComputeConfig(ctx);

      // --- 6. Create the new matmul (without activation) ---
      auto newMatmul = builder.create<MatmulOp>(
          matmulOp.getLoc(), l1OutType, in0Reshard.getResult(),
          weightReshard.getResult(), matmulOp.getTransposeA(),
          matmulOp.getTransposeB(),
          /*matmulProgramConfig=*/progConfig,
          /*activation=*/nullptr,
          /*computeConfig=*/computeConfig);

      // --- 7. Reshard output to the original matmul's output layout ---
      // The optimizer may have placed this matmul's output in any layout
      // (DRAM interleaved, L1 width_sharded, etc.). Build a memory config
      // that matches the original output layout so the to_memory_config op
      // produces a result tensor whose encoding matches its memory_config.
      auto outMemConfig = MemoryConfigAttr::get(outLayout, deviceGrid);
      auto outputDeshard = builder.create<ToMemoryConfigOp>(
          matmulOp.getLoc(), outType, newMatmul.getResult(), outMemConfig);

      // --- 8. Insert separate activation op if needed ---
      Value finalResult = outputDeshard.getResult();
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
