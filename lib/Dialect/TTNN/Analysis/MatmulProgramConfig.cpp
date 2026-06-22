// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MatmulProgramConfig.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {

static inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Compute the maximum number of tiles that can fit in the destination register.
// This depends on the compute kernel config settings:
//   - Base: 16 tiles (for standard 32x32 tiles)
//   - If dst_full_sync_en = false: divide by 2 -> 8 tiles (typical case)
//   - If fp32_dest_acc_en = true: divide by 2 -> 4 tiles
// If config is null or properties are not set, returns default of 8.
static int64_t getMaxSubblockSize(DeviceComputeKernelConfigAttr computeConfig) {
  // Default max subblock size (assumes dst_full_sync_en=false which is typical)
  int64_t maxSubblockSize = 8;

  if (!computeConfig) {
    return maxSubblockSize;
  }

  // If dst_full_sync_en is explicitly set to true, we have double the registers
  if (auto dstFullSyncAttr = computeConfig.getDstFullSyncEn()) {
    if (dstFullSyncAttr.getValue()) {
      maxSubblockSize *= 2;
    }
  }

  // If fp32_dest_acc_en is explicitly set to true, registers are halved
  if (auto fp32DestAccAttr = computeConfig.getFp32DestAccEn()) {
    if (fp32DestAccAttr.getValue()) {
      maxSubblockSize /= 2;
    }
  }

  return maxSubblockSize;
}

// Find the largest divisor of 'value' that is <= 'maxDivisor'.
static inline int64_t largestDivisorUpTo(int64_t value, int64_t maxDivisor) {
  for (int64_t d = std::min(value, maxDivisor); d >= 1; --d) {
    if (value % d == 0) {
      return d;
    }
  }
  return 1;
}

// Matmul Program Config Constraints (from matmul_op.cpp):
// --------------------------------------------------------
// Non-zero constraints:
//   - in0_block_w != 0
//   - out_subblock_h != 0
//   - out_subblock_w != 0
//   - out_block_h != 0
//   - out_block_w != 0
//   - per_core_M != 0
//   - per_core_N != 0
//
// Divisibility constraints:
//   - Kt % in0_block_w == 0
//   - per_core_M % out_subblock_h == 0
//   - per_core_N % out_subblock_w == 0
//   - per_core_M % out_block_h == 0
//   - per_core_N % out_block_w == 0
//   - out_block_h % out_subblock_h == 0
//   - out_block_w % out_subblock_w == 0
//
// Register constraints:
//   - out_subblock_w * out_subblock_h <= available_reg_count
//     (4-16 depending on fp32_dest_acc_en and dst_full_sync_en)
//
// L1 memory constraints:
//   - Circular buffers for input/output tiles must fit in L1 memory.
//   - out_block_h * out_block_w determines output CB size per core.
//   - in0_block_w * out_block_h determines in0 CB size per core.
//
// TODO(rpavlovicTT): Currently we set out_block_h = per_core_M and out_block_w
// = per_core_N, which may exceed L1 capacity for large tensors. A follow-up
// improvement is to generate multiple configs with different out_block_h/w
// values (divisors of per_core_M/N) and use OpModel validation to find a config
// that fits in L1. This would enable handling larger matmuls by trading off
// reuse for memory.
//
// Generate MatmulMultiCoreReuseMultiCast1DProgramConfig for width/height
// sharded output.
static mlir::Attribute
generateMatmul1DProgramConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                              int64_t Kt, TTNNLayoutAttr outputLayout,
                              TensorMemoryLayout outputMemLayout,
                              UnaryWithParamAttr fusedActivation,
                              int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
  int64_t numCores = gridX * gridY;

  bool mcastIn0 = (outputMemLayout == TensorMemoryLayout::WidthSharded);
  int64_t perCoreM, perCoreN;

  if (mcastIn0) {
    perCoreM = Mt;
    perCoreN = divUp(Nt, numCores);
  } else {
    perCoreM = divUp(Mt, numCores);
    perCoreN = Nt;
  }

  constexpr int64_t kLargeNtThreshold = 128;
  int64_t in0BlockW;
  if (!mcastIn0) {
    in0BlockW = Kt;
  } else {
    if (Nt > kLargeNtThreshold) {
      in0BlockW = (Kt % 2 == 0) ? 2 : 1;
    } else {
      if (Kt % 8 == 0) {
        in0BlockW = 8;
      } else if (Kt % 4 == 0) {
        in0BlockW = 4;
      } else if (Kt % 2 == 0) {
        in0BlockW = 2;
      } else {
        in0BlockW = 1;
      }
    }
  }

  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;
  int64_t outSubblockH = 1;
  // out_subblock_w must divide out_block_w (== perCoreN) evenly.
  // See matmul_op.cpp constraints: out_block_w % out_subblock_w == 0.
  int64_t outSubblockW = largestDivisorUpTo(outBlockW, maxSubblockSize);

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);
  auto hopCoresAttr = CoreRangeSetAttr::get(ctx, {});

  return MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      fuseBatch, /*fusedActivation=*/fusedActivation, mcastIn0,
      /*gather_in0=*/false, hopCoresAttr, /*num_global_cb_receivers=*/0,
      /*untilize_out=*/false);
}

// Generate MatmulMultiCoreReuseMultiCastProgramConfig for block sharded output.
static mlir::Attribute
generateMatmul2DProgramConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                              int64_t Kt, TTNNLayoutAttr outputLayout,
                              UnaryWithParamAttr fusedActivation,
                              int64_t maxSubblockSize, bool fuseBatch) {
  auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);

  int64_t perCoreM = divUp(Mt, gridY);
  int64_t perCoreN = divUp(Nt, gridX);

  int64_t in0BlockW = (Kt % 2 == 0) ? 2 : 1;
  int64_t outSubblockH = 1;

  // out_subblock_w must divide out_block_w (== perCoreN) evenly.
  // See matmul_op.cpp constraints: out_block_w % out_subblock_w == 0.
  int64_t outSubblockW = largestDivisorUpTo(perCoreN, maxSubblockSize);
  int64_t outBlockH = perCoreM;
  int64_t outBlockW = perCoreN;

  auto gridAttr = CoreCoordAttr::get(ctx, gridX, gridY);

  return MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      ctx, gridAttr, static_cast<uint64_t>(in0BlockW),
      static_cast<uint64_t>(outSubblockH), static_cast<uint64_t>(outSubblockW),
      static_cast<uint64_t>(outBlockH), static_cast<uint64_t>(outBlockW),
      static_cast<uint64_t>(perCoreM), static_cast<uint64_t>(perCoreN),
      /*transpose_mcast=*/false, /*fusedActivation=*/fusedActivation,
      fuseBatch);
}

std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout) {
  if (!outputLayout || !outputLayout.hasShardedL1TensorMemoryLayout()) {
    return std::nullopt;
  }

  TensorMemoryLayout outputMemLayout = outputLayout.getMemLayout().getValue();
  if (outputMemLayout != TensorMemoryLayout::WidthSharded &&
      outputMemLayout != TensorMemoryLayout::HeightSharded &&
      outputMemLayout != TensorMemoryLayout::BlockSharded) {
    return std::nullopt;
  }

  auto resultType =
      mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> outShape = resultType.getShape();
  if (outShape.size() < 2) {
    return std::nullopt;
  }

  // Get input A, input B shapes and activation from the op.
  auto [inputA, inputB, activation] =
      llvm::TypeSwitch<Operation *,
                       std::tuple<Value, Value, std::optional<StringRef>>>(op)
          .Case<ttnn::MatmulOp, ttnn::LinearOp>([](auto matmulOp) {
            std::optional<StringRef> act;
            if (auto actAttr = matmulOp.getActivationAttr()) {
              act = actAttr.getValue();
            }
            return std::make_tuple(matmulOp.getA(), matmulOp.getB(), act);
          })
          .Default([](Operation *) {
            return std::make_tuple(nullptr, nullptr,
                                   std::optional<StringRef>{});
          });

  if (!inputA || !inputB) {
    return std::nullopt;
  }

  auto inputAType = mlir::dyn_cast<RankedTensorType>(inputA.getType());
  auto inputBType = mlir::dyn_cast<RankedTensorType>(inputB.getType());
  if (!inputAType || !inputBType) {
    return std::nullopt;
  }
  llvm::ArrayRef<int64_t> aShape = inputAType.getShape();
  llvm::ArrayRef<int64_t> bShape = inputBType.getShape();
  if (aShape.size() < 2 || bShape.size() < 2) {
    return std::nullopt;
  }

  // Check if all batch dimensions of input B are 1.
  // fuse_batch can only be true when all batch dims of B are 1.
  bool fuseBatch = true;
  for (size_t i = 0; i < bShape.size() - 2; ++i) {
    if (bShape[i] != 1) {
      fuseBatch = false;
      break;
    }
  }

  int64_t M = outShape[outShape.size() - 2];
  int64_t N = outShape[outShape.size() - 1];
  int64_t K = aShape[aShape.size() - 1];
  int64_t Mt = divUp(M, TILE_HEIGHT);
  int64_t Nt = divUp(N, TILE_WIDTH);
  int64_t Kt = divUp(K, TILE_WIDTH);

  MLIRContext *ctx = op->getContext();
  UnaryWithParamAttr fusedActivation =
      ttnn::utils::getActivationAttr(ctx, activation);

  // Get compute kernel config from the operation to determine max subblock size
  DeviceComputeKernelConfigAttr computeConfig = nullptr;
  if (auto computeConfigOp = dyn_cast<TTNNComputeKernelConfigOpInterface>(op)) {
    computeConfig = computeConfigOp.getComputeConfigAttr();
  }
  int64_t maxSubblockSize = getMaxSubblockSize(computeConfig);

  // Batched matmul (fuse_batch=false, i.e. in1 has a non-unit batch dim):
  // sharded outputs hit a hang with the mcast program configs
  // (https://github.com/tenstorrent/tt-metal/issues/42572), so we opt out
  // of emitting any program config here and let tt-metal's runtime auto-picker
  // choose. Once the issue is resolved, op itself would reject such configs
  // and we can remove the fuse_batch check here.
  if (!fuseBatch) {
    return std::nullopt;
  }

  if (outputMemLayout == TensorMemoryLayout::BlockSharded) {
    // 2D mcast requires both grid dims > 1. On a degenerate grid (one dim == 1)
    // fall back to 1D mcast: tt-metal treats BlockSharded + 1D-grid as an
    // intentional alias for WidthSharded/HeightSharded (see
    // ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp
    // ~L197), and the 1D mcast direction must follow which grid dim is
    // degenerate, not outputMemLayout (otherwise mcast_in0 picks the wrong
    // direction).
    auto [gridX, gridY] = utils::getPhysicalGridDimensions(outputLayout);
    if (gridX > 1 && gridY > 1) {
      return generateMatmul2DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                           fusedActivation, maxSubblockSize,
                                           fuseBatch);
    }
    outputMemLayout = (gridY == 1) ? TensorMemoryLayout::WidthSharded
                                   : TensorMemoryLayout::HeightSharded;
  }

  return generateMatmul1DProgramConfig(ctx, Mt, Nt, Kt, outputLayout,
                                       outputMemLayout, fusedActivation,
                                       maxSubblockSize, fuseBatch);
}

// ============================================================================
// DRAM-sharded matmul config generation
// ============================================================================

static constexpr int64_t kTileSize = 32;

static int64_t padToDRAMBanks(int64_t n, int64_t numBanks) {
  int64_t lcm = kTileSize * numBanks;
  return ((n + lcm - 1) / lcm) * lcm;
}

std::optional<DRAMShardParams>
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

TTNNLayoutAttr buildDRAMShardedWeightLayout(MLIRContext *ctx,
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
TTNNLayoutAttr buildL1ShardedLayout(MLIRContext *ctx, TTNNLayoutAttr origLayout,
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

MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
buildDRAMShardedProgramConfig(MLIRContext *ctx, const DRAMShardParams &p,
                              UnaryWithParamAttr fusedAct) {
  return MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
      ctx, p.in0BlockW, p.perCoreM, p.perCoreN, fusedAct);
}

DeviceComputeKernelConfigAttr
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

} // namespace mlir::tt::ttnn
