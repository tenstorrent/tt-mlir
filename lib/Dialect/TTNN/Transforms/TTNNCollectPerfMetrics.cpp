// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Support/Logger.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <sys/types.h>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNCOLLECTPERFMETRICS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

struct OperationMetrics {
  std::string opName;
  std::string location;
  bool isSharded = false;
  bool isSpilledToDRAM = false;
  bool hasSystemMemory = false;
  std::string layoutInfo;
};

struct AggregatedMetrics {
  uint64_t totalOps = 0;
  uint64_t totalOpsWithOutputTensor = 0;
  uint64_t totalShardableOps = 0;
  uint64_t shardedOps = 0;
  uint64_t effectivelyShardedOps = 0;
  uint64_t shardedAndSpilledOps = 0;
  uint64_t dramSpilledOps = 0;
  uint64_t systemMemoryOps = 0;

  double shardedPercentage = 0.0;
  double effectivelyShardedPercentage = 0.0;
  double shardedAndSpilledPercentage = 0.0;
  double spilledPercentage = 0.0;
  double systemMemoryPercentage = 0.0;

  void calculatePercentages() {
    if (totalShardableOps == 0) {
      return;
    }
    shardedPercentage =
        (static_cast<double>(shardedOps) / totalShardableOps) * 100.0;
    effectivelyShardedPercentage =
        (static_cast<double>(effectivelyShardedOps) / totalShardableOps) *
        100.0;
    shardedAndSpilledPercentage =
        (static_cast<double>(shardedAndSpilledOps) / totalShardableOps) * 100.0;
    spilledPercentage =
        (static_cast<double>(dramSpilledOps) / totalShardableOps) * 100.0;
    systemMemoryPercentage =
        (static_cast<double>(systemMemoryOps) / totalShardableOps) * 100.0;
  }
};

inline bool isValidWeightForTargetCalculation(mlir::Value v);
bool constEvalDoesNotChangeTensorVolume(ttcore::LoadCachedOp loadCachedOp);
uint64_t bytesAtBfp8(RankedTensorType t);
uint64_t getNumTileMatmuls(Value lhs, Value result, bool transposeA);

// Tile dimensions used to round matmul / SDPA dims up to whole 32x32 tiles.
constexpr int64_t kTileHeight = ttcore::TileType::getDefaultShape()[0];
constexpr int64_t kTileWidth = ttcore::TileType::getDefaultShape()[1];

// Round `x` up to a whole number of tiles along a 32-wide dimension.
inline uint64_t roundUpToTiles(uint64_t x) {
  return (x + kTileWidth - 1) / kTileWidth;
}

//===----------------------------------------------------------------------===//
// Performance Targets and Roofline Calculation
//
// Used for estimating top perf for any model on the current hardware.
//
// This calculation should be extremely conservative and always try to stay on
// the side of overestimating the time an op takes (i.e. underestimating
// performance). Underestimating the time reports false, optimistic results
// which can cause a lot of harm.
//
// Process:
// For each matmul op (including linear and SDPA) we calculate the time it would
// take to read its weights from DRAM at theoretical bandwidth and the time it
// would take to compute the matmul at 100% utilization. The max of these is the
// roofline. The roofline is then divided by a utilization factor (0.7) to get a
// realistic time estimate, since peak hardware throughput is rarely achievable.
//
// Assumptions and limitations:
// * Single chip only, skip multichip for now
// * All weights are assumed to be bfp8
// * All compute cores are to be used for every matmul
// * All matmuls use HiFi2 math fidelity (32 cycles per tile-mul)
// * All ops other than matmul are skipped (vision models with convs will have
// inaccurate estimates)
//
// Specs used:
// * Wormhole N150, N300 - 288 GB/s DRAM bandwidth, 1.0 GHz AICLK
// * Wormhole Galaxy - 336 GB/s DRAM bandwidth, 1.0 GHz AICLK
// * Blackhole P150, P300, QB2, Galaxy - 512 GB/s DRAM bandwidth, 1.35 GHz AICLK
// * The rest is read from the system descriptor
//
//===----------------------------------------------------------------------===//
struct PerfTargets {
  ttcore::Arch arch = ttcore::Arch::WormholeB0;
  unsigned numChips = 0;
  uint64_t dramBandwidthBytesPerSec = 0;
  uint64_t aiclkHz = 0;
  uint64_t numTensixCores = 0;
  uint64_t cyclesPerTileMatmul = 0;

  MathFidelity defaultMathFidelity = MathFidelity::HiFi2;

  // Per-op classification across every weight matmul / linear / SDPA op
  // in the forward function. Counts use peak numbers before the
  // utilization derate.
  uint64_t dramBoundOps = 0;
  uint64_t computeBoundOps = 0;

  // Sum of weight (rhs) scalars / bytes across every counted op.
  uint64_t paramCount = 0;
  uint64_t paramMemoryBytes = 0;

  // Ops dropped because their weight source couldn't be verified see
  // isValidWeightForTargetCalculation.
  uint64_t skippedOps = 0;

  // rooflineMs is the pure theoretical ceiling — sum of per-op
  // max(dram_us, compute_us) at peak hardware, in milliseconds.
  // topPerfEstimateMs = rooflineMs / kUtilizationFactor is the
  // realistic single-step wall-clock estimate.
  double rooflineMs = 0.0;
  double topPerfEstimateMs = 0.0;

  double getUsToReadWeightsFromDRAM(uint64_t weightBytes) {
    if (dramBandwidthBytesPerSec == 0) {
      return 0.0;
    }
    double us = static_cast<double>(weightBytes) / dramBandwidthBytesPerSec;
    return us * 1e6; // convert to us
  }

  // Total cycles for `numMatmulTiles` 32x32x32 tile matmuls, scaled by the
  // per-tile cost of the given math fidelity.
  uint64_t getNumCycles(uint64_t numMatmulTiles, MathFidelity mathFidelity) {
    switch (mathFidelity) {
    case MathFidelity::LoFi:
      return numMatmulTiles * 16;
    case MathFidelity::HiFi2:
      return numMatmulTiles * 32;
    case MathFidelity::HiFi3:
      return numMatmulTiles * 48;
    case MathFidelity::HiFi4:
      return numMatmulTiles * 64;
    }
    llvm_unreachable("Unsupported MathFidelity");
  }

  // Walk all ops in the function and calculate the dram and compute time
  // estimates for each.
  void calculatePerfTargets(func::FuncOp funcOp) {
    funcOp->walk([&](Operation *op) {
      SmallVector<Value, 2> weights;
      uint64_t tileMuls = 0;

      bool matched =
          llvm::TypeSwitch<Operation *, bool>(op)
              .Case<MatmulOp, LinearOp>([&](auto m) {
                weights = {m.getB()};
                tileMuls = getNumTileMatmuls(m.getA(), m.getResult(),
                                             m.getTransposeA());
                return true;
              })
              .Case<ScaledDotProductAttentionDecodeOp,
                    PagedScaledDotProductAttentionDecodeOp>([&](auto m) {
                weights = {m.getKey(), m.getValue()};
                // Assume decode compute to be negligible and calculate dram
                // time only.
                tileMuls = 0;
                return true;
              })
              .Case<ScaledDotProductAttentionOp>(
                  [&](ScaledDotProductAttentionOp m) {
                    if (m.getAttentionMask()) {
                      // If there is an attention mask, we don't know what the
                      // roofline for this op is so we skip it to avoid
                      // overestimating.
                      skippedOps++;
                      llvm::errs() << "TTNNCollectPerfMetrics: skipping "
                                   << op->getName()
                                   << " — dynamic attention mask present\n";
                      return false;
                    }

                    weights = {m.getKey(), m.getValue()};
                    tileMuls = sdpaPrefillTileMuls(
                        m.getQuery(), m.getKey(), m.getValue(),
                        m.getSlidingWindowSize(), m.getIsCausal());
                    return true;
                  })
              .Default([](Operation *) { return false; });

      if (!matched) {
        return;
      }

      for (Value w : weights) {
        if (!isValidWeightForTargetCalculation(w)) {
          skippedOps++;
          llvm::errs() << "TTNNCollectPerfMetrics: skipping " << op->getName()
                       << " — input is not DRAM-resident (producer: "
                       << (w.getDefiningOp()
                               ? w.getDefiningOp()->getName().getStringRef()
                               : "<block-arg>")
                       << ")\n";
          return;
        }
      }

      // Count params and size at BFP8 bytes across all weights.
      uint64_t weightScalars = 0;
      uint64_t weightBytes = 0;
      for (Value w : weights) {
        auto t = mlir::cast<RankedTensorType>(w.getType());
        weightScalars += static_cast<uint64_t>(t.getNumElements());
        weightBytes += bytesAtBfp8(t);
      }
      paramCount += weightScalars;
      paramMemoryBytes += weightBytes;

      // DRAM time estimate in us
      double dramUs = getUsToReadWeightsFromDRAM(weightBytes);

      // Compute time estimate in us
      uint64_t computeCycles = getNumCycles(tileMuls, defaultMathFidelity);
      double computeUs =
          (numTensixCores == 0 || aiclkHz == 0)
              ? 0.0
              : static_cast<double>(computeCycles) * 1.0e6 /
                    static_cast<double>(numTensixCores * aiclkHz);

      // Take max, accumulate, classify.
      double opUs = std::max(dramUs, computeUs);
      rooflineMs += opUs / 1000.0;
      bool dramBound = dramUs >= computeUs;
      if (dramBound) {
        dramBoundOps++;
      } else {
        computeBoundOps++;
      }

      TTMLIR_DEBUG(::ttmlir::LogComponent::PerfTargets,
                   "{0}\n"
                   "  inputs        = [{1}]\n"
                   "  outputs       = [{2}]\n"
                   "  weights       = [{3}]\n"
                   "  weight_scalars= {4}\n"
                   "  weight_bytes  = {5} (BFP8)\n"
                   "  tile_muls     = {6}\n"
                   "  compute_cycles= {7}\n"
                   "  dram_us       = {8}\n"
                   "  compute_us    = {9}\n"
                   "  op_us         = {10} ({11}-bound)\n"
                   "  running: roofline_ms={12} dram_bound_ops={13} "
                   "compute_bound_ops={14} skipped_ops={15}",
                   op->getName().getStringRef(), op->getOperandTypes(),
                   op->getResultTypes(),
                   mlir::TypeRange(mlir::ValueRange(weights)), weightScalars,
                   weightBytes, tileMuls, computeCycles, dramUs, computeUs,
                   opUs, (dramBound ? "DRAM" : "compute"), rooflineMs,
                   dramBoundOps, computeBoundOps, skippedOps);
    });
  }

private:
  // === Per-op-type extractors ===

  uint64_t sdpaPrefillTileMuls(Value query, Value key, Value value,
                               std::optional<int32_t> slidingWindowSize,
                               bool isCausal) {
    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

    assert(qType && kType && vType && qType.getShape().size() == 4 &&
           kType.getShape().size() == 4 && vType.getShape().size() == 4 &&
           "SDPA Q/K/V must be ranked 4-D tensors");

    auto qShape = qType.getShape();
    auto kShape = kType.getShape();
    auto vShape = vType.getShape();

    // query: `[B x Hq x Sq x Dk]`.
    // key: `[B x Hkv x Sk x Dk]`.
    // value: `[B x Hkv x Sk x Dv]`
    // Hq - number of query heads
    // Hkv - number of key/value heads, can be different from Hq in multi-query
    // Sq - sequence length of queries
    // Sk - sequence length of keys/values
    // Dk - hidden dimension of keys
    // Dv - hidden dimension of values (usually Dk = Dv)

    auto B = qShape[0];
    auto Hq = qShape[1];
    auto Sq = qShape[2];
    auto Dk = qShape[3];
    auto Sk = kShape[2];
    auto Dv = vShape[3];

    auto effectiveSk = Sk;
    if (slidingWindowSize.has_value()) {
      effectiveSk = std::min<int64_t>(slidingWindowSize.value(), Sk);
    }

    // Make sure the shapes for all inputs match
    assert(B == kShape[0] && B == vShape[0] &&
           "Batch size must match across Q, K, V");
    assert(Dk == kShape[3] && Dv == vShape[3] &&
           "Head dimension of Q must match K and V");
    assert(Sk == vShape[2] && "Sequence length of K and V must match");

    auto SqTiles = roundUpToTiles(Sq);
    auto DkTiles = roundUpToTiles(Dk);
    auto effectiveSkTiles = roundUpToTiles(effectiveSk);
    auto DvTiles = roundUpToTiles(Dv);

    // QK^T matmul compute
    // B * Hq * Sq * Dk * Sk
    auto qktTileMuls = B * Hq * SqTiles * DkTiles * effectiveSkTiles;

    // V matmul compute
    // B * Hq * Sq * Sk * Dv
    auto vTileMuls = B * Hq * SqTiles * effectiveSkTiles * DvTiles;

    auto totalTileMuls = qktTileMuls + vTileMuls;
    if (isCausal && !slidingWindowSize.has_value()) {
      // Rough estimate.
      totalTileMuls /= 2;
    }

    return totalTileMuls;
  }
};

// In order to not overestimate the time it takes to read weights from DRAM, we
// need to be sure the tensor is actually representative of the real model
// weights. We check the following conditions:
// * Tensor needs to be in DRAM.
// * A tensor that is a direct arg marked as a weight
// (Parameter/Constant/kv_cache) is valid.
// * A tensor produced by a load_cached op is valid, but only when the tensor
// volume is not changed in the const eval func. This is so that we prevent
// counting a tensor that has been broadcasted.
inline bool isValidWeightForTargetCalculation(mlir::Value v) {
  auto vType = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!vType) {
    return false;
  }

  if (utils::getBufferTypeFromTensor(vType) != BufferType::DRAM) {
    return false;
  }

  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
    auto funcOp =
        mlir::dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
    if (!funcOp) {
      return false;
    }

    unsigned idx = blockArg.getArgNumber();
    ttcore::ArgumentType argType = ttcore::getFunctionArgumentType(funcOp, idx);
    return argType == ttcore::ArgumentType::Parameter ||
           argType == ttcore::ArgumentType::Constant ||
           ttcore::isKVCacheArgument(funcOp, idx);
  }

  Operation *definingOp = v.getDefiningOp();

  if (!definingOp) {
    return false;
  }

  ttcore::LoadCachedOp loadCachedOp =
      mlir::dyn_cast<ttcore::LoadCachedOp>(definingOp);
  if (!loadCachedOp) {
    return false;
  }

  if (!constEvalDoesNotChangeTensorVolume(loadCachedOp)) {
    return false;
  }

  return true;
}

// Compare the total scalar volume across the load_cached op's inputs vs
// its results. Equal totals mean the const-eval function only reshaped /
// retiled / fused weights — the post-const-eval tensor is still backed
// by the same number of bytes streamed from DRAM, so it's safe to count
// against the roofline. A larger output volume implies a broadcast (e.g.
// repeat_interleave Hkv → Hq); in that case the post-const-eval tensor
// is *bigger* than what DRAM actually carries, so we reject it.
bool constEvalDoesNotChangeTensorVolume(ttcore::LoadCachedOp loadCachedOp) {
  auto totalVolume = [](auto range) {
    uint64_t total = 0;
    for (mlir::Value v : range) {
      if (auto rt = mlir::dyn_cast<RankedTensorType>(v.getType())) {
        total += static_cast<uint64_t>(rt.getNumElements());
      }
    }
    return total;
  };
  return totalVolume(loadCachedOp.getInputs()) ==
         totalVolume(loadCachedOp.getResults());
}

// BFP8 (block floating-point 8-bit) packs 16 scalars per row into 16 mantissa
// bytes + 1 shared 8-bit exponent byte (17 B per 16 scalars). A 32x32 tile is
// 4 faces of 16x16, each face = 16 rows × (16 mantissa + 1 exponent) B =
// 272 B → 4 × 272 = 1088 B for 1024 scalars = 17/16 = 1.0625 B / scalar.
// Matches BFLOAT8_B_TILE_HW = TILE_HW + 64 in tt-metal constants.hpp.
uint64_t bytesAtBfp8(RankedTensorType t) {
  return static_cast<uint64_t>(
      static_cast<double>(t.getNumElements()) * 17.0 / 16.0 + 0.5);
}

uint64_t getNumTileMatmuls(Value lhs, Value result, bool transposeA) {
  ArrayRef<int64_t> lhsShape =
      mlir::cast<RankedTensorType>(lhs.getType()).getShape();
  ArrayRef<int64_t> resultShape =
      mlir::cast<RankedTensorType>(result.getType()).getShape();

  // Both operands must be at least rank 2 to extract M/K/N.
  if (lhsShape.size() < 2 || resultShape.size() < 2) {
    return 0;
  }

  int64_t M = resultShape[resultShape.size() - 2];
  // The contraction dim K is the last dim of A, unless A is transposed, in
  // which case it is the second-to-last dim.
  int64_t K = transposeA ? lhsShape[lhsShape.size() - 2]
                         : lhsShape[lhsShape.size() - 1];
  int64_t N = resultShape[resultShape.size() - 1];
  int64_t batch = 1;
  for (size_t i = 0; i < resultShape.size() - 2; i++) {
    batch *= resultShape[i];
  }

  // Tilize M, K, N
  int64_t tilesM = (M + kTileHeight - 1) / kTileHeight;
  int64_t tilesK = (K + kTileWidth - 1) / kTileWidth;
  int64_t tilesN = (N + kTileWidth - 1) / kTileWidth;

  return static_cast<uint64_t>(batch) * tilesM * tilesK * tilesN;
}

//===----------------------------------------------------------------------===//
// Per-op FLOP accounting
//
// This is a different quantity from the roofline above. The roofline is a
// property of *this* compiled graph — it moves whenever the compiler changes
// the graph (fusion, layout, padding). The FLOP count is the algorithmic work
// of the model, which is graph-invariant: a matmul is 2*M*K*N FLOPs no matter
// how it is tiled or laid out.
//
// FLOPs are the MFU numerator. Downstream tooling (e.g. tt-xla) that has a
// measured wall-clock time `t` reports:
//   MFU = total_flops / (t * peak_flops_per_sec).
// Because total_flops is fixed, a better-compiled graph runs in less `t` and
// therefore reports a *higher* MFU — which is exactly the signal we want.
//
// FLOPs are counted on LOGICAL, un-padded shapes, matching the convention used
// by tt-metal's model FLOP counters (2*M*K*N for matmul, 2*B*H*Sq*Sk*(Dk+Dv)
// for attention, etc.). Tile padding is a compilation artifact and is
// deliberately excluded so the number stays graph-invariant.
//
// Coverage: matmul / linear, conv (1d/2d/3d/transpose), SDPA prefill,
// elementwise (unary/binary), reduction and softmax. Data-movement / layout
// ops contribute zero FLOPs and are dropped from the report.
//===----------------------------------------------------------------------===//

// A single 32x32x32 tile-mul performs 32^3 MACs = 2*32^3 FLOPs.
constexpr double kFlopsPerTileMul = 2.0 * kTileHeight * kTileWidth * kTileWidth;

// Cycles the matrix engine spends per 32x32x32 tile-mul at a given fidelity.
// Same table as PerfTargets::getNumCycles; peak throughput scales as 1/cycles.
uint64_t cyclesPerTileFor(MathFidelity mathFidelity) {
  switch (mathFidelity) {
  case MathFidelity::LoFi:
    return 16;
  case MathFidelity::HiFi2:
    return 32;
  case MathFidelity::HiFi3:
    return 48;
  case MathFidelity::HiFi4:
    return 64;
  }
  llvm_unreachable("Unsupported MathFidelity");
}

// Peak matrix-engine throughput of the whole chip at a given fidelity, in
// FLOP/s: cores * clock * (FLOPs per tile-mul) / (cycles per tile-mul).
double peakFlopsPerSec(uint64_t numCores, uint64_t aiclkHz,
                       MathFidelity mathFidelity) {
  uint64_t cpt = cyclesPerTileFor(mathFidelity);
  if (numCores == 0 || aiclkHz == 0 || cpt == 0) {
    return 0.0;
  }
  return static_cast<double>(numCores) * static_cast<double>(aiclkHz) *
         kFlopsPerTileMul / static_cast<double>(cpt);
}

// Logical scalar count of a ranked-tensor value (0 for non-tensors).
uint64_t numScalars(Value v) {
  auto t = mlir::dyn_cast<RankedTensorType>(v.getType());
  return t ? static_cast<uint64_t>(t.getNumElements()) : 0;
}

// Bytes backing a tensor value at its actual on-device dtype. Uses the TTNN
// layout's element size (which handles BFP8's fractional 1.0625 B/scalar and
// tile encodings); falls back to the scalar bit width when there is no layout.
// Returns 0 for non-tensor operands (e.g. the !ttnn.device operand).
uint64_t tensorBytes(Value v) {
  auto t = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!t) {
    return 0;
  }
  uint64_t vol = static_cast<uint64_t>(t.getNumElements());
  if (auto layout = mlir::dyn_cast_or_null<TTNNLayoutAttr>(t.getEncoding())) {
    // getElementSizeBytes() returns per-tile bytes when tiled; divide back down
    // to a per-scalar figure so the byte count tracks the logical volume.
    double scalarBytes =
        layout.isTiled() ? static_cast<double>(layout.getElementSizeBytes()) /
                               static_cast<double>(kTileHeight * kTileWidth)
                         : static_cast<double>(layout.getElementSizeBytes());
    return static_cast<uint64_t>(vol * scalarBytes + 0.5);
  }
  Type et = t.getElementType();
  if (et.isIntOrFloat()) {
    return vol * (et.getIntOrFloatBitWidth() / 8);
  }
  return vol * 2; // unknown dtype: assume 2 bytes/scalar
}

// True when a value is a DRAM-resident ranked tensor.
bool isDramTensor(Value v) {
  auto t = mlir::dyn_cast<RankedTensorType>(v.getType());
  return t && utils::getBufferTypeFromTensor(t) == BufferType::DRAM;
}

// DRAM input + output byte traffic of an op at actual dtypes. Only DRAM-
// resident operands/results are charged against DRAM bandwidth; L1-resident
// tensors are on-chip scratch and treated as free for this theoretical floor,
// so a well-fused / sharded graph is not penalized. This is the conservative
// direction for a *best-case* estimate.
uint64_t opDramBytes(Operation *op) {
  uint64_t bytes = 0;
  for (Value operand : op->getOperands()) {
    if (isDramTensor(operand)) {
      bytes += tensorBytes(operand);
    }
  }
  for (Value result : op->getResults()) {
    if (isDramTensor(result)) {
      bytes += tensorBytes(result);
    }
  }
  return bytes;
}

// 2*M*K*N over the logical (un-padded) shapes. K is the last dim of A, or the
// second-to-last when A is transposed.
uint64_t matmulFlops(Value lhs, Value result, bool transposeA) {
  ArrayRef<int64_t> lhsShape =
      mlir::cast<RankedTensorType>(lhs.getType()).getShape();
  ArrayRef<int64_t> resultShape =
      mlir::cast<RankedTensorType>(result.getType()).getShape();
  if (lhsShape.size() < 2 || resultShape.size() < 2) {
    return 0;
  }
  int64_t M = resultShape[resultShape.size() - 2];
  int64_t K = transposeA ? lhsShape[lhsShape.size() - 2]
                         : lhsShape[lhsShape.size() - 1];
  int64_t N = resultShape[resultShape.size() - 1];
  uint64_t batch = 1;
  for (size_t i = 0; i + 2 < resultShape.size(); ++i) {
    batch *= static_cast<uint64_t>(resultShape[i]);
  }
  return 2ULL * batch * static_cast<uint64_t>(M) * static_cast<uint64_t>(K) *
         static_cast<uint64_t>(N);
}

// Convolution FLOPs from the op's *logical* attributes (not the weight tensor,
// which gets padded/reblocked during weight preparation). Every output scalar
// costs (in_channels / groups) * prod(kernel) MACs, so
// flops = 2 * output_scalars * (in_channels / groups) * prod(kernel).
// `result` is the flattened (1, 1, N*spatial_out, O) tensor, whose element
// count is already N * spatial_out * out_channels.
uint64_t convFlopsFromAttrs(int64_t inChannels, int64_t groups,
                            uint64_t kernelProduct, Value result) {
  if (groups <= 0) {
    groups = 1;
  }
  uint64_t macsPerOutputScalar =
      static_cast<uint64_t>(inChannels / groups) * kernelProduct;
  return 2ULL * numScalars(result) * macsPerOutputScalar;
}

// Scaled-dot-product attention (prefill) FLOPs: the QK^T and the *V matmuls.
// QK^T = 2*B*Hq*Sq*Sk*Dk, scores*V = 2*B*Hq*Sq*Sk*Dv. Halved (rough) for a
// plain causal mask. A sliding window caps the effective key length.
uint64_t sdpaFlops(Value query, Value key, Value value, bool isCausal,
                   std::optional<int32_t> slidingWindowSize) {
  auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
  auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
  auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!qType || !kType || !vType || qType.getShape().size() != 4 ||
      kType.getShape().size() != 4 || vType.getShape().size() != 4) {
    return 0;
  }
  auto qShape = qType.getShape();
  int64_t B = qShape[0], Hq = qShape[1], Sq = qShape[2], Dk = qShape[3];
  int64_t Sk = kType.getShape()[2], Dv = vType.getShape()[3];
  int64_t effectiveSk = slidingWindowSize.has_value()
                            ? std::min<int64_t>(slidingWindowSize.value(), Sk)
                            : Sk;
  uint64_t flops = 2ULL * static_cast<uint64_t>(B) * static_cast<uint64_t>(Hq) *
                   static_cast<uint64_t>(Sq) *
                   static_cast<uint64_t>(effectiveSk) *
                   static_cast<uint64_t>(Dk + Dv);
  if (isCausal && !slidingWindowSize.has_value()) {
    flops /= 2;
  }
  return flops;
}

// Logical FLOPs + coarse category for one op. Returns flops == 0 for ops that
// do no arithmetic (reshape, layout, data movement, etc.), which the caller
// drops from the report.
std::pair<StringRef, uint64_t> classifyOpFlops(Operation *op) {
  using Pair = std::pair<StringRef, uint64_t>;
  Pair result =
      llvm::TypeSwitch<Operation *, Pair>(op)
          .Case<MatmulOp, LinearOp>([](auto m) -> Pair {
            return {"matmul",
                    matmulFlops(m.getA(), m.getResult(), m.getTransposeA())};
          })
          .Case<Conv2dOp, Conv3dOp, ConvTranspose2dOp>([](auto c) -> Pair {
            // kernel_size is a DenseI32Array on 2d/3d/transpose convs.
            uint64_t kernelProduct = 1;
            for (int32_t k : c.getKernelSize()) {
              kernelProduct *= static_cast<uint64_t>(k);
            }
            return {"conv", convFlopsFromAttrs(c.getInChannels(), c.getGroups(),
                                               kernelProduct, c.getResult())};
          })
          .Case<Conv1dOp>([](Conv1dOp c) -> Pair {
            // conv1d's kernel_size is a scalar i32.
            return {"conv",
                    convFlopsFromAttrs(c.getInChannels(), c.getGroups(),
                                       static_cast<uint64_t>(c.getKernelSize()),
                                       c.getResult())};
          })
          .Case<ScaledDotProductAttentionOp>(
              [](ScaledDotProductAttentionOp m) -> Pair {
                return {"sdpa",
                        sdpaFlops(m.getQuery(), m.getKey(), m.getValue(),
                                  m.getIsCausal(), m.getSlidingWindowSize())};
              })
          .Case<SoftmaxOp>([](SoftmaxOp s) -> Pair {
            // exp + max + subtract + sum + divide over every element.
            return {"softmax", 5ULL * numScalars(s.getInput())};
          })
          .Case<SumOp, MeanOp, MaxOp, MinOp>([](auto r) -> Pair {
            // One accumulate per input scalar.
            return {"reduction", numScalars(r->getOperand(0))};
          })
          .Default([](Operation *) -> Pair { return {"other", 0}; });

  if (result.second == 0 && result.first == "other") {
    // Fall back to the elementwise op interfaces for the (many) unary/binary
    // ops not named explicitly above: one FLOP per output scalar.
    if (mlir::isa<ElementwiseUnary, ElementwiseBinary>(op) &&
        op->getNumResults() > 0) {
      return {"elementwise", numScalars(op->getResult(0))};
    }
  }
  return result;
}

// One accounted op: its logical FLOPs plus a secondary per-op roofline.
struct FlopRecord {
  std::string opName;
  std::string location;
  StringRef category;
  uint64_t flops = 0;
  uint64_t memoryBytes = 0;
  MathFidelity mathFidelity = MathFidelity::HiFi2;
  double computeUs = 0.0; // flops / peak_flops(fidelity)   (ideal, un-padded)
  double memoryUs = 0.0;  // memoryBytes / dram_bandwidth
  double rooflineUs = 0.0;
  bool computeBound = false;
};

// Aggregate FLOP report for one function.
struct FlopSummary {
  uint64_t totalFlops = 0;
  llvm::StringMap<uint64_t> flopsByCategory;
  uint64_t computeBoundOps = 0;
  uint64_t memoryBoundOps = 0;
  double idealComputeMs = 0.0;  // Σ flops / peak_flops(fidelity)
  double idealMemoryMs = 0.0;   // Σ bytes / dram_bandwidth
  double idealRooflineMs = 0.0; // Σ max(compute, memory)
  std::vector<FlopRecord> perOp;
};

class TTNNCollectPerfMetrics
    : public impl::TTNNCollectPerfMetricsBase<TTNNCollectPerfMetrics> {

private:
  // Get layout information as string for individual op analysis
  std::string getLayoutInfo(Operation *op) {
    assert(op->getNumResults());
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    assert(tensorType);
    auto encoding = tensorType.getEncoding();
    assert(encoding);
    return llvm::formatv("{0}", encoding).str();
  }

  // Get location as string for individual op analysis
  std::string getLocationString(Operation *op) {
    if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc())) {
      return nameLoc.getName().str();
    }
    return "unknown";
  }

  // Analyze ops individually
  OperationMetrics analyzeOperation(Operation *op) {
    OperationMetrics metrics;
    metrics.opName = op->getName().getStringRef().str();
    metrics.location = getLocationString(op);
    metrics.layoutInfo = getLayoutInfo(op);
    metrics.isSharded = utils::producesShardedL1Layout(op);
    metrics.hasSystemMemory = utils::producesSystemMemoryLayout(op);

    return metrics;
  }

  // Find operations that are spilled to DRAM by analyzing toMemoryConfigOps
  void identifyDRAMSpills(func::FuncOp funcOp,
                          llvm::DenseSet<Value> &spilledValues) {
    funcOp.walk([&](ttnn::ToMemoryConfigOp toMemoryConfigOp) {
      if (utils::producesDRAMLayout(toMemoryConfigOp)) {
        // Mark the input operand's defining op's result as spilled
        Value input = toMemoryConfigOp.getInput();
        if (input.getDefiningOp() &&
            utils::producesShardedL1Layout(input.getDefiningOp())) {
          spilledValues.insert(input);
        }
      }
    });
  }

  void addSummaryToJson(llvm::json::Object &jsonOutput,
                        const AggregatedMetrics &aggregatedMetrics) {
    llvm::json::Object summary;
    summary["total_ops"] = static_cast<int64_t>(aggregatedMetrics.totalOps);
    summary["total_ops_with_output_tensor"] =
        static_cast<int64_t>(aggregatedMetrics.totalOpsWithOutputTensor);
    summary["total_shardable_ops"] =
        static_cast<int64_t>(aggregatedMetrics.totalShardableOps);
    summary["sharded_ops"] = static_cast<int64_t>(aggregatedMetrics.shardedOps);
    summary["effectively_sharded_ops"] =
        static_cast<int64_t>(aggregatedMetrics.effectivelyShardedOps);
    summary["sharded_and_spilled_ops"] =
        static_cast<int64_t>(aggregatedMetrics.shardedAndSpilledOps);
    summary["dram_spilled_ops"] =
        static_cast<int64_t>(aggregatedMetrics.dramSpilledOps);
    summary["system_memory_ops"] =
        static_cast<int64_t>(aggregatedMetrics.systemMemoryOps);
    summary["sharded_percentage"] = aggregatedMetrics.shardedPercentage;
    summary["effectively_sharded_percentage"] =
        aggregatedMetrics.effectivelyShardedPercentage;
    summary["spilled_percentage"] = aggregatedMetrics.spilledPercentage;
    summary["system_memory_percentage"] =
        aggregatedMetrics.systemMemoryPercentage;
    jsonOutput["summary"] = std::move(summary);
  }

  void
  addVerboseOutputToJson(llvm::json::Object &jsonOutput,
                         const std::vector<OperationMetrics> &operationDetails,
                         const llvm::StringMap<int> &operationTypeCounts) {
    llvm::json::Array operations;
    for (const auto &opMetrics : operationDetails) {
      llvm::json::Object opJson;
      opJson["operation"] = opMetrics.opName;
      opJson["location"] = opMetrics.location;
      opJson["is_sharded"] = opMetrics.isSharded;
      opJson["is_spilled_to_dram"] = opMetrics.isSpilledToDRAM;
      opJson["has_system_memory"] = opMetrics.hasSystemMemory;
      opJson["layout_info"] = opMetrics.layoutInfo;
      operations.push_back(std::move(opJson));
    }
    jsonOutput["shardable_operations"] = std::move(operations);

    // Counts of all operation types
    llvm::json::Object operationTypeBreakdown;
    for (const auto &pair : operationTypeCounts) {
      operationTypeBreakdown[pair.first()] = pair.second;
    }
    jsonOutput["operation_type_breakdown"] = std::move(operationTypeBreakdown);
  }

  // Per-arch DRAM bandwidth lookup. Not exposed on ChipDescAttr today, so
  // hardcoded. Wormhole B0 n150: 12 channels of GDDR6 ≈ 288 GB/s aggregate.
  // Blackhole: 512 GB/s placeholder, refine when we evaluate on hardware.
  // On galaxy systems the effective DRAM bandwidth is different. Single chip
  // calculation only for now.
  LogicalResult populateHardwareLimits(PerfTargets &t,
                                       ttcore::ChipDescAttr chipDesc,
                                       ModuleOp module) {
    t.arch = chipDesc.getArch().getValue();
    // Worker-grid Tensix-core count from the chip desc.
    t.numTensixCores = 1;
    for (int64_t d : chipDesc.getGrid()) {
      t.numTensixCores *= static_cast<uint64_t>(d);
    }
    // HiFi2 = 32 cycles / 32x32x32 tile-mul. Same on WH and BH per
    // tech_reports/matrix_engine/matrix_engine.md.
    t.cyclesPerTileMatmul = 32;
    switch (t.arch) {
    case ttcore::Arch::WormholeB0:
      t.dramBandwidthBytesPerSec = 288ULL * 1000ULL * 1000ULL * 1000ULL;
      t.aiclkHz = 1000ULL * 1000ULL * 1000ULL; // 1.0 GHz
      return success();
    case ttcore::Arch::Blackhole:
      t.dramBandwidthBytesPerSec = 512ULL * 1000ULL * 1000ULL * 1000ULL;
      t.aiclkHz = 1350ULL * 1000ULL * 1000ULL; // 1.35 GHz
      return success();
    case ttcore::Arch::Quasar:
      return module.emitError()
             << "TTNNCollectPerfMetrics: perf target estimate not calibrated "
                "for arch 'quasar'.";
    }
    llvm_unreachable("unknown ttcore::Arch value");
  }

  FailureOr<PerfTargets> computePerfTargets(ModuleOp module) {
    PerfTargets perfTargets;
    auto systemDescAttr = module->getAttrOfType<ttcore::SystemDescAttr>(
        ttcore::SystemDescAttr::name);
    assert(systemDescAttr &&
           "system_desc presence must be checked by the caller");
    auto chipDescs = systemDescAttr.getChipDescs();
    // Single-chip only. On anything other than exactly one chip we leave
    // the rest of `perfTargets` unpopulated and return; the caller's
    // `numChips == 1` gate suppresses emission. Multi-chip rooflines need
    // tensor-parallel-aware accounting we don't have yet, and silently
    // estimating against chip 0 would give a misleading number.
    perfTargets.numChips = chipDescs.size();
    if (perfTargets.numChips != 1) {
      llvm::errs() << "TTNNCollectPerfMetrics: perf target estimate requires "
                      "a single-chip system desc; got "
                   << chipDescs.size() << " chips. Skipping.\n";
      return perfTargets;
    }
    if (failed(populateHardwareLimits(perfTargets, chipDescs[0], module))) {
      return failure();
    }
    return perfTargets;
  }

  void addPerfTargetsToJson(llvm::json::Object &jsonOutput,
                            const PerfTargets &t) {
    llvm::json::Object pt;
    pt["arch"] = ttcore::stringifyArch(t.arch).str();
    pt["num_chips"] = static_cast<int64_t>(t.numChips);
    pt["dram_bandwidth_bytes_per_sec"] =
        static_cast<int64_t>(t.dramBandwidthBytesPerSec);
    pt["aiclk_hz"] = static_cast<int64_t>(t.aiclkHz);
    pt["num_tensix_cores"] = static_cast<int64_t>(t.numTensixCores);
    pt["cycles_per_tile_matmul"] = static_cast<int64_t>(t.cyclesPerTileMatmul);

    pt["params_count"] = static_cast<int64_t>(t.paramCount);
    pt["params_memory_bytes"] = static_cast<int64_t>(t.paramMemoryBytes);

    pt["dram_bound_ops"] = static_cast<int64_t>(t.dramBoundOps);
    pt["compute_bound_ops"] = static_cast<int64_t>(t.computeBoundOps);
    pt["skipped_ops"] = static_cast<int64_t>(t.skippedOps);

    pt["roofline_ms"] = t.rooflineMs;
    pt["top_perf_estimate_ms"] = t.topPerfEstimateMs;

    jsonOutput["perf_targets"] = std::move(pt);
  }

  // Walk the function and build the per-op FLOP report using the hardware
  // limits already populated on `t` (cores, clock, DRAM bandwidth).
  FlopSummary computeFlopSummary(func::FuncOp funcOp, const PerfTargets &t) {
    FlopSummary summary;
    double dramBw = static_cast<double>(t.dramBandwidthBytesPerSec);

    funcOp->walk([&](Operation *op) {
      auto [category, flops] = classifyOpFlops(op);
      if (flops == 0) {
        // No arithmetic (data movement / layout / unrecognized): skip.
        return;
      }

      FlopRecord rec;
      rec.opName = op->getName().getStringRef().str();
      rec.location = getLocationString(op);
      rec.category = category;
      rec.flops = flops;
      rec.memoryBytes = opDramBytes(op);

      // Per-op math fidelity, if the op carries a compute-kernel config;
      // otherwise the conservative HiFi2 default (matches the roofline).
      rec.mathFidelity = t.defaultMathFidelity;
      if (auto ckc = mlir::dyn_cast<TTNNComputeKernelConfigOpInterface>(op)) {
        if (auto cfg = ckc.getComputeConfigAttr()) {
          if (auto mf = cfg.getMathFidelity()) {
            rec.mathFidelity = *mf;
          }
        }
      }

      // Ideal compute time: logical FLOPs at 100% matrix-engine utilization.
      double peak =
          peakFlopsPerSec(t.numTensixCores, t.aiclkHz, rec.mathFidelity);
      rec.computeUs =
          peak > 0.0 ? static_cast<double>(rec.flops) / peak * 1e6 : 0.0;
      rec.memoryUs = dramBw > 0.0
                         ? static_cast<double>(rec.memoryBytes) / dramBw * 1e6
                         : 0.0;
      rec.rooflineUs = std::max(rec.computeUs, rec.memoryUs);
      rec.computeBound = rec.computeUs >= rec.memoryUs;

      summary.totalFlops += rec.flops;
      summary.flopsByCategory[category] += rec.flops;
      if (rec.computeBound) {
        summary.computeBoundOps++;
      } else {
        summary.memoryBoundOps++;
      }
      summary.idealComputeMs += rec.computeUs / 1000.0;
      summary.idealMemoryMs += rec.memoryUs / 1000.0;
      summary.idealRooflineMs += rec.rooflineUs / 1000.0;
      summary.perOp.push_back(std::move(rec));
    });

    return summary;
  }

  // Emit the FLOP report. `total_flops` + `peak_flops_per_sec` are the
  // graph-invariant MFU inputs; the ideal_*_ms fields and per-op array are a
  // secondary roofline view of this specific graph.
  void addFlopsToJson(llvm::json::Object &jsonOutput, const PerfTargets &t,
                      const FlopSummary &summary, bool verbose) {
    llvm::json::Object flops;
    flops["total_flops"] = static_cast<int64_t>(summary.totalFlops);

    llvm::json::Object peak;
    peak["lofi"] =
        peakFlopsPerSec(t.numTensixCores, t.aiclkHz, MathFidelity::LoFi);
    peak["hifi2"] =
        peakFlopsPerSec(t.numTensixCores, t.aiclkHz, MathFidelity::HiFi2);
    peak["hifi3"] =
        peakFlopsPerSec(t.numTensixCores, t.aiclkHz, MathFidelity::HiFi3);
    peak["hifi4"] =
        peakFlopsPerSec(t.numTensixCores, t.aiclkHz, MathFidelity::HiFi4);
    flops["peak_flops_per_sec"] = std::move(peak);

    llvm::json::Object byCategory;
    for (const auto &entry : summary.flopsByCategory) {
      // Pass an owned std::string: json::ObjectKey does not copy a StringRef,
      // and entry.first() points into the StringMap.
      byCategory[entry.first().str()] = static_cast<int64_t>(entry.second);
    }
    flops["flops_by_category"] = std::move(byCategory);

    flops["ideal_compute_ms"] = summary.idealComputeMs;
    flops["ideal_memory_ms"] = summary.idealMemoryMs;
    flops["ideal_roofline_ms"] = summary.idealRooflineMs;
    flops["compute_bound_ops"] = static_cast<int64_t>(summary.computeBoundOps);
    flops["memory_bound_ops"] = static_cast<int64_t>(summary.memoryBoundOps);

    if (verbose) {
      llvm::json::Array perOp;
      for (const FlopRecord &rec : summary.perOp) {
        llvm::json::Object opJson;
        opJson["operation"] = rec.opName;
        opJson["location"] = rec.location;
        opJson["category"] = rec.category.str();
        opJson["flops"] = static_cast<int64_t>(rec.flops);
        opJson["dram_bytes"] = static_cast<int64_t>(rec.memoryBytes);
        opJson["math_fidelity"] = stringifyMathFidelity(rec.mathFidelity).str();
        opJson["compute_us"] = rec.computeUs;
        opJson["memory_us"] = rec.memoryUs;
        opJson["roofline_us"] = rec.rooflineUs;
        opJson["bound"] = rec.computeBound ? "compute" : "memory";
        perOp.push_back(std::move(opJson));
      }
      flops["per_op"] = std::move(perOp);
    }

    jsonOutput["flops"] = std::move(flops);
  }

  std::string generateAutoFilename(ModuleOp module) {
    std::string baseName = "UnnamedModule";

    // Try to get module name
    if (auto moduleSymName = module.getSymName()) {
      baseName = moduleSymName->str();
      if (baseName.front() == '@') {
        baseName = baseName.substr(1);
      }
    } else {
      // Try to get the first function name if module has no name
      bool foundFunction = false;
      module->walk([&](func::FuncOp funcOp) {
        if (!foundFunction && !ttmlir::utils::isConstEvalFunc(funcOp)) {
          baseName = funcOp.getName().str();
          foundFunction = true;
        }
      });
    }

    return "perf_metrics/" + baseName + "_perf_metrics.json";
  }

  void ensureDirectoryExists(StringRef filePath) {
    llvm::SmallString<256> dirPath(filePath);
    llvm::sys::path::remove_filename(dirPath);

    if (dirPath.empty()) {
      return; // Current directory always exists
    }
    std::error_code ec = llvm::sys::fs::create_directories(dirPath);
    if (ec) {
      llvm::report_fatal_error(Twine("Failed to create directory: ") + dirPath +
                               " - " + ec.message());
    }
  }

  void writeJsonToFile(llvm::json::Object jsonOutput, ModuleOp module) {
    std::string outputPath;
    if (!ttnnPerfMetricsOutputFile.empty()) {
      // User specified a custom output file
      outputPath = ttnnPerfMetricsOutputFile.getValue();
      if (!llvm::StringRef(outputPath).ends_with(".json")) {
        outputPath += ".json";
      }
    } else {
      // Generate automatic filename
      outputPath = generateAutoFilename(module);
    }

    ensureDirectoryExists(outputPath);

    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      llvm::report_fatal_error(Twine("Failed to open output file: ") +
                               outputPath + " - " + ec.message());
    }

    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonOutput)))
       << "\n";
    os.close();
  }

public:
  using impl::TTNNCollectPerfMetricsBase<
      TTNNCollectPerfMetrics>::TTNNCollectPerfMetricsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::vector<OperationMetrics> operationDetails;
    AggregatedMetrics aggregatedMetrics;
    llvm::DenseSet<Value> spilledValues;
    llvm::StringMap<int> operationTypeCounts;
    PerfTargets perfTargets;
    // The single function whose weight ops feed the perf-target roofline.
    // Counting only this function avoids double-counting weights when more
    // than one function matches the walk filter below.
    func::FuncOp perfTargetFunc;

    // Identify the outer forward function for perf-target collection. All
    // inputs / weight tensors should be visible from the argument list of this
    // function. Nothing else is needed for the DRAM bound roofline calculation.
    {
      llvm::SmallVector<func::FuncOp> outerFuncs;
      module->walk([&](func::FuncOp funcOp) {
        if (ttmlir::utils::isForwardDeviceFunc(funcOp) && !funcOp.isPrivate()) {
          outerFuncs.push_back(funcOp);
        }
      });

      if (outerFuncs.size() != 1) {
        module.emitError() << "TTNNCollectPerfMetrics: expected exactly one "
                              "outer forward-device function for perf target "
                              "estimate; got "
                           << outerFuncs.size() << ".";
        signalPassFailure();
        return;
      }

      if (!module->getAttrOfType<ttcore::SystemDescAttr>(
              ttcore::SystemDescAttr::name)) {
        module.emitError() << "TTNNCollectPerfMetrics: no ttcore.system_desc "
                              "on the module; cannot compute perf target "
                              "estimate.";
        signalPassFailure();
        return;
      }

      FailureOr<PerfTargets> computed = computePerfTargets(module);
      if (failed(computed)) {
        signalPassFailure();
        return;
      }
      perfTargets = std::move(*computed);

      // Pick the one function to account for. When tracing is enabled the
      // weight ops live in the trace-main function; otherwise they live in
      // the validated outer forward-device function. Other matching
      // functions (e.g. private forward funcs) are intentionally skipped so
      // their weights are not counted twice.
      if (ttnnEnableTrace) {
        module->walk([&](func::FuncOp funcOp) {
          if (!perfTargetFunc && ttmlir::utils::isTraceMainFunc(funcOp)) {
            perfTargetFunc = funcOp;
          }
        });
      }
      // Fall back to the outer forward-device function when tracing is
      // disabled, or enabled but no trace-main function is present (otherwise
      // perfTargetFunc would stay null and we'd emit an all-zero roofline).
      if (!perfTargetFunc) {
        perfTargetFunc = outerFuncs.front();
      }
    }

    // Walk every weight-consuming op (matmul/linear + SDPA variants) in the
    // single selected function through a shared roofline kernel. Accounting
    // exactly one function avoids double-counting weights.
    perfTargets.calculatePerfTargets(perfTargetFunc);

    module->walk([&](func::FuncOp funcOp) {
      if (ttnnEnableTrace) {
        if (!ttmlir::utils::isTraceMainFunc(funcOp)) {
          return;
        }
      } else {
        if (!ttmlir::utils::isForwardDeviceFunc(funcOp)) {
          return;
        }
      }

      // First pass: identify DRAM spills
      identifyDRAMSpills(funcOp, spilledValues);

      // Second pass: analyze all operations
      funcOp->walk([&](Operation *op) {
        // Debug: Count operations by type
        std::string opName = op->getName().getStringRef().str();
        operationTypeCounts[opName]++;

        // Skip operations which never change (some appear only for
        // enable-trace=true)
        if (isa<ttnn::GetDeviceOp>(op)) {
          return;
        }
        aggregatedMetrics.totalOps++;

        // Skip operations without tensor results
        if (op->getNumResults() == 0) {
          return;
        }
        if (!mlir::isa<RankedTensorType>(op->getResult(0).getType())) {
          return;
        }
        aggregatedMetrics.totalOpsWithOutputTensor++;

        // Skip operations which make no sense to shard/spill, as they are
        // helpers
        if (isa<ttnn::ToMemoryConfigOp>(op)) {
          return;
        }

        // Only analyze TTNN operations, skip ttcore::load_cached
        if (!isa<TTNNDialect>(op->getDialect())) {
          return;
        }

        aggregatedMetrics.totalShardableOps++;

        OperationMetrics opMetrics = analyzeOperation(op);

        // Check if this operation's result is spilled and update aggregated
        // metrics
        if (opMetrics.isSharded) {
          Value result = op->getResult(0);
          if (spilledValues.count(result)) {
            opMetrics.isSpilledToDRAM = true;
            aggregatedMetrics.shardedAndSpilledOps++;
          } else {
            aggregatedMetrics.effectivelyShardedOps++;
          }
          aggregatedMetrics.shardedOps++;
        }

        if (utils::producesDRAMLayout(op)) {
          aggregatedMetrics.dramSpilledOps++;
        }

        if (opMetrics.hasSystemMemory) {
          aggregatedMetrics.systemMemoryOps++;
        }

        operationDetails.push_back(opMetrics);
      });
    });

    aggregatedMetrics.calculatePercentages();

    // Derate the per-matmul roofline by a fixed utilization factor. The peak
    // DRAM math assumes 288 GB/s on WH B0 but tt-metal's
    // tech_reports/Saturating_DRAM_bandwidth.md measures real matmul reads at
    // ~83-90% of peak; 0.7 is the conservative end of that range and lines up
    // with published Llama 3 8B decode tok/sec on n150. Apply once at the
    // end so the per-op classification still uses the peak numbers.
    // rooflineMs stays as the pure theoretical ceiling — sum of per-op
    // max(dram_us, compute_us) at peak hardware. topPerfEstimateMs is
    // the realistic estimate derated by kUtilizationFactor, where peak
    // is rarely achievable.
    constexpr double kUtilizationFactor = 0.7;
    perfTargets.topPerfEstimateMs = perfTargets.rooflineMs / kUtilizationFactor;

    llvm::json::Object jsonOutput;
    addSummaryToJson(jsonOutput, aggregatedMetrics);

    // Only emit perf_targets / flops when computePerfTargets fully populated
    // the struct; a soft-skip (e.g. multi-chip) leaves numChips at 0 and the
    // numbers would all be zero.
    if (perfTargets.numChips == 1) {
      addPerfTargetsToJson(jsonOutput, perfTargets);

      // Per-op FLOP report over the same single selected function. Uses the
      // hardware limits populated on perfTargets.
      FlopSummary flopSummary = computeFlopSummary(perfTargetFunc, perfTargets);
      addFlopsToJson(jsonOutput, perfTargets, flopSummary,
                     ttnnPerfMetricsVerboseOutputEnabled);
    }

    if (ttnnPerfMetricsVerboseOutputEnabled) {
      addVerboseOutputToJson(jsonOutput, operationDetails, operationTypeCounts);
    }

    writeJsonToFile(std::move(jsonOutput), module);
  }
};

} // namespace

} // namespace mlir::tt::ttnn
