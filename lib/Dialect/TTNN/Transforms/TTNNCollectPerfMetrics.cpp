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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <sys/types.h>

#define DEBUG_TYPE "ttnn-collect-perf-metrics"

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

//===----------------------------------------------------------------------===//
// Per-matmul roofline
//
// Walks every ttnn.matmul in the forward function and, for each op, computes
// the time to fetch the weight (rhs) from DRAM and the time to do the
// tile-multiplies at peak compute, takes max(dram_us, compute_us), and sums.
// The sum is divided by a fixed utilization factor at the end to land on a
// realistic single-step wall-clock estimate.
//
// Process
//   1. Pick the entry point: the unique public forward-device function.
//      Hard fail if not exactly one, if `ttcore.system_desc` is missing,
//      or if the arch (currently Quasar) has no calibrated constants.
//   2. Per-arch hardware constants: DRAM bandwidth (B/s), AICLK (Hz), and
//      the worker-grid Tensix-core count from `ChipDescAttr.getGrid()`.
//   3. Per matmul:
//        - rhs scalar count → BFP8 bytes (1056 / 1024 ≈ 1.03125 B/scalar).
//        - M/K/N from scalar shapes, batch from result shape; tile-mul work
//          = batch × ceil(M/32) × ceil(K/32) × ceil(N/32).
//        - dram_us  = rhsBytes / dramBandwidth.
//        - compute_us = (tileMuls × 32) / (numTensixCores × aiclkHz)
//          (HiFi2 baseline: 32 cycles per 32×32×32 tile-mul on one Tensix
//           matrix engine; full-grid parallelism).
//        - bound = dram_us ≥ compute_us ? DRAM : COMPUTE.
//        - Accumulate max(dram_us, compute_us) (converted to ms) into
//          rooflineMs, and rhs scalars / bytes into paramCount /
//          paramMemoryBytes.
//   4. After the walk, rooflineMs is the pure theoretical ceiling.
//      topPerfEstimateMs = rooflineMs / kUtilizationFactor (0.7 —
//      conservative end of the 83–90 % measured range in
//      tt-metal/tech_reports/Saturating_DRAM_bandwidth.md) is the
//      realistic single-step wall-clock estimate.
//
// Assumptions
//   - DRAM-bound regime is typical (and is observed on LLM decode), but the
//     bound classification picks the per-op winner; compute-bound ops
//     contribute their compute time to the sum.
//   - All weights are treated as BFP8 for byte accounting, independent of
//     how the IR has encoded the rhs dtype.
//   - Per-arch hardware constants (hardcoded; not pulled from ChipDescAttr):
//       Wormhole B0 = 288 GB/s, 1.00 GHz, grid from chip desc (e.g. 64).
//       Blackhole   = 512 GB/s, 1.35 GHz, grid from chip desc (e.g. 130).
//       Quasar      = uncalibrated (hard fail).
//     Not derated for attainable bandwidth, memory-controller efficiency,
//     or access pattern; Galaxy effective bandwidth differs.
//   - Single-chip only. Multi-chip system_desc is a soft skip (multi-chip
//     / tensor-parallel accounting is not yet implemented).
//   - 32 cycles per 32×32×32 tile-mul = HiFi2 baseline (LoFi 16, HiFi3 48,
//     HiFi4 64; tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:101).
//   - Compute is assumed to fully parallelize across the worker grid.
//   - Only the matmul rhs (weight) DRAM read is counted. Activation reuse,
//     L1 traffic, dispatch overhead, host transfers, KV-cache reads outside
//     of attention matmuls, and inter-chip traffic are all ignored.
//   - One sample == one invocation. `samples_per_sec` assumes a single
//     forward call yields a single sample (one token for decode, one image
//     for vision); consumers must interpret accordingly.
//===----------------------------------------------------------------------===//

// Forward declarations for free helpers used by PerfTargets methods below.
// Their definitions live right after PerfTargets.
uint64_t getNumMatmulTiles(RankedTensorType lhs, RankedTensorType output);
inline bool isValidWeightForTargetCalculation(mlir::Value v);
bool constEvalDoesNotChangeTensorVolume(ttcore::LoadCachedOp loadCachedOp);

struct PerfTargets {
  ttcore::Arch arch = ttcore::Arch::WormholeB0;
  unsigned numChips = 0;
  uint64_t dramBandwidthBytesPerSec = 0;
  // AICLK (Tensix core clock). Used to convert peak DRAM B/s into B/cycle so
  // we can compare against per-tile compute cycle counts. Sources:
  // tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md (WH 1 GHz, BH 1.35 GHz).
  uint64_t aiclkHz = 0;
  // Worker-grid Tensix-core count (product of ChipDescAttr.getGrid()). Used
  // to convert serial compute cycles into chip-level cycles for the bound
  // comparison.
  uint64_t numTensixCores = 0;
  // Cycles per 32x32x32 tile-mul on one Tensix matrix engine. HiFi2 baseline
  // = 32 cycles (LoFi 16, HiFi3 48, HiFi4 64; see
  // tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:101).
  uint64_t cyclesPerTileMatmul = 0;

  MathFidelity defaultMathFidelity = MathFidelity::HiFi2;
  ttcore::DataType defaultWeightDataType = ttcore::DataType::BFP_BFloat4;

  // Per-op classification across every weight matmul / linear / SDPA op
  // in the forward function. Counts use peak numbers before the
  // utilization derate.
  uint64_t dramBoundOps = 0;
  uint64_t computeBoundOps = 0;

  // Sum of weight (rhs) scalars / bytes across every counted op.
  uint64_t paramCount = 0;
  uint64_t paramMemoryBytes = 0;

  // Ops dropped because their DRAM-stream source couldn't be verified —
  // see isValidWeightForTargetCalculation. Mirrored to JSON so silent skips
  // remain auditable.
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

  // Cycles per `numMatmulTiles` 32x32x32 tile-muls, scaled by the
  // configured math fidelity. HiFi2 baseline = 32 cycles per tile
  // (LoFi 16, HiFi3 48, HiFi4 64; see
  // tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:101).
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

  // BFP8 byte count of a tensor — same accounting everywhere
  // (1056 bytes / 1024 scalars = 1.03125 B / scalar). Caller pre-casts.
  uint64_t bytesAtBfp8(RankedTensorType t) {
    return static_cast<uint64_t>(
        static_cast<double>(t.getNumElements()) * 1056.0 / 1024.0 + 0.5);
  }

  // Single entry point: walk every interesting op in funcOp, dispatch by
  // op type to extract its DRAM-resident weights and tile-mul work, then
  // run the shared roofline kernel below.
  //
  // Per-op extractors only encode WHAT the op consumes from DRAM and HOW
  // much compute it does — the validate / max(dram, compute) / classify
  // logic lives in one place.
  void accountOp(func::FuncOp funcOp) {
    funcOp->walk([&](Operation *op) {
      SmallVector<Value, 2> weights;
      uint64_t tileMuls = 0;

      bool matched =
          llvm::TypeSwitch<Operation *, bool>(op)
              .Case<MatmulOp, LinearOp>([&](auto m) {
                weights = {m.getB()};
                tileMuls = matmulTileMuls(m.getA(), m.getResult());
                return true;
              })
              .Case<ScaledDotProductAttentionDecodeOp,
                    PagedScaledDotProductAttentionDecodeOp>([&](auto m) {
                weights = {m.getKey(), m.getValue()};
                // SDPA decode: M=1 → compute negligible; DRAM-only.
                tileMuls = 0;
                return true;
              })
              .Case<ScaledDotProductAttentionOp>([&](auto m) {
                weights = {m.getKey(), m.getValue()};
                tileMuls = sdpaPrefillTileMuls(m.getQuery(), m.getKey());
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

      // 2. Sum weight scalars + BFP8 bytes across all weights.
      uint64_t weightScalars = 0;
      uint64_t weightBytes = 0;
      for (Value w : weights) {
        auto t = mlir::cast<RankedTensorType>(w.getType());
        weightScalars += static_cast<uint64_t>(t.getNumElements());
        weightBytes += bytesAtBfp8(t);
      }
      paramCount += weightScalars;
      paramMemoryBytes += weightBytes;

      // 3+4. DRAM us and compute us.
      double dramUs = getUsToReadWeightsFromDRAM(weightBytes);
      uint64_t computeCycles = getNumCycles(tileMuls, defaultMathFidelity);
      double computeUs =
          (numTensixCores == 0 || aiclkHz == 0)
              ? 0.0
              : static_cast<double>(computeCycles) * 1.0e6 /
                    static_cast<double>(numTensixCores * aiclkHz);

      // 5. Take max, accumulate, classify.
      double opUs = std::max(dramUs, computeUs);
      rooflineMs += opUs / 1000.0;
      bool dramBound = dramUs >= computeUs;
      if (dramBound) {
        dramBoundOps++;
      } else {
        computeBoundOps++;
      }

      // Debug log — every counted op, with every value we computed.
      // Enable via `-debug-only=ttnn-collect-perf-metrics`.
      LLVM_DEBUG({
        llvm::dbgs() << "[perf-metrics] " << op->getName() << " loc=";
        op->getLoc().print(llvm::dbgs());
        llvm::dbgs() << "\n  weights        =";
        for (Value w : weights) {
          llvm::dbgs() << " " << w.getType();
        }
        llvm::dbgs() << "\n  weight_scalars = " << weightScalars
                     << "\n  weight_bytes   = " << weightBytes << " (BFP8)"
                     << "\n  tile_muls      = " << tileMuls
                     << "\n  compute_cycles = " << computeCycles
                     << "\n  dram_us        = " << dramUs
                     << "\n  compute_us     = " << computeUs
                     << "\n  op_us          = " << opUs << " ("
                     << (dramBound ? "DRAM" : "compute") << "-bound)"
                     << "\n  running totals: roofline_ms=" << rooflineMs
                     << " dram_bound_ops=" << dramBoundOps
                     << " compute_bound_ops=" << computeBoundOps
                     << " skipped_ops=" << skippedOps << "\n";
      });
    });
  }

private:
  // === Per-op-type extractors ===

  // Tile-mul count for matmul/linear (M·K·N tiles, batched).
  uint64_t matmulTileMuls(Value lhs, Value result) {
    return getNumMatmulTiles(mlir::cast<RankedTensorType>(lhs.getType()),
                             mlir::cast<RankedTensorType>(result.getType()));
  }

  // Tile-mul count for SDPA prefill: 2·B·Hq·up32(Sq)·up32(D)·up32(Sk).
  // Returns 0 (compute-free) for unexpected ranks — caller falls back to
  // DRAM-only via the shared max(dramUs, computeUs).
  uint64_t sdpaPrefillTileMuls(Value query, Value key) {
    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    if (!qType || !kType || qType.getShape().size() != 4 ||
        kType.getShape().size() != 4) {
      return 0;
    }
    ArrayRef<int64_t> q = qType.getShape();
    ArrayRef<int64_t> k = kType.getShape();
    auto up32 = [](uint64_t x) { return (x + 31) / 32; };
    return 2ULL * static_cast<uint64_t>(q[0]) * static_cast<uint64_t>(q[1]) *
           up32(static_cast<uint64_t>(q[2])) *
           up32(static_cast<uint64_t>(q[3])) *
           up32(static_cast<uint64_t>(k[2]));
  }
};

// Tensor needs to be in DRAM.
// A direct arg marked as a weight (Parameter/Constant/kv_cache) is valid.
// A tensor produced by a load_cached op is valid, but only when the tensor
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
    // No function-type check here: the caller's outer walk has already
    // restricted us to either a forward-device func (no trace) or a
    // trace-main func (trace enabled). Both carry the same
    // ttcore.argument_type metadata on their block args.
    unsigned idx = blockArg.getArgNumber();
    ttcore::ArgumentType argType = ttcore::getFunctionArgumentType(funcOp, idx);
    return argType == ttcore::ArgumentType::Parameter ||
           argType == ttcore::ArgumentType::Constant ||
           ttcore::isKVCacheArgument(funcOp, idx);
  }

  auto definingOp = v.getDefiningOp();

  if (!definingOp) {
    return false;
  }

  auto loadCachedOp = mlir::dyn_cast<ttcore::LoadCachedOp>(definingOp);
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

uint64_t getNumMatmulTiles(RankedTensorType lhs, RankedTensorType output) {
  ArrayRef<int64_t> lhsShape = lhs.getShape();
  ArrayRef<int64_t> resultShape = output.getShape();

  auto M = resultShape[resultShape.size() - 2];
  auto K = lhsShape[lhsShape.size() - 1];
  auto N = resultShape[resultShape.size() - 1];
  auto batch = 1;
  for (size_t i = 0; i < resultShape.size() - 2; i++) {
    batch *= resultShape[i];
  }

  // Tilize M, K, N
  auto tilesM = (M + 31) / 32;
  auto tilesK = (K + 31) / 32;
  auto tilesN = (N + 31) / 32;

  return batch * tilesM * tilesK * tilesN;
}

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

    // Identify the outer forward function for perf-target collection. All
    // inputs / wegiht tensors should be visible from the argument list of this
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
    }

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

      // Walk every weight-consuming op (matmul/linear + SDPA variants)
      // through a single dispatch + shared roofline kernel.
      perfTargets.accountOp(funcOp);

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

    // Only emit perf_targets when computePerfTargets fully populated the
    // struct; a soft-skip (e.g. multi-chip) leaves numChips at 0 and the
    // numbers would all be zero.
    if (perfTargets.numChips == 1) {
      addPerfTargetsToJson(jsonOutput, perfTargets);
    }

    if (ttnnPerfMetricsVerboseOutputEnabled) {
      addVerboseOutputToJson(jsonOutput, operationDetails, operationTypeCounts);
    }

    writeJsonToFile(std::move(jsonOutput), module);
  }
};

} // namespace

} // namespace mlir::tt::ttnn
