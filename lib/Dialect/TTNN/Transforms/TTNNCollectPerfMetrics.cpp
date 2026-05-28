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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Builders.h"
#include <algorithm>
#include <cassert>
#include <string>

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
//        - Accumulate max(dram_us, compute_us) into rooflineTimeUs,
//          and rhs scalars / bytes into paramCount / paramMemoryBytes.
//   4. After the walk, divide rooflineTimeUs by kUtilizationFactor
//      (0.7 — conservative end of the 83–90 % measured range in
//       tt-metal/tech_reports/Saturating_DRAM_bandwidth.md). Mirror into
//      topPerfTimeMs / topPerfSamplesPerSec.
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

  // Per-matmul roofline classification, summed across all matmuls in the
  // forward function. rooflineTimeUs is the sum of per-matmul
  // max(dram_us, compute_us) — the simplest single-op roofline — divided
  // by a fixed utilization factor (see kUtilizationFactor in
  // runOnOperation). The bound counts are classified against the peak
  // numbers, before the derate.
  uint64_t dramBoundOps = 0;
  uint64_t computeBoundOps = 0;
  double rooflineTimeUs = 0.0;

  // SDPA-shape matmuls (4D operands: batch × heads × M × K • K × N) are
  // tracked separately so consumers can see how much of the roofline is
  // attention-driven (decode-step KV-cache reads, which scale linearly
  // with context length) vs weight-driven. They are still included in
  // rooflineTimeUs above; sdpaTimeUs is a subset.
  uint64_t sdpaOps = 0;
  double sdpaTimeUs = 0.0;

  // Sum of weight (rhs) scalars / bytes across every matmul. Reflects the
  // total weight stream the forward pass reads from DRAM, accumulated
  // alongside rooflineTimeUs in the matmul walk.
  uint64_t paramCount = 0;
  uint64_t paramMemoryBytes = 0;

  double topPerfTimeMs = 0.0;
  double topPerfSamplesPerSec = 0.0;
};

// Bytes per scalar element. Caller must ensure elementType is a plain
// scalar (non-TileType); TTNN forward args reach this pass with scalar
// element type and TTNNLayoutAttr encoding carrying the tile/dtype layout.
inline uint64_t getBytesPerScalarElement(mlir::Type elementType) {
  return elementType.getIntOrFloatBitWidth() / 8;
}

inline uint64_t getTensorMemoryBytes(mlir::RankedTensorType tt) {
  // RankedTensorType.getNumElements() returns the logical scalar count even
  // when the element type is a TileType (the tile-typing only affects how
  // the data is laid out in memory). For tile-typed tensors compute bytes
  // from the tile size; for scalar-typed tensors use the dtype byte width.
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(tt.getElementType())) {
    uint64_t scalarsPerTile = static_cast<uint64_t>(tileType.getHeight()) *
                              static_cast<uint64_t>(tileType.getWidth());
    return static_cast<uint64_t>(tt.getNumElements()) / scalarsPerTile *
           tileType.getSizeBytes();
  }
  return static_cast<uint64_t>(tt.getNumElements()) *
         getBytesPerScalarElement(tt.getElementType());
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

    llvm::json::Object params;
    params["count"] = static_cast<int64_t>(t.paramCount);
    params["memory_bytes"] = static_cast<int64_t>(t.paramMemoryBytes);
    pt["params"] = std::move(params);

    llvm::json::Object rl;
    rl["top_perf_time_ms"] = t.topPerfTimeMs;
    rl["top_perf_samples_per_sec"] = t.topPerfSamplesPerSec;
    pt["roofline"] = std::move(rl);

    llvm::json::Object mm;
    mm["dram_bound_ops"] = static_cast<int64_t>(t.dramBoundOps);
    mm["compute_bound_ops"] = static_cast<int64_t>(t.computeBoundOps);
    mm["sdpa_ops"] = static_cast<int64_t>(t.sdpaOps);
    mm["sdpa_time_us"] = t.sdpaTimeUs;
    mm["roofline_time_us"] = t.rooflineTimeUs;
    pt["matmul"] = std::move(mm);

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

      funcOp->walk([&](MatmulOp op) {
        // For each matmul calculate the number of bytes read from the weight
        // (rhs) and the number of 32x32x32 tile-multiply units of work.
        auto rhsType = mlir::dyn_cast<RankedTensorType>(op.getB().getType());
        auto lhsType = mlir::dyn_cast<RankedTensorType>(op.getA().getType());
        auto resultType =
            mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
        assert(rhsType && lhsType && resultType &&
               "TTNNCollectPerfMetrics: matmul operands and result must be "
               "ranked tensors.");
        // Operands may carry either a scalar element type or a
        // ttcore::TileType<H, W, dtype>. In both cases the shape is the
        // logical scalar shape — TileType only affects in-memory layout.
        // getTensorMemoryBytes handles both via the tile.getSizeBytes() path.

        ArrayRef<int64_t> lhsShape = lhsType.getShape();
        ArrayRef<int64_t> rhsShape = rhsType.getShape();
        ArrayRef<int64_t> resultShape = resultType.getShape();
        assert(lhsShape.size() >= 2 && rhsShape.size() >= 2 &&
               resultShape.size() >= 2 &&
               "TTNNCollectPerfMetrics: matmul shapes must be at least 2D.");

        // M, K, N from the scalar shapes. Last two dims of lhs are (M, K),
        // last dim of rhs is N. transpose_a/transpose_b are ignored here —
        // they swap which dim is which but don't change the total work.
        uint64_t M = static_cast<uint64_t>(lhsShape[lhsShape.size() - 2]);
        uint64_t K = static_cast<uint64_t>(lhsShape[lhsShape.size() - 1]);
        uint64_t N = static_cast<uint64_t>(rhsShape[rhsShape.size() - 1]);

        // Round M, K, N up to 32 and count 32x32x32 tile-multiply units.
        auto roundUp32 = [](uint64_t x) { return (x + 31) & ~uint64_t{31}; };
        uint64_t mUnits = roundUp32(M) / 32;
        uint64_t kUnits = roundUp32(K) / 32;
        uint64_t nUnits = roundUp32(N) / 32;

        // Batch dims come from the result shape (already reflects any lhs/rhs
        // broadcasting).
        uint64_t batch = 1;
        for (size_t i = 0; i + 2 < resultShape.size(); ++i) {
          batch *= static_cast<uint64_t>(resultShape[i]);
        }
        uint64_t tileMuls = batch * mUnits * kUnits * nUnits;

        // Bytes read from the weight (rhs) — scalar element count at BFP8
        // (1056 bytes / 1024 scalars = 1.03125 B / scalar).
        assert(utils::getBufferTypeFromTensor(rhsType) == BufferType::DRAM &&
               "TTNNCollectPerfMetrics: expected matmul rhs (weight) to be in "
               "DRAM.");
        uint64_t rhsScalars = static_cast<uint64_t>(rhsType.getNumElements());
        uint64_t rhsBytes = static_cast<uint64_t>(
            static_cast<double>(rhsScalars) * 1056.0 / 1024.0 + 0.5);

        // Accumulate the rhs weight stream — useful as a sanity-check
        // alongside the time-based roofline.
        perfTargets.paramCount += rhsScalars;
        perfTargets.paramMemoryBytes += getTensorMemoryBytes(rhsType);

        // Compute vs DRAM-bound classification, directly in us.
        //
        // Skip when populateHardwareLimits hasn't populated these (e.g.
        // multi-chip soft-skip leaves dramBandwidthBytesPerSec / aiclkHz /
        // numTensixCores all zero).
        if (perfTargets.dramBandwidthBytesPerSec == 0 ||
            perfTargets.aiclkHz == 0 || perfTargets.numTensixCores == 0) {
          return;
        }

        // DRAM time: bytes / (bytes/sec) = sec, × 1e6 = us. Aggregate chip
        // bandwidth.
        double dramUs =
            static_cast<double>(rhsBytes) * 1.0e6 /
            static_cast<double>(perfTargets.dramBandwidthBytesPerSec);

        // Compute time at HiFi2: 32 cycles per 32x32x32 tile-mul on one
        // Tensix matrix engine; the chip runs numTensixCores tile-muls in
        // parallel at AICLK. Assumes the matmul parallelizes across the full
        // worker grid (best-case roofline).
        constexpr uint64_t kCyclesPerTileMatmulHiFi2 = 32;
        double computeUs =
            static_cast<double>(tileMuls * kCyclesPerTileMatmulHiFi2) * 1.0e6 /
            static_cast<double>(perfTargets.numTensixCores *
                                perfTargets.aiclkHz);

        double opUs = std::max(dramUs, computeUs);
        perfTargets.rooflineTimeUs += opUs;
        if (dramUs >= computeUs) {
          perfTargets.dramBoundOps++;
        } else {
          perfTargets.computeBoundOps++;
        }
        // SDPA shape signature at low opt levels: rank-4 operands of the
        // form (B1, B2, M, K) × (B1, B2, K, N) → (B1, B2, M, N). At higher
        // optimization levels SDPA is fused into a dedicated
        // scaled_dot_product_attention_decode op — handled below.
        if (lhsShape.size() == 4 && rhsShape.size() == 4) {
          perfTargets.sdpaOps++;
          perfTargets.sdpaTimeUs += opUs;
        }
      });

      // Fused SDPA decode (emitted at optimization-level >= 1). The op
      // packages QK^T + softmax + attn@V; for the roofline, the dominant
      // cost is reading the K and V caches from DRAM (M=1 decode means
      // the compute math collapses to negligible). Treat compute as 0 and
      // bill the DRAM stream of both caches at the BFP8 byte rate to stay
      // consistent with the matmul weight path.
      auto handleSdpaDecode = [&](Value key, Value value) {
        if (perfTargets.dramBandwidthBytesPerSec == 0) {
          return;
        }
        auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
        auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());
        if (!kType || !vType) {
          return;
        }
        auto bytesAtBfp8 = [](RankedTensorType t) {
          return static_cast<uint64_t>(
              static_cast<double>(t.getNumElements()) * 1056.0 / 1024.0 + 0.5);
        };
        uint64_t kvBytes = bytesAtBfp8(kType) + bytesAtBfp8(vType);
        double dramUs =
            static_cast<double>(kvBytes) * 1.0e6 /
            static_cast<double>(perfTargets.dramBandwidthBytesPerSec);
        perfTargets.rooflineTimeUs += dramUs;
        perfTargets.dramBoundOps++;
        perfTargets.sdpaOps++;
        perfTargets.sdpaTimeUs += dramUs;
      };
      funcOp->walk([&](ScaledDotProductAttentionDecodeOp op) {
        handleSdpaDecode(op.getKey(), op.getValue());
      });
      funcOp->walk([&](PagedScaledDotProductAttentionDecodeOp op) {
        handleSdpaDecode(op.getKey(), op.getValue());
      });

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
    constexpr double kUtilizationFactor = 0.7;
    perfTargets.rooflineTimeUs /= kUtilizationFactor;
    perfTargets.sdpaTimeUs /= kUtilizationFactor;

    // Mirror the per-matmul total into the ms-scale roofline fields so the
    // perf_targets.roofline block carries the realistic wall-clock estimate.
    perfTargets.topPerfTimeMs = perfTargets.rooflineTimeUs / 1000.0;
    perfTargets.topPerfSamplesPerSec = perfTargets.topPerfTimeMs > 0.0
                                           ? 1000.0 / perfTargets.topPerfTimeMs
                                           : 0.0;

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
