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
// DRAM-bound top-perf calculation
//
// Emits a theoretical performance ceiling under the assumption that every
// forward pass is bottlenecked by reading weights and KV cache from DRAM.
// The result lands in the `perf_targets` section of the JSON output.
//
// Process
//   1. Pick the entry point: the unique public forward-device function.
//      Hard fail if not exactly one, if `ttcore.system_desc` is missing,
//      or if the arch (currently Quasar) has no calibrated bandwidth.
//   2. Walk *only the forward function's argument list* (no op-graph
//      walk). Each tensor arg is bucketed via existing ttcore attributes:
//        - `ttcore.kv_cache` present       → KV cache bytes
//        - `ttcore.argument_type != Input` → parameter bytes
//                                            (Parameter / Constant / Default)
//        - Input args                      → ignored (activations).
//   3. Normalize both totals to BFP8-equivalent storage
//      (1056 B / 32x32 tile = 1.03125 B / scalar), independent of how the
//      IR has encoded the arg dtype.
//   4. `dram_time_ms = total_bytes / dram_peak_bandwidth`,
//      `top_perf_time_ms = dram_time_ms`,
//      `top_perf_samples_per_sec = 1000 / top_perf_time_ms`.
//
// Assumptions
//   - DRAM-bound regime. Compute (FLOPs vs TFLOPS), L1/SRAM, dispatch,
//     kernel-launch overhead, host transfers, and inter-chip traffic are
//     all ignored.
//   - All weights assumed to be BFP8.
//   - Relies on the front end labelling Parameter / Constant / kv_cache on
//     args correctly.
//   - Peak per-arch DRAM bandwidth, hardcoded:
//       Wormhole B0 = 288 GB/s (12-channel GDDR6 aggregate),
//       Blackhole  = 512 GB/s (placeholder until measured),
//       Quasar     = uncalibrated (hard fail).
//     Not pulled from `ChipDescAttr`; not derated for attainable / memory
//     controller efficiency / access pattern; Galaxy effective bandwidth
//     differs.
//   - Single-chip only. Multi-chip system_desc is a soft skip (multi-chip
//     / tensor-parallel accounting is not yet implemented).
//   - Strict roofline, alpha = 1.0. `top_perf_time_ms = dram_time_ms`
//     directly — no safety factor, no min(dram, compute) blend. The
//     emitted number is a pure theoretical ceiling for consumers to
//     compare measurements against.
//   - Static shapes only. Dynamic-shape args contribute zero bytes;
//     today's decode entry points are fully static.
//   - One sample == one invocation. `samples_per_sec` assumes a single
//     forward call yields a single sample (one token for decode, one
//     image for vision); consumers must interpret accordingly.
//===----------------------------------------------------------------------===//
struct PerfTargets {
  std::string arch;
  unsigned numChips = 0;
  uint64_t dramBandwidthBytesPerSec = 0;

  uint64_t paramCount = 0;
  uint64_t paramMemoryBytes = 0;
  uint64_t paramMemoryBytesBfp8 = 0;

  uint64_t kvCacheCount = 0;
  uint64_t kvCacheMemoryBytes = 0;
  uint64_t kvCacheMemoryBytesBfp8 = 0;

  double dramRooflineTimeMs = 0.0;
  double topPerfTimeMs = 0.0;
  double topPerfSamplesPerSec = 0.0;
};

// Bytes per scalar element. Caller must ensure elementType is a plain
// scalar (non-TileType); TTNN forward args reach this pass with scalar
// element type and TTNNLayoutAttr encoding carrying the tile/dtype layout.
inline double getBytesPerScalarElement(mlir::Type elementType) {
  return static_cast<double>(elementType.getIntOrFloatBitWidth()) / 8.0;
}

inline uint64_t getTensorMemoryBytes(mlir::RankedTensorType tt) {
  if (!tt.hasStaticShape()) {
    return 0;
  }
  double bytes = static_cast<double>(tt.getNumElements()) *
                 getBytesPerScalarElement(tt.getElementType());
  return static_cast<uint64_t>(bytes + 0.5);
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
    switch (chipDesc.getArch().getValue()) {
    case ttcore::Arch::WormholeB0:
      t.arch = "wormhole_b0";
      t.dramBandwidthBytesPerSec = 288ULL * 1000ULL * 1000ULL * 1000ULL;
      return success();
    case ttcore::Arch::Blackhole:
      t.arch = "blackhole";
      t.dramBandwidthBytesPerSec = 512ULL * 1000ULL * 1000ULL * 1000ULL;
      return success();
    case ttcore::Arch::Quasar:
      return module.emitError()
             << "TTNNCollectPerfMetrics: perf target estimate not calibrated "
                "for arch 'quasar'.";
    }
    llvm_unreachable("unknown ttcore::Arch value");
  }

  // Walk forward-function arguments only — no op-graph traversal. Classify
  // each arg via the existing ttcore.argument_type and ttcore.kv_cache attrs.
  void collectArgumentStats(PerfTargets &t, func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorType) {
        continue;
      }
      // TTNN forward args carry a scalar element type with TTNNLayoutAttr
      // encoding. A TileType element would mean the outer shape is in
      // tile counts, breaking the scalarVol/byte math below.
      assert(!mlir::isa<ttcore::TileType>(tensorType.getElementType()) &&
             "TTNNCollectPerfMetrics: unexpected TileType element on "
             "forward arg; expected scalar element + TTNNLayoutAttr "
             "encoding.");
      uint64_t scalarVol =
          tensorType.hasStaticShape()
              ? static_cast<uint64_t>(tensorType.getNumElements())
              : 0;
      uint64_t bytes = getTensorMemoryBytes(tensorType);
      unsigned idx = arg.getArgNumber();

      if (ttcore::isKVCacheArgument(funcOp, idx)) {
        t.kvCacheCount += scalarVol;
        t.kvCacheMemoryBytes += bytes;
        continue;
      }

      // Parameter / Constant / Default — everything that isn't a runtime
      // input contributes to the DRAM-bound weight stream.
      if (ttcore::getFunctionArgumentType(funcOp, idx) !=
          ttcore::ArgumentType::Input) {
        t.paramCount += scalarVol;
        t.paramMemoryBytes += bytes;
      }
    }
  }

  FailureOr<PerfTargets> computePerfTargets(ModuleOp module,
                                            func::FuncOp forwardFunc) {
    PerfTargets perfTargets;
    auto systemDescAttr = module->getAttrOfType<ttcore::SystemDescAttr>(
        ttcore::SystemDescAttr::name);
    assert(systemDescAttr &&
           "system_desc presence must be checked by the caller");
    auto chipDescs = systemDescAttr.getChipDescs();
    // Single-chip only. We *fail the calculation* (return without
    // populating `perfTargets`, so addPerfTargetsToJson will not emit) on
    // anything other than exactly one chip — multi-chip rooflines need
    // tensor-parallel-aware accounting we don't have yet, and silently
    // estimating against chip 0 would give a misleading number.
    if (chipDescs.size() != 1) {
      llvm::errs() << "TTNNCollectPerfMetrics: perf target estimate requires "
                      "a single-chip system desc; got "
                   << chipDescs.size() << " chips. Skipping.\n";
      return perfTargets;
    }
    perfTargets.numChips = 1;
    if (failed(populateHardwareLimits(perfTargets, chipDescs[0], module))) {
      return failure();
    }
    collectArgumentStats(perfTargets, forwardFunc);

    // Normalize weight / KV bytes to BFP8 storage independent of the
    // encoded dtype in the IR. The tt-mlir TTNN runtime pipeline pushes
    // weights through BFP8 by default; the few tensors that stay BF16
    // (norms, RoPE inv_freq) are small enough that rounding them down
    // doesn't move the DRAM math materially. This keeps the ceiling stable
    // across front-end dtype annotations.
    // BFP8 tile: 1056 B per 32x32 tile = 1056/1024 ≈ 1.03125 B / scalar.
    constexpr double kBfp8BytesPerElement = 1056.0 / 1024.0;
    auto bytesAtBfp8 = [](uint64_t count) {
      return static_cast<uint64_t>(
          static_cast<double>(count) * kBfp8BytesPerElement + 0.5);
    };
    perfTargets.paramMemoryBytesBfp8 = bytesAtBfp8(perfTargets.paramCount);
    perfTargets.kvCacheMemoryBytesBfp8 = bytesAtBfp8(perfTargets.kvCacheCount);

    uint64_t totalDramBytes =
        perfTargets.paramMemoryBytesBfp8 + perfTargets.kvCacheMemoryBytesBfp8;
    if (perfTargets.dramBandwidthBytesPerSec > 0) {
      double dramTimeSec =
          static_cast<double>(totalDramBytes) /
          static_cast<double>(perfTargets.dramBandwidthBytesPerSec);
      perfTargets.dramRooflineTimeMs = dramTimeSec * 1000.0;
    }
    // Strict roofline: top_perf_time_ms = dram_time_ms (alpha = 1.0). The
    // emitted number is a theoretical ceiling consumers can compare against.
    perfTargets.topPerfTimeMs = perfTargets.dramRooflineTimeMs;
    perfTargets.topPerfSamplesPerSec = perfTargets.topPerfTimeMs > 0.0
                                           ? 1000.0 / perfTargets.topPerfTimeMs
                                           : 0.0;
    return perfTargets;
  }

  void addPerfTargetsToJson(llvm::json::Object &jsonOutput,
                            const PerfTargets &t) {
    llvm::json::Object pt;
    pt["arch"] = t.arch;
    pt["num_chips"] = static_cast<int64_t>(t.numChips);
    pt["dram_bandwidth_bytes_per_sec"] =
        static_cast<int64_t>(t.dramBandwidthBytesPerSec);

    llvm::json::Object params;
    params["count"] = static_cast<int64_t>(t.paramCount);
    params["memory_bytes"] = static_cast<int64_t>(t.paramMemoryBytes);
    params["memory_bytes_bfp8"] = static_cast<int64_t>(t.paramMemoryBytesBfp8);
    pt["params"] = std::move(params);

    llvm::json::Object kv;
    kv["count"] = static_cast<int64_t>(t.kvCacheCount);
    kv["memory_bytes"] = static_cast<int64_t>(t.kvCacheMemoryBytes);
    kv["memory_bytes_bfp8"] = static_cast<int64_t>(t.kvCacheMemoryBytesBfp8);
    pt["kv_cache"] = std::move(kv);

    llvm::json::Object rl;
    rl["dram_time_ms"] = t.dramRooflineTimeMs;
    rl["top_perf_time_ms"] = t.topPerfTimeMs;
    rl["top_perf_samples_per_sec"] = t.topPerfSamplesPerSec;
    pt["roofline"] = std::move(rl);

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
    bool perfTargetsComputed = false;

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

      FailureOr<PerfTargets> computed =
          computePerfTargets(module, outerFuncs.front());
      if (failed(computed)) {
        signalPassFailure();
        return;
      }
      perfTargets = std::move(*computed);
      perfTargetsComputed = true;
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

    llvm::json::Object jsonOutput;
    addSummaryToJson(jsonOutput, aggregatedMetrics);

    // Only emit perf_targets when we actually computed against a system
    // desc; an empty `arch` means computePerfTargets early-returned and the
    // numbers would all be zero.
    if (perfTargetsComputed && !perfTargets.arch.empty()) {
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
