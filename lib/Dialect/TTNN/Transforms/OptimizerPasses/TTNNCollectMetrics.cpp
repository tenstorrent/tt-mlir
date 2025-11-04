// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNCOLLECTMETRICS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

struct OperationMetrics {
  std::string opName;
  std::string location;
  bool isSharded = false;
  bool isSpilledToDRAM = false;
  bool hasSystemMemory = false;
  std::string layoutInfo;
  std::string memoryConfigInfo;
};

struct AggregatedMetrics {
  uint64_t totalOps = 0;
  uint64_t shardedOps = 0;
  uint64_t effectivelyShardedOps = 0;
  uint64_t shardedAndSpilledOps = 0;
  uint64_t dramSpilledOps = 0;
  uint64_t systemMemoryOps = 0;

  double shardedPercentage = 0.0;
  double effectivelyShardedPercentage = 0.0;
  double spilledPercentage = 0.0;
  double systemMemoryPercentage = 0.0;

  void calculatePercentages() {
    if (totalOps == 0) {
      return;
    }
    shardedPercentage = (static_cast<double>(shardedOps) / totalOps) * 100.0;
    effectivelyShardedPercentage =
        (static_cast<double>(effectivelyShardedOps) / totalOps) * 100.0;
    spilledPercentage =
        (static_cast<double>(dramSpilledOps) / totalOps) * 100.0;
    systemMemoryPercentage =
        (static_cast<double>(systemMemoryOps) / totalOps) * 100.0;
  }
};

class TTNNCollectMetrics
    : public impl::TTNNCollectMetricsBase<TTNNCollectMetrics> {

private:
  // Get layout information as string for debugging
  std::string getLayoutInfo(Operation *op) {
    if (op->getNumResults() == 0) {
      return "";
    }

    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType) {
      return "";
    }

    if (auto encoding = tensorType.getEncoding()) {
      return llvm::formatv("{0}", encoding).str();
    }
    return "";
  }

  // Get memory config information as string for debugging
  std::string getMemoryConfigInfo(Operation *op) {
    // Try to find memory_config attribute
    if (auto memConfigAttr = op->getAttr("memory_config")) {
      return llvm::formatv("{0}", memConfigAttr).str();
    }
    return "";
  }

  // Get location as string for debugging
  std::string getLocationString(Operation *op) {
    if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc())) {
      return nameLoc.getName().str();
    }
    if (auto fileLoc = dyn_cast<FileLineColLoc>(op->getLoc())) {
      return llvm::formatv("{0}:{1}:{2}", fileLoc.getFilename(),
                           fileLoc.getLine(), fileLoc.getColumn())
          .str();
    }
    return "unknown";
  }

  // Analyze an operation and determine its metrics
  OperationMetrics analyzeOperation(Operation *op) {
    OperationMetrics metrics;
    metrics.opName = op->getName().getStringRef().str();
    metrics.location = getLocationString(op);
    metrics.layoutInfo = getLayoutInfo(op);
    metrics.memoryConfigInfo = getMemoryConfigInfo(op);
    metrics.isSharded = utils::producesShardedL1Layout(op);
    metrics.hasSystemMemory = utils::producesSystemMemoryLayout(op);

    return metrics;
  }

  // Find operations that are spilled to DRAM by analyzing toLayoutOps
  void identifyDRAMSpills(func::FuncOp funcOp,
                          llvm::DenseMap<Value, bool> &spilledValues) {
    int totalToLayoutOps = 0;
    int dramSpillOps = 0;
    funcOp.walk([&](ttnn::ToLayoutOp toLayoutOp) {
      totalToLayoutOps++;
      if (utils::producesDRAMLayout(toLayoutOp)) {
        dramSpillOps++;
        llvm::outs() << "Identified DRAM spill in ToLayoutOp at "
                     << getLocationString(toLayoutOp) << "\n";
        // Mark the input operand's defining op's result as spilled
        Value input = toLayoutOp.getInput();
        if (input.getDefiningOp()) {
          spilledValues[input] = true;
        }
      }
    });
    llvm::outs() << "Total ToLayoutOps: " << totalToLayoutOps << "\n";
    llvm::outs() << "DRAM spill ops: " << dramSpillOps << "\n";
  }

public:
  using impl::TTNNCollectMetricsBase<
      TTNNCollectMetrics>::TTNNCollectMetricsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::vector<OperationMetrics> operationDetails;
    AggregatedMetrics aggregatedMetrics;
    llvm::DenseMap<Value, bool> spilledValues;

    // Debug: Count operations by type
    llvm::StringMap<int> operationTypeCounts;

    module->walk([&](func::FuncOp func) {
      // Filter out all const-eval functions.
      if (ttmlir::utils::isConstEvalFunc(func)) {
        llvm::outs() << "Skipping const-eval function: " << func.getName().str()
                     << "\n";
        return;
      }

      // First pass: identify DRAM spills
      identifyDRAMSpills(func, spilledValues);

      // Second pass: analyze all operations
      func->walk([&](Operation *op) {
        // Skip operations without tensor results
        if (op->getNumResults() == 0) {
          return;
        }

        // Check if result is a tensor
        auto tensorType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!tensorType) {
          return;
        }

        // Skip excluded operations
        if (isa<ttnn::GetDeviceOp, ttnn::ToLayoutOp, ttnn::PrepareConv2dBiasOp,
                ttnn::PrepareConv2dWeightsOp>(op)) {
          return;
        }

        // Only analyze TTNN operations
        if (!isa<TTNNDialect>(op->getDialect())) {
          return;
        }

        // Debug: Count operations by type
        std::string opName = op->getName().getStringRef().str();
        operationTypeCounts[opName]++;

        aggregatedMetrics.totalOps++;

        OperationMetrics opMetrics = analyzeOperation(op);

        // Check if this operation's result is spilled
        if (op->getNumResults() > 0) {
          Value result = op->getResult(0);
          if (spilledValues.count(result) && spilledValues[result]) {
            opMetrics.isSpilledToDRAM = true;
          }
        }

        // Update aggregated metrics
        if (opMetrics.isSharded) {
          aggregatedMetrics.shardedOps++;
          if (!opMetrics.isSpilledToDRAM) {
            aggregatedMetrics.effectivelyShardedOps++;
          } else {
            aggregatedMetrics.shardedAndSpilledOps++;
          }
        }

        if (opMetrics.isSpilledToDRAM) {
          aggregatedMetrics.dramSpilledOps++;
        }

        if (opMetrics.hasSystemMemory) {
          aggregatedMetrics.systemMemoryOps++;
        }

        operationDetails.push_back(opMetrics);
      });
    });

    // Calculate percentages
    aggregatedMetrics.calculatePercentages();

    // Generate JSON output
    llvm::json::Object jsonOutput;

    // Add aggregated metrics
    llvm::json::Object summary;
    summary["total_ops"] = static_cast<int64_t>(aggregatedMetrics.totalOps);
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

    // Add detailed operation information if verbose mode is enabled
    if (optimizerMetricsVerboseOutputEnabled) {
      llvm::json::Array operations;
      for (const auto &opMetrics : operationDetails) {
        llvm::json::Object opJson;
        opJson["operation"] = opMetrics.opName;
        opJson["location"] = opMetrics.location;
        opJson["is_sharded"] = opMetrics.isSharded;
        opJson["is_spilled_to_dram"] = opMetrics.isSpilledToDRAM;
        opJson["has_system_memory"] = opMetrics.hasSystemMemory;
        opJson["layout_info"] = opMetrics.layoutInfo;
        opJson["memory_config_info"] = opMetrics.memoryConfigInfo;
        operations.push_back(std::move(opJson));
      }
      jsonOutput["operations"] = std::move(operations);
    }

    // Write JSON to file
    std::error_code ec;
    llvm::raw_fd_ostream os(optimizerMetricsOutputFile, ec);
    if (ec) {
      module.emitError("Failed to open output file: " +
                       optimizerMetricsOutputFile.getValue());
      // pass failure
      return;
    }

    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(jsonOutput)))
       << "\n";
    os.close();

    // Print summary to stdout for immediate feedback
    llvm::outs() << "TTNN Metrics Collection Complete:\n";
    llvm::outs() << "  Total operations: " << aggregatedMetrics.totalOps
                 << "\n";
    llvm::outs() << "  Sharded operations: " << aggregatedMetrics.shardedOps
                 << " ("
                 << llvm::format("%.2f", aggregatedMetrics.shardedPercentage)
                 << "%)\n";
    llvm::outs() << "  Effectively sharded: "
                 << aggregatedMetrics.effectivelyShardedOps << " ("
                 << llvm::format("%.2f",
                                 aggregatedMetrics.effectivelyShardedPercentage)
                 << "%)\n";
    llvm::outs() << "  Sharded and spilled: "
                 << aggregatedMetrics.shardedAndSpilledOps << "\n";
    llvm::outs() << "  DRAM spilled: " << aggregatedMetrics.dramSpilledOps
                 << " ("
                 << llvm::format("%.2f", aggregatedMetrics.spilledPercentage)
                 << "%)\n";
    llvm::outs() << "  System memory: " << aggregatedMetrics.systemMemoryOps
                 << " ("
                 << llvm::format("%.2f",
                                 aggregatedMetrics.systemMemoryPercentage)
                 << "%)\n";
    llvm::outs() << "  Metrics exported to: " << optimizerMetricsOutputFile
                 << "\n";

    // Debug: Print operation type breakdown
    llvm::outs() << "\nOperation type breakdown:\n";
    for (const auto &pair : operationTypeCounts) {
      llvm::outs() << "  " << pair.first() << ": " << pair.second << "\n";
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
