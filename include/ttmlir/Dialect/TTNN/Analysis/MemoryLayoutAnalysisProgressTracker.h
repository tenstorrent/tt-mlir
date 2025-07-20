// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPROGRESSTRACKER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPROGRESSTRACKER_H

#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>

namespace mlir::tt::ttnn {

class MemoryLayoutAnalysisProgressTracker {
private:
  std::chrono::steady_clock::time_point startTime;
  size_t totalL1Chains = 0;
  bool enabled = false;

public:
  MemoryLayoutAnalysisProgressTracker()
      : startTime(std::chrono::steady_clock::now()) {
    // For now, disable progress tracking by default. Once we integrate
    // tt-logger with tt-mlir we can use it and log progress always (and remove
    // this env var). This way, we will have progress visible in every build.
    enabled = std::getenv("MLA_TRACK_PROGRESS") != nullptr;
  }

  void startAnalysis(Operation *rootOp, size_t numL1Chains,
                     const std::string &context = "") {
    totalL1Chains = numL1Chains;

    emitMsg(mlir::tt::ttnn::utils::getOpLocName(rootOp),
            "MemoryLayoutAnalysis: Starting Analysis: " + context +
                ", Total L1 Chains: " + std::to_string(numL1Chains));
  }

  void startL1Chain(Operation *firstOp, size_t chainIndex,
                    size_t numOpsInChain) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTime);

    emitMsg(mlir::tt::ttnn::utils::getOpLocName(firstOp),
            "MemoryLayoutAnalysis: Starting L1Chain " +
                std::to_string(chainIndex + 1) + "/" +
                std::to_string(totalL1Chains) +
                " (ops: " + std::to_string(numOpsInChain) + "), elapsed " +
                std::to_string(elapsed.count()) + "s");
  }

  void finishL1Chain(Operation *firstOp, size_t chainIndex, bool success) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTime);

    emitMsg(mlir::tt::ttnn::utils::getOpLocName(firstOp),
            "MemoryLayoutAnalysis: L1Chain " + std::to_string(chainIndex + 1) +
                "/" + std::to_string(totalL1Chains) +
                (success ? " COMPLETED" : " FAILED") + ", elapsed " +
                std::to_string(elapsed.count()) + "s");
  }

  void finishAnalysis(Operation *rootOp) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - startTime);

    emitMsg(mlir::tt::ttnn::utils::getOpLocName(rootOp),
            "MemoryLayoutAnalysis: Analysis Complete, total time: " +
                std::to_string(elapsed.count()) + "s");
  }

private:
  void emitMsg(std::string loc, std::string_view msg) {
    if (!enabled) {
      return;
    }
    llvm::outs() << loc << ": " << msg << "\n";
    llvm::outs().flush();
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSISPROGRESSTRACKER_H
