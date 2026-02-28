// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_COMPILETIMESTATSOBSERVER_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_COMPILETIMESTATSOBSERVER_H

#include "ttmlir/Dialect/TTNN/Analysis/LayoutPropagationObserver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace mlir::tt::ttnn {

/// Observer that collects per-op compile-time statistics for the greedy
/// optimizer. Output is printed at DEBUG level (GreedyOptimizer component)
/// at the end of layout propagation.
///
/// Used instead of DecisionTraceObserver (not alongside it).
class CompileTimeStatsObserver : public LayoutPropagationObserver {
public:
  void onStart(llvm::StringRef funcName, size_t beamWidth) override {
    funcName_ = funcName.str();
  }

  void onOpSetup(Operation *op,
                 const std::vector<std::vector<InputCandidate>> &inputSets,
                 const OutputHints &hints, size_t crossProductSize) override {
    current_ = OpStats();
    current_.name = op->getName().getStringRef().str();
    std::string locStr;
    llvm::raw_string_ostream locOS(locStr);
    locOS << op->getLoc();
    current_.loc = locStr;
    current_.crossProduct = crossProductSize;
    opStart_ = std::chrono::steady_clock::now();
  }

  void onEvaluation(Operation *op, const OpConfig &hint, size_t hintIdx,
                    llvm::ArrayRef<TTNNLayoutAttr> inputLayouts, bool valid,
                    const BeamCandidate *candidate,
                    llvm::StringRef failureReason) override {
    ++current_.validations;
    if (valid) {
      ++current_.validResults;
    }
  }

  void onBeamResult(Operation *op, llvm::ArrayRef<BeamCandidate> beam,
                    bool usedDramFallback) override {
    auto now = std::chrono::steady_clock::now();
    current_.elapsedMs =
        std::chrono::duration<double, std::milli>(now - opStart_).count();
    allStats_.push_back(current_);
  }

  void onEnd(size_t totalOps) override {
    // Sort by elapsed time descending.
    std::sort(allStats_.begin(), allStats_.end(),
              [](const OpStats &a, const OpStats &b) {
                return a.elapsedMs > b.elapsedMs;
              });

    double totalMs = 0;
    size_t totalValidations = 0;
    size_t totalValid = 0;
    for (const auto &s : allStats_) {
      totalMs += s.elapsedMs;
      totalValidations += s.validations;
      totalValid += s.validResults;
    }

    llvm::raw_ostream &os = llvm::errs();
    os << "\n=== Greedy Optimizer Compile-Time Stats (func: " << funcName_
       << ") ===\n";
    os << "Total: " << llvm::format("%.0f", totalMs) << "ms across "
       << allStats_.size() << " ops, " << totalValidations << " validations ("
       << totalValid << " valid)\n\n";
    os << "    Time(ms)  Valid/Total   CrossProd  Op\n";

    for (const auto &s : allStats_) {
      std::string ratio =
          std::to_string(s.validResults) + "/" + std::to_string(s.validations);
      os << llvm::format("  %10.1f  %12s  %10zu  ", s.elapsedMs, ratio.c_str(),
                         s.crossProduct)
         << s.name << " (" << s.loc << ")\n";
    }
    os << "\n";
  }

private:
  struct OpStats {
    std::string name;
    std::string loc;
    size_t crossProduct = 0;
    size_t validations = 0;
    size_t validResults = 0;
    double elapsedMs = 0.0;
  };

  std::vector<OpStats> allStats_;
  OpStats current_;
  std::chrono::steady_clock::time_point opStart_;
  std::string funcName_;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_COMPILETIMESTATSOBSERVER_H
