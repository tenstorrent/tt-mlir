// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_TTIRANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_TTIRANALYSIS_H

#include "mlir/IR/Operation.h"
#include "ttmlir/Dialect/TTIR/Analysis/OverrideMap.h"

namespace mlir::tt::ttir {
// Base class for all TTIR analyses.
//
template <class I, class R> class TTIRAnalysis {
protected:
  Operation *op;
  bool is_valid = false;
  std::string attributeName;
  R analysis_result;
  I analysis_input;

  TTIRAnalysis(Operation *op) : op(op) {}

  // Actual implementation of the analysis.
  // Must be implemented by every analysis type.
  //
  virtual void analysisImplementation() = 0;
  virtual void handleOverride(llvm::json::Object *override) = 0;

public:
  virtual ~TTIRAnalysis() {};

  // Initialize the analysis with the input if needed.
  //
  void init(const I &input) {
    // Analysis can be cached and reused. Check that input remained the same.
    //
    if (analysis_input != input) {
      analysis_input = input;
      is_valid = false;
    }
  }

  // Get the analysis result.
  //
  const R &getResult() {
    auto overrideMap = tt::OverrideMap::getInstance();
    if (overrideMap->containsOverride(attributeName))
      handleOverride(overrideMap->getOverride(attributeName));
    else
      runAnalysis();
    return analysis_result;
  }

private:
  // Run the analysis.
  //
  void runAnalysis() {
    // Skip the analysis if it was already run and input params haven't changed.
    //
    if (!is_valid) {
      analysisImplementation();
      is_valid = true;
    }
  }
};
} // namespace mlir::tt::ttir

#endif
