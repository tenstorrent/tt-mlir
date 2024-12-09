// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_TTNNANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_TTNNANALYSIS_H

#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn {
// Base class for all TTNN analyses.
//
template <class I, class R>
class TTNNAnalysis {
protected:
  Operation *op;
  bool isValid = false;
  R analysisResult;
  I analysisInput;

  TTNNAnalysis(Operation *op) : op(op) {}

  // Actual implementation of the analysis.
  // Must be implemented by every analysis type.
  //
  virtual void analysisImplementation() = 0;

  // Load overrides if they exist.
  // Must be implemented by every analysis type.
  // Returns true if analysis should be skipped.
  //
  virtual bool applyOverrides() = 0;

public:
  virtual ~TTNNAnalysis() {};

  // Initialize the analysis with the input if needed.
  //
  void init(const I &input) {
    // Analysis can be cached and reused. Check that input remained the same.
    //
    if (analysisInput != input) {
      analysisInput = input;
      isValid = false;
    }
  }

  // Get the analysis result.
  //
  const R &getResult() {
    runAnalysis();
    return analysisResult;
  }

private:
  // Run the analysis.
  //
  void runAnalysis() {
    // Skip the analysis if it was already run and input params haven't changed.
    //
    if (!isValid) {
      // Apply overrides if needed.
      //
      bool skipAnalysis = applyOverrides();

      if (!skipAnalysis) {
        analysisImplementation();
      }

      isValid = true;
    }
  }
};
} // namespace mlir::tt::ttnn

#endif
