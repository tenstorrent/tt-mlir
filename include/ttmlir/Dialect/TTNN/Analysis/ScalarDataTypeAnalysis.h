// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_SCALARDATATYPEANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_SCALARDATATYPEANALYSIS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {

struct ScalarDataTypeAnalysisInput {
  llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides;

  ScalarDataTypeAnalysisInput() : outputLayoutOverrides(nullptr) {}

  ScalarDataTypeAnalysisInput(
      llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides)
      : outputLayoutOverrides(outputLayoutOverrides) {}

  bool operator==(const ScalarDataTypeAnalysisInput &rhs) const {
    return outputLayoutOverrides == rhs.outputLayoutOverrides;
  }

  bool operator!=(const ScalarDataTypeAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class ScalarDataTypeAnalysis
    : public TTNNAnalysis<ScalarDataTypeAnalysisInput, llvm::DenseSet<Type>> {
public:
  ScalarDataTypeAnalysis(Operation *op)
      : TTNNAnalysis<ScalarDataTypeAnalysisInput, llvm::DenseSet<Type>>(op) {}

private:
  void analysisImplementation() override;
  bool applyOverrides() override { return false; }

  // Helper to extract scalar type from a tensor type.
  Type getScalarType(Type type) const;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_SCALARDATATYPEANALYSIS_H
