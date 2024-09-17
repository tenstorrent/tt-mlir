// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"

namespace mlir::tt::ttir {

struct GenericAnalysisInput {
  DeviceAttr device;

  GenericAnalysisInput() : device(nullptr) {}
  GenericAnalysisInput(DeviceAttr device) : device(device) {}

  bool operator==(const GenericAnalysisInput &other) const {
    return device == other.device;
  }

  bool operator!=(const GenericAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct GenericAnalysisResult {
  llvm::DenseMap<Operation *, llvm::SmallVector<std::int64_t>> gridShapes;

  GenericAnalysisResult() = default;
};

// Determine shard chain configs.
//
class GenericAnalysis
    : public TTIRAnalysis<GenericAnalysisInput, GenericAnalysisResult> {
private:
  virtual void analysisImplementation() override;
  virtual bool applyOverrides() override;

public:
  GenericAnalysis(Operation *op) : TTIRAnalysis(op) {}
};
} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_GENERICANALYSIS_H
