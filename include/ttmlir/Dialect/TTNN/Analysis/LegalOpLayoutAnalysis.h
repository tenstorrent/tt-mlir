// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "llvm/ADT/StringMap.h"

namespace mlir::tt::ttnn {

struct LegalOpLayoutAnalysisInput {
  TensorTypeLayoutsForScalarType *possibleLayouts;
  uint64_t maxShardedConfigs;
  llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides;
  bool rowMajorEnabled;

  LegalOpLayoutAnalysisInput()
      : possibleLayouts(nullptr), maxShardedConfigs(0),
        outputLayoutOverrides(nullptr), rowMajorEnabled(false) {}

  LegalOpLayoutAnalysisInput(
      TensorTypeLayoutsForScalarType *tensorTypePossibleLayouts,
      uint64_t maxShardedConfigs,
      llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides,
      bool rowMajorEnabled)
      : possibleLayouts(tensorTypePossibleLayouts),
        maxShardedConfigs(maxShardedConfigs),
        outputLayoutOverrides(outputLayoutOverrides),
        rowMajorEnabled(rowMajorEnabled) {}

  bool operator==(const LegalOpLayoutAnalysisInput &rhs) const {
    return possibleLayouts == rhs.possibleLayouts &&
           maxShardedConfigs == rhs.maxShardedConfigs &&
           outputLayoutOverrides == rhs.outputLayoutOverrides &&
           rowMajorEnabled == rhs.rowMajorEnabled;
  }

  bool operator!=(const LegalOpLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// This analysis takes all possible layouts for the output tensor of the op and
// applies output layout overrides. It picks top configurations based on grid
// volume and returns at most maxShardedConfigs sharded layouts plus
// interleaved layouts.
class LegalOpLayoutAnalysis
    : public TTNNAnalysis<LegalOpLayoutAnalysisInput, std::vector<OpConfig>> {

public:
  LegalOpLayoutAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

  void fillTTNNLayoutAttrs(TTNNLayoutAttr layout);
  void fillOpSpecificAttrs();
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALOPLAYOUTANALYSIS_H
