// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerOverrides.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "llvm/ADT/StringMap.h"

namespace mlir::tt::ttnn {

struct LegalLayoutAnalysisInput {
  TensorTypeLayoutsForScalarType *possibleLayouts;
  uint64_t maxShardedConfigs;
  llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides;
  llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides;
  bool rowMajorEnabled;

  LegalLayoutAnalysisInput()
      : possibleLayouts(nullptr), maxShardedConfigs(0),
        outputLayoutOverrides(nullptr), conv2dConfigOverrides(nullptr),
        rowMajorEnabled(false) {}

  LegalLayoutAnalysisInput(
      TensorTypeLayoutsForScalarType *tensorTypePossibleLayouts,
      uint64_t maxShardedConfigs,
      llvm::StringMap<OutputLayoutOverrideParams> *outputLayoutOverrides,
      llvm::StringMap<Conv2dConfigOverrideParams> *conv2dConfigOverrides,
      bool rowMajorEnabled)
      : possibleLayouts(tensorTypePossibleLayouts),
        maxShardedConfigs(maxShardedConfigs),
        outputLayoutOverrides(outputLayoutOverrides),
        conv2dConfigOverrides(conv2dConfigOverrides),
        rowMajorEnabled(rowMajorEnabled) {}

  bool operator==(const LegalLayoutAnalysisInput &rhs) const {
    return possibleLayouts == rhs.possibleLayouts &&
           maxShardedConfigs == rhs.maxShardedConfigs &&
           outputLayoutOverrides == rhs.outputLayoutOverrides &&
           conv2dConfigOverrides == rhs.conv2dConfigOverrides &&
           rowMajorEnabled == rhs.rowMajorEnabled;
  }

  bool operator!=(const LegalLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class LegalLayoutAnalysis
    : public TTNNAnalysis<LegalLayoutAnalysisInput, std::vector<OpConfig>> {
private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  LegalLayoutAnalysis(Operation *op) : TTNNAnalysis(op) {}
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALLAYOUTANALYSIS_H
