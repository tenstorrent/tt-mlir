// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_ALLPOSSIBLELAYOUTSANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_ALLPOSSIBLELAYOUTSANALYSIS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Support/Logger.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {

struct AllPossibleLayoutsAnalysisInput {
  ttcore::GridAttr maxGrid;
  llvm::DenseSet<Type> *allowedScalarTypes;
  bool rowMajorAllowed;

  AllPossibleLayoutsAnalysisInput()
      : maxGrid(nullptr), allowedScalarTypes(nullptr), rowMajorAllowed(false) {}

  AllPossibleLayoutsAnalysisInput(ttcore::GridAttr maxGrid,
                                  llvm::DenseSet<Type> *allowedScalarTypes,
                                  bool rowMajorAllowed)
      : maxGrid(maxGrid), allowedScalarTypes(allowedScalarTypes),
        rowMajorAllowed(rowMajorAllowed) {}

  bool operator==(const AllPossibleLayoutsAnalysisInput &rhs) const {
    return maxGrid == rhs.maxGrid &&
           allowedScalarTypes == rhs.allowedScalarTypes;
  }

  bool operator!=(const AllPossibleLayoutsAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

class AllPossibleLayoutsAnalysis
    : public TTNNAnalysis<AllPossibleLayoutsAnalysisInput,
                          TensorTypeLayoutsMap> {
public:
  using TTNNAnalysis<AllPossibleLayoutsAnalysisInput,
                     TensorTypeLayoutsMap>::TTNNAnalysis;

  AllPossibleLayoutsAnalysis(Operation *op) : TTNNAnalysis(op) {}

private:
  // Main analysis implementation
  void analysisImplementation() override;

  // Apply overrides
  bool applyOverrides() override { return false; }

  // Helper to process a single tensor type
  void processTensorType(RankedTensorType tensorType);

  // Generate layouts for a tensor type
  std::vector<TTNNLayoutAttr> generateLayouts(RankedTensorType tensorType);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_ALLPOSSIBLELAYOUTSANALYSIS_H
