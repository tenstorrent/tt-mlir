// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALTENSORLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALTENSORLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {

struct LegalTensorLayoutAnalysisInput {
  GridAttr maxGrid;
  llvm::DenseSet<Type> *allowedScalarTypes;
  bool rowMajorAllowed;

  LegalTensorLayoutAnalysisInput()
      : maxGrid(nullptr), allowedScalarTypes(nullptr), rowMajorAllowed(false) {}

  LegalTensorLayoutAnalysisInput(GridAttr maxGrid,
                                 llvm::DenseSet<Type> *allowedScalarTypes,
                                 bool rowMajorAllowed)
      : maxGrid(maxGrid), allowedScalarTypes(allowedScalarTypes),
        rowMajorAllowed(rowMajorAllowed) {}

  bool operator==(const LegalTensorLayoutAnalysisInput &rhs) const {
    return maxGrid == rhs.maxGrid &&
           allowedScalarTypes == rhs.allowedScalarTypes &&
           rowMajorAllowed == rhs.rowMajorAllowed;
  }

  bool operator!=(const LegalTensorLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// This analysis generates all possible layouts for all the ranked tensors in
// the graph under the given module op.
class LegalTensorLayoutAnalysis
    : public TTNNAnalysis<LegalTensorLayoutAnalysisInput,
                          TensorTypeLayoutsMap> {
public:
  using TTNNAnalysis<LegalTensorLayoutAnalysisInput,
                     TensorTypeLayoutsMap>::TTNNAnalysis;

  LegalTensorLayoutAnalysis(Operation *op) : TTNNAnalysis(op) {}

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

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_LEGALTENSORLAYOUTANALYSIS_H
