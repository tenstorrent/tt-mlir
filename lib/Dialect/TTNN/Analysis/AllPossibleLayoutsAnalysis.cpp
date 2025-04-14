// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/AllPossibleLayoutsAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/OptimizerUtils.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

void AllPossibleLayoutsAnalysis::analysisImplementation() {
  TensorTypeLayoutsMap result;
  mlir::ModuleOp moduleOp = mlir::cast<mlir::ModuleOp>(op);
  llvm::DenseSet<RankedTensorType> processedTypes;

  // Walk through module and collect all tensor types
  moduleOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (!mlir::isa<RankedTensorType>(operand.getType())) {
        continue;
      }
      auto tensorType = mlir::cast<RankedTensorType>(operand.getType());
      // Only process each unique tensor type once
      if (processedTypes.insert(tensorType).second) {
        processTensorType(tensorType);
      }
    }

    for (Value result : op->getResults()) {
      if (!mlir::isa<RankedTensorType>(result.getType())) {
        continue;
      }
      auto tensorType = mlir::cast<RankedTensorType>(result.getType());
      // Only process each unique tensor type once
      if (processedTypes.insert(tensorType).second) {
        processTensorType(tensorType);
      }
    }
  });
}

void AllPossibleLayoutsAnalysis::processTensorType(
    RankedTensorType tensorType) {
  // Generate all possible layouts for this tensor type
  auto layouts = generateLayouts(tensorType);

  // Categorize layouts by scalar type, memory layout, and data layout
  for (auto layout : layouts) {
    Type scalarType = layout.getScalarElementType();

    TensorMemoryLayout memLayout = TensorMemoryLayout::Interleaved;
    if (auto memLayoutAttr = layout.getMemLayout()) {
      memLayout = memLayoutAttr.getValue();
    }

    // Use helper functions to get indices (now directly as std::size_t)
    auto dataLayoutIndex = getDataLayoutIndex(layout.getLayout());
    auto memLayoutIndex = getMemoryLayoutIndex(memLayout);

    // Store layout in the categorized map
    analysisResult[tensorType][scalarType][dataLayoutIndex][memLayoutIndex]
        .push_back(layout);
  }
}

std::vector<TTNNLayoutAttr>
AllPossibleLayoutsAnalysis::generateLayouts(RankedTensorType tensorType) {

  std::vector<TTNNLayoutAttr> allLayouts;

  // Assert that we have allowedScalarTypes and it's not empty
  assert(
      analysisInput.allowedScalarTypes &&
      "AllPossibleLayoutsAnalysis requires allowedScalarTypes to be non-null");
  assert(!analysisInput.allowedScalarTypes->empty() &&
         "AllPossibleLayoutsAnalysis requires at least one scalar type");

  // Generate layouts for each allowed scalar type
  for (Type scalarType : *analysisInput.allowedScalarTypes) {
    auto layoutsForType = optimizer_utils::generateAllPossibleLayouts(
        tensorType.getContext(), tensorType, analysisInput.maxGrid, scalarType);

    allLayouts.insert(allLayouts.end(), layoutsForType.begin(),
                      layoutsForType.end());
  }

  return allLayouts;
}

} // namespace mlir::tt::ttnn
