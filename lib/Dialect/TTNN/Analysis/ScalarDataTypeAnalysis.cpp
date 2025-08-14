// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ScalarDataTypeAnalysis.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"

namespace mlir::tt::ttnn {

Type ScalarDataTypeAnalysis::getScalarType(Type type) const {
  // Assert that the type is a RankedTensorType
  assert(mlir::isa<RankedTensorType>(type) && "Expected RankedTensorType");

  auto tensorType = mlir::cast<RankedTensorType>(type);
  if (auto layoutAttr =
          mlir::dyn_cast_or_null<TTNNLayoutAttr>(tensorType.getEncoding())) {
    // If the tensor type is quantized, compare the storage type of the
    // quantized type to the scalar element type of the layout.
    if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(
            tensorType.getElementType())) {
      assert(quantType.getStorageType() == layoutAttr.getScalarElementType() &&
             "Expected quantized storage type to match scalar element type in "
             "TTNNLayoutAttr");
    }
    // Otherwise, require an exact match between the scalar element type of the
    // layout and the tensor element type.
    else {
      assert(layoutAttr.getScalarElementType() == tensorType.getElementType() &&
             "Expected scalar element type in TTNNLayoutAttr to match tensor "
             "element type");
    }
    // Prefer the layout's scalar element type for downstream analyses.
    return layoutAttr.getScalarElementType();
  }
  return tensorType.getElementType();
}

void ScalarDataTypeAnalysis::analysisImplementation() {
  Operation *moduleOp = op;
  llvm::DenseSet<Type> scalarTypes;

  // Helper to process data type overrides
  auto processOverrides = [&](Operation *op) {
    if (!isa<NameLoc>(op->getLoc())) {
      return;
    }

    StringRef opLocName = mlir::cast<NameLoc>(op->getLoc()).getName();
    if (analysisInput.outputLayoutOverrides &&
        analysisInput.outputLayoutOverrides->find(opLocName) !=
            analysisInput.outputLayoutOverrides->end()) {
      auto &override =
          analysisInput.outputLayoutOverrides->find(opLocName)->getValue();
      if (override.dataType.has_value()) {
        Type scalarType =
            dataTypeToElementType(op->getContext(), override.dataType.value());
        scalarTypes.insert(scalarType);
      }
    }
  };

  // Walk through all operations
  moduleOp->walk([&](Operation *op) {
    // Check overrides first
    processOverrides(op);

    // Process result types
    for (Value result : op->getResults()) {
      if (mlir::isa<RankedTensorType>(result.getType())) {
        scalarTypes.insert(getScalarType(result.getType()));
      }
    }

    // Process operand types
    for (Value operand : op->getOperands()) {
      if (mlir::isa<RankedTensorType>(operand.getType())) {
        scalarTypes.insert(getScalarType(operand.getType()));
      }
    }
  });

  analysisResult = std::move(scalarTypes);
}

} // namespace mlir::tt::ttnn
