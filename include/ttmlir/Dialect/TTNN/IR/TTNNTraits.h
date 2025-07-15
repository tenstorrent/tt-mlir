// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::ttnn {

template <typename ConcreteType>
class HasMemoryConfigTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, HasMemoryConfigTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {

    // Check if the operation defines memory config attribute.
    auto attributeNames = ConcreteType::getAttributeNames();
    if (std::find(attributeNames.begin(), attributeNames.end(),
                  MemoryConfigAttr::getMnemonic()) == attributeNames.end()) {
      return op->emitOpError("Operation must define memory_config attribute.");
    }

    // Retrieve memory config attribute if it exist because it is optional.
    MemoryConfigAttr memoryConfigAttr =
        op->getAttrOfType<MemoryConfigAttr>(MemoryConfigAttr::getMnemonic());

    if (!memoryConfigAttr) {
      return mlir::success();
    }

    // Verify the output layout with the memory config attribute if exist.
    assert(op->getResults().size() == 1 &&
           "Operation should define only one result.");

    // Retrieve output layout.
    RankedTensorType output =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    TTNNLayoutAttr outputLayoutAttr =
        mlir::cast<TTNNLayoutAttr>(output.getEncoding());

    // Verify if the buffer type is the same.
    if (memoryConfigAttr.getBufferType().getValue() !=
        outputLayoutAttr.getBufferType()) {
      return op->emitOpError()
             << "Output tensor buffer type "
             << stringifyBufferType(outputLayoutAttr.getBufferType())
             << " must match memory config buffer type "
             << stringifyBufferType(
                    memoryConfigAttr.getBufferType().getValue());
    }

    // Tensor memory layout is optional, if not set, it is assumed to be the
    // same as the output tensor memory layout.
    if (memoryConfigAttr.getTensorMemoryLayout() &&
        memoryConfigAttr.getTensorMemoryLayout() !=
            outputLayoutAttr.getMemLayout()) {
      return op->emitOpError()
             << "Output tensor layout memory space "
             << stringifyTensorMemoryLayout(
                    outputLayoutAttr.getMemLayout().getValue())
             << " must match memory config memory space "
             << stringifyTensorMemoryLayout(
                    memoryConfigAttr.getTensorMemoryLayout().getValue());
    }

    if (memoryConfigAttr.getShardSpec()) {
      if (memoryConfigAttr.getShardSpec()->getShape() !=
          ShapeAttr::get(op->getContext(),
                         outputLayoutAttr.getScalarShardShape())) {
        return op->emitOpError()
               << "Output tensor scalar shard shape ("
               << outputLayoutAttr.getScalarShardShape()
               << ") must match memory config shard spec shape ("
               << memoryConfigAttr.getShardSpec()->getShape().getShape() << ")";
      }
    }

    return mlir::success();
  }
};

class HasOutputDTypeTraitBase {
public:
  static constexpr ::llvm::StringLiteral getOutputDTypeAttributeName() {
    return "output_dtype";
  }
};

template <typename ConcreteType>
class HasOutputDTypeTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, HasOutputDTypeTrait>,
      public HasOutputDTypeTraitBase {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    // Check if the operation defines output data type attribute.
    auto attributeNames = ConcreteType::getAttributeNames();
    if (std::find(attributeNames.begin(), attributeNames.end(),
                  getOutputDTypeAttributeName()) == attributeNames.end()) {
      return op->emitOpError(
          "Operation must define output data type attribute.");
    }

    // Retrieve output layout.
    RankedTensorType output =
        mlir::cast<RankedTensorType>(op->getResult(0).getType());
    TTNNLayoutAttr outputLayoutAttr =
        mlir::dyn_cast_or_null<TTNNLayoutAttr>(output.getEncoding());

    // If output layout isn't present, skip the verification.
    if (!outputLayoutAttr) {
      return mlir::success();
    }

    // Retrieve output data type attribute.
    auto outputDTypeAttr =
        op->getAttrOfType<ttcore::DataTypeAttr>(getOutputDTypeAttributeName());
    if (!outputDTypeAttr) {
      return op->emitOpError("Output data type attribute is not defined for op "
                             "that has output layout attribute.");
    }

    // Compare output data type attribute with output tensor data type.
    if (outputDTypeAttr.getValue() != outputLayoutAttr.getDataType()) {
      return op->emitOpError()
             << "Output tensor layout data type "
             << DataTypeEnumToString(outputLayoutAttr.getDataType())
             << " must match output data type attribute "
             << DataTypeEnumToString(outputDTypeAttr.getValue());
    }

    return mlir::success();
  }
};

template <typename ConcreteType>
class ExplicateOperandBroadcastsTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      ExplicateOperandBroadcastsTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return mlir::success();
  }
};

template<typename ConcreteType>
class CheckBFloat8BTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CheckBFloat8BTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() == 0) {
      return mlir::success();
    }

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
      return mlir::success();
    }

    TTNNLayoutAttr layoutAttr =
        mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());
    Type elementType = resultType.getElementType();
    Type scalarElementType = layoutAttr.getScalarElementType();

    if (elementType != scalarElementType) {
      return op->emitOpError()
             << "Expected element type to match scalar element type, but got "
             << elementType << " and " << scalarElementType;
    }

    if (!isa<ttcore::BFloat8BType>(elementType)) {
      return mlir::success();
    }

    if (layoutAttr.getLayout() != Layout::Tile) {
      return op->emitOpError()
             << "BFloat8BType is only supported with Tile layout, but got "
             << stringifyLayout(layoutAttr.getLayout());
    }

    return mlir::success();
  }
};

} // namespace mlir::tt::ttnn

#endif
