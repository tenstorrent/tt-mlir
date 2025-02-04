// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TTMLIR_DIALECT_TTNN_IR_VERIFICATIONINTERFACE_H
#define TTMLIR_DIALECT_TTNN_IR_VERIFICATIONINTERFACE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/Operation.h"

namespace mlir::tt::ttnn {
// Verifies the TTNNVerificationInterface
mlir::LogicalResult verifyTTNNVerificationInterface(mlir::Operation *op);
} // namespace mlir::tt::ttnn

namespace mlir::OpTrait {
// Verifies that operation changes output data type
template <typename ConcreteType>
class ChangesDataType : public TraitBase<ConcreteType, ChangesDataType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
    RankedTensorType outputType =
        mlir::cast<RankedTensorType>(output.getType());

    if (inputType.getElementType() == outputType.getElementType()) {
      return op->emitOpError(
          "Output data type should be different from input.");
    }

    tt::ttnn::TTNNLayoutAttr inputLayout =
        cast<tt::ttnn::TTNNLayoutAttr>(inputType.getEncoding());
    tt::ttnn::TTNNLayoutAttr outputLayout =
        cast<tt::ttnn::TTNNLayoutAttr>(outputType.getEncoding());

    if (!changesDataType(inputLayout, outputLayout)) {
      return op->emitOpError("Output layout data type should be different from "
                             "input layout data type.");
    }

    return mlir::success();
  }
};

// Verifies that operation changes output buffer type
template <typename ConcreteType>
class ChangesBufferType : public TraitBase<ConcreteType, ChangesBufferType> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(input.getType()));
    auto outputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(output.getType()));

    if (!changesBufferType(inputLayout, outputLayout)) {
      return op->emitOpError(
          "Output buffer type should be different from input buffer type.");
    }

    return mlir::success();
  }
};

// Verifies that operation changes output memory layout
template <typename ConcreteType>
class ChangesMemoryLayout
    : public TraitBase<ConcreteType, ChangesMemoryLayout> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(input.getType()));
    auto outputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(output.getType()));

    if (!changesMemoryLayout(inputLayout, outputLayout)) {
      return op->emitOpError(
          "Output memory layout should be different from input memory layout.");
    }

    return mlir::success();
  }
};

template <typename ConcreteType>
class ChangesTensorLayout
    : public TraitBase<ConcreteType, ChangesTensorLayout> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(input.getType()));
    auto outputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(output.getType()));

    if (!changesTensorLayout(inputLayout, outputLayout)) {
      return op->emitOpError(
          "Output tensor layout should be different from input tensor layout.");
    }

    return mlir::success();
  }
};

template <typename ConcreteType>
class DoesNotChangeTensorEncoding
    : public TraitBase<ConcreteType, DoesNotChangeTensorEncoding> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    if (op->hasTrait<ChangesDataType>() || op->hasTrait<ChangesBufferType>() ||
        op->hasTrait<ChangesMemoryLayout>() ||
        op->hasTrait<ChangesTensorLayout>()) {
      return success();
    }

    if (op->getNumOperands() == 0 || op->getNumResults() == 0) {
      return success();
    }

    Value input = op->getOperand(0);
    Value output = op->getResult(0);

    auto inputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(input.getType()));
    auto outputLayout = tt::ttnn::utils::getLayoutAttrFromTensor(
        cast<RankedTensorType>(output.getType()));

    if (inputLayout != outputLayout) {
      return op->emitOpError("Output tensor encoding should be the same as "
                             "input tensor encoding.");
    }

    return success();
  }
};

} // namespace mlir::OpTrait

#include "ttmlir/Dialect/TTNN/IR/TTNNVerificationInterface.h.inc"

#endif
