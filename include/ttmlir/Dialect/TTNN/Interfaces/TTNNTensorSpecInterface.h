// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {
// Verifies the TTNNDtypeOpInterface
template <typename ConcreteType>
mlir::LogicalResult verifyTTNNDtypeOpInterface(mlir::Operation *op) {
  // Check if the operation defines output dtype attribute.
  auto outputDTypeAttr = mlir::cast<ConcreteType>(op).getDtypeAttr();

  // Retrieve output layout.
  for (Value result : op->getResults()) {
    RankedTensorType output = mlir::cast<RankedTensorType>(result.getType());
    TTNNLayoutAttr outputLayoutAttr =
        mlir::dyn_cast_if_present<TTNNLayoutAttr>(output.getEncoding());

    // If output layout isn't present, skip the verification.
    if (!outputLayoutAttr) {
      return mlir::success();
    }

    if (!outputDTypeAttr) {
      return op->emitOpError()
             << "output data type attribute is not defined for op "
             << "that has output layout data attribute "
             << DataTypeEnumToString(outputLayoutAttr.getDataType());
    }

    // Compare output data type attribute with output tensor data type.
    if (outputDTypeAttr.getValue() != outputLayoutAttr.getDataType()) {
      return op->emitOpError()
             << "output tensor layout data type "
             << DataTypeEnumToString(outputLayoutAttr.getDataType())
             << " must match output data type attribute "
             << DataTypeEnumToString(outputDTypeAttr.getValue());
    }
  }

  return mlir::success();
}

// Verifies the TTNNLayoutInterface
template <typename ConcreteType>
mlir::LogicalResult verifyTTNNLayoutInterface(mlir::Operation *op) {
  // Check if the operation defines output layout attribute.
  auto outputLayoutAttr = mlir::cast<ConcreteType>(op).getLayoutAttr();

  // Retrieve output layout.
  for (Value result : op->getResults()) {
    RankedTensorType output = mlir::cast<RankedTensorType>(result.getType());
    TTNNLayoutAttr outputTTNNLayoutAttr =
        mlir::dyn_cast_if_present<TTNNLayoutAttr>(output.getEncoding());

    // If output layout isn't present, skip the verification.
    if (!outputTTNNLayoutAttr) {
      return mlir::success();
    }

    // Retrieve output layout attribute.
    if (!outputLayoutAttr) {
      return op->emitOpError("output layout attribute is not defined for op "
                             "that has output layout attribute.");
    }

    // Compare output layout attribute with output tensor layout.
    if (outputLayoutAttr.getValue() != outputTTNNLayoutAttr.getLayout()) {
      return op->emitOpError()
             << "output tensor layout "
             << stringifyLayout(outputTTNNLayoutAttr.getLayout())
             << " must match output layout attribute "
             << stringifyLayout(outputLayoutAttr.getValue());
    }
  }

  return mlir::success();
}

// Verifies the TTNNMemoryConfigInterface
template <typename ConcreteType>
mlir::LogicalResult verifyTTNNMemoryConfigInterface(mlir::Operation *op) {
  // Check if the operation defines output memory config attribute.
  auto memoryConfigAttr = mlir::cast<ConcreteType>(op).getMemoryConfigAttr();

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
           << stringifyBufferType(memoryConfigAttr.getBufferType().getValue());
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
} // namespace mlir::tt::ttnn

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h.inc"

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H
