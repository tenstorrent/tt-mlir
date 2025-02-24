// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/STLExtras.h"

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
      return op->emitOpError(
          "Output tensor buffer type must match memory config buffer type.");
    }

    // Tensor memory layout is optional, if not set, it is assumed to be the
    // same as the output tensor memory layout.
    if (memoryConfigAttr.getTensorMemoryLayout() &&
        memoryConfigAttr.getTensorMemoryLayout() !=
            outputLayoutAttr.getMemLayout()) {
      return op->emitOpError("Output tensor layout memory space must match "
                             "memory config memory space.");
    }

    // Verify if the shard spec is the same.
    if (!llvm::equal(memoryConfigAttr.getShardSpec().getShardShape().getShape(),
                     outputLayoutAttr.getShardShape())) {
      return op->emitOpError(
          "Output tensor shard spec must match memory config shard spec.");
    }

    return mlir::success();
  }
};

// Trait to verify that operations have exactly N operands.
template <unsigned N>
class NOperandTTNN {
public:
  template <typename ConcreteType>
  class Impl
      : public mlir::OpTrait::TraitBase<ConcreteType, NOperandTTNN<N>::Impl> {
    static LogicalResult verifyTrait(Operation *op) {
      if (op->getNumOperands() != N) {
        return op->emitOpError() << "Operation " << op->getName()
                                 << " must have exactly " << N << " operands.";
      }
      return success();
    }
  };
};

} // namespace mlir::tt::ttnn

#endif
