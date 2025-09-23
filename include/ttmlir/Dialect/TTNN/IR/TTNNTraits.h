// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H
#define TTMLIR_DIALECT_TTNN_IR_TTNNTRAITS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/OpDefinition.h"

namespace mlir::tt::ttnn {
template <typename ConcreteType>
class ExplicateOperandBroadcastsTrait
    : public mlir::OpTrait::TraitBase<ConcreteType,
                                      ExplicateOperandBroadcastsTrait> {};

// bfp8_b is tile type so we need some way to check if op which returns
// it has correct layout information in encoding.
template <typename ConcreteType>
class CheckBFloat8BTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CheckBFloat8BTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    if (op->getNumResults() == 0) {
      return mlir::success();
    }

    for (mlir::Value result : op->getResults()) {
      auto resultType = mlir::dyn_cast<RankedTensorType>(result.getType());
      if (!resultType) {
        continue;
      }

      mlir::Type elementType = resultType.getElementType();
      if (!mlir::isa<ttcore::TileType>(elementType)) {
        continue;
      }

      TTNNLayoutAttr layoutAttr =
          mlir::cast<TTNNLayoutAttr>(resultType.getEncoding());
      mlir::Type scalarElementType = layoutAttr.getScalarElementType();
      if (elementType != scalarElementType) {
        return op->emitOpError()
               << "Output element type must match the scalar "
                  "element type from encoding."
               << " Element type: " << elementType
               << ", Scalar element type: " << scalarElementType << ".";
      }
    }

    return mlir::success();
  }
};

// Trait to specify if op supports execution on host.
template <typename ConcreteType>
class CanExecuteOnHostTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, CanExecuteOnHostTrait> {};
} // namespace mlir::tt::ttnn

#endif
