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

// Marker trait for ops that do NOT have golden_function implementation.
// By default, all TTNN ops are assumed to have golden functions.
template <typename ConcreteType>
class NoGoldenFunctionTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, NoGoldenFunctionTrait> {};

// Helper function to check if a value is a "torch tensor" by tracing its
// defining op chain. A tensor is a torch tensor if it comes from:
// 1. A to_torch op
// 2. Any ttnn op marked with use_golden attribute
inline bool isTorchTensorValue(mlir::Value value) {
  mlir::Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    // Block argument - not a torch tensor
    return false;
  }

  // Check if the defining op is a to_torch op
  if (definingOp->getName().getStringRef() == "ttnn.to_torch") {
    return true;
  }

  // Check if the defining op has use_golden attribute
  if (definingOp->hasAttr("use_golden")) {
    return true;
  }

  return false;
}

// Verify use_golden attribute requirements.
// When use_golden attribute is present:
// 1. The op must NOT have the NoGoldenFunctionTrait
// 2. All tensor operands must come from a to_torch op or an op with use_golden
template <typename ConcreteType>
class VerifyUseGoldenTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, VerifyUseGoldenTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    // Check if use_golden attribute is present
    if (!op->hasAttr("use_golden")) {
      return mlir::success();
    }

    // Verify the op does not have NoGoldenFunctionTrait
    if (op->hasTrait<NoGoldenFunctionTrait>()) {
      return op->emitOpError()
             << "cannot use 'use_golden' attribute on op that has "
             << "NoGoldenFunctionTrait";
    }

    // Verify all tensor operands come from torch tensor sources
    for (mlir::Value operand : op->getOperands()) {
      auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType) {
        continue; // Skip non-tensor operands (e.g., device)
      }

      if (!isTorchTensorValue(operand)) {
        return op->emitOpError()
               << "when 'use_golden' attribute is present, all tensor operands "
               << "must come from a 'to_torch' op or another op with "
               << "'use_golden' attribute. "
               << "Operand does not satisfy this requirement.";
      }
    }

    return mlir::success();
  }
};
} // namespace mlir::tt::ttnn

#endif
