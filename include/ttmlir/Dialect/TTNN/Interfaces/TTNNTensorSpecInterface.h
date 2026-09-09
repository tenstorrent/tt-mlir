// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::ttnn {
inline MemoryConfigAttr getMemoryConfigFromResult(mlir::Operation *op) {
  if (op->getNumResults() == 0) {
    return nullptr;
  }

  auto output = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!output) {
    return nullptr;
  }

  if (!output.getEncoding()) {
    return nullptr;
  }

  return llvm::TypeSwitch<mlir::Attribute, MemoryConfigAttr>(
             output.getEncoding())
      .Case<TTNNLayoutAttr>([](TTNNLayoutAttr layoutAttr) {
        return MemoryConfigAttr::get(layoutAttr);
      })
      .Case<TTNNNDLayoutAttr>([](TTNNNDLayoutAttr layoutAttr) {
        return MemoryConfigAttr::get(layoutAttr);
      })
      .Default([](mlir::Attribute) { return MemoryConfigAttr(); });
}

// Derives the target page layout from the targeted result's TTNN(ND)LayoutAttr
// encoding. Returns a null LayoutAttr if the result is missing encoding.
inline LayoutAttr getLayoutAttrFromResult(mlir::Operation *op,
                                          unsigned int resultIndex = 0) {
  if (resultIndex >= op->getNumResults()) {
    return nullptr;
  }

  auto output =
      mlir::dyn_cast<RankedTensorType>(op->getResult(resultIndex).getType());
  if (!output || !output.getEncoding()) {
    return nullptr;
  }

  return llvm::TypeSwitch<mlir::Attribute, LayoutAttr>(output.getEncoding())
      .Case<TTNNLayoutAttr>([](TTNNLayoutAttr layoutAttr) {
        return LayoutAttr::get(layoutAttr.getContext(), layoutAttr.getLayout());
      })
      .Case<TTNNNDLayoutAttr>([](TTNNNDLayoutAttr layoutAttr) {
        return LayoutAttr::get(layoutAttr.getContext(), layoutAttr.getLayout());
      })
      .Default([](mlir::Attribute) { return LayoutAttr(); });
}

// Derives the data type carried by a tensor-typed Value. Prefers the
// TTNN(ND)LayoutAttr encoding's dataType when present, and otherwise falls
// back to the tensor's element type. Returns a null DataTypeAttr only if the
// value is not a RankedTensorType, or the element type is not a recognized
// TTMLIR data type.
inline ttcore::DataTypeAttr getDtypeFromValue(mlir::Value value) {
  auto tensor = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!tensor) {
    return nullptr;
  }

  mlir::MLIRContext *ctx = value.getContext();

  // Prefer the TTNN(ND)LayoutAttr encoding's dataType when present.
  if (mlir::Attribute encoding = tensor.getEncoding()) {
    if (auto layoutAttr = mlir::dyn_cast<TTNNLayoutAttr>(encoding)) {
      return ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
    }
    if (auto layoutAttr = mlir::dyn_cast<TTNNNDLayoutAttr>(encoding)) {
      return ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
    }
  }

  // Fall back to deriving from the tensor's element type.
  std::optional<ttcore::DataType> dataType =
      ttcore::elementTypeToDataTypeImpl(tensor.getElementType());
  assert(dataType && "element type must be a recognized TTMLIR data type");
  return ttcore::DataTypeAttr::get(ctx, *dataType);
}

// Derives the output data type from the targeted result. Returns a null
// DataTypeAttr if the op has no result, otherwise delegates to
// getDtypeFromValue on the targeted result.
inline ttcore::DataTypeAttr getDtypeFromResult(mlir::Operation *op,
                                               unsigned int resultIndex) {
  if (op->getNumResults() == 0 && resultIndex == 0) {
    return nullptr;
  }
  assert(resultIndex < op->getNumResults() && "result index out of bounds");
  return getDtypeFromValue(op->getResult(resultIndex));
}

// Convenience wrapper to convert a (possibly null) DataTypeAttr into the
// std::optional<ttcore::DataType> form.
inline std::optional<ttcore::DataType>
dataTypeAttrToOptional(ttcore::DataTypeAttr attr) {
  if (!attr) {
    return std::nullopt;
  }
  return attr.getValue();
}

inline std::optional<Layout> layoutAttrToOptional(LayoutAttr attr) {
  if (!attr) {
    return std::nullopt;
  }
  return attr.getValue();
}

// Verifies that tensor results carry a TTNN(ND)LayoutAttr encoding.
template <typename ConcreteType>
mlir::LogicalResult verifyTTNNLayoutInterface(mlir::Operation *op) {
  for (Value result : op->getResults()) {
    auto output = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (!output) {
      continue;
    }
    if (!mlir::dyn_cast_if_present<TTNNLayoutAttr>(output.getEncoding()) &&
        !mlir::dyn_cast_if_present<TTNNNDLayoutAttr>(output.getEncoding())) {
      return op->emitOpError("Output tensor type missing layout attribute");
    }
  }
  return mlir::success();
}

// Verifies the TTNNComputeKernelConfigInterface
template <typename ConcreteType>
mlir::LogicalResult
verifyTTNNComputeKernelConfigInterface(mlir::Operation *op) {
  // ComputeKernelConfig is a runtime execution hint that doesn't affect IR
  // semantics, so there are no invariants to verify at the IR
  // level. Unlike MemoryConfig which must match tensor layouts in the type
  // system, ComputeKernelConfig parameters (math_fidelity, fp32_dest_acc_en,
  // etc.) are purely optimization hints for the backend runtime. If specific
  // validation is needed in the future (e.g., checking compatibility with
  // operation types), it can be added here.
  return mlir::success();
}
} // namespace mlir::tt::ttnn

#include "ttmlir/Dialect/TTNN/Interfaces/TTNNTensorSpecInterface.h.inc"

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H
