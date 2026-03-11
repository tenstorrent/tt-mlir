// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNTENSORSPECINTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
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

  auto getFromLayout = [&](TTNNLayoutAttr layoutAttr) {
    auto bufferTypeAttr =
        mlir::cast<BufferTypeAttr>(layoutAttr.getMemref().getMemorySpace());
    return MemoryConfigAttr::get(op->getContext(), layoutAttr.getMemLayout(),
                                 bufferTypeAttr, /*shardSpec=*/std::nullopt);
  };

  auto getFromNDLayout = [&](TTNNNDLayoutAttr layoutAttr) {
    auto bufferTypeAttr =
        mlir::cast<BufferTypeAttr>(layoutAttr.getMemref().getMemorySpace());
    return MemoryConfigAttr::get(
        op->getContext(), layoutAttr.getMemLayout(), bufferTypeAttr,
        /*shardSpec=*/std::nullopt, /*ndShardSpec=*/std::nullopt);
  };

  if (!output.getEncoding()) {
    return nullptr;
  }
  return llvm::TypeSwitch<mlir::Attribute, MemoryConfigAttr>(
             output.getEncoding())
      .Case<TTNNLayoutAttr>(getFromLayout)
      .Case<TTNNNDLayoutAttr>(getFromNDLayout)
      .Default([&](mlir::Attribute) { return MemoryConfigAttr(); });
}

inline void setMemoryConfigOnResult(mlir::Operation *op,
                                    MemoryConfigAttr memoryConfigAttr) {
  if (!memoryConfigAttr || op->getNumResults() == 0) {
    return;
  }

  Value result = op->getResult(0);
  auto output = mlir::dyn_cast<RankedTensorType>(result.getType());
  if (!output) {
    return;
  }

  mlir::Attribute encoding = output.getEncoding();
  if (!encoding) {
    return;
  }
  if (auto layoutAttr = mlir::dyn_cast<TTNNLayoutAttr>(encoding)) {
    TTNNLayoutAttr updatedLayout =
        layoutAttr.withBufferType(memoryConfigAttr.getBufferType().getValue());
    if (auto tensorMemoryLayout = memoryConfigAttr.getTensorMemoryLayout()) {
      updatedLayout = updatedLayout.withMemoryLayout(tensorMemoryLayout);
    }
    if (auto shardSpec = memoryConfigAttr.getShardSpec()) {
      updatedLayout = updatedLayout.withShardShape(
          llvm::SmallVector<int64_t>(shardSpec->getShape().getShape()));
    }
    result.setType(RankedTensorType::get(
        output.getShape(), output.getElementType(), updatedLayout));
    return;
  }

  if (auto layoutAttr = mlir::dyn_cast<TTNNNDLayoutAttr>(encoding)) {
    auto tensorMemoryLayout = memoryConfigAttr.getTensorMemoryLayout()
                                  ? memoryConfigAttr.getTensorMemoryLayout()
                                  : layoutAttr.getMemLayout();
    auto memref =
        MemRefType::get(layoutAttr.getMemref().getShape(),
                        layoutAttr.getMemref().getElementType(), AffineMap(),
                        memoryConfigAttr.getBufferType());
    TTNNNDLayoutAttr updatedLayout = TTNNNDLayoutAttr::get(
        op->getContext(), layoutAttr.getGrid(), memref, tensorMemoryLayout,
        layoutAttr.getShardOrientation(),
        layoutAttr.getShardDistributionStrategy());
    result.setType(RankedTensorType::get(
        output.getShape(), output.getElementType(), updatedLayout));
  }
}

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

    // Some ops derive output dtype from output layout encoding.
    if (!outputDTypeAttr) {
      continue;
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

// Generically support different LayoutAttr types.
template <typename LayoutAttrType>
mlir::LogicalResult
verifyMemoryConfigWithLayout(mlir::Operation *op,
                             MemoryConfigAttr memoryConfigAttr,
                             LayoutAttrType outputLayoutAttr) {
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

// Verifies the TTNNMemoryConfigInterface
template <typename ConcreteType>
mlir::LogicalResult verifyTTNNMemoryConfigInterface(mlir::Operation *op) {
  // Memory config is derived directly from output layout encoding, so there is
  // no independent op attribute to cross-validate here.
  (void)op;
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
