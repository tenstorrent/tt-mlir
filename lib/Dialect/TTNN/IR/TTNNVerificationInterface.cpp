// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "ttmlir/Dialect/TTNN/IR/TTNNVerificationInterface.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {
#include "ttmlir/Dialect/TTNN/IR/TTNNVerificationInterface.cpp.inc"

// Verifier function for TTNN verification interface.
mlir::LogicalResult verifyTTNNVerificationInterface(mlir::Operation *op) {
  if (llvm::isa<GetDeviceOp>(op) ||
      llvm::isa<ToLayoutOp>(
          op) // To layout is composite op so we cannot check it for now
      ||
      llvm::isa<SliceOp>(op) // Slice op has hack to change input into row major
      || llvm::isa<EmbeddingOp>(op)         // Has workaround
      || llvm::isa<EmbeddingBackwardOp>(op) // Has workaround
      || llvm::isa<Conv2dOp>(op) // Has workaround in TTNNLayout (need to remove
                                 // this to check how it works)
      ||
      llvm::isa<ConvTranspose2dOp>(op) // Has workaround in TTNNLayout (need to
                                       // remove this to check how it works)
      || llvm::isa<EmptyOp>(op) || llvm::isa<DeallocateOp>(op) ||
      llvm::isa<OnesOp>(op) || llvm::isa<FullOp>(op)) {
    return mlir::success();
  }

  bool changesDataTypeTrait = op->hasTrait<mlir::OpTrait::ChangesDataType>();
  bool changesBufferTypeTrait =
      op->hasTrait<mlir::OpTrait::ChangesBufferType>();
  bool changesMemoryLayoutTrait =
      op->hasTrait<mlir::OpTrait::ChangesMemoryLayout>();
  bool changesTensorLayoutTrait =
      op->hasTrait<mlir::OpTrait::ChangesTensorLayout>();

  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto output = dyn_cast<RankedTensorType>(op->getResult(0).getType());

  TTNNLayoutAttr inputLayout =
      dyn_cast_or_null<TTNNLayoutAttr>(input.getEncoding());
  TTNNLayoutAttr outputLayout =
      dyn_cast_or_null<TTNNLayoutAttr>(output.getEncoding());

  // This should be error in normal case, but there are a lot of tests
  // that don't have the layout attributes set. These tests might run
  // canonicalization pass, or some other specific pass which is needed for
  // testing.
  if (!inputLayout || !outputLayout) {
    return mlir::success();
  }

  if (!changesBufferTypeTrait && changesBufferType(inputLayout, outputLayout)) {
    return op->emitOpError("TTNN op " + op->getName().getStringRef() +
                           " changes buffer type, but does not have the "
                           "ChangesBufferType trait");
  }

  if (!changesMemoryLayoutTrait &&
      changesMemoryLayout(inputLayout, outputLayout)) {
    return op->emitOpError("TTNN op " + op->getName().getStringRef() +
                           " changes memory layout, but does not have the "
                           "ChangesMemoryLayout trait");
  }

  if (!changesTensorLayoutTrait &&
      changesTensorLayout(inputLayout, outputLayout)) {
    return op->emitOpError("TTNN op " + op->getName().getStringRef() +
                           " changes tensor layout, but does not have the "
                           "ChangesTensorLayout trait");
  }

  if (!changesDataTypeTrait &&
      (changesDataType(inputLayout, outputLayout) ||
       input.getElementType() != output.getElementType())) {
    return op->emitOpError("TTNN op " + op->getName().getStringRef() +
                           " changes data type, but does not have the "
                           "ChangesDataType trait");
  }

  return mlir::success();
}
} // namespace mlir::tt::ttnn
