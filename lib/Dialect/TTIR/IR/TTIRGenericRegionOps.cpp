// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TileMatmulBlockOp
//===----------------------------------------------------------------------===//

// TileMatmulBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileMatmulBlockOp::verify() {

  if (!llvm::isa<mlir::tt::TileType>(getA().getType().getElementType()) ||
      !llvm::isa<mlir::tt::TileType>(getB().getType().getElementType())) {
    return emitOpError("MemRef operands to TileMatmulBlock must have tt.tile "
                       "element type");
  }

  return success();
}

// TileTilizeBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileTilizeBlockOp::verify() {

  if (llvm::isa<mlir::tt::TileType>(getInput().getType().getElementType())) {
    return emitOpError(
        "MemRef operand to TileTilizeBlock must not have tt.tile "
        "element type");
  }

  if (!llvm::isa<mlir::tt::TileType>(getOutput().getType().getElementType())) {
    return emitOpError("MemRef result of TileTilizeBlock must have tt.tile "
                       "element type");
  }

  return success();
}

// TileUntilizeBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileUntilizeBlockOp::verify() {

  if (!llvm::isa<mlir::tt::TileType>(getInput().getType().getElementType())) {
    return emitOpError("MemRef operand to TileUntilizeBlock must have tt.tile "
                       "element type");
  }

  if (llvm::isa<mlir::tt::TileType>(getOutput().getType().getElementType())) {
    return emitOpError(
        "MemRef result of TileUntilizeBlock must not have tt.tile "
        "element type");
  }

  return success();
}

void mlir::tt::ttir::TileMatmulBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  return mlir::tt::ttir::getDpsEffects(*this, effects);
}

//===----------------------------------------------------------------------===//
// DMAOp
//===----------------------------------------------------------------------===//
::mlir::LogicalResult mlir::tt::ttir::DMAOp::verify() {
  MemRefType srcType = getSrc().getType();
  MemRefType dstType = getDst().getType();
  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError(
        "MemRef operands to DMA must have the same element type");
  }

  if (isSrcRemote() && isDstRemote()) {
    return emitOpError("DMA cannot have both src and dst remote");
  }

  if (getSrcAffineMap() && getSrcIndices().size()) {
    return emitOpError("DMA cannot have both src affine map and indices");
  }

  if (getDstAffineMap() && getDstIndices().size()) {
    return emitOpError("DMA cannot have both dst affine map and indices");
  }

  int64_t srcIndices = getSrcAffineMap() ? getSrcAffineMap()->getNumResults()
                                         : getSrcIndices().size();
  int64_t dstIndices = getDstAffineMap() ? getDstAffineMap()->getNumResults()
                                         : getDstIndices().size();

  if (srcIndices > srcType.getRank()) {
    return emitOpError("Invalid number of src indices, expected less than ")
           << srcType.getRank();
  }

  if (dstIndices > dstType.getRank()) {
    return emitOpError("Invalid number of dst indices, expected less than ")
           << dstType.getRank();
  }

  if ((srcType.getRank() - srcIndices) != (dstType.getRank() - dstIndices)) {
    return emitOpError(
        "MemRef operands to DMA must have the same post-index rank");
  }

  if (!std::equal(srcType.getShape().begin() + srcIndices,
                  srcType.getShape().end(),
                  dstType.getShape().begin() + dstIndices)) {
    return emitOpError(
        "MemRef operands to DMA must have the same post-index shape");
  }

  return success();
}

void mlir::tt::ttir::DMAOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void mlir::tt::ttir::DMAOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
}

void mlir::tt::ttir::IterIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "iter" + std::to_string(dim));
}

void mlir::tt::ttir::CoreIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "core" + std::to_string(dim));
}
