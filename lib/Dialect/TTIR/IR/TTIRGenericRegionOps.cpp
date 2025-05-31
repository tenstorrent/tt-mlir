// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/Support/MathExtras.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

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
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  //
  // Tensor style DMA needs additional refactoring to support the device layout
  // shapes. These skips can be removed once this issue is resolved:
  // https://github.com/tenstorrent/tt-mlir/issues/3389
  //
  bool skipChecksForTensorType = mlir::isa<RankedTensorType>(srcType) ||
                                 mlir::isa<RankedTensorType>(dstType);

  if (!skipChecksForTensorType &&
      srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("operands to DMA must have the same element type");
  }

  if (isSrcRemote() && isDstRemote()) {
    return emitOpError("cannot have both src and dst remote");
  }

  if (isDstRemote() && isMcast()) {
    return emitOpError("cannot mcast to remote dst");
  }

  if (getSrcAffineMap() && !getSrcIndices().empty()) {
    return emitOpError("cannot have both src affine map and indices");
  }

  if (getDstAffineMap() && !getDstIndices().empty()) {
    return emitOpError("cannot have both dst affine map and indices");
  }

  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast start index requires mcast shape");
  }

  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast shape requires mcast start index");
  }

  //
  // Skip below verification steps for tensor style or lowered DMA.
  //
  if (isLowered() || skipChecksForTensorType) {
    return success();
  }

  int64_t srcIndices = getSrcAffineMap() ? getSrcAffineMap()->getNumResults()
                                         : getSrcIndices().size();
  int64_t dstIndices = getDstAffineMap() ? getDstAffineMap()->getNumResults()
                                         : getDstIndices().size();

  if (srcIndices > srcType.getRank()) {
    return emitOpError("invalid number of src indices, expected less than ")
           << srcType.getRank();
  }

  if (dstIndices > dstType.getRank()) {
    return emitOpError("invalid number of dst indices, expected less than ")
           << dstType.getRank();
  }

  if ((srcType.getRank() - srcIndices) != (dstType.getRank() - dstIndices)) {
    return emitOpError("memref operands must have the same post-index rank");
  }

  if (!std::equal(srcType.getShape().begin() + srcIndices,
                  srcType.getShape().end(),
                  dstType.getShape().begin() + dstIndices)) {
    return emitOpError("memref operands must have the same post-index shape");
  }

  if (getSrcAffineMap() && !isSrcRemote()) {
    return emitOpError("if src affine map is provided, src must be remote");
  }

  if (getDstAffineMap() && !isDstRemote()) {
    return emitOpError("if dst affine map is provided, dst must be remote");
  }

  return success();
}

int64_t mlir::tt::ttir::DMAOp::getNumElems() {
  if (getOptNumElems()) {
    return *getOptNumElems();
  }

  ArrayRef<int64_t> txShape =
      getSrcMemRefType().getShape().drop_front(getSrcIndices().size());
  return ttmlir::utils::volume(txShape);
}

size_t mlir::tt::ttir::DMAOp::getSizeBytes() {
  auto elementSizeBytes =
      getElementSizeBytes(getSrcMemRefType().getElementType());
  return getNumElems() * elementSizeBytes;
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

bool mlir::tt::ttir::DMAOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getSrc();
}

bool mlir::tt::ttir::DMAOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getDst();
}

mlir::LogicalResult mlir::tt::ttir::DMAOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options) {
  Value src = nullptr;
  // NOLINTNEXTLINE
  if (isSrcRemote()) {
    auto maybeSrc = mlir::bufferization::getBuffer(rewriter, getSrc(), options);
    if (failed(maybeSrc)) {
      return maybeSrc;
    }
    src = *maybeSrc;
  } else {
    src = getSrc();
  }

  Value dst = nullptr;
  // NOLINTNEXTLINE
  if (isDstRemote()) {
    auto maybeDst = mlir::bufferization::getBuffer(rewriter, getDst(), options);
    if (failed(maybeDst)) {
      return maybeDst;
    }
    dst = *maybeDst;
  } else {
    dst = getDst();
  }

  ::llvm::SmallVector<mlir::Value> invocationStack;
  // NOLINTNEXTLINE
  mlir::bufferization::replaceOpWithNewBufferizedOp<mlir::tt::ttir::DMAOp>(
      rewriter, *this, getResult().getType(), src, getSrcAffineMapAttr(),
      getSrcIndices(), dst, getDstAffineMapAttr(), getDstIndices(),
      getOptNumElemsAttr(), getMcastStartIndex(), getMcastShape());

  return mlir::success();
}

mlir::bufferization::AliasingValueList mlir::tt::ttir::DMAOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType> mlir::tt::ttir::DMAOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    ::llvm::SmallVector<mlir::Value> &) {
  auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  return mlir::cast<tt::MetalLayoutAttr>(rankedTensorType.getEncoding())
      .getBufferType(/*isView=*/true);
}

mlir::OpFoldResult mlir::tt::ttir::IterIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

void mlir::tt::ttir::IterIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "iter" + std::to_string(dim));
}

void mlir::tt::ttir::IterIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

void mlir::tt::ttir::CoreIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "core" + std::to_string(dim));
}

//===----------------------------------------------------------------------===//
// CoreIndexOp
//===----------------------------------------------------------------------===//

mlir::OpFoldResult mlir::tt::ttir::CoreIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

void mlir::tt::ttir::CoreIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}
