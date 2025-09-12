// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/Support/MathExtras.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

static mlir::Type getElemType(mlir::Type t) {
  auto shaped = mlir::cast<mlir::ShapedType>(t);
  return shaped.getElementType();
};

// Helper function to wrap a value in ToTensorOp (if needed) to conform to
// bufferization API expectations
static mlir::Value wrapValueInTensorCompatibleType(mlir::RewriterBase &rewriter,
                                                   mlir::Value value,
                                                   mlir::Location loc) {
  if (mlir::isa<mlir::RankedTensorType>(value.getType())) {
    return value;
  }

  // We need to convert from memref to tensor, so the result type should be a
  // tensor type and the input should be the memref value
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType) {
    return value; // Return original if not a shaped type
  }

  auto tensorType = mlir::RankedTensorType::get(shapedType.getShape(),
                                                shapedType.getElementType());

  return rewriter
      .create<mlir::bufferization::ToTensorOp>(loc, tensorType, value)
      .getResult();
}

//===----------------------------------------------------------------------===//
// TileMatmulBlockOp
//===----------------------------------------------------------------------===//

// TileMatmulBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileMatmulBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getA().getType())) ||
      !llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getB().getType()))) {
    return emitOpError("operands to TileMatmulBlock must have ttcore.tile "
                       "element type");
  }

  int numAttrsSet = getBlockM().has_value() + getBlockK().has_value() +
                    getBlockN().has_value() + getBBlockStride().has_value();
  if (numAttrsSet != 0 && numAttrsSet != 4) {
    return emitOpError(
        "all or none of the block dim attributes must be present");
  }

  return success();
}

bool mlir::tt::ttir::TileMatmulBlockOp::hasBlockDims() {
  return getBlockM() && getBlockK() && getBlockN() && getBBlockStride();
}

// TileTilizeBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileTilizeBlockOp::verify() {
  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError("operand to TileTilizeBlock must not have ttcore.tile "
                       "element type");
  }

  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError("result of TileTilizeBlock must have ttcore.tile "
                       "element type");
  }

  return success();
}

// TileUntilizeBlockOp verification
::mlir::LogicalResult mlir::tt::ttir::TileUntilizeBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError("operand to TileUntilizeBlock must have ttcore.tile "
                       "element type");
  }

  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError("result of TileUntilizeBlock must not have ttcore.tile "
                       "element type");
  }

  return success();
}

void mlir::tt::ttir::TileMatmulBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getAMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getBMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

void mlir::tt::ttir::AcquireDstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "dst");
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
  if (skipChecksForTensorType) {
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
    return emitOpError("operands must have the same post-index rank");
  }

  if (!std::equal(srcType.getShape().begin() + srcIndices,
                  srcType.getShape().end(),
                  dstType.getShape().begin() + dstIndices)) {
    return emitOpError("operands must have the same post-index shape");
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
      ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
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
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  Value src = nullptr;
  // NOLINTNEXTLINE
  if (isSrcRemote()) {
    auto srcToTensor =
        wrapValueInTensorCompatibleType(rewriter, getSrc(), getLoc());

    auto maybeSrc =
        mlir::bufferization::getBuffer(rewriter, srcToTensor, options, state);
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
    auto dstToTensor =
        wrapValueInTensorCompatibleType(rewriter, getDst(), getLoc());

    auto maybeDst =
        mlir::bufferization::getBuffer(rewriter, dstToTensor, options, state);
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
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  return ttir::getBufferType(rankedTensorType, /*isView=*/true);
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
// DMAWriteOp and DMAReadOp
//===----------------------------------------------------------------------===//

void mlir::tt::ttir::DMAWriteOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void mlir::tt::ttir::DMAWriteOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
}

::mlir::LogicalResult mlir::tt::ttir::DMAWriteOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  auto isLocal = [&](auto operand) {
    Block *block = operand.getParentBlock();
    Block::BlockArgListType blockArgs = block->getArguments();
    return std::find(blockArgs.begin(), blockArgs.end(), operand) !=
           blockArgs.end();
  };
  auto isRemote = [&](auto operand) { return !isLocal(operand); };

  if (isRemote(getSrc())) {
    return emitOpError("For DMAWrite, src must be local");
  }

  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("Operands to DMAWrite must have the same element type");
  }

  if (isDstRemote() && isMcast()) {
    return emitOpError("cannot mcast to remote dst");
  }

  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast shape defined but mcast start index is not");
  }

  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast start index defined but mcast shape is not");
  }

  constexpr int64_t kExpectedIndicesRemote = 3;
  constexpr int64_t kExpectedIndicesLocal = 1;

  int64_t numDstIndices = getDstIndices().size();
  int64_t numSrcIndices = getSrcIndices().size();

  if (isDstRemote()) {
    if (numDstIndices != kExpectedIndicesRemote) {
      return emitOpError("Must have 3 dst indices for remote dst operand");
    }
  } else {
    if (numDstIndices != kExpectedIndicesLocal) {
      return emitOpError("Must have 1 dst index for local dst operand");
    }
  }

  if (numSrcIndices != kExpectedIndicesLocal) {
    return emitOpError("Must have 1 src index for local src operand");
  }

  return success();
}

void mlir::tt::ttir::DMAReadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void mlir::tt::ttir::DMAReadOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
}

::mlir::LogicalResult mlir::tt::ttir::DMAReadOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  auto isLocal = [&](auto operand) {
    Block *block = operand.getParentBlock();
    Block::BlockArgListType blockArgs = block->getArguments();
    return std::find(blockArgs.begin(), blockArgs.end(), operand) !=
           blockArgs.end();
  };
  auto isRemote = [&](auto operand) { return !isLocal(operand); };

  if (!(isRemote(getSrc()) && isLocal(getDst()))) {
    return emitOpError("For DMARead, src must be remote and dst must be local");
  }

  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError("Operands to DMAWrite must have the same element type");
  }

  constexpr int64_t kExpectedIndicesRemote = 3;
  constexpr int64_t kExpectedIndicesLocal = 1;

  int64_t numDstIndices = getDstIndices().size();
  int64_t numSrcIndices = getSrcIndices().size();

  if (numSrcIndices != kExpectedIndicesRemote) {
    return emitOpError("Must have 3 src indices for remote src operand");
  }

  if (numDstIndices != kExpectedIndicesLocal) {
    return emitOpError("Must have 1 dst index for local dst operand");
  }

  return success();
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

void mlir::tt::ttir::TileTilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

void mlir::tt::ttir::TileUntilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult mlir::tt::ttir::TileTilizeBlockOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(getOperation());

  mlir::Value in = getInput();
  mlir::Value out = getOutput();
  if (mlir::isa<mlir::RankedTensorType>(in.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  mlir::Operation *old = getOperation();
  auto newOp = rewriter.create<mlir::tt::ttir::TileTilizeBlockOp>(old->getLoc(),
                                                                  in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool mlir::tt::ttir::TileTilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool mlir::tt::ttir::TileTilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList
mlir::tt::ttir::TileTilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType>
mlir::tt::ttir::TileTilizeBlockOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult mlir::tt::ttir::TileUntilizeBlockOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(getOperation());

  mlir::Value in = getInput();
  mlir::Value out = getOutput();
  if (mlir::isa<mlir::RankedTensorType>(in.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  mlir::Operation *old = getOperation();
  auto newOp = rewriter.create<mlir::tt::ttir::TileUntilizeBlockOp>(
      old->getLoc(), in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool mlir::tt::ttir::TileUntilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool mlir::tt::ttir::TileUntilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList
mlir::tt::ttir::TileUntilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType>
mlir::tt::ttir::TileUntilizeBlockOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}
