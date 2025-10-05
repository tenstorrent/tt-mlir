// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

void AcquireDstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "dst");
}

// Helper to extract element type from ranked tensor or memref
static Type getElemType(Type ty) {
  if (auto rt = dyn_cast<RankedTensorType>(ty)) {
    return rt.getElementType();
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    return mr.getElementType();
  }
  return ty;
}

// Helper function to wrap a value in ToTensorOp (if needed) to conform to
// bufferization API expectations (mirrors D2M)
static mlir::Value wrapValueInTensorCompatibleType(mlir::RewriterBase &rewriter,
                                                   mlir::Value value,
                                                   mlir::Location loc) {
  if (mlir::isa<mlir::RankedTensorType>(value.getType())) {
    return value;
  }
  auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
  if (!shapedType) {
    return value;
  }
  auto tensorType = mlir::RankedTensorType::get(shapedType.getShape(),
                                                shapedType.getElementType());
  return rewriter
      .create<mlir::bufferization::ToTensorOp>(loc, tensorType, value)
      .getResult();
}

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

//===----------------------------------------------------------------------===//
// DMA Operations
//===----------------------------------------------------------------------===//

// DMA base verification and helpers
mlir::LogicalResult DMAOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

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

int64_t DMAOp::getNumElems() {
  if (getOptNumElems()) {
    return *getOptNumElems();
  }
  ArrayRef<int64_t> txShape =
      getSrcMemRefType().getShape().drop_front(getSrcIndices().size());
  return ttmlir::utils::volume(txShape);
}

size_t DMAOp::getSizeBytes() {
  auto elementSizeBytes =
      ttcore::getElementSizeBytes(getSrcMemRefType().getElementType());
  return getNumElems() * elementSizeBytes;
}

void DMAOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

bool DMAOp::isNotConflicting(mlir::OpOperand *, mlir::OpOperand *,
                             const mlir::bufferization::AnalysisState &) {
  // Return true to avoid forcing out of place bufferization.
  return true;
}

// Comprehensive verifiers matching D2M
::mlir::LogicalResult DMAWriteOp::verify() {
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

::mlir::LogicalResult DMAReadOp::verify() {
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

void DMAReadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void DMAWriteOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "tx");
}

void DMAOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                           mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(),
                       0 /*stage*/, true /*effectOnFullRegion*/,
                       mlir::SideEffects::DefaultResource::get());
}

void DMAReadOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
}

void DMAWriteOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
}

bool DMAOp::bufferizesToMemoryRead(mlir::OpOperand &operand,
                                   const mlir::bufferization::AnalysisState &) {
  return operand.get() == getSrc();
}

bool DMAOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getDst();
}

mlir::bufferization::AliasingValueList
DMAOp::getAliasingValues(mlir::OpOperand &,
                         const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
DMAOp::getBufferType(mlir::Value value,
                     const mlir::bufferization::BufferizationOptions &,
                     const mlir::bufferization::BufferizationState &,
                     ::llvm::SmallVector<mlir::Value> &) {
  auto rankedTensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
  return mlir::tt::d2m::getBufferType(rankedTensorType, /*isView=*/true);
}

mlir::LogicalResult
DMAOp::bufferize(mlir::RewriterBase &rewriter,
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
  mlir::bufferization::replaceOpWithNewBufferizedOp<mlir::tt::d2m::DMAOp>(
      rewriter, *this, getResult().getType(), src, getSrcAffineMapAttr(),
      getSrcIndices(), dst, getDstAffineMapAttr(), getDstIndices(),
      getOptNumElemsAttr(), getMcastStartIndex(), getMcastShape());

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Index Operations
//===----------------------------------------------------------------------===//

void IterIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "iter" + std::to_string(dim));
}

void IterIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

void CoreIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "core" + std::to_string(dim));
}

void CoreIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult IterIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

mlir::OpFoldResult CoreIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

// TileMatmulBlockOp verification
::mlir::LogicalResult TileMatmulBlockOp::verify() {
  if (getElemType(getA().getType()) != getElemType(getB().getType())) {
    return emitOpError(
        "operands to TileMatmulBlock must have same element type");
  }

  int numAttrsSet = getBlockM().has_value() + getBlockK().has_value() +
                    getBlockN().has_value() + getBBlockStride().has_value();
  if (numAttrsSet != 0 && numAttrsSet != 4) {
    return emitOpError(
        "all or none of the block dim attributes must be present");
  }

  return success();
}

bool TileMatmulBlockOp::hasBlockDims() {
  return getBlockM() && getBlockK() && getBlockN() && getBBlockStride();
}

void TileMatmulBlockOp::getEffects(
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

mlir::LogicalResult TileTilizeBlockOp::bufferize(
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
  auto newOp =
      rewriter.create<mlir::tt::d2m::TileTilizeBlockOp>(old->getLoc(), in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool TileTilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool TileTilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList TileTilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
TileTilizeBlockOp::getBufferType(
    mlir::Value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult TileTilizeBlockOp::verify() {
  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError(
        "operand to TileTilizeBlock must not have ttcore.tile element type");
  }

  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError(
        "result of TileTilizeBlock must have ttcore.tile element type");
  }

  return success();
}

void TileTilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

mlir::LogicalResult TileUntilizeBlockOp::bufferize(
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
  auto newOp = rewriter.create<mlir::tt::d2m::TileUntilizeBlockOp>(
      old->getLoc(), in, out);
  rewriter.replaceOp(old, newOp->getResults());
  return mlir::success();
}

bool TileUntilizeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getInput();
}

bool TileUntilizeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput();
}

mlir::bufferization::AliasingValueList TileUntilizeBlockOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
TileUntilizeBlockOp::getBufferType(
    mlir::Value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult TileUntilizeBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getInput().getType()))) {
    return emitOpError(
        "operand to TileUntilizeBlock must have ttcore.tile element type");
  }
  if (llvm::isa<mlir::tt::ttcore::TileType>(
          getElemType(getOutput().getType()))) {
    return emitOpError(
        "result of TileUntilizeBlock must not have ttcore.tile element type");
  }
  return success();
}

void TileUntilizeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// YieldOp / AwaitOp
//===----------------------------------------------------------------------===//

static bool valueInRegionArguments(mlir::Value value, mlir::Region *region) {
  return llvm::is_contained(region->getArguments(), value);
}

static mlir::Value recurseThroughMemrefCollapse(mlir::Value value) {
  while (auto memrefCastOp =
             value.getDefiningOp<::mlir::memref::CollapseShapeOp>()) {
    value = memrefCastOp.getOperand();
  }
  return value;
}

static ::mlir::LogicalResult operandsInRegionArguments(mlir::Operation *op,
                                                       mlir::Region *region) {
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    mlir::Value value = recurseThroughMemrefCollapse(operand.get());
    if (!valueInRegionArguments(value, region)) {
      return op->emitOpError() << "operand[" << operand.getOperandNumber()
                               << "] not in region arguments";
    }
  }
  return ::mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  return success();
  auto generic = getOperation()->getParentOfType<GenericOp>();
  if (generic && generic.hasPureTensorSemantics()) {
    return ::mlir::success();
  }

  return operandsInRegionArguments(
      getOperation(),
      ttmlir::utils::getRegionWithParentOfType<GenericOp, func::FuncOp>(
          getOperation()));
}

mlir::LogicalResult AwaitOp::verify() {
  return success();
  auto generic = getOperation()->getParentOfType<GenericOp>();
  if (generic && generic.hasPureTensorSemantics()) {
    return emitOpError(
        "await op illegal to use inside generic with pure tensor semantics");
  }

  return operandsInRegionArguments(
      getOperation(),
      ttmlir::utils::getRegionWithParentOfType<GenericOp, func::FuncOp>(
          getOperation()));
}

mlir::LogicalResult PopOp::verify() {
  // Verify that the circular buffer operand is in the region arguments
  mlir::Value cb = getCb();
  mlir::Region *region =
      ttmlir::utils::getRegionWithParentOfType<GenericOp, func::FuncOp>(
          getOperation());

  if (!valueInRegionArguments(cb, region)) {
    return emitOpError() << "circular buffer operand not in region arguments";
  }

  // Verify that the result type matches the wrapped memref type
  auto cbType = llvm::cast<CBType>(cb.getType());
  if (cbType.getUnderlying() != getResult().getType()) {
    return emitOpError() << "result type does not match circular buffer's "
                            "wrapped memref type";
  }

  return ::mlir::success();
}

mlir::LogicalResult ReserveOp::verify() {
  // Verify that the circular buffer operand is in the region arguments
  mlir::Value cb = getCb();
  mlir::Region *region =
      ttmlir::utils::getRegionWithParentOfType<GenericOp, func::FuncOp>(
          getOperation());

  if (!valueInRegionArguments(cb, region)) {
    return emitOpError() << "circular buffer operand not in region arguments";
  }

  // Verify that the result type matches the wrapped memref type
  auto cbType = llvm::cast<CBType>(cb.getType());
  if (cbType.getUnderlying() != getResult().getType()) {
    return emitOpError() << "result type does not match circular buffer's "
                            "wrapped memref type";
  }

  return ::mlir::success();
}
