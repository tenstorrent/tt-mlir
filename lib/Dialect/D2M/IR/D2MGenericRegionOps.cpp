// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/APInt.h"
#include <limits>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

// Template helper for bufferizing CB ops that take a CB operand but no result
// (PushOp, PopOp). These ops unwrap to_tensor by creating to_buffer.
template <typename OpTy>
static mlir::LogicalResult
bufferizeCBOp(OpTy op, mlir::RewriterBase &rewriter,
              const mlir::bufferization::BufferizationOptions &options) {
  auto cbBufferType =
      mlir::cast<bufferization::TensorLikeType>(op.getCbType())
          .getBufferType(options, [&]() { return op.emitOpError(); });
  assert(succeeded(cbBufferType));
  auto toBuffer = rewriter.create<bufferization::ToBufferOp>(
      op.getLoc(), *cbBufferType, op.getCb());
  mlir::bufferization::replaceOpWithNewBufferizedOp<OpTy>(rewriter, op,
                                                          toBuffer.getResult());
  return mlir::success();
}

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

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// DMA Operations
//===----------------------------------------------------------------------===//

// Comprehensive verifiers matching D2M
::mlir::LogicalResult DMAWriteOp::verify() {
  ShapedType srcType = mlir::cast<ShapedType>(getSrc().getType());
  ShapedType dstType = mlir::cast<ShapedType>(getDst().getType());

  auto isRemote = [&](auto operand) {
    return ttcore::hasDeviceLayout(operand);
  };

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
  auto isRemote = [&](auto operand) {
    return ttcore::hasDeviceLayout(operand);
  };
  auto isLocal = [&](auto operand) { return !isRemote(operand); };
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

//===----------------------------------------------------------------------===//
// Remote Load/Store Operations
//===----------------------------------------------------------------------===//

::mlir::LogicalResult RemoteLoadOp::verify() {
  auto shapedType = getShapedType();
  bool hasCbOperand = static_cast<bool>(getCb());
  bool hasResultValue = static_cast<bool>(getResult());
  Value localBuffer = getLocalBuffer();
  bool hasLocalBuffer = static_cast<bool>(localBuffer);

  // Verify XOR constraint: exactly one of localBuffer or cb must be present
  if (hasCbOperand == hasLocalBuffer) {
    if (hasCbOperand) {
      return emitOpError(
          "cannot have both circular buffer and local buffer; exactly one must "
          "be present");
    }
    return emitOpError(
        "must have either circular buffer or local buffer; exactly one must be "
        "present");
  }

  if (hasCbOperand && hasResultValue) {
    return emitOpError(
        "explicit CB form cannot have a result; result must only be present in "
        "implicit form.");
  }
  if (hasLocalBuffer && !hasResultValue) {
    return emitOpError("implicit form (with localBuffer) must have a result.");
  }

  // Verify that tensor parameters are not allowed in explicit CB form
  if (isExplicitCBForm()) {
    if (mlir::isa<RankedTensorType>(getMemref().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; memref "
          "operand must be a memref type");
    }
  }

  // Verify localBuffer type matches shard shape (only in implicit form)
  if (hasLocalBuffer) {
    auto localBufferType = mlir::cast<ShapedType>(localBuffer.getType());
    auto deviceLayout = ttcore::getDeviceLayout(getMemref());
    if (!deviceLayout) {
      return emitOpError("failed to get device layout from memref/tensor");
    }
    auto shardShape = deviceLayout.getShardShape(shapedType);

    if (localBufferType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError("localBuffer rank must match shard shape rank, got ")
             << localBufferType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != localBufferType.getDimSize(i)) {
        return emitOpError(
                   "localBuffer shape must match shard shape at dimension ")
               << i << ", got " << localBufferType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    if (localBufferType.getElementType() != shapedType.getElementType()) {
      return emitOpError(
          "localBuffer element type must match memref/tensor element type");
    }
  }

  // Verify that the memref/tensor is remote (has device layout)
  if (!ttcore::hasDeviceLayout(getMemref())) {
    return emitOpError("memref/tensor must have a device layout.");
  }

  // Verify memref/tensor rank is even (grid + shard dimensions)
  if (shapedType.getRank() % 2 != 0) {
    return emitOpError("memref/tensor rank must be even for device shape (grid "
                       "+ shard dimensions)");
  }

  // Verify indices count matches grid dimensions (first N/2 dimensions)
  int64_t gridRank = shapedType.getRank() / 2;
  if (static_cast<int64_t>(getIndices().size()) != gridRank) {
    return emitOpError("number of indices must equal grid rank (N/2 where N is "
                       "memref/tensor rank), got ")
           << getIndices().size() << " indices but expected " << gridRank;
  }

  // Verify multicast parameters: both must be provided or neither
  if (!getMcastStartIndex().empty() && getMcastShape().empty()) {
    return emitOpError("mcast start index requires mcast shape");
  }
  if (!getMcastShape().empty() && getMcastStartIndex().empty()) {
    return emitOpError("mcast shape requires mcast start index");
  }

  // Verify mutual exclusivity between low-level and high-level multicast forms
  if (!getMcastShape().empty() && !getMcastDims().empty()) {
    return emitOpError(
        "cannot use both low-level multicast (mcore/mshape) and high-level "
        "multicast (mcast dims); they are mutually exclusive");
  }

  // Verify high-level mcast dimensions are constant indices
  SmallVector<int64_t> mcastDimIndices;
  for (Value dimValue : getMcastDims()) {
    auto constantOp = dimValue.getDefiningOp<arith::ConstantOp>();
    if (!constantOp) {
      return emitOpError("mcast dimension must be a constant index");
    }
    auto indexAttr = mlir::dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!indexAttr) {
      return emitOpError("mcast dimension must be an integer attribute");
    }
    mcastDimIndices.push_back(indexAttr.getInt());
  }

  // Verify that memref references a generic op operand or scratch allocation
  // when inside a generic
  if (auto genericOp = getOperation()->getParentOfType<GenericOp>()) {
    Value memrefOperand = getMemref();
    std::optional<unsigned> operandIndex;
    for (auto [idx, operand] : llvm::enumerate(genericOp.getOperands())) {
      if (operand == memrefOperand) {
        operandIndex = idx;
        break;
      }
    }
    // Also allow scratch allocations
    if (!operandIndex &&
        !isa_and_nonnull<ScratchAllocateOp>(memrefOperand.getDefiningOp())) {
      return emitOpError(
          "memref operand must reference one of the parent generic op's "
          "operands or a scratch allocation");
    }

    // Forbid high-level mcast form in explicit datamovement form
    if (genericOp.isExplicitDatamovementForm() && !getMcastDims().empty()) {
      return emitOpError(
          "high-level multicast form (mcast dims) is not allowed in explicit "
          "datamovement form; use low-level multicast (mcore/mshape) instead");
    }

    // Skip checks that rely on indexing maps and iterator types when in
    // explicit datamovement form
    if (!genericOp.isExplicitDatamovementForm()) {
      // Verify mcast dimensions are parallel iterator type
      if (!mcastDimIndices.empty()) {
        AffineMap indexingMap = genericOp.getIndexingMap(*operandIndex);
        ArrayAttr iteratorTypes = genericOp.getIteratorTypes();

        for (int64_t gridDim : mcastDimIndices) {
          if (gridDim < 0 ||
              gridDim >= static_cast<int64_t>(indexingMap.getNumResults())) {
            return emitOpError("mcast dimension index ")
                   << gridDim << " is out of bounds for grid rank "
                   << indexingMap.getNumResults();
          }

          AffineExpr expr = indexingMap.getResult(gridDim);
          if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
            int64_t iterDimPos = dimExpr.getPosition();
            auto iterType =
                mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[iterDimPos]);
            if (iterType.getValue() != ttcore::IteratorType::Parallel) {
              return emitOpError("mcast dimension index ")
                     << gridDim
                     << " must correspond to a parallel iterator type, but "
                        "found reduction";
            }
          }
        }
      }
    }
  }

  // CB-specific verification (only when CB is present)
  if (hasCbOperand) {
    auto cbType = mlir::cast<CBType>(getCb().getType());
    auto cbUnderlyingType = cbType.getUnderlying();
    auto deviceLayout = ttcore::getDeviceLayout(getMemref());
    if (!deviceLayout) {
      return emitOpError("failed to get device layout from memref/tensor");
    }
    auto shardShape = deviceLayout.getShardShape(shapedType);

    // Verify CB underlying shape matches shard shape
    if (cbUnderlyingType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError(
                 "circular buffer underlying rank must match shard shape "
                 "rank, got ")
             << cbUnderlyingType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != cbUnderlyingType.getDimSize(i)) {
        return emitOpError("circular buffer underlying shape must match "
                           "shard shape at dimension ")
               << i << ", got " << cbUnderlyingType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    if (cbUnderlyingType.getElementType() != shapedType.getElementType()) {
      return emitOpError(
          "circular buffer element type must match memref/tensor element type");
    }
  }

  return mlir::success();
}

void WriteRowMaskTileOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

void WriteColMaskTileOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

::mlir::LogicalResult RemoteStoreOp::verify() {
  auto shapedType = getShapedType();
  bool hasCbOperand = static_cast<bool>(getCb());
  bool hasLocalBufferOperand = static_cast<bool>(getLocalBuffer());
  bool hasResultValue = static_cast<bool>(getResult());

  // Verify XOR constraint: exactly one of localBuffer or cb must be present
  if (hasCbOperand == hasLocalBufferOperand) {
    if (hasCbOperand) {
      return emitOpError(
          "cannot have both circular buffer and local buffer; exactly one must "
          "be present");
    }
    return emitOpError(
        "must have either circular buffer or local buffer; exactly one must be "
        "present");
  }

  // Verify result presence constraints (mirror RemoteLoadOp)
  // Explicit CB form: CB present, localBuffer absent, result must NOT be
  // present Implicit form: localBuffer present, CB absent, result MUST be
  // present
  if (hasCbOperand && hasResultValue) {
    return emitOpError(
        "explicit CB form cannot have a result; result must only be present in "
        "implicit form.");
  }
  if (hasLocalBufferOperand && !hasResultValue) {
    return emitOpError("implicit form (with localBuffer) must have a result.");
  }

  // Verify that tensor parameters are not allowed in explicit CB form
  if (isExplicitCBForm()) {
    if (mlir::isa<RankedTensorType>(getMemref().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; memref "
          "operand must be a memref type");
    }
    if (hasLocalBufferOperand &&
        mlir::isa<RankedTensorType>(getLocalBuffer().getType())) {
      return emitOpError(
          "tensor parameters are not allowed in explicit CB form; localBuffer "
          "operand must be a memref type");
    }
  }

  // Verify that the memref/tensor is remote (has device layout)
  if (!ttcore::hasDeviceLayout(getMemref())) {
    return emitOpError("memref/tensor must be remote (have a device layout)");
  }

  // Verify memref/tensor rank is even (grid + shard dimensions)
  if (shapedType.getRank() % 2 != 0) {
    return emitOpError("memref/tensor rank must be even for device shape (grid "
                       "+ shard dimensions)");
  }

  // Verify indices count matches grid dimensions (first N/2 dimensions)
  int64_t gridRank = shapedType.getRank() / 2;
  if (static_cast<int64_t>(getIndices().size()) != gridRank) {
    return emitOpError("number of indices must equal grid rank (N/2 where N is "
                       "memref/tensor rank), got ")
           << getIndices().size() << " indices but expected " << gridRank;
  }

  // Verify that memref references a generic op operand or scratch allocation
  // when inside a generic
  if (auto genericOp = getOperation()->getParentOfType<GenericOp>()) {
    Value memrefOperand = getMemref();
    bool foundInOperands = false;
    for (Value operand : genericOp.getOperands()) {
      if (operand == memrefOperand) {
        foundInOperands = true;
        break;
      }
    }
    // Also allow scratch allocations
    if (!foundInOperands &&
        !isa_and_nonnull<ScratchAllocateOp>(memrefOperand.getDefiningOp())) {
      return emitOpError(
          "memref operand must reference one of the parent generic op's "
          "operands or a scratch allocation");
    }
  }

  // Get device layout for shape verification
  auto deviceLayout = ttcore::getDeviceLayout(getMemref());
  if (!deviceLayout) {
    return emitOpError("failed to get device layout from memref/tensor");
  }
  auto shardShape = deviceLayout.getShardShape(shapedType);

  // CB-specific verification (only when CB is present)
  if (hasCbOperand) {
    auto cbType = mlir::cast<CBType>(getCb().getType());
    auto cbUnderlyingType = cbType.getUnderlying();

    // Verify CB underlying shape matches shard shape
    if (cbUnderlyingType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError(
                 "circular buffer underlying rank must match shard shape "
                 "rank, got ")
             << cbUnderlyingType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != mlir::ShapedType::kDynamic &&
          cbUnderlyingType.getDimSize(i) != mlir::ShapedType::kDynamic &&
          shardShape[i] != cbUnderlyingType.getDimSize(i)) {
        return emitOpError("circular buffer underlying shape must match shard "
                           "shape at dimension ")
               << i << ", got " << cbUnderlyingType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    Type cbElementType = cbUnderlyingType.getElementType();
    Type shapedElementType = shapedType.getElementType();
    if (cbElementType != shapedElementType) {
      return emitOpError(
          "circular buffer element type must match memref/tensor element type");
    }
  }

  // Local buffer-specific verification (only when localBuffer is present)
  if (hasLocalBufferOperand) {
    auto localBufferType = mlir::cast<ShapedType>(getLocalBuffer().getType());

    // Verify local buffer shape matches shard shape
    if (localBufferType.getRank() != static_cast<int64_t>(shardShape.size())) {
      return emitOpError("local buffer rank must match shard shape rank, got ")
             << localBufferType.getRank() << " but expected "
             << shardShape.size();
    }

    for (size_t i = 0; i < shardShape.size(); ++i) {
      if (shardShape[i] != mlir::ShapedType::kDynamic &&
          localBufferType.getDimSize(i) != mlir::ShapedType::kDynamic &&
          shardShape[i] != localBufferType.getDimSize(i)) {
        return emitOpError(
                   "local buffer shape must match shard shape at dimension ")
               << i << ", got " << localBufferType.getDimSize(i)
               << " but expected " << shardShape[i];
      }
    }

    // Verify element types match
    Type localBufferElementType = localBufferType.getElementType();
    Type shapedElementType = shapedType.getElementType();
    if (localBufferElementType != shapedElementType) {
      return emitOpError(
          "local buffer element type must match memref/tensor element type");
    }
  }

  // Verify result type matches memref type if result is present and in tensor
  // mode
  if (hasResultValue) {
    Type memrefType = getMemref().getType();

    // Only verify result type in tensor mode
    if (mlir::isa<RankedTensorType>(memrefType)) {
      Type resultType = getResult().getType();
      // Result should match the destination memref type
      if (resultType != memrefType) {
        return emitOpError("result type must match memref type");
      }
    } else {
      // In memref mode, result should be unused
      if (!getResult().use_empty()) {
        return emitOpError(
            "result must be unused when memref operand has memref type");
      }
    }
  }

  return mlir::success();
}

void RemoteLoadOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Load operations don't have results anymore (they load into CB)
}

//===----------------------------------------------------------------------===//
// RemoteLoadOp Custom Assembly Format
//===----------------------------------------------------------------------===//

ParseResult RemoteLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  // Try to parse localBuffer (optional, only in implicit form)
  // Format: explicit CB form: %memref[...] into %cb
  //         implicit form: %localBuffer %memref[...]
  OpAsmParser::UnresolvedOperand localBuffer;
  OpAsmParser::UnresolvedOperand memref;
  bool hasLocalBuffer = false;

  // Parse first operand - could be localBuffer or memref
  OpAsmParser::UnresolvedOperand firstOperand;
  if (parser.parseOperand(firstOperand)) {
    return failure();
  }

  // Try to parse another operand - if it succeeds, firstOperand was localBuffer
  // If it fails (next token is '['), firstOperand was memref
  OpAsmParser::UnresolvedOperand secondOperand;
  OptionalParseResult secondOperandResult =
      parser.parseOptionalOperand(secondOperand);

  if (secondOperandResult.has_value() && succeeded(*secondOperandResult)) {
    // Second operand parsed successfully, so first was localBuffer, second is
    // memref
    localBuffer = firstOperand;
    memref = secondOperand;
    hasLocalBuffer = true;
  } else {
    // Second operand parse failed (next token is '['), so first was memref
    memref = firstOperand;
    hasLocalBuffer = false;
  }

  // Parse indices
  if (parser.parseLSquare()) {
    return failure();
  }
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  if (parser.parseOperandList(indices)) {
    return failure();
  }
  if (parser.parseRSquare()) {
    return failure();
  }

  // Parse optional "into" keyword and CB
  OpAsmParser::UnresolvedOperand cb;
  bool hasCb = succeeded(parser.parseOptionalKeyword("into")) &&
               succeeded(parser.parseOperand(cb));

  // Parse optional multicast parameters
  SmallVector<OpAsmParser::UnresolvedOperand> mcastStartIndex;
  SmallVector<OpAsmParser::UnresolvedOperand> mcastShape;
  bool hasLowLevelMcast = false;
  if (succeeded(parser.parseOptionalKeyword("mcore"))) {
    hasLowLevelMcast = true;
    if (parser.parseLSquare() || parser.parseOperandList(mcastStartIndex) ||
        parser.parseRSquare() || parser.parseKeyword("mshape") ||
        parser.parseLSquare() || parser.parseOperandList(mcastShape) ||
        parser.parseRSquare()) {
      return failure();
    }
  }

  SmallVector<OpAsmParser::UnresolvedOperand> mcastDims;
  bool hasHighLevelMcast = succeeded(parser.parseOptionalKeyword("mcast")) &&
                           succeeded(parser.parseLSquare()) &&
                           succeeded(parser.parseOperandList(mcastDims)) &&
                           succeeded(parser.parseRSquare());

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Parse types
  SmallVector<Type> types;
  if (parser.parseColon()) {
    return failure();
  }

  // Parse localBuffer type if present
  if (hasLocalBuffer) {
    Type localBufferType;
    if (parser.parseType(localBufferType)) {
      return failure();
    }
    types.push_back(localBufferType);
    if (parser.parseComma()) {
      return failure();
    }
  }

  // Parse memref type
  Type memrefType;
  if (parser.parseType(memrefType)) {
    return failure();
  }
  types.push_back(memrefType);

  // Parse CB type if present
  if (hasCb) {
    if (parser.parseKeyword("into")) {
      return failure();
    }
    Type cbType;
    if (parser.parseType(cbType)) {
      return failure();
    }
    types.push_back(cbType);
  }

  // Parse result type if present
  Type resultType;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultType)) {
      return failure();
    }
    result.addTypes(resultType);
  }

  // Resolve operands
  if (hasLocalBuffer) {
    if (parser.resolveOperand(localBuffer, types[0], result.operands)) {
      return failure();
    }
    if (parser.resolveOperand(memref, types[1], result.operands)) {
      return failure();
    }
  } else {
    if (parser.resolveOperand(memref, types[0], result.operands)) {
      return failure();
    }
  }

  if (parser.resolveOperands(indices, parser.getBuilder().getIndexType(),
                             result.operands)) {
    return failure();
  }

  if (hasCb) {
    unsigned cbTypeIdx = hasLocalBuffer ? 2 : 1;
    if (parser.resolveOperand(cb, types[cbTypeIdx], result.operands)) {
      return failure();
    }
  }

  if (hasLowLevelMcast) {
    if (parser.resolveOperands(mcastStartIndex,
                               parser.getBuilder().getIndexType(),
                               result.operands) ||
        parser.resolveOperands(mcastShape, parser.getBuilder().getIndexType(),
                               result.operands)) {
      return failure();
    }
  }

  if (hasHighLevelMcast) {
    if (parser.resolveOperands(mcastDims, parser.getBuilder().getIndexType(),
                               result.operands)) {
      return failure();
    }
  }

  // Set operandSegmentSizes attribute for AttrSizedOperandSegments
  // Segments: [localBuffer, memref, indices, cb, mcastStartIndex, mcastShape,
  // mcastDims]
  SmallVector<int32_t> segmentSizes = {
      hasLocalBuffer ? 1 : 0,                       // localBuffer
      1,                                            // memref
      static_cast<int32_t>(indices.size()),         // indices
      hasCb ? 1 : 0,                                // cb
      static_cast<int32_t>(mcastStartIndex.size()), // mcastStartIndex
      static_cast<int32_t>(mcastShape.size()),      // mcastShape
      static_cast<int32_t>(mcastDims.size())        // mcastDims
  };
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));

  return success();
}

void RemoteLoadOp::print(OpAsmPrinter &p) {
  // Print localBuffer if present (implicit form)
  if (Value localBuffer = getLocalBuffer()) {
    p << " ";
    p.printOperand(localBuffer);
  }

  // Print memref and indices
  p << " ";
  p.printOperand(getMemref());
  p << "[";
  p.printOperands(getIndices());
  p << "]";

  // Print CB if present (explicit CB form)
  if (Value cb = getCb()) {
    p << " into ";
    p.printOperand(cb);
  }

  // Print multicast parameters
  if (!getMcastStartIndex().empty()) {
    p << " mcore[";
    p.printOperands(getMcastStartIndex());
    p << "] mshape[";
    p.printOperands(getMcastShape());
    p << "]";
  }

  if (!getMcastDims().empty()) {
    p << " mcast[";
    p.printOperands(getMcastDims());
    p << "]";
  }

  // Print attributes (excluding operandSegmentSizes which is an internal
  // attribute)
  llvm::StringRef elidedAttrs[] = {"operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print types
  p << " : ";
  if (Value localBuffer = getLocalBuffer()) {
    p.printType(localBuffer.getType());
    p << ", ";
  }
  p.printType(getMemref().getType());
  if (Value cb = getCb()) {
    p << " into ";
    p.printType(cb.getType());
  }
  if (Value result = getResult()) {
    p << " -> ";
    p.printType(result.getType());
  }
}

//===----------------------------------------------------------------------===//
// RemoteLoadOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool RemoteLoadOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Only result-form (no CB) operations should exist during bufferization
  // The memref operand is read from
  return operand.get() == getMemref();
}

//===----------------------------------------------------------------------===//
// FillArangeTileOp Implementation
//===----------------------------------------------------------------------===//

void FillArangeTileOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ArangeBlockOp Implementation
//===----------------------------------------------------------------------===//

mlir::LogicalResult ArangeBlockOp::verify() {
  // Output and index_tile_tensor must have the same element type category
  // (both tensor or both memref).
  Type outputType = getOutput().getType();
  Type indexType = getIndexTileTensor().getType();

  bool outputIsTensor = mlir::isa<mlir::RankedTensorType>(outputType);
  bool indexIsTensor = mlir::isa<mlir::RankedTensorType>(indexType);

  if (outputIsTensor != indexIsTensor) {
    return emitOpError(
        "output and index_tile_tensor must both be tensors or both be memrefs");
  }
  return mlir::success();
}

void ArangeBlockOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  // Read and write index tile tensor (written by
  // WriteFullLinearIndexTileOp, then read for tile arithmetic).
  effects.emplace_back(mlir::MemoryEffects::Read::get(),
                       &getIndexTileTensorMutable(), 0, true,
                       mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getIndexTileTensorMutable(), 0, true,
                       mlir::SideEffects::DefaultResource::get());
  // Write to the output tensor.
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
}

bool ArangeBlockOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getIndexTileTensor();
}

bool ArangeBlockOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.get() == getOutput() || operand.get() == getIndexTileTensor();
}

mlir::bufferization::AliasingValueList
ArangeBlockOp::getAliasingValues(mlir::OpOperand &operand,
                                 const mlir::bufferization::AnalysisState &) {
  // The result aliases the output operand (DPS style).
  if (operand.get() == getOutput()) {
    return {{getResult(), mlir::bufferization::BufferRelation::Equivalent,
             /*isDefinite=*/true}};
  }
  return {};
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
ArangeBlockOp::getBufferType(mlir::Value value,
                             const mlir::bufferization::BufferizationOptions &,
                             const mlir::bufferization::BufferizationState &,
                             ::llvm::SmallVector<mlir::Value> &) {
  // The result type is derived from the output tensor's type.
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (!tensorType) {
    // Already a memref.
    return mlir::bufferization::BufferLikeType(
        mlir::cast<mlir::MemRefType>(getOutput().getType()));
  }
  auto memrefType =
      mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType);
  return mlir::bufferization::BufferLikeType(memrefType);
}

// NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
mlir::LogicalResult ArangeBlockOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // Skip if already bufferized.
  if (!mlir::isa<mlir::RankedTensorType>(getOutput().getType())) {
    return mlir::failure();
  }

  // Get bufferized versions of the operands.
  auto maybeOutputBuffer =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(maybeOutputBuffer)) {
    return maybeOutputBuffer;
  }

  auto maybeIndexTileBuffer = mlir::bufferization::getBuffer(
      rewriter, getIndexTileTensor(), options, state);
  if (failed(maybeIndexTileBuffer)) {
    return maybeIndexTileBuffer;
  }

  // Create new op with memref operands.
  auto newOp = rewriter.create<ArangeBlockOp>(
      getLoc(), *maybeIndexTileBuffer, *maybeOutputBuffer, getNumElements(),
      getStart(), getStep());

  // Replace uses and erase (DPS pattern - result aliases output buffer).
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, getOperation(),
                                                     newOp.getResult());
  return mlir::success();
}
// NOLINTEND(clang-analyzer-core.StackAddressEscape)

bool RemoteLoadOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Only result-form (no CB) operations should exist during bufferization
  // The localBuffer operand is written to (only present in implicit form)
  Value localBuffer = getLocalBuffer();
  return localBuffer && operand.get() == localBuffer;
}

mlir::bufferization::AliasingValueList
RemoteLoadOp::getAliasingValues(mlir::OpOperand &operand,
                                const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList aliasList;
  // Result aliases localBuffer since we load into it in-place.
  // This is necessary so that downstream ops (like RemoteStoreOp) that use
  // this op's result can correctly resolve to the original buffer allocation
  // via getBuffer().
  Value localBuffer = getLocalBuffer();
  Value resultValue = getResult();
  if (localBuffer && resultValue && operand.get() == localBuffer) {
    aliasList.addAlias(
        {resultValue, mlir::bufferization::BufferRelation::Equivalent});
  }
  return aliasList;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
RemoteLoadOp::getBufferType(mlir::Value value,
                            const mlir::bufferization::BufferizationOptions &,
                            const mlir::bufferization::BufferizationState &,
                            ::llvm::SmallVector<mlir::Value> &) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return mlir::failure();
  }
  Value localBuffer = getLocalBuffer();
  if (value == getMemref() || (localBuffer && value == localBuffer)) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  return mlir::failure();
}

mlir::LogicalResult RemoteLoadOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return emitOpError(
        "RemoteLoadOp with CB should not exist during bufferization");
  }

  // Result-only mode: no CB, just the result
  Value result = getResult();
  if (!result) {
    return emitOpError("Expected result when CB is not present");
  }

  // Bufferize the memref/tensor operand
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  mlir::FailureOr<Value> memrefBuffer =
      mlir::bufferization::getBuffer(rewriter, getMemref(), options, state);
  if (failed(memrefBuffer)) {
    return memrefBuffer;
  }

  // Bufferize the localBuffer operand
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  mlir::FailureOr<Value> localBufferBuffer = mlir::bufferization::getBuffer(
      rewriter, getLocalBuffer(), options, state);
  if (failed(localBufferBuffer)) {
    return localBufferBuffer;
  }

  // Convert result type to memref type
  Type resultBufferType =
      ttcore::getBufferType(result.getType(), /*isView=*/false);

  // RemoteLoadOp always loads into L1 memory, so ensure the result type has
  // L1 memory space. If the result type is a memref without memory space,
  // add the L1 memory space attribute.
  if (auto memrefType = mlir::dyn_cast<MemRefType>(resultBufferType)) {
    if (!memrefType.getMemorySpace()) {
      auto l1Attr = ttcore::MemorySpaceAttr::get(getContext(),
                                                 ttcore::MemorySpace::DeviceL1);
      resultBufferType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          memrefType.getLayout(), l1Attr);
    }
  }

  // Create a new RemoteLoadOp with bufferized operands (no CB, with result)
  // Preserve the multicast form - either high-level (mcastDims) or low-level
  // (mcastStartIndex/mcastShape)
  RemoteLoadOp newOp;
  if (isHighLevelMcast()) {
    // High-level mcast form: use mcastDims builder
    newOp = rewriter.create<RemoteLoadOp>(getLoc(), resultBufferType,
                                          *localBufferBuffer, *memrefBuffer,
                                          getIndices(), getMcastDims());
  } else {
    // Low-level mcast form or no mcast: use mcastStartIndex/mcastShape builder
    newOp = rewriter.create<RemoteLoadOp>(
        getLoc(), resultBufferType, *localBufferBuffer, *memrefBuffer,
        getIndices(), getMcastStartIndex(), getMcastShape());
  }

  // Create a ToTensorOp wrapper to maintain tensor semantics for downstream
  // ops. This ensures that operations like linalg.generic still see tensors
  // until they are bufferized. When they call getBuffer() during bufferization,
  // they'll get the underlying memref (*localBufferBuffer).
  auto toTensor = rewriter.create<bufferization::ToTensorOp>(
      getLoc(), result.getType(), *localBufferBuffer);
  rewriter.replaceAllUsesWith(result, toTensor.getResult());
  rewriter.eraseOp(*this);

  return mlir::success();
}

bool RemoteLoadOp::hasTensorSemantics() {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return false;
  }

  // Check if the memref operand is a tensor (needs bufferization)
  bool memrefIsTensor = mlir::isa<RankedTensorType>(getMemref().getType());
  // Check if the result is a tensor (needs bufferization)
  Value result = getResult();
  bool resultIsTensor = result && mlir::isa<RankedTensorType>(result.getType());
  return memrefIsTensor || resultIsTensor;
}

//===----------------------------------------------------------------------===//
// RemoteStoreOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool RemoteStoreOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Only localBuffer-form (no CB) operations should exist during bufferization
  // The localBuffer operand is read from
  return operand.get() == getLocalBuffer();
}

bool RemoteStoreOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // The memref operand is written to
  return operand.get() == getMemref();
}

mlir::bufferization::AliasingValueList
RemoteStoreOp::getAliasingValues(mlir::OpOperand &operand,
                                 const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList aliasList;
  // Result aliases memref operand since it represents the destination after
  // the store operation.
  Value result = getResult();
  if (result && operand.get() == getMemref()) {
    aliasList.addAlias(
        {result, mlir::bufferization::BufferRelation::Equivalent});
  }
  return aliasList;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
RemoteStoreOp::getBufferType(mlir::Value value,
                             const mlir::bufferization::BufferizationOptions &,
                             const mlir::bufferization::BufferizationState &,
                             ::llvm::SmallVector<mlir::Value> &) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return mlir::failure();
  }

  Value localBuffer = getLocalBuffer();
  if (!localBuffer) {
    return mlir::failure();
  }

  Value result = getResult();
  if (result && value == result) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }

  if (value == localBuffer) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  if (value == getMemref()) {
    return ttcore::getBufferType(value.getType(), /*isView=*/false);
  }
  return mlir::failure();
}

// NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
mlir::LogicalResult RemoteStoreOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return emitOpError(
        "RemoteStoreOp with CB should not exist during bufferization");
  }

  // Implicit form: localBuffer mode
  Value localBuffer = getLocalBuffer();
  if (!localBuffer) {
    return emitOpError("Expected localBuffer when CB is not present");
  }

  // Bufferize the memref/tensor operand
  mlir::FailureOr<Value> memrefBuffer =
      mlir::bufferization::getBuffer(rewriter, getMemref(), options, state);
  if (failed(memrefBuffer)) {
    return memrefBuffer;
  }

  // Bufferize the localBuffer operand (only if it's a tensor)
  Value localBufferBufferized = localBuffer;
  if (mlir::isa<RankedTensorType>(localBuffer.getType())) {
    mlir::FailureOr<Value> localBufferMaybe =
        mlir::bufferization::getBuffer(rewriter, localBuffer, options, state);
    if (failed(localBufferMaybe)) {
      return localBufferMaybe;
    }
    localBufferBufferized = *localBufferMaybe;
  }

  // Convert result type to memref type
  // In implicit form (localBuffer present), result is always required
  Value result = getResult();
  if (!result) {
    return emitOpError("Expected result in implicit form during bufferization");
  }

  Type resultBufferType =
      ttcore::getBufferType(result.getType(), /*isView=*/false);

  // Create a new RemoteStoreOp with bufferized operands and result
  mlir::bufferization::replaceOpWithNewBufferizedOp<RemoteStoreOp>(
      rewriter, *this, resultBufferType, *memrefBuffer, getIndices(),
      localBufferBufferized, /*cb=*/Value{});

  return mlir::success();
}
// NOLINTEND(clang-analyzer-core.StackAddressEscape)

bool RemoteStoreOp::hasTensorSemantics() {
  // CB-form operations should not exist during bufferization
  if (getCb()) {
    return false;
  }

  // Check if the memref operand is a tensor (needs bufferization)
  bool memrefIsTensor = mlir::isa<RankedTensorType>(getMemref().getType());
  // Check if the localBuffer is a tensor (needs bufferization)
  Value localBuffer = getLocalBuffer();
  bool localBufferIsTensor =
      localBuffer && mlir::isa<RankedTensorType>(localBuffer.getType());
  // Check if the result is a tensor (needs bufferization)
  Value result = getResult();
  bool resultIsTensor = result && mlir::isa<RankedTensorType>(result.getType());
  return memrefIsTensor || localBufferIsTensor || resultIsTensor;
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

mlir::OpFoldResult IterIndexOp::fold(FoldAdaptor adaptor) {
  // Cannot fold: runtime value depends on loop iteration, not the dimension
  // index itself.
  return {};
}

//===----------------------------------------------------------------------===//
// BlockIndexOp
//===----------------------------------------------------------------------===//

void BlockIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "block" + std::to_string(dim));
}

void BlockIndexOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult BlockIndexOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

//===----------------------------------------------------------------------===//
// BlockOffsetOp
//===----------------------------------------------------------------------===//

void BlockOffsetOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "block_offset" + std::to_string(dim));
}

void BlockOffsetOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult BlockOffsetOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
}

//===----------------------------------------------------------------------===//
// GetBlockFactorOp
//===----------------------------------------------------------------------===//

void GetBlockFactorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  int64_t dim = getDim();
  setNameFn(getResult(), "block_factor" + std::to_string(dim));
}

void GetBlockFactorOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

mlir::OpFoldResult GetBlockFactorOp::fold(FoldAdaptor adaptor) {
  return adaptor.getDimAttr();
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

mlir::OpFoldResult CoreIndexOp::fold(FoldAdaptor adaptor) {
  // Only fold to the constant `dim` when no virtualization map is present.
  // If a map is present, the result depends on runtime core coordinates and
  // must not be folded.
  if (adaptor.getPhysToVirtMapAttr()) {
    return {};
  }
  return adaptor.getDimAttr();
}

// TileMatmulBlockOp verification
::mlir::LogicalResult TileMatmulBlockOp::verify() {
  if (!llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getA().getType())) ||
      !llvm::isa<mlir::tt::ttcore::TileType>(getElemType(getB().getType()))) {
    return emitOpError("operands to TileMatmulBlock must have ttcore.tile "
                       "element type");
  }

  return success();
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
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  rewriter.create<mlir::tt::d2m::TileTilizeBlockOp>(getLoc(), out.getType(), in,
                                                    out);
  // DPS-style op: replace uses of result with the output buffer, not the new
  // op's result. This ensures downstream ops correctly use the original buffer
  // allocation.
  rewriter.replaceAllUsesWith(getResult(), out);
  rewriter.eraseOp(*this);
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
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList aliasList;
  // Result aliases output operand since this is a DPS-style op that writes
  // in-place to the output buffer.
  if (operand.get() == getOutput()) {
    aliasList.addAlias(
        {getResult(), mlir::bufferization::BufferRelation::Equivalent});
  }
  return aliasList;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
TileTilizeBlockOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
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
        "output of TileTilizeBlock must have ttcore.tile element type");
  }

  // Verify result type matches output type (DPS style)
  if (getResult().getType() != getOutput().getType()) {
    return emitOpError("result type must match output parameter type");
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
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
    auto maybe = mlir::bufferization::getBuffer(rewriter, in, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    in = *maybe;
  }
  if (mlir::isa<mlir::RankedTensorType>(out.getType())) {
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
    auto maybe = mlir::bufferization::getBuffer(rewriter, out, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    out = *maybe;
  }

  rewriter.create<mlir::tt::d2m::TileUntilizeBlockOp>(getLoc(), out.getType(),
                                                      in, out);
  // DPS-style op: replace uses of result with the output buffer, not the new
  // op's result. This ensures downstream ops correctly use the original buffer
  // allocation.
  rewriter.replaceAllUsesWith(getResult(), out);
  rewriter.eraseOp(*this);
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
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList aliasList;
  // Result aliases output operand since this is a DPS-style op that writes
  // in-place to the output buffer.
  if (operand.get() == getOutput()) {
    aliasList.addAlias(
        {getResult(), mlir::bufferization::BufferRelation::Equivalent});
  }
  return aliasList;
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
        "output of TileUntilizeBlock must not have ttcore.tile element type");
  }

  // Verify result type matches output type (DPS style)
  if (getResult().getType() != getOutput().getType()) {
    return emitOpError("result type must match output parameter type");
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

template <typename Pred>
static mlir::OpFoldResult foldScalarIdentity(mlir::Operation *op,
                                             mlir::Attribute rhsAttr,
                                             Pred isIdentity) {
  if (!rhsAttr) {
    return nullptr;
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(rhsAttr)) {
    return isIdentity(floatAttr.getValue()) ? op->getOperand(0) : nullptr;
  }
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(rhsAttr)) {
    return isIdentity(intAttr.getValue()) ? op->getOperand(0) : nullptr;
  }
  return nullptr;
}

mlir::OpFoldResult TileAddOp::fold(FoldAdaptor adaptor) {
  return foldScalarIdentity(getOperation(), adaptor.getRhs(),
                            [](auto v) { return v.isZero(); });
}

mlir::OpFoldResult TileSubOp::fold(FoldAdaptor adaptor) {
  return foldScalarIdentity(getOperation(), adaptor.getRhs(),
                            [](auto v) { return v.isZero(); });
}

mlir::OpFoldResult TileMulOp::fold(FoldAdaptor adaptor) {
  return foldScalarIdentity(getOperation(), adaptor.getRhs(), [](auto v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, mlir::APFloat>) {
      return v.isExactlyValue(1.0);
    } else {
      return v.isOne();
    }
  });
}

mlir::OpFoldResult TileDivOp::fold(FoldAdaptor adaptor) {
  return foldScalarIdentity(getOperation(), adaptor.getRhs(), [](auto v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, mlir::APInt>) {
      return v.isOne();
    } else {
      return v.isExactlyValue(1.0);
    }
  });
}

mlir::OpFoldResult TilePowOp::fold(FoldAdaptor adaptor) {
  return foldScalarIdentity(getOperation(), adaptor.getRhs(), [](auto v) {
    using T = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<T, mlir::APInt>) {
      return v.isOne();
    } else {
      return v.isExactlyValue(1.0);
    }
  });
}

//===----------------------------------------------------------------------===//
// BlockMaskOp
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
mlir::LogicalResult
BlockMaskOp::bufferize(mlir::RewriterBase &rewriter,
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

  // Bufferize mask tensors if present.
  mlir::Value rowMaskCb = getRowMaskCb();
  if (rowMaskCb && mlir::isa<mlir::RankedTensorType>(rowMaskCb.getType())) {
    auto maybe =
        mlir::bufferization::getBuffer(rewriter, rowMaskCb, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    rowMaskCb = *maybe;
  }

  mlir::Value colMaskCb = getColMaskCb();
  if (colMaskCb && mlir::isa<mlir::RankedTensorType>(colMaskCb.getType())) {
    auto maybe =
        mlir::bufferization::getBuffer(rewriter, colMaskCb, options, state);
    if (failed(maybe)) {
      return maybe;
    }
    colMaskCb = *maybe;
  }

  rewriter.create<mlir::tt::d2m::BlockMaskOp>(
      getLoc(), out.getType(), in, out, rowMaskCb, colMaskCb, getLogicalRows(),
      getLogicalCols(), getFillValue());
  rewriter.replaceAllUsesWith(getResult(), out);
  rewriter.eraseOp(*this);
  return mlir::success();
}
// NOLINTEND(clang-analyzer-core.StackAddressEscape)

bool BlockMaskOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // Input and mask CBs (if present) are read.
  return operand.get() == getInput() || operand.get() == getRowMaskCb() ||
         operand.get() == getColMaskCb();
}

bool BlockMaskOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // We technically write to the scratch CBs as well as output.
  return operand.get() == getOutput() || operand.get() == getRowMaskCb() ||
         operand.get() == getColMaskCb();
}

mlir::bufferization::AliasingValueList
BlockMaskOp::getAliasingValues(mlir::OpOperand &operand,
                               const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList aliasList;
  // Result aliases output operand since this is a DPS-style op that writes
  // in-place to the output buffer.
  if (operand.get() == getOutput()) {
    aliasList.addAlias(
        {getResult(), mlir::bufferization::BufferRelation::Equivalent});
  }
  return aliasList;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
BlockMaskOp::getBufferType(mlir::Value,
                           const mlir::bufferization::BufferizationOptions &,
                           const mlir::bufferization::BufferizationState &,
                           ::llvm::SmallVector<mlir::Value> &) {
  assert(false && "should already have bufferized types via parent generic op "
                  "bufferization");
  return mlir::failure();
}

mlir::LogicalResult BlockMaskOp::verify() {
  // Verify input and output have compatible types.
  auto inType = getInput().getType();
  auto outType = getOutput().getType();

  auto inShapedType = mlir::dyn_cast<mlir::ShapedType>(inType);
  auto outShapedType = mlir::dyn_cast<mlir::ShapedType>(outType);

  if (!inShapedType || !outShapedType) {
    return emitOpError("input and output must be shaped types");
  }

  if (inShapedType.getShape() != outShapedType.getShape()) {
    return emitOpError("input and output must have the same shape");
  }

  if (inShapedType.getElementType() != outShapedType.getElementType()) {
    return emitOpError("input and output must have the same element type");
  }

  // Mask CBs must be either both provided or both absent.
  bool hasRowMask = getRowMaskCb() != nullptr;
  bool hasColMask = getColMaskCb() != nullptr;
  if (hasRowMask != hasColMask) {
    return emitOpError("row_mask_cb and col_mask_cb must both be provided or "
                       "both be absent");
  }

  return success();
}

void BlockMaskOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInputMutable(), 0,
                       true, mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getOutputMutable(),
                       0, true, mlir::SideEffects::DefaultResource::get());
  // Mask CBs are read-only inputs. Since they're optional and tracked via
  // AttrSizedOperandSegments, we handle them via bufferizesToMemoryRead()
  // rather than effects, which is sufficient for bufferization analysis.
}

//===----------------------------------------------------------------------===//
// YieldOp / WaitOp / ReserveOp / PushOp / PopOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  auto generic = getOperation()->getParentOfType<GenericOp>();
  if (!generic || !generic.hasPureTensorSemantics()) {
    return emitOpError()
           << "used outside of generic op with pure tensor semantics";
  }

  return ::mlir::success();
}

bool PushOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool PushOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::bufferization::AliasingValueList
PushOp::getAliasingValues(mlir::OpOperand &,
                          const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
PushOp::getBufferType(mlir::Value,
                      const mlir::bufferization::BufferizationOptions &,
                      const mlir::bufferization::BufferizationState &,
                      ::llvm::SmallVector<mlir::Value> &) {
  llvm_unreachable(
      "intentionally unimplemented, this op can only accept block arguments "
      "which should have already been converted");
}

mlir::LogicalResult
PushOp::bufferize(mlir::RewriterBase &rewriter,
                  const mlir::bufferization::BufferizationOptions &options,
                  mlir::bufferization::BufferizationState &) {
  return bufferizeCBOp(*this, rewriter, options);
}

bool PopOp::bufferizesToMemoryRead(mlir::OpOperand &,
                                   const mlir::bufferization::AnalysisState &) {
  return false;
}

bool PopOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::bufferization::AliasingValueList
PopOp::getAliasingValues(mlir::OpOperand &,
                         const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
PopOp::getBufferType(mlir::Value,
                     const mlir::bufferization::BufferizationOptions &,
                     const mlir::bufferization::BufferizationState &,
                     ::llvm::SmallVector<mlir::Value> &) {
  llvm_unreachable(
      "intentionally unimplemented, this op can only accept block arguments "
      "which should have already been converted");
}

mlir::LogicalResult
PopOp::bufferize(mlir::RewriterBase &rewriter,
                 const mlir::bufferization::BufferizationOptions &options,
                 mlir::bufferization::BufferizationState &) {
  return bufferizeCBOp(*this, rewriter, options);
}
