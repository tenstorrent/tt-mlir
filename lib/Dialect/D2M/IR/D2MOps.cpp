// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

namespace mlir::tt::d2m {

// Extract grid/shard extents from a shaped type without requiring an SSA value.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
getGridAndShardFromShapedType(ShapedType shapedType) {
  if (auto memrefType = mlir::dyn_cast<MemRefType>(shapedType)) {
    if (auto layout = mlir::dyn_cast<ttcore::DeviceLayoutInterface>(
            memrefType.getLayout())) {
      return {llvm::to_vector(layout.getGridShape(memrefType)),
              llvm::to_vector(layout.getShardShape(memrefType))};
    }
    auto shape = memrefType.getShape();
    TT_assert(shape.size() % 2 == 0u);
    SmallVector<int64_t> gridShape(shape.begin(),
                                   shape.begin() + shape.size() / 2);
    SmallVector<int64_t> shardShape(shape.begin() + shape.size() / 2,
                                    shape.end());
    return {gridShape, shardShape};
  }

  auto tensorType = mlir::cast<RankedTensorType>(shapedType);
  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  return {llvm::to_vector(layout.getGridShape(tensorType)),
          llvm::to_vector(layout.getShardShape(tensorType))};
}

// Convenience wrapper for call sites that start from a Value instead of a Type.
static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
getGridAndShardFromValue(Value v) {
  return getGridAndShardFromShapedType(mlir::cast<ShapedType>(v.getType()));
}

// Derive this operand's target grid shape from the new generic execution
// grid and block factors using the generic's indexing maps.
static FailureOr<SmallVector<int64_t>>
computeReblockedOperandGridShape(d2m::GenericOp genericOp, int64_t operandIndex,
                                 ArrayRef<int64_t> newGridShape,
                                 ArrayRef<int64_t> newBlockFactors) {
  unsigned numLoopDims = genericOp.getNumDims();
  if (newBlockFactors.size() != numLoopDims) {
    return failure();
  }

  // Map the output grid to loop space via the output indexing map.
  AffineMap outputMap = genericOp.getOutputIndexingMap();
  SmallVector<int64_t> loopTotals(numLoopDims, 0);
  for (auto [outputDim, expr] : llvm::enumerate(outputMap.getResults())) {
    if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
      if (outputDim < newGridShape.size()) {
        loopTotals[dimExpr.getPosition()] = newGridShape[outputDim];
      }
    }
  }

  // Parallel dims start from the output grid extent and then multiply in the
  // block factor.
  // Reduction dims (i.e. not in the output map) contribute only their block
  // factor.
  for (unsigned i = 0; i < numLoopDims; ++i) {
    if (loopTotals[i] == 0) {
      loopTotals[i] = newBlockFactors[i];
    } else {
      loopTotals[i] *= newBlockFactors[i];
    }
  }

  // Map loop totals to the operand's grid shape via its indexing map.
  // This function is only used for operand grids which are positionally
  // aligned with indexing_maps by the op verifier. So operandIndex ->
  // getIndexingMap(operandIndex) is valid.
  AffineMap operandMap = genericOp.getIndexingMap(operandIndex);
  SmallVector<int64_t> operandGrid = operandMap.compose(loopTotals);
  for (int64_t &dim : operandGrid) {
    if (dim == 0) {
      dim = 1;
    }
  }
  return operandGrid;
}

void d2m::GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}

mlir::tt::ttcore::DeviceAttr d2m::GenericOp::getDevice() {
  return ttcore::lookupDevice(*this);
}

//===----------------------------------------------------------------------===//
// EmptyOp Builder
//===----------------------------------------------------------------------===//

void d2m::EmptyOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         ArrayRef<int64_t> shape, Type elementType,
                         Attribute encoding,
                         ArrayRef<int64_t> targetGridShape) {
  auto resultType = RankedTensorType::get(shape, elementType, encoding);
  AffineMapAttr invAttr = nullptr;
  AffineMapAttr fwdAttr = nullptr;

  if (auto metalLayout =
          mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(encoding)) {
    auto gridShape = llvm::to_vector(metalLayout.getGridShape(resultType));
    if (ttmlir::d2m::utils::grids::requiresVirtualGrid(gridShape,
                                                       targetGridShape)) {
      auto squareGrid = utils::getSquareTargetGrid(targetGridShape);
      auto physGrid = utils::findLegalPhysicalGridForVolume(
          ttmlir::utils::volume<int64_t>(gridShape), squareGrid);
      TT_assertv(!physGrid.empty(),
                 "Virtual grid required but no legal physical grid found for "
                 "volume {}; target grid [{},{}]",
                 ttmlir::utils::volume<int64_t>(gridShape), targetGridShape[0],
                 targetGridShape[1]);
      auto *ctx = builder.getContext();
      auto [fwdMap, invMap] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
          ctx, gridShape, physGrid);
      invAttr = AffineMapAttr::get(invMap);
      fwdAttr = AffineMapAttr::get(fwdMap);
    }
  }

  build(builder, state, resultType, invAttr, fwdAttr);
}

//===----------------------------------------------------------------------===//
// EmptyOp Memory Effects
//===----------------------------------------------------------------------===//

void d2m::EmptyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// EmptyOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool d2m::EmptyOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool d2m::EmptyOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::LogicalResult d2m::EmptyOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  if (getOperation()->getUses().empty()) {
    rewriter.eraseOp(*this);
    return success();
  }

  // Don't bufferize if tensor has a ttnn_layout; lowering to ttnn generic.
  if (options.allowUnknownOps) {
    auto encoding = getResult().getType().getEncoding();
    if (encoding && (mlir::isa<ttnn::TTNNLayoutAttr>(encoding) ||
                     mlir::isa<ttnn::TTNNNDLayoutAttr>(encoding))) {
      return success();
    }
  }
  ::llvm::SmallVector<mlir::Value> invocationStack;
  auto bufferType = mlir::cast<MemRefType>(
      *getBufferType(getResult(), options, state, invocationStack));
  auto allocOp = rewriter.create<memref::AllocOp>(getLoc(), bufferType);

  // Propagate virtualGridInverseMapping (inverse) and virtualGridForwardMapping
  // (forward) as discardable attributes on memref::AllocOp (we don't own
  // AllocOp so we can't add declared attributes).
  if (auto vgm = getVirtualGridInverseMappingAttr()) {
    allocOp->setAttr(d2m::utils::kVirtualGridInverseMappingAttr, vgm);
  }
  if (auto fwd = getVirtualGridForwardMappingAttr()) {
    allocOp->setAttr(d2m::utils::kVirtualGridForwardMappingAttr, fwd);
  }

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     allocOp.getResult());
  return mlir::success();
}

mlir::bufferization::AliasingValueList
d2m::EmptyOp::getAliasingValues(mlir::OpOperand &,
                                const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::EmptyOp::getBufferType(mlir::Value value,
                            const mlir::bufferization::BufferizationOptions &,
                            const mlir::bufferization::BufferizationState &,
                            ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// CreateGlobalSemaphoreOp
//===----------------------------------------------------------------------===//

bool d2m::CreateGlobalSemaphoreOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool d2m::CreateGlobalSemaphoreOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return true;
}

mlir::bufferization::AliasingValueList
d2m::CreateGlobalSemaphoreOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return {};
}

mlir::LogicalResult d2m::CreateGlobalSemaphoreOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  // Only bufferize the input.
  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  rewriter.replaceOpWithNewOp<CreateGlobalSemaphoreOp>(
      *this, getResult().getType(), *maybeInput, getValueAttr());
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
  return success();
}

::mlir::LogicalResult d2m::CreateGlobalSemaphoreOp::verify() {
  // Verify that the grid shape of the input tensor matches the device grid
  // shape.
  ttcore::DeviceAttr device = ttcore::lookupDevice(getOperation());
  auto deviceGridShape = device.getWorkerGrid().getShape();
  auto tensorGridShape = ttcore::getGridShape(getInput());
  if (!llvm::equal(deviceGridShape, tensorGridShape)) {
    return emitOpError() << "input tensor grid shape (" << tensorGridShape
                         << ") does not match device grid shape ("
                         << deviceGridShape
                         << ") for create_global_semaphore op";
  }

  // Check that the shard shape is 1x1.
  auto shardShape = ttcore::getShardShape(getInput());
  if (shardShape != llvm::ArrayRef<int64_t>({1, 1})) {
    return emitOpError()
           << "input tensor shard shape (" << shardShape
           << ") does not match 1x1 for create_global_semaphore op";
  }

  // Check that the element type is ui32.
  auto expectedType = mlir::IntegerType::get(getOperation()->getContext(), 32,
                                             mlir::IntegerType::Unsigned);
  if (mlir::cast<ShapedType>(getInput().getType()).getElementType() !=
      expectedType) {
    return emitOpError()
           << "input tensor element type ("
           << mlir::cast<ShapedType>(getInput().getType()).getElementType()
           << ") does not match ui32 for create_global_semaphore op";
  }

  if (!mlir::isa<d2m::GlobalSemaphoreType>(getResult().getType())) {
    return emitOpError()
           << "CreateGlobalSemaphoreOp result should be a GlobalSemaphore type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

bool d2m::FullOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool d2m::FullOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::LogicalResult
d2m::FullOp::bufferize(mlir::RewriterBase &rewriter,
                       const mlir::bufferization::BufferizationOptions &options,
                       mlir::bufferization::BufferizationState &state) {
  ::llvm::SmallVector<mlir::Value> invocationStack;
  // Same as d2m::empty, don't bufferize if tensor has a ttnn_layout.
  if (options.allowUnknownOps) {
    auto encoding = getResult().getType().getEncoding();
    if (encoding && (mlir::isa<ttnn::TTNNLayoutAttr>(encoding) ||
                     mlir::isa<ttnn::TTNNNDLayoutAttr>(encoding))) {
      return mlir::success();
    }
  }
  auto memrefType = mlir::cast<mlir::MemRefType>(
      getBufferType(getResult(), options, state, invocationStack).value());

  auto eltType = getResult().getType().getElementType();
  mlir::Attribute fillValue = getFillValueAttr();
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(fillValue);
      floatAttr && floatAttr.getType() != eltType) {
    fillValue = mlir::FloatAttr::get(eltType, floatAttr.getValueAsDouble());
  } else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(fillValue);
             intAttr && intAttr.getType() != eltType) {
    fillValue = mlir::IntegerAttr::get(eltType, intAttr.getValue());
  }
  auto denseAttr =
      mlir::DenseElementsAttr::get(getResult().getType(), fillValue);

  mlir::memref::GlobalOp global = ttcore::createGlobal(
      getOperation()->getParentOfType<ModuleOp>(), memrefType, denseAttr);
  mlir::bufferization::replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
      rewriter, *this, global.getType(), global.getName());

  return mlir::success();
}

mlir::bufferization::AliasingValueList
d2m::FullOp::getAliasingValues(mlir::OpOperand &,
                               const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::FullOp::getBufferType(mlir::Value value,
                           const mlir::bufferization::BufferizationOptions &,
                           const mlir::bufferization::BufferizationState &,
                           ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/false);
}

::mlir::LogicalResult d2m::FullOp::verify() {
  // Verify that the shape attribute matches the result type's shape.
  if (!llvm::equal(getShape(), getType().getShape())) {
    return emitOpError() << "expected shape (" << getType().getShape()
                         << "), got (" << getShape() << ")";
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// MeshShardOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult d2m::MeshShardOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  if (mlir::isa<::mlir::MemRefType>(getInput().getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeResult =
      mlir::bufferization::getBuffer(rewriter, getResult(), options, state);
  if (failed(maybeResult)) {
    return maybeResult;
  }

  mlir::bufferization::replaceOpWithNewBufferizedOp<d2m::MeshShardOp>(
      rewriter, *this, maybeResult->getType(), *maybeInput, getShardType(),
      getShardDirection(), getShardShape(), getShardDims());

  return success();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::MeshShardOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), false);
}

::mlir::OpFoldResult d2m::MeshShardOp::fold(FoldAdaptor adaptor) {
  auto shardShapeArray = getShardShape();
  auto shardType = getShardType();
  if (shardType != mlir::tt::ttcore::MeshShardType::Replicate &&
      ttmlir::utils::volume(shardShapeArray) == 1) {
    return getInput();
  }

  return {};
}

::mlir::LogicalResult d2m::MeshShardOp::verify() {
  auto shardType = getShardType();

  // Currently, we are not supporting maximal from StableHLO.
  if (shardType == mlir::tt::ttcore::MeshShardType::Maximal) {
    return emitOpError("Invalid shard_type (maximal) for mesh_shard op.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

// Helper: return true if two MetalLayoutAttrs are identical except for OOBVal.
static bool layoutsMatchExceptOOB(ttcore::MetalLayoutAttr a,
                                  ttcore::MetalLayoutAttr b) {
  return a.getLogicalShape() == b.getLogicalShape() &&
         a.getDimAlignments() == b.getDimAlignments() &&
         a.getCollapsedIntervals() == b.getCollapsedIntervals() &&
         a.getMemorySpace() == b.getMemorySpace() &&
         a.getMemoryLayout() == b.getMemoryLayout();
}

// Fold away a to_layout whose only effect is changing the OOB fill value.
// OOB is metadata that controls padding behaviour during LowerToLayout; when
// two layouts are otherwise identical a to_layout between them is a no-op
// because the underlying data arrangement is the same.
struct ToLayoutFoldOOBUndefPattern : public OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ToLayoutFoldOOBUndefPattern(MLIRContext *context)
      : OpRewritePattern<ToLayoutOp>(context) {
    setDebugName("d2m.ToLayoutFoldOOBUndefPattern");
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputType = mlir::dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType =
        mlir::dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType) {
      return failure();
    }

    auto inputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        inputType.getEncoding());
    auto outputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        outputType.getEncoding());
    if (!inputLayout || !outputLayout) {
      return failure();
    }

    // OOB must actually differ (same OOB is handled by identity fold),
    // and at least one side must be undef.
    if (inputLayout.getOobVal() == outputLayout.getOobVal()) {
      return failure();
    }
    if (inputLayout.getOobVal() != ttcore::OOBVal::Undef &&
        outputLayout.getOobVal() != ttcore::OOBVal::Undef) {
      return failure();
    }

    if (inputType.getShape() != outputType.getShape() ||
        inputType.getElementType() != outputType.getElementType() ||
        !layoutsMatchExceptOOB(inputLayout, outputLayout)) {
      return failure();
    }

    // Layouts match except OOB — the to_layout is a no-op.
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

struct ToLayoutFoldRedundantPattern : public OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ToLayoutFoldRedundantPattern(MLIRContext *context)
      : OpRewritePattern<ToLayoutOp>(context) {
    setDebugName("d2m.ToLayoutFoldRedundantPattern");
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
    ToLayoutOp producerLayoutOp = op.getInput().getDefiningOp<ToLayoutOp>();
    if (!producerLayoutOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, producerLayoutOp.getInput(),
                                            op.getOutput());
    return success();
    // NOLINTEND(clang-analyzer-core.StackAddressEscape)
  }
};

static ::mlir::LogicalResult
verifyLayoutOp(mlir::Operation *op, const char *aName, const char *bName,
               mlir::Type aType, mlir::Type bType, bool checkSameElementType,
               bool checkSameMemorySpace, bool checkSameRank,
               bool checkSameGridShape, bool checkSameShardShape) {
  mlir::ShapedType a = mlir::cast<mlir::ShapedType>(aType);
  mlir::ShapedType b = mlir::cast<mlir::ShapedType>(bType);
  ttcore::DeviceLayoutInterface aLayout = ttcore::getDeviceLayout(a);
  ttcore::DeviceLayoutInterface bLayout = ttcore::getDeviceLayout(b);
  if (!aLayout || !bLayout) {
    // If no layout present, we can early exit.
    return mlir::success();
  }

  if (checkSameElementType && (a.getElementType() != b.getElementType())) {
    return op->emitOpError()
           << aName << " and " << bName << "'s element types must be the same ("
           << a.getElementType() << " != " << b.getElementType() << ")";
  }

  if (checkSameMemorySpace && mlir::isa<MemRefType>(a)) {
    if (mlir::cast<MemRefType>(a).getMemorySpace() !=
        mlir::cast<MemRefType>(b).getMemorySpace()) {
      return op->emitOpError()
             << aName << " and " << bName
             << "'s memref memory spaces must be the same ("
             << mlir::cast<MemRefType>(a).getMemorySpace()
             << " != " << mlir::cast<MemRefType>(b).getMemorySpace() << ")";
    }
  }

  if (checkSameRank && a.getRank() != b.getRank()) {
    return op->emitOpError()
           << aName << " and " << bName << "'s ranks must be the same ("
           << a.getRank() << " != " << b.getRank() << ")";
  }

  if (checkSameGridShape &&
      (aLayout.getGridShape(a) != bLayout.getGridShape(b))) {
    return op->emitOpError()
           << aName << " and " << bName << "'s grid shape must be the same ("
           << aLayout.getGridShape(a) << " != " << bLayout.getGridShape(b)
           << ")";
  }

  if (checkSameShardShape &&
      (aLayout.getShardShape(a) != bLayout.getShardShape(b))) {
    return op->emitOpError()
           << aName << " and " << bName << "'s shard shape must be the same ("
           << aLayout.getShardShape(a) << " != " << bLayout.getShardShape(b)
           << ")";
  }

  return mlir::success();
}

// ToLayoutOp verification
::mlir::LogicalResult ToLayoutOp::verify() {
  return verifyLayoutOp(*this, "input", "output", getInput().getType(),
                        getOutput().getType(),
                        /*checkSameElementType*/ false,
                        /*checkSameMemorySpace*/ false,
                        /*checkSameRank*/ false,
                        /*checkSameGridShape*/ false,
                        /*checkSameShardShape*/ false);
}

mlir::LogicalResult
ToLayoutOp::fold(FoldAdaptor,
                 llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  mlir::RankedTensorType inputType =
      dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType outputType =
      dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (inputType && outputType && inputType == outputType) {
    // Don't fold if the input is a view — the remapping it carries
    // must be materialized by this to_layout, even if the types match.
    if (getInput().getDefiningOp<ViewLayoutOp>()) {
      return mlir::failure();
    }
    // Don't fold when the virtualGridInverseMappings of the input and output
    // differ.  Different TTNN shard strategies (e.g. height_sharded vs
    // block_sharded) can map to the same MetalLayoutAttr after the
    // indexAffineMap refactor, so we compare VGMs to mirror main's
    // behavior where the indexAffineMap made the types structurally
    // different.
    if (utils::getVirtualGridInverseMapping(getInput()) !=
        utils::getVirtualGridInverseMapping(getOutput())) {
      return mlir::failure();
    }
    results.push_back(getInput());
    return mlir::success();
  }
  return mlir::failure();
}

void ToLayoutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    if (operand.getOperandNumber() == 0) { // Input operand
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    } else { // Output operand
      effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    }
  }
}

void ToLayoutOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *context) {
  // Fold into d2m.empty w/ desired layout
  patterns.add(+[](ToLayoutOp op, mlir::PatternRewriter &rewriter) {
    EmptyOp emptyOp = op.getInput().getDefiningOp<EmptyOp>();
    if (!emptyOp) {
      return failure();
    }
    // Propagate VGM attributes from the output operand's EmptyOp.
    AffineMapAttr invMap = nullptr;
    AffineMapAttr fwdMap = nullptr;
    if (auto outputEmpty = op.getOutput().getDefiningOp<EmptyOp>()) {
      invMap = outputEmpty.getVirtualGridInverseMappingAttr();
      fwdMap = outputEmpty.getVirtualGridForwardMappingAttr();
    }
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType(), invMap,
                                         fwdMap);
    return success();
  });

  patterns.add(std::make_unique<ToLayoutFoldRedundantPattern>(context));
  patterns.add(std::make_unique<ToLayoutFoldOOBUndefPattern>(context));
}

bool ToLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 0; // Input operand
}

bool ToLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 1; // Output operand
}

mlir::LogicalResult
ToLayoutOp::bufferize(mlir::RewriterBase &rewriter,
                      const mlir::bufferization::BufferizationOptions &options,
                      mlir::bufferization::BufferizationState &state) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "ToLayoutOp should have exactly one result");

  if (!mlir::isa<::mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeOutput =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(maybeOutput)) {
    return maybeOutput;
  }

  // ToLayoutOp is now only for device-to-device transfers. Host transfers
  // use ToDeviceOp and ToHostOp instead.
  rewriter.create<ToLayoutOp>(getLoc(), TypeRange(), *maybeInput, *maybeOutput);

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     *maybeOutput);
  return success();
}

mlir::bufferization::AliasingValueList
mlir::tt::d2m::ToLayoutOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  if (operand.getOperandNumber() == 1) { // Output operand aliases with result
    result.addAlias({getResult(0), bufferization::BufferRelation::Equivalent});
  }
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
ToLayoutOp::getBufferType(mlir::Value value,
                          const mlir::bufferization::BufferizationOptions &,
                          const mlir::bufferization::BufferizationState &,
                          ::llvm::SmallVector<mlir::Value> &) {
  // ToLayoutOp is now only for device-to-device transfers, no layout override
  // needed.
  return ttcore::getBufferType(value.getType(), /*isView=*/false);
}

bool ToLayoutOp::isHostToDevice() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return hostInput && !hostOutput;
}

bool ToLayoutOp::isDeviceToHost() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getInput().getType())
              .getEncoding());
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
          nullptr ||
      mlir::isa<mlir::tt::ttcore::TensorMeshAttr>(
          mlir::cast<mlir::RankedTensorType>(getOutput().getType())
              .getEncoding());
  return !hostInput && hostOutput;
}

//===----------------------------------------------------------------------===//
// ToDeviceOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ToDeviceOp::verify() {
  // Skip verification for memrefs (post-bufferization) - let bufferization
  // handle the semantics.
  if (mlir::isa<MemRefType>(getInput().getType())) {
    return success();
  }

  // Input must be host (no encoding or TensorMeshAttr).
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto outputType = mlir::cast<RankedTensorType>(getOutput().getType());

  bool hostInput = inputType.getEncoding() == nullptr ||
                   mlir::isa<ttcore::TensorMeshAttr>(inputType.getEncoding());

  if (!hostInput) {
    return emitOpError("input must be a host tensor (no encoding or "
                       "TensorMeshAttr encoding)");
  }

  // Output must be device (has MetalLayoutAttr).
  if (!mlir::isa_and_present<ttcore::MetalLayoutAttr>(
          outputType.getEncoding())) {
    return emitOpError("output must be a device tensor with MetalLayoutAttr");
  }

  return success();
}

void ToDeviceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    if (operand.getOperandNumber() == 0) {
      // Input operand case.
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    } else {
      // Output operand case.
      effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    }
  }
}

bool ToDeviceOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 0; // Input operand
}

bool ToDeviceOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 1; // Output operand
}

mlir::LogicalResult
ToDeviceOp::bufferize(mlir::RewriterBase &rewriter,
                      const mlir::bufferization::BufferizationOptions &options,
                      mlir::bufferization::BufferizationState &state) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "ToDeviceOp should have exactly one result");

  if (!mlir::isa<::mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeOutput =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(maybeOutput)) {
    return maybeOutput;
  }

  // For unaligned H2D, copy the unaligned host tensor to an aligned & padded
  // bounce buffer, then write to the device.
  llvm::SmallVector<mlir::Value> invocationStack;
  MemRefType alignedHostMemref = mlir::cast<MemRefType>(
      *getBufferType(getInput(), options, state, invocationStack));

  if (mlir::cast<ttcore::HostLayoutAttr>(alignedHostMemref.getLayout())
          .isPadded()) {
    auto alignedHostTensor =
        rewriter.create<memref::AllocOp>(getLoc(), alignedHostMemref);
    rewriter.create<memref::CopyOp>(getLoc(), *maybeInput, alignedHostTensor);
    maybeInput = alignedHostTensor.getResult();
  }

  rewriter.create<ToDeviceOp>(getLoc(), TypeRange(), *maybeInput, *maybeOutput,
                              getLayout());

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     *maybeOutput);
  return success();
}

mlir::bufferization::AliasingValueList
ToDeviceOp::getAliasingValues(mlir::OpOperand &operand,
                              const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  if (operand.getOperandNumber() == 1) { // Output operand aliases with result
    result.addAlias({getResult(0), bufferization::BufferRelation::Equivalent});
  }
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
ToDeviceOp::getBufferType(mlir::Value value,
                          const mlir::bufferization::BufferizationOptions &,
                          const mlir::bufferization::BufferizationState &,
                          ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/false, getLayout());
}

//===----------------------------------------------------------------------===//
// ToHostOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ToHostOp::verify() {
  // Skip verification for memrefs (post-bufferization) - let bufferization
  // handle the semantics.
  if (mlir::isa<MemRefType>(getInput().getType())) {
    return success();
  }

  // Input must be device (has MetalLayoutAttr).
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto outputType = mlir::cast<RankedTensorType>(getOutput().getType());

  if (!mlir::isa_and_present<ttcore::MetalLayoutAttr>(
          inputType.getEncoding())) {
    return emitOpError("input must be a device tensor with MetalLayoutAttr");
  }

  // Output must be host (no encoding or TensorMeshAttr).
  bool hostOutput = outputType.getEncoding() == nullptr ||
                    mlir::isa<ttcore::TensorMeshAttr>(outputType.getEncoding());

  if (!hostOutput) {
    return emitOpError("output must be a host tensor (no encoding or "
                       "TensorMeshAttr encoding)");
  }

  return success();
}

void ToHostOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    if (!llvm::isa<MemRefType>(operand.get().getType())) {
      continue;
    }
    if (operand.getOperandNumber() == 0) {
      // Input operand case.
      effects.emplace_back(MemoryEffects::Read::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    } else {
      // Output operand case.
      effects.emplace_back(MemoryEffects::Write::get(), &operand, /*stage*/ 0,
                           /*effectOnFullRegion*/ true,
                           SideEffects::DefaultResource::get());
    }
  }
}

bool ToHostOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 0;
}

bool ToHostOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 1;
}

mlir::LogicalResult
ToHostOp::bufferize(mlir::RewriterBase &rewriter,
                    const mlir::bufferization::BufferizationOptions &options,
                    mlir::bufferization::BufferizationState &state) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "ToHostOp should have exactly one result");

  if (!mlir::isa<::mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeOutput =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(maybeOutput)) {
    return maybeOutput;
  }

  // For unaligned D2H, read the device tensor to an aligned & padded bounce
  // buffer, then do strided memcpy to copy the data into the unaligned tensor.
  llvm::SmallVector<mlir::Value> invocationStack;
  MemRefType alignedHostMemref = mlir::cast<MemRefType>(
      *getBufferType(getOutput(), options, state, invocationStack));

  auto hostLayout =
      mlir::dyn_cast<ttcore::HostLayoutAttr>(alignedHostMemref.getLayout());
  if (hostLayout && hostLayout.isPadded()) {
    auto alignedHostTensor =
        rewriter.create<memref::AllocOp>(getLoc(), alignedHostMemref);

    rewriter.create<ToHostOp>(getLoc(), TypeRange(), *maybeInput,
                              alignedHostTensor, getLayout());

    rewriter.create<memref::CopyOp>(getLoc(), alignedHostTensor, *maybeOutput);
  } else {
    rewriter.create<ToHostOp>(getLoc(), TypeRange(), *maybeInput, *maybeOutput,
                              getLayout());
  }

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     *maybeOutput);
  return success();
}

mlir::bufferization::AliasingValueList
ToHostOp::getAliasingValues(mlir::OpOperand &operand,
                            const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  if (operand.getOperandNumber() == 1) { // Output operand aliases with result
    result.addAlias({getResult(0), bufferization::BufferRelation::Equivalent});
  }
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
ToHostOp::getBufferType(mlir::Value value,
                        const mlir::bufferization::BufferizationOptions &,
                        const mlir::bufferization::BufferizationState &,
                        ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/false, getLayout());
}

//===----------------------------------------------------------------------===//
// ViewLayoutOp
//===----------------------------------------------------------------------===//

void d2m::ViewLayoutOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view");
}

void d2m::ViewLayoutOp::build(OpBuilder &builder, OperationState &state,
                              Type outputType, Value input) {
  auto inputType = mlir::cast<ShapedType>(input.getType());
  auto viewType = mlir::cast<ShapedType>(outputType);
  TT_assertv(ttmlir::utils::volume<int64_t>(inputType.getShape()) ==
                 ttmlir::utils::volume<int64_t>(viewType.getShape()),
             "input volume {} does not match view volume {}",
             ttmlir::utils::volume<int64_t>(inputType.getShape()),
             ttmlir::utils::volume<int64_t>(viewType.getShape()));

  AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      inputType.getShape(), viewType.getShape(), builder.getContext());
  build(builder, state, outputType, input, reblockMap,
        /*reinterpretLayout=*/false);
}

mlir::LogicalResult d2m::ViewLayoutOp::verify() {
  auto inputType = mlir::cast<mlir::ShapedType>(getInput().getType());
  auto resultType = mlir::cast<mlir::ShapedType>(getResult().getType());

  if (getReinterpretLayout()) {
    // For reinterpret, verify grid doesn't change; only shard (for tilizing
    // etc).
    if (auto inputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
      auto resultTensor = mlir::cast<mlir::RankedTensorType>(resultType);
      auto inputLayout = mlir::cast<mlir::tt::ttcore::MetalLayoutAttr>(
          inputTensor.getEncoding());
      auto resultLayout = mlir::cast<mlir::tt::ttcore::MetalLayoutAttr>(
          resultTensor.getEncoding());

      if (inputLayout.getGridShape(inputType) !=
          resultLayout.getGridShape(resultType)) {
        return emitOpError("reinterpret_layout cannot change grid shape");
      }
    }
    // Can change shard shape for tiled <-> untiled
  } else {
    // For affine map-based views, verify logical shapes are preserved.
    // Device tensor shapes can differ (grid redistribution, alignment changes),
    // but the underlying logical data must be the same.

    auto inputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
    auto resultTensor = mlir::dyn_cast<mlir::RankedTensorType>(resultType);

    if (inputTensor && resultTensor) {
      bool hasInputLayout = mlir::isa_and_nonnull<ttcore::MetalLayoutAttr>(
          inputTensor.getEncoding());
      bool hasResultLayout = mlir::isa_and_nonnull<ttcore::MetalLayoutAttr>(
          resultTensor.getEncoding());

      if (!hasInputLayout && !hasResultLayout) {
        // Neither has layout: verify total element count matches.
        if (ttmlir::utils::volume<int64_t>(inputType.getShape()) !=
            ttmlir::utils::volume<int64_t>(resultType.getShape())) {
          return emitOpError("view must preserve total number of elements");
        }
      }
    }

    // We also should not change element type unless reinterpretting.
    if (inputType.getElementType() != resultType.getElementType()) {
      return emitOpError("view must not change dtype");
    }

    if (auto inputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
      auto resultTensor = mlir::cast<mlir::RankedTensorType>(resultType);
      auto inputLayout = mlir::cast<mlir::tt::ttcore::MetalLayoutAttr>(
          inputTensor.getEncoding());
      auto resultLayout = mlir::cast<mlir::tt::ttcore::MetalLayoutAttr>(
          resultTensor.getEncoding());
      if (inputLayout.getOobVal() != resultLayout.getOobVal()) {
        return emitOpError("view cannot change oob_val");
      }

      if (inputLayout.getMemorySpace() != resultLayout.getMemorySpace()) {
        return emitOpError("view cannot change memory space");
      }
    }
  }

  return mlir::success();
}

bool d2m::ViewLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an input, it is a bufferized to a memory read.
  return false;
}

bool d2m::ViewLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an output, it is a bufferized to a memory write.
  return false;
}

mlir::LogicalResult d2m::ViewLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  if (mlir::isa<mlir::MemRefType>(getInput().getType())) {
    return mlir::failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  // Build the memref result type from the tensor result encoding.
  ::llvm::SmallVector<mlir::Value> dummy;
  auto outMemrefTypeOr = getBufferType(getResult(), options, state, dummy);
  if (mlir::failed(outMemrefTypeOr)) {
    return outMemrefTypeOr;
  }

  auto outMemrefType = mlir::cast<mlir::MemRefType>(*outMemrefTypeOr);
  auto newOp = rewriter.create<d2m::ViewLayoutOp>(getLoc(), outMemrefType,
                                                  *maybeInput, getRemapping(),
                                                  getReinterpretLayout());

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     newOp.getResult());
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)

  return mlir::success();
}

mlir::bufferization::AliasingValueList d2m::ViewLayoutOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  result.addAlias({getResult(), bufferization::BufferRelation::Equivalent});
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::ViewLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/true);
}

bool d2m::ViewLayoutOp::isReblockOnly() {
  auto inputShape =
      mlir::cast<mlir::ShapedType>(getInput().getType()).getShape();
  auto resultShape =
      mlir::cast<mlir::ShapedType>(getResult().getType()).getShape();
  // A reblock map requires equal volumes; views that change the logical shape
  // (TM views) are never reblock-only.
  if (ttmlir::utils::volume<int64_t>(inputShape) !=
      ttmlir::utils::volume<int64_t>(resultShape)) {
    return false;
  }
  mlir::AffineMap reblockMap =
      ttmlir::utils::calculateReblockMap(inputShape, resultShape, getContext());
  return getRemapping() == reblockMap;
}

mlir::OpFoldResult d2m::ViewLayoutOp::fold(FoldAdaptor adaptor) {
  // A view is a no-op when the types match and its own remapping is identity.
  // The input's associated remapping is irrelevant — applying the same
  // non-identity map twice should compose, not fold.
  if (getInput().getType() == getType() && getRemapping().isIdentity()) {
    return getInput();
  }

  ViewLayoutOp consecutiveView = getInput().getDefiningOp<d2m::ViewLayoutOp>();
  if (!consecutiveView) {
    return nullptr;
  }

  if (auto inputType = mlir::dyn_cast<MemRefType>(consecutiveView.getType())) {
    // Replace the input through the consecutive view.
    setOperand(consecutiveView.getInput());

    auto composedMap = consecutiveView.getRemapping().compose(getRemapping());
    setRemappingAttr(AffineMapAttr::get(composedMap));

    return getResult();
  }

  // For tensor types, only fold if both are reblock-only views.
  // This avoids complexities around layout composing affine maps
  // with a bunch of irreducible operations like floorDiv and mod.
  if (!consecutiveView.isReblockOnly() || !isReblockOnly()) {
    return nullptr;
  }

  // Don't fold through reinterpretLayout views - they legitimately change
  // logical shape and cannot be skipped.
  if (consecutiveView.getReinterpretLayout()) {
    return nullptr;
  }

  // Replace the input through the consecutive view.
  setOperand(consecutiveView.getInput());

  // Recompute the reblock map from the original input to the final result.
  mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      mlir::cast<mlir::ShapedType>(consecutiveView.getInput().getType())
          .getShape(),
      mlir::cast<mlir::ShapedType>(getType()).getShape(), getContext());

  // Update the remapping attribute on this op (layouts no longer carry
  // index maps).
  setRemappingAttr(AffineMapAttr::get(reblockMap));

  return getResult();
}

void d2m::ViewLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  // No patterns currently needed.
}

//===----------------------------------------------------------------------===//
// CompositeViewOp
//===----------------------------------------------------------------------===//

Value d2m::CompositeViewOp::getInput() {
  llvm_unreachable("Composite view must have all its inputs handled");
}

bool d2m::CompositeViewOp::isComposite() { return true; }

SmallVector<Value> d2m::CompositeViewOp::getCompositeInputs() {
  return SmallVector<Value>(getInputs().begin(), getInputs().end());
}

bool d2m::CompositeViewOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool d2m::CompositeViewOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

LogicalResult d2m::CompositeViewOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  SmallVector<Value> bufferizedInputs;
  for (Value input : getInputs()) {
    auto maybeBuffer =
        mlir::bufferization::getBuffer(rewriter, input, options, state);
    if (failed(maybeBuffer)) {
      return maybeBuffer;
    }
    bufferizedInputs.push_back(*maybeBuffer);
  }

  SmallVector<Value> invocationStack;
  auto outMemrefTypeOr =
      getBufferType(getResult(), options, state, invocationStack);
  if (failed(outMemrefTypeOr)) {
    return outMemrefTypeOr;
  }

  auto outMemrefType = mlir::cast<MemRefType>(*outMemrefTypeOr);
  auto newOp = rewriter.create<d2m::CompositeViewOp>(
      getLoc(), outMemrefType, bufferizedInputs, getDim());
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     newOp.getResult());
  return success();
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

// This isn't aliasing any single input and is meant to create a new tensor.
mlir::bufferization::AliasingValueList d2m::CompositeViewOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  mlir::bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::CompositeViewOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &, SmallVector<Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/true);
}

mlir::LogicalResult d2m::CompositeViewOp::verify() {
  auto resultType = this->getResult().getType();
  const bool isMemrefType = mlir::isa<MemRefType>(resultType);

  auto outShape = isMemrefType
                      ? mlir::cast<MemRefType>(resultType).getShape()
                      : mlir::cast<RankedTensorType>(resultType).getShape();
  const int32_t rank = static_cast<int32_t>(outShape.size()) / 2;
  const int32_t compositeDim = this->getDim();
  if (compositeDim < 0 || compositeDim >= rank) {
    return emitOpError("Composite view dim out of range.");
  }

  if (this->getInputs().size() < 2) {
    return emitOpError("Composite view should have at least two inputs.");
  }

  int64_t accum = 0;
  for (auto input : this->getInputs()) {
    auto inShape =
        isMemrefType ? mlir::cast<MemRefType>(input.getType()).getShape()
                     : mlir::cast<RankedTensorType>(input.getType()).getShape();
    if (inShape.size() != static_cast<size_t>(2 * rank)) {
      return emitOpError("Incompatible input/output shapes.");
    }

    for (int32_t i = 0; i < rank; i++) {
      if (i == compositeDim) {
        accum += inShape[i] * inShape[i + rank];
      } else if (inShape[i] * inShape[i + rank] !=
                 outShape[i] * outShape[i + rank]) {
        return emitOpError("Incompatible non-composite dim.");
      }
    }
  }

  // The output's composite dim could have been aligned-up.
  if (accum > outShape[compositeDim] * outShape[compositeDim + rank]) {
    return emitOpError("Incompatible composite dim.");
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

void d2m::GenericOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, ValueRange inputs,
                           ValueRange outputs, ValueRange additionalArgs,
                           ArrayAttr indexingMaps, ArrayAttr iteratorTypes,
                           ThreadType singleThreadType, ttcore::GridAttr grid,
                           ArrayRef<int64_t> blockFactors) {
  TT_assertv(!indexingMaps.empty(), "expected non-empty indexing maps");
  TT_assertv(outputs.size() == 1u, "expected single output");

  if (!grid) {
    auto output = outputs[0];
    SmallVector<int64_t> gridShape;
    TT_assert(ttcore::hasDeviceLayout(output));
    gridShape = llvm::to_vector(ttcore::getGridShape(output));

    auto layout =
        ttcore::getDeviceLayout(mlir::dyn_cast<ShapedType>(output.getType()));
    auto metalLayout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(layout);

    if (metalLayout) {
      // 1. Check for an explicit virtualGridInverseMapping (inverse map) on the
      //    output's EmptyOp.  Use the stored map directly — it encodes the
      //    correct physical grid from the TTNN layout.
      if (auto invMap = utils::getVirtualGridInverseMapping(output)) {
        // Get virtual to physical map from output as well.
        auto fwdMap = *utils::getVirtualGridForwardMapping(output);
        fwdMap = ttmlir::utils::affineMapDropBackResults(fwdMap, 2);
        fwdMap = ttmlir::utils::dropDim(fwdMap, 3);
        fwdMap = ttmlir::utils::dropDim(fwdMap, 2);
        fwdMap = fwdMap.shiftDims(/*shift=*/1, /*offset=*/0);

        grid = builder.getAttr<ttcore::GridAttr>(gridShape, fwdMap, *invMap);
      }

      // 2. Check for a 2D→2D permutation reblocking on a ViewLayoutOp.
      //    After the refactor, associated remappings are always reblockings
      //    (never virtual grids), so we only need the permutation check.
      if (!grid) {
        auto existingRemapping = utils::getAssociatedRemapping(output);
        if (existingRemapping.has_value() && !existingRemapping->isEmpty() &&
            !existingRemapping->isIdentity()) {
          auto indexMap = *existingRemapping;
          constexpr size_t kExpectedDimsFor2DDeviceShape = 2 * 2;
          bool indexMapIs2DPermutation =
              indexMap.isPermutation() &&
              indexMap.getNumResults() == kExpectedDimsFor2DDeviceShape &&
              indexMap.getNumInputs() == kExpectedDimsFor2DDeviceShape;

          if (indexMapIs2DPermutation) {
            auto [fwdMap, invMap] =
                ttmlir::utils::createGridForwardAndInverseMapFor2DPermutation(
                    indexMap, gridShape.size(), builder.getContext());
            grid = builder.getAttr<ttcore::GridAttr>(gridShape, fwdMap, invMap);
          }
        }
      }

      if (!grid) {
        // Output aligns with its underlying physical grid shape and has no
        // permuted indices; no need to have a virtualization mapping.
        grid = builder.getAttr<ttcore::GridAttr>(gridShape);
      }
    } else {
      grid = builder.getAttr<ttcore::GridAttr>(gridShape);
    }
  }

  ArrayAttr blockFactorsAttr;
  if (blockFactors.empty()) {
    auto maps =
        llvm::to_vector(llvm::map_range(indexingMaps, [](Attribute attr) {
          return cast<AffineMapAttr>(attr).getValue();
        }));
    SmallVector<SmallVector<int64_t>> operandGridShapes;
    const auto values = llvm::to_vector(llvm::concat<Value>(inputs, outputs));
    operandGridShapes.reserve(values.size());
    for (Value v : values) {
      auto shapedType = mlir::cast<ShapedType>(v.getType());
      ttcore::DeviceLayoutInterface layout =
          ttcore::getDeviceLayout(shapedType);
      TT_assertv(
          layout,
          "This generic constructor expects operands to be in device layout");
      operandGridShapes.emplace_back(layout.getGridShape(shapedType).begin(),
                                     layout.getGridShape(shapedType).end());
    }
    blockFactorsAttr =
        builder.getI64ArrayAttr(d2m::utils::deriveBlockFactorsFromOperandGrids(
            maps, operandGridShapes, grid.getShape()));
  } else {
    blockFactorsAttr = builder.getI64ArrayAttr(blockFactors);
  }

  auto threads =
      builder.getArrayAttr(builder.getAttr<ThreadAttr>(singleThreadType));

  build(builder, state, TypeRange(outputs), inputs, outputs, additionalArgs,
        grid, blockFactorsAttr, indexingMaps, iteratorTypes, threads,
        /*scratch_inputs=*/nullptr, 1);
}

void d2m::GenericOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ValueRange inputs,
    ValueRange outputs, ValueRange additionalArgs, ArrayAttr indexingMaps,
    ArrayAttr iteratorTypes,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)>
        singleThreadRegionBuilder,
    ThreadType singleThreadType, ttcore::GridAttr grid,
    ArrayRef<int64_t> blockFactors) {
  build(builder, state, inputs, outputs, additionalArgs, indexingMaps,
        iteratorTypes, singleThreadType, grid, blockFactors);

  auto inputOutputOperands = llvm::SmallVector<Value>(
      state.operands.begin(),
      state.operands.begin() + inputs.size() + outputs.size());

  // Create an empty block (no block arguments) and populate it with
  // tensor.empty ops for each operand's shard shape. The region builder
  // callback receives these tensor.empty values instead of CB block args.
  Region &region = *state.regions.front().get();
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&region, region.end());

  llvm::SmallVector<Value> operandAllocs;
  for (Value operand : inputOutputOperands) {
    auto tensorType = mlir::cast<RankedTensorType>(operand.getType());
    auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        tensorType.getEncoding());

    // If the operand is a view, get the layout from its source.
    if (!layout) {
      if (auto viewOp = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
        auto inputType =
            mlir::cast<RankedTensorType>(viewOp.getInput().getType());
        layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
            inputType.getEncoding());
      }
    }

    assert(layout && "Expected MetalLayoutAttr on operand or its view source");
    auto shardShape = layout.getShardShape(tensorType);
    auto emptyOp = builder.create<mlir::tensor::EmptyOp>(
        state.location, shardShape, tensorType.getElementType());
    operandAllocs.push_back(emptyOp.getResult());
  }

  singleThreadRegionBuilder(builder, state.location, operandAllocs);
}

void d2m::GenericOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ValueRange inputs,
    ValueRange outputs, ValueRange additionalArgs,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)>
        singleThreadRegionBuilder,
    ThreadType singleThreadType, ttcore::GridAttr grid,
    ArrayRef<int64_t> blockFactors) {
  TT_assertv(outputs.size() == 1u, "expected single output");
  RankedTensorType tensorType =
      mlir::cast<RankedTensorType>(outputs[0].getType());
  ttcore::MetalLayoutAttr maybeLayout =
      mlir::dyn_cast_or_null<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
  const size_t rank = (maybeLayout != nullptr)
                          ? maybeLayout.getShardShape(tensorType).size()
                          : tensorType.getShape().size();
  auto [indexingMaps, iteratorTypes] = buildParallelAffineMapsAndIteratorTypes(
      builder, inputs.size() + outputs.size(), rank);
  build(builder, state, inputs, outputs, additionalArgs, indexingMaps,
        iteratorTypes, singleThreadRegionBuilder, singleThreadType, grid);
}

bool d2m::GenericOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an input, it is bufferized to a memory read.
  return isDpsInput(&operand);
}

bool d2m::GenericOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an output, it is bufferized to a memory write.
  return isDpsInit(&operand);
}

mlir::bufferization::AliasingValueList
d2m::GenericOp::getAliasingValues(mlir::OpOperand &,
                                  const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

bool d2m::GenericOp::isWritable(mlir::Value value,
                                const mlir::bufferization::AnalysisState &) {
  return mlir::isa<mlir::OpResult>(value) ||
         mlir::isa<mlir::BlockArgument>(value);
}

bool d2m::GenericOp::hasTensorSemantics() {
  auto isaTensor = [](Type t) { return isa<bufferization::TensorLikeType>(t); };
  if (any_of(getResultTypes(), isaTensor)) {
    return true;
  }
  return any_of(getOperandTypes(), isaTensor);
}

static std::optional<int64_t>
isNotEqualOrBroadcast(mlir::ArrayRef<int64_t> as, mlir::ArrayRef<int64_t> bs) {
  for (auto [dim, a, b] : llvm::enumerate(as, bs)) {
    if (a != b && a != 0 && b != 0) {
      return dim;
    }
  }
  return std::nullopt;
}

static mlir::LogicalResult verifyAffineShapesPermutation(
    const char *shapeName, mlir::ArrayRef<mlir::AffineMap> indexingMaps,
    mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes,
    llvm::function_ref<mlir::InFlightDiagnostic()> diagFn) {
  assert(indexingMaps.size() == shapes.size());

  for (size_t operandA = 0; operandA < indexingMaps.size(); ++operandA) {
    for (size_t operandB = 0; operandB < indexingMaps.size(); ++operandB) {
      if (operandA == operandB) {
        continue;
      }

      auto shapeMapA =
          inverseAndBroadcastProjectedPermutation(indexingMaps[operandA]);
      auto shapeMapB =
          inverseAndBroadcastProjectedPermutation(indexingMaps[operandB]);
      auto shapeA = shapeMapA.compose(shapes[operandA]);
      auto shapeB = shapeMapB.compose(shapes[operandB]);

      if (auto dim = isNotEqualOrBroadcast(shapeA, shapeB)) {
        return diagFn() << shapeName << " shape mismatch between operand["
                        << operandA << "] " << shapeName << "_shape=["
                        << shapes[operandA] << "] and operand[" << operandB
                        << "] " << shapeName << "_shape=[" << shapes[operandB]
                        << "] at affine dim d" << *dim;
      }
    }
  }

  return mlir::success();
}

static mlir::LogicalResult verifyAffineBlocking(
    const char *shapeName, mlir::ArrayRef<mlir::AffineMap> indexingMaps,
    mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes,
    mlir::ArrayRef<int64_t> blockingFactors, mlir::AffineMap opGridIndexingMap,
    mlir::ArrayRef<int64_t> computeGridShape,
    llvm::function_ref<mlir::InFlightDiagnostic()> diagFn) {
  assert(indexingMaps.size() == shapes.size());

  // Skip verification if blockingFactors is empty (lowered loop form after
  // GenerateOuterLoops).
  if (blockingFactors.empty()) {
    return mlir::success();
  }

  // Invert the opGridIndexingMap. e.g. matmul map might be:
  //   (m, n, k) -> (m, n)
  //
  // Its inverse is:
  //   (m, n) -> (m, n, 0)
  //
  // We take this inverse and multiply out the blocking factors to calculate
  // the expected operand grid shapes.
  auto inverseOpGridMap =
      inverseAndBroadcastProjectedPermutation(opGridIndexingMap);
  mlir::SmallVector<int64_t> factors =
      inverseOpGridMap.compose(computeGridShape);
  assert(factors.size() == blockingFactors.size());
  for (size_t i = 0; i < blockingFactors.size(); ++i) {
    if (factors[i] == 0) {
      // The "Broadcast" part of inverseAndBroadcastProjectedPermutation will 0
      // fill unparticipating dims.  Promote these to 1's so that we can
      // multiply by blocking factor.
      factors[i] = 1;
    }
    factors[i] *= blockingFactors[i];
  }

  for (size_t operand = 0; operand < indexingMaps.size(); ++operand) {
    auto shape = shapes[operand];
    auto factor = indexingMaps[operand].compose(factors);
    if (shape.size() != factor.size()) {
      return diagFn() << shapeName << " rank mismatch for operand[" << operand
                      << "] " << shapeName << "_shape=[" << shapes[operand]
                      << "] expected " << shapeName << "_shape=[" << factor
                      << "]";
    }

    if (auto dim = isNotEqualOrBroadcast(shape, factor)) {
      return diagFn() << shapeName << " dim unexpected for operand[" << operand
                      << "] " << shapeName << "_shape=[" << shapes[operand]
                      << "] expected " << shapeName << "_shape=[" << factor
                      << "] at affine dim d" << *dim;
    }
  }

  return mlir::success();
}

Operation::operand_range d2m::GenericOp::getInputsAndOutputs() {
  return Operation::operand_range(getOperands().begin(),
                                  getOperands().begin() + getInputs().size() +
                                      getOutputs().size());
}

MutableArrayRef<OpOperand> d2m::GenericOp::getInputsAndOutputsMutable() {
  return MutableArrayRef<OpOperand>(getOperation()->getOpOperands().begin(),
                                    getOperation()->getOpOperands().begin() +
                                        getInputs().size() +
                                        getOutputs().size());
}

// GenericOp verification
::mlir::LogicalResult d2m::GenericOp::verify() {
  if (hasPureTensorSemantics()) {
    if (this->getNumRegions() != 1 && !isExplicitDatamovementForm()) {
      return emitOpError(
          "generic op with pure tensor semantics must have exactly 1 region "
          "when not in explicit data movement form");
    }

    // Only check yield terminator for non-explicit-datamovement form.
    // Explicit datamovement form allows users to manage terminators themselves.
    if (!isExplicitDatamovementForm() && !this->getRegion(0).empty()) {
      Region &region = this->getRegion(0);

      Block &block = region.front();
      if (block.getOperations().empty() || !mlir::isa<YieldOp>(&block.back())) {
        return emitOpError(
            "generic op with pure tensor semantics must have yield terminator");
      }

      if (block.back().getNumOperands() != getNumResults()) {
        return emitOpError("yield terminator must have the same number of "
                           "arguments as generic results");
      }
    }
  }

  if (getOutputs().size() != 1) {
    return emitOpError("must currently have exactly one output operand");
  }

  if (getThreads().empty()) {
    return emitOpError("must have at least one thread");
  }

  if (!getRegions().empty() && getRegions().size() != getThreads().size()) {
    return emitOpError("number of regions must match the number of threads");
  }

  // Output grid shape must equal the GenericOp grid shape.
  auto opGridShape = getGrid().getShape();
  for (auto output : getOutputs()) {
    mlir::ShapedType outputType =
        mlir::cast<mlir::ShapedType>(output.getType());
    ttcore::DeviceLayoutInterface layout = ttcore::getDeviceLayout(outputType);
    if (!layout) {
      // Skip layout checks if the tensor type does not have a layout yet.
      // Turn this into an error, generic op should enforce operands have
      // device layout: https://github.com/tenstorrent/tt-mlir/issues/5445
      continue;
    }
    ArrayRef<int64_t> outputGridShape = layout.getGridShape(outputType);
    if (!llvm::all_of(llvm::zip(outputGridShape, opGridShape), [](auto pair) {
          auto [out, op] = pair;
          return out % op == 0;
        })) {
      return emitOpError("output grid shape must be divisible by the generic "
                         "op's grid shape");
    }
  }

  if (!getGrid().getPhysicalToVirtMap().isEmpty()) {

    if (getGrid().getPhysicalToVirtMap().getNumInputs() != 2ul) {
      return emitOpError(
          "GenericOp virtual grid affine map must have 2 inputs, or be empty.");
    }

    // Verify that the grid volume fits within the device's worker grid.
    auto device = ttcore::lookupDevice(*this);
    int64_t gridVolume = getGrid().getGridVolume();
    int64_t deviceVolume = device.getWorkerGrid().getGridVolume();
    if (gridVolume > deviceVolume) {
      return emitOpError("grid volume (")
             << gridVolume << ") exceeds device worker grid capacity ("
             << deviceVolume << ")";
    }

    auto isDRAM = [](Value output) {
      if (auto memrefType = mlir::dyn_cast<MemRefType>(output.getType())) {
        return ttcore::getMemorySpace(memrefType) ==
               ttcore::MemorySpace::DeviceDRAM;
      }
      if (auto tensorType =
              mlir::dyn_cast<RankedTensorType>(output.getType())) {
        if (auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                tensorType.getEncoding())) {
          return layout.getMemorySpace() == ttcore::MemorySpace::DeviceDRAM;
        }
      }
      return false;
    };
    // Verify per-output VGM consistency:
    // 1. For non-DRAM outputs, the output's inverse VGM must match the
    // GridAttr's inverse map.
    // 2. The inverse map applied to the physical grid shape must produce
    //    a virtual grid shape matching the output's grid shape.
    AffineMap gridInvMap = getGrid().getPhysicalToVirtMap();
    for (Value output : getOutputs()) {
      if (!isDRAM(output)) {
        auto outputInvMap = utils::getVirtualGridInverseMapping(output);
        if (outputInvMap && *outputInvMap != gridInvMap) {
          return emitOpError("grid inverse map does not match output operand's "
                             "inverse VGM");
        }
        if (!outputInvMap && !gridInvMap.isEmpty()) {
          return emitOpError("grid has an inverse map but output operand "
                             "does not have a VGM");
        }
      }

      SmallVector<int64_t> physicalGridShape =
          d2m::utils::getPhysicalGridShape(output);

      // Drop the deviceID result (first result) from the inverse map.
      AffineMap invMapNoDevice = gridInvMap.dropResult(0);

      SmallVector<int64_t> impliedVirtShape =
          ttmlir::utils::evalShape(invMapNoDevice, physicalGridShape);

      SmallVector<int64_t> outputGridShape =
          llvm::to_vector(ttcore::getGridShape(output));

      if (outputGridShape != impliedVirtShape) {
        return emitOpError("output grid shape does not match implied virtual "
                           "grid shape from physical grid and inverse mapping");
      }
    }
  }

  if (!llvm::all_equal(llvm::map_range(getIndexingMapsValue(), [](AffineMap m) {
        return m.getNumDims();
      }))) {
    return emitOpError(
        "all indexing maps must have the same number of dimensions");
  }

  size_t numIterators = getIteratorTypes().size();
  SmallVector<int64_t> blockFactors = getBlockFactorsValue();
  if (numIterators > 0) {
    if (blockFactors.size() != numIterators) {
      return emitOpError("number of block factors[")
             << blockFactors.size() << "] must match the number of iterators["
             << numIterators << "]";
    }
  }

  auto rankedTensorType =
      mlir::dyn_cast<RankedTensorType>(getOutputs().front().getType());
  bool hasGrid = mlir::isa<MemRefType>(getOutputs().front().getType()) ||
                 (rankedTensorType && rankedTensorType.getEncoding());
  SmallVector<AffineMap> indexingMaps = getIndexingMapsValue();
  if (hasGrid && !indexingMaps.empty()) {
    // Validate that all operands have device layouts before calling
    // getInputOutputOperandGridShapes(), which assumes layouts are present.
    for (Value operand : getInputsAndOutputs()) {
      auto result =
          llvm::TypeSwitch<Type, LogicalResult>(operand.getType())
              .Case<MemRefType>([&](MemRefType memrefType) -> LogicalResult {
                if (mlir::isa<ttcore::ViewLayoutAttr>(memrefType.getLayout())) {
                  return success();
                }
                if (!mlir::dyn_cast<ttcore::DeviceLayoutInterface>(
                        memrefType.getLayout())) {
                  return emitOpError("memref operand must have a device layout "
                                     "attribute "
                                     "(e.g., #ttcore.shard or "
                                     "#ttcore.interleaved), "
                                     "but got: ")
                         << memrefType;
                }
                return success();
              })
              .Case<RankedTensorType>(
                  [&](RankedTensorType tensorType) -> LogicalResult {
                    if (!mlir::dyn_cast<ttcore::MetalLayoutAttr>(
                            tensorType.getEncoding())) {
                      return emitOpError("tensor operand must have a metal "
                                         "layout encoding, "
                                         "but got: ")
                             << tensorType;
                    }
                    return success();
                  })
              .Default([](Type) { return success(); });

      if (failed(result)) {
        return failure();
      }
    }

    auto emitDiag = [&]() -> InFlightDiagnostic { return this->emitOpError(); };
    SmallVector<SmallVector<int64_t>> gridShapes =
        getInputOutputOperandGridShapes();
    LogicalResult gridResult = verifyAffineShapesPermutation(
        "grid", indexingMaps, gridShapes, emitDiag);

    if (failed(gridResult)) {
      return gridResult;
    }

    SmallVector<SmallVector<int64_t>> scalarShardShapes =
        getInputOutputOperandShardShapes(/*convertTileToScalar=*/true);
    LogicalResult shardResult = verifyAffineShapesPermutation(
        "shard", indexingMaps, scalarShardShapes, emitDiag);
    if (failed(shardResult)) {
      return shardResult;
    }

    assert(getNumDpsInits() == 1);
    ::mlir::OpOperand *output = getDpsInitOperand(0);
    // Op grid map is implicitly derived from the output operand.
    AffineMap opGridMap = indexingMaps[output->getOperandNumber()];
    LogicalResult blockFactorResult =
        verifyAffineBlocking("grid", indexingMaps, gridShapes, blockFactors,
                             opGridMap, opGridShape, emitDiag);
    if (failed(blockFactorResult)) {
      return blockFactorResult;
    }
  }

  // Unified form will be replicated across compute and datamovement threads.
  // Reject semaphore ops that would create race conditions when replicated.
  if (isUnifiedForm() && !this->getRegion(0).empty()) {
    if (failed(utils::checkForIllegalSemaphoreOps(&getRegion(0).front()))) {
      return failure();
    }
  }

  auto *firstRegion = getRegions().begin();
  for (Region &region : getRegions()) {
    if (!region.hasOneBlock()) {
      return emitOpError("region must have a single block");
    }

    // All regions must have the same number of arguments and signature.
    if (region.getNumArguments() != firstRegion->getNumArguments()) {
      return emitOpError("all regions must have the same number of arguments");
    }

    // Block arguments may only be semaphore type.
    // Semaphore block args are added by PreallocateMcastSemaphores.
    for (BlockArgument arg : region.getArguments()) {
      if (!mlir::isa<d2m::SemaphoreType>(arg.getType())) {
        return emitOpError(
            "region block arguments must be of 'semaphore' type");
      }

      if (arg.getType() !=
          firstRegion->getArgument(arg.getArgNumber()).getType()) {
        return emitOpError("all regions must have the same argument types");
      }
    }

    if (isExplicitDatamovementForm()) {
      // Explicit datamovement form: skip shape validation.
      continue;
    }

    // When not in explicit datamovement form, indexing_maps must be non-empty.
    if (indexingMaps.empty()) {
      return emitOpError(
          "indexing_maps must be non-empty unless in explicit "
          "datamovement form (all of block_factors, indexing_maps, "
          "and iterator_types are empty)");
    }
  }

  if (isExternalSymbolForm()) {
    if (llvm::any_of(getThreads(), [](Attribute thread) {
          return !mlir::cast<ThreadAttr>(thread).getKernelSymbol();
        })) {
      return emitOpError("threads must have a kernel symbol in external symbol "
                         "form (i.e. without regions)");
    }
  }

  // Verify that any values used in regions that are defined outside
  // must be operands of this GenericOp.
  getOperation()->walk([&](Operation *nestedOp) {
    // Skip block arguments - they're defined within the region.
    for (Value operand : nestedOp->getOperands()) {
      if (mlir::isa<BlockArgument>(operand)) {
        continue;
      }

      // Check if the operand is defined outside this GenericOp.
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && !getOperation()->isAncestor(definingOp)) {
        // It's defined outside - check if it's one of our operands.
        bool isOurOperand = llvm::is_contained(getOperands(), operand);
        if (!isOurOperand) {
          emitOpError("region uses value defined outside that is not an "
                      "operand of this generic op: ")
              << operand;
        }
      }
    }
  });

  return success();
}
// Spatialop verification
::mlir::LogicalResult d2m::SpatialOp::verify() {
  mlir::ArrayAttr gridRangesAttr = getGridRanges();

  // Check that grid_ranges has at least 1 CoreRange
  if (!gridRangesAttr || gridRangesAttr.empty()) {
    return emitOpError("grid_ranges must contain at least one CoreRange");
  }

  // Check that the number of CoreRanges matches the number of Regions
  size_t numCoreRanges = gridRangesAttr.size();
  size_t numRegions = getNumRegions();

  if (numCoreRanges != numRegions) {
    return emitOpError("number of CoreRanges (")
           << numCoreRanges << ") must match the number of Regions ("
           << numRegions << ")";
  }

  // Each region has exactly one generic op. Verify its grid (2D only) fits
  // within the region: when mapping is empty, compare shape only; when
  // mapping is present, require virtual grid contained in region's virtual
  // bbox.
  for (auto [region, rangeAttr] :
       llvm::zip(getRegions(), gridRangesAttr.getValue())) {
    auto coreRange = mlir::cast<ttcore::CoreRangeAttr>(rangeAttr);
    utils::BoundingBox regionBox;
    regionBox.start = {coreRange.getStartCoord().getY(),
                       coreRange.getStartCoord().getX()};
    regionBox.end = {coreRange.getEndCoord().getY(),
                     coreRange.getEndCoord().getX()};
    int64_t regionShapeY = regionBox.end[0] - regionBox.start[0] + 1;
    int64_t regionShapeX = regionBox.end[1] - regionBox.start[1] + 1;

    if (region.empty()) {
      return emitOpError("each spatial region must not be empty");
    }
    auto genericOps = llvm::to_vector(region.getOps<GenericOp>());
    if (genericOps.size() != 1) {
      return emitOpError(
                 "each region must contain exactly one d2m.generic op, got ")
             << genericOps.size();
    }
    GenericOp genericOp = genericOps.front();

    auto grid = genericOp.getGrid();
    llvm::ArrayRef<int64_t> gridShape = grid.getShape();

    if (gridShape.size() != 2) {
      return emitOpError("d2m.generic inside d2m.spatial: only 2D "
                         "grid is considered for now, got ")
             << gridShape.size() << "D";
    }

    if (grid.getVirtToPhysicalMap().isEmpty()) {
      if (gridShape[0] > regionShapeY || gridShape[1] > regionShapeX) {
        return emitOpError("generic op grid shape [")
               << gridShape[0] << ", " << gridShape[1]
               << "] exceeds region CoreRange shape [" << regionShapeY << ", "
               << regionShapeX << "]";
      }
    } else {
      AffineMap physicalToVirtual = grid.getVirtToPhysicalMap();
      if (physicalToVirtual.getNumResults() == 3u) {
        physicalToVirtual = physicalToVirtual.dropResult(0);
      }
      utils::BoundingBox regionVirtualBbox =
          utils::getProjectedBoundingBox(regionBox, physicalToVirtual);
      int64_t gridEndY = gridShape[0] - 1, gridEndX = gridShape[1] - 1;
      if (regionVirtualBbox.start[0] > 0 ||
          regionVirtualBbox.end[0] < gridEndY ||
          regionVirtualBbox.start[1] > 0 ||
          regionVirtualBbox.end[1] < gridEndX) {
        return emitOpError(
                   "generic op grid not contained in region grid_ranges [")
               << regionBox.start[0] << ", " << regionBox.start[1] << "] to ["
               << regionBox.end[0] << ", " << regionBox.end[1] << "]";
      }
    }
  }

  return success();
}

void GenericOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::MLIRContext *context) {
  patterns.add(
      // Check to see if any linalg generic passes have wired up d2m generic op
      // inputs to the outputs (i.e. inplace) which we currently do not support.
      // This canonicalization pattern will catch these cases and rewire the
      // inputs.
      +[](GenericOp op, mlir::PatternRewriter &rewriter) {
        if (op->getNumRegions() == 0) {
          return mlir::failure();
        }

        auto replaceWithOutputAlloc =
            [op](PatternRewriter &rewriter, Region &region, Operation *regionOp,
                 OpOperand &initOperand, int64_t dpsIOBoundary) -> bool {
          Operation *origDefiningOp = initOperand.get().getDefiningOp();
          if (origDefiningOp &&
              !mlir::isa<EmptyOp, mlir::tensor::EmptyOp>(origDefiningOp)) {
            return false;
          }

          // Find the output alloc by positional counting.
          Value outputAlloc = GenericOp::getOperandAlloc(region, dpsIOBoundary);

          if (!outputAlloc || outputAlloc.use_empty()) {
            return false;
          }

          // Find a wait/reserve that dominates the DPS operation.
          Operation *waitOrReserve = nullptr;

          // Get the parent function to compute dominance.
          Operation *parentOp = op->getParentOp();
          while (parentOp && !mlir::isa<FunctionOpInterface>(parentOp)) {
            parentOp = parentOp->getParentOp();
          }

          assert(parentOp && "d2m.generic must be nested within a function");

          // Use DominanceInfo for cross-block dominance checking.
          DominanceInfo domInfo(parentOp);
          for (Operation *user : outputAlloc.getUsers()) {
            if (!mlir::isa<d2m::WaitOp, d2m::ReserveOp>(user)) {
              continue;
            }
            // Check if this wait/reserve dominates the regionOp.
            if (domInfo.dominates(user, regionOp)) {
              waitOrReserve = user;
              break;
            }
          }

          if (!waitOrReserve) {
            return false;
          }

          rewriter.modifyOpInPlace(regionOp, [&]() {
            initOperand.assign(waitOrReserve->getResult(0));
          });

          if (mlir::isa_and_nonnull<EmptyOp, mlir::tensor::EmptyOp>(
                  origDefiningOp)) {
            rewriter.replaceAllUsesWith(origDefiningOp->getResult(0),
                                        initOperand.get());
          }

          return true;
        };

        int64_t dpsIOBoundary = op.getNumDpsInputs();
        bool updated = false;
        for (Region &region : op->getRegions()) {
          if (op.getRegionThreadType(region.getRegionNumber()) !=
              ThreadType::Compute) {
            continue;
          }

          region.walk([&](Operation *regionOp) {
            if (DestinationStyleOpInterface dps =
                    mlir::dyn_cast<DestinationStyleOpInterface>(regionOp);
                dps) {
              for (OpOperand &initOperand : dps.getDpsInitsMutable()) {
                assert(op.getNumDpsInits() == dps.getNumDpsInits());
                assert(op.getNumDpsInits() == 1);

                updated |= replaceWithOutputAlloc(rewriter, region, regionOp,
                                                  initOperand, dpsIOBoundary);
              }
            } else if (TileMatmulBlockOp tmb =
                           mlir::dyn_cast<TileMatmulBlockOp>(regionOp);
                       tmb) {
              updated |=
                  replaceWithOutputAlloc(rewriter, region, regionOp,
                                         tmb.getOutputMutable(), dpsIOBoundary);
            }
          });
        }

        return updated ? mlir::success() : mlir::failure();
      });
}

unsigned d2m::GenericOp::getNumLoops() { return getNumDims(); }

unsigned d2m::GenericOp::getNumDims() {
  assert(!getIndexingMaps().empty() && "GenericOp must be pre-loop generated "
                                       "with indexing maps to use this method");
  return getIndexingMap(0).getNumDims();
}

unsigned d2m::GenericOp::getNumBlockFactors() {
  return static_cast<unsigned>(getBlockFactors().size());
}

mlir::AffineMap d2m::GenericOp::getIndexingMap(int64_t operandIndex) {
  TT_debugv(!isExplicitDatamovementForm(),
            "Attempting to access indexing map while in explicit "
            "datamovement form.");
  return getIndexingMapsValue()[operandIndex];
}

AffineMap d2m::GenericOp::getIndexingMapForOperand(Value operand) {
  for (unsigned i = 0; i < getNumOperands(); ++i) {
    if (getOperand(i) == operand) {
      return getIndexingMap(i);
    }
  }
  llvm_unreachable("Operand not found in GenericOp");
}

AffineMap d2m::GenericOp::getOutputIndexingMap() {
  TT_assertv(getNumDpsInits() == 1,
             "getOutputIndexingMap expects exactly one output operand");
  return getIndexingMapForOperand(getOutputs().front());
}

std::optional<unsigned> d2m::GenericOp::getOutputOperandIndex(Value operand) {
  for (OpOperand &output : getOutputsMutable()) {
    if (output.get() == operand) {
      return output.getOperandNumber();
    }
  }
  return std::nullopt;
}

mlir::SmallVector<int64_t> d2m::GenericOp::getOutputGridDimPositions() {
  TT_assertv(getNumDpsInits() == 1,
             "getOutputGridDimPositions expects exactly one output operand");
  auto outputOperandIndex = getOperandIndex(getOutputs().front());
  return getParticipatingLoopDims(outputOperandIndex);
}

int64_t d2m::GenericOp::getOperandIndex(Value operand) {
  for (unsigned i = 0; i < getNumOperands(); ++i) {
    if (getOperand(i) == operand) {
      return i;
    }
  }
  llvm_unreachable("Operand not found in GenericOp");
}

mlir::SmallVector<mlir::AffineMap> d2m::GenericOp::getIndexingMapsValue() {
  return llvm::map_to_vector(getIndexingMaps(), [](Attribute a) {
    return mlir::cast<AffineMapAttr>(a).getValue();
  });
}

mlir::SmallVector<mlir::tt::ttcore::IteratorType>
d2m::GenericOp::getIteratorTypesValue() {
  return llvm::map_to_vector(getIteratorTypes(), [](Attribute a) {
    return mlir::cast<ttcore::IteratorTypeAttr>(a).getValue();
  });
}

mlir::SmallVector<int64_t> d2m::GenericOp::getBlockFactorsValue() {
  return llvm::map_to_vector(getBlockFactors(), [](Attribute a) {
    return mlir::cast<IntegerAttr>(a).getInt();
  });
}

// Rebuild operands and record any reblock views inserted for them.
static std::pair<SmallVector<Value>, SmallVector<d2m::ViewLayoutOp>>
createReblockedOperands(d2m::GenericOp thisOp, OpBuilder &builder,
                        ArrayRef<Type> reblockedTypes) {
  SmallVector<Value> reblockedOperands;
  SmallVector<d2m::ViewLayoutOp> operandViews;
  reblockedOperands.reserve(thisOp.getInputsAndOutputs().size());
  operandViews.reserve(thisOp.getInputsAndOutputs().size());

  for (auto [operand, reblockedType] :
       llvm::zip(thisOp.getInputsAndOutputsMutable(), reblockedTypes)) {
    // No work to do.
    if (reblockedType == operand.get().getType()) {
      operandViews.push_back(nullptr);
      reblockedOperands.push_back(operand.get());
      continue;
    }

    d2m::ViewLayoutOp view = builder.create<d2m::ViewLayoutOp>(
        thisOp.getLoc(), reblockedType, operand.get());
    operandViews.push_back(view);
    reblockedOperands.push_back(view.getResult());
  }

  return {std::move(reblockedOperands), std::move(operandViews)};
}

// Recreate the generic op shell around the reblocked operand list.
static d2m::GenericOp
createParallelizedGenericShell(d2m::GenericOp thisOp, OpBuilder &builder,
                               ArrayRef<Value> reblockedOperands,
                               ttcore::GridAttr newGrid,
                               ArrayRef<int64_t> newBlockFactors) {
  const std::size_t numInputs = thisOp.getInputs().size();
  const std::size_t numOutputs = thisOp.getOutputs().size();
  SmallVector<Value> newInputs(reblockedOperands.begin(),
                               reblockedOperands.begin() + numInputs);
  SmallVector<Value> newOutputs(reblockedOperands.begin() + numInputs,
                                reblockedOperands.begin() + numInputs +
                                    numOutputs);

  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(thisOp.getNumResults());
  for (std::size_t resultIndex = 0; resultIndex < thisOp.getNumResults();
       ++resultIndex) {
    newResultTypes.push_back(newOutputs[resultIndex].getType());
  }

  return builder.create<d2m::GenericOp>(
      thisOp.getLoc(), TypeRange(newResultTypes), newInputs, newOutputs,
      thisOp.getAdditionalArgs(), newGrid,
      builder.getI64ArrayAttr(newBlockFactors), thisOp.getIndexingMaps(),
      thisOp.getIteratorTypes(), thisOp.getThreads(),
      thisOp.getScratchInputsAttr(), thisOp.getNumRegions());
}

// Clone one generic region and retarget its block args to reblocked operands.
static Block *cloneParallelizedRegion(d2m::GenericOp thisOp,
                                      d2m::GenericOp newGenericOp,
                                      OpBuilder &builder, Region &oldRegion,
                                      Region &newRegion,
                                      ArrayRef<Value> reblockedOperands) {
  Block &oldBlock = oldRegion.front();
  SmallVector<Type> newArgTypes;
  SmallVector<Location> newArgLocs;
  newArgTypes.reserve(oldBlock.getNumArguments());
  newArgLocs.reserve(oldBlock.getNumArguments());
  for (BlockArgument oldArg : oldBlock.getArguments()) {
    newArgTypes.push_back(oldArg.getType());
    newArgLocs.push_back(oldArg.getLoc());
  }

  Block *newBlock =
      builder.createBlock(&newRegion, newRegion.end(), newArgTypes, newArgLocs);

  IRMapping mapping;
  for (auto [oldVal, newVal] :
       llvm::zip(thisOp.getInputsAndOutputs(), reblockedOperands)) {
    mapping.map(oldVal, newVal);
  }
  for (auto [oldVal, newVal] : llvm::zip(thisOp.getAdditionalArgs(),
                                         newGenericOp.getAdditionalArgs())) {
    mapping.map(oldVal, newVal);
  }
  for (auto [oldArg, newArg] :
       llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
    mapping.map(oldArg, newArg);
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(newBlock);
  for (Operation &op : oldBlock) {
    builder.clone(op, mapping);
  }

  return newBlock;
}

// Use operand_index or remote_load/remote_store binding to find the associated
// operand for a get_cb op.
static Value findAssocOperandForGetCB(d2m::GetCBOp getCbOp) {
  GenericOp genericOp = getCbOp->getParentOfType<GenericOp>();
  if (!genericOp) {
    return Value();
  }

  if (std::optional<int64_t> operandIndex = getCbOp.getOperandIndex()) {
    if (*operandIndex >= 0 && static_cast<size_t>(*operandIndex) <
                                  genericOp.getInputsAndOutputs().size()) {
      return genericOp.getInputsAndOutputs()[*operandIndex];
    }
    return Value();
  }

  Value cb = getCbOp.getResult();
  for (Operation *userOp : cb.getUsers()) {
    if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(userOp)) {
      if (loadOp.isExplicitCBForm() && loadOp.getCb() == cb) {
        return loadOp.getMemref();
      }
    }
    if (auto storeOp = mlir::dyn_cast<RemoteStoreOp>(userOp)) {
      if (storeOp.isExplicitCBForm() && storeOp.getCb() == cb) {
        return storeOp.getMemref();
      }
    }
  }

  return Value();
}

// Repair cloned ops whose result types depend on reblocked operand types.
static void repairParallelizedRegionTypes(Block *newBlock) {
  for (Operation &clonedOp : *newBlock) {
    if (auto allocOp = mlir::dyn_cast<memref::AllocOp>(&clonedOp)) {
      Value associatedOperand = d2m::GenericOp::findAssocOperand(allocOp);
      if (associatedOperand) {
        Type newAllocType = d2m::utils::cloneWithShardShape(associatedOperand,
                                                            allocOp.getType());
        if (newAllocType != allocOp.getType()) {
          auto newMemRefType = mlir::dyn_cast<MemRefType>(newAllocType);
          TT_assert(newMemRefType);
          allocOp.getResult().setType(newMemRefType);
        }
      }
    } else if (auto tensorEmptyOp =
                   mlir::dyn_cast<tensor::EmptyOp>(&clonedOp)) {
      Value associatedOperand = d2m::GenericOp::findAssocOperand(tensorEmptyOp);
      if (associatedOperand) {
        Type newEmptyType = d2m::utils::cloneWithShardShape(
            associatedOperand, tensorEmptyOp.getType());
        if (newEmptyType != tensorEmptyOp.getType()) {
          tensorEmptyOp.getResult().setType(
              cast<RankedTensorType>(newEmptyType));
        }
      }
    } else if (auto getCbOp = mlir::dyn_cast<d2m::GetCBOp>(&clonedOp)) {
      Value associatedOperand = findAssocOperandForGetCB(getCbOp);
      if (associatedOperand) {
        auto oldCbType = mlir::cast<d2m::CBType>(getCbOp.getResult().getType());
        Type newUnderlyingType = d2m::utils::cloneWithShardShape(
            associatedOperand, oldCbType.getUnderlying());
        if (newUnderlyingType != oldCbType.getUnderlying()) {
          getCbOp.getResult().setType(d2m::CBType::get(
              getCbOp.getContext(), mlir::cast<ShapedType>(newUnderlyingType)));
        }
      }
    } else if (auto remoteLoadOp =
                   mlir::dyn_cast<d2m::RemoteLoadOp>(&clonedOp)) {
      if (Value localBuffer = remoteLoadOp.getLocalBuffer()) {
        remoteLoadOp.getResult().setType(localBuffer.getType());
      }
    } else if (auto remoteStoreOp =
                   mlir::dyn_cast<d2m::RemoteStoreOp>(&clonedOp)) {
      if (remoteStoreOp.hasResultForm()) {
        remoteStoreOp.getResult().setType(remoteStoreOp.getMemref().getType());
      }
    } else if (auto dstOp =
                   mlir::dyn_cast<DestinationStyleOpInterface>(&clonedOp)) {
      unsigned numIns = dstOp.getNumDpsInputs();
      unsigned numOuts = clonedOp.getNumResults();
      for (unsigned i = 0; i < numOuts; ++i) {
        clonedOp.getResult(i).setType(
            clonedOp.getOperand(numIns + i).getType());
      }
    }
  }
}

// Build a return view for the rebuilt generic when the caller requests one.
static d2m::ViewLayoutOp createReturnView(d2m::GenericOp thisOp,
                                          d2m::GenericOp newGenericOp,
                                          OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(newGenericOp);
  if (thisOp.getNumResults() > 0) {
    return builder.create<d2m::ViewLayoutOp>(thisOp.getLoc(),
                                             thisOp.getResult(0).getType(),
                                             newGenericOp.getResult(0));
  }
  if (!thisOp.getOutputs().empty()) {
    return builder.create<d2m::ViewLayoutOp>(
        thisOp.getLoc(), thisOp.getOutputs().front().getType(),
        newGenericOp.getOutputs().front());
  }
  return d2m::ViewLayoutOp();
}

// Rewrite the IR:
// - insert operand views
// - create the new generic
// - clone/retype the region ops
// - optionally create the return view.
static FailureOr<d2m::ParallelizedGeneric>
withParallelizationImpl(d2m::GenericOp thisOp, OpBuilder &builder,
                        ArrayRef<Type> reblockedTypes, ttcore::GridAttr newGrid,
                        ArrayRef<int64_t> newBlockFactors,
                        bool generateReturnView) {
  // Reblock operands via explicit view_layout ops.
  auto [reblockedOperands, operandViews] =
      createReblockedOperands(thisOp, builder, reblockedTypes);

  // Reconstruct the GenericOp shell with new attrs + operands.
  auto newGenericOp = createParallelizedGenericShell(
      thisOp, builder, reblockedOperands, newGrid, newBlockFactors);

  // Reconstruct regions, remapping values and repairing type-dependent
  // ops.
  for (unsigned regionIndex = 0; regionIndex < thisOp.getNumRegions();
       ++regionIndex) {
    Region &oldRegion = thisOp.getRegion(regionIndex);
    Region &newRegion = newGenericOp.getRegion(regionIndex);
    if (oldRegion.empty()) {
      continue;
    }

    Block *newBlock = cloneParallelizedRegion(
        thisOp, newGenericOp, builder, oldRegion, newRegion, reblockedOperands);
    repairParallelizedRegionTypes(newBlock);
  }

  // Only return a view into the old generic op result if requested.
  d2m::ViewLayoutOp returnView = nullptr;
  if (generateReturnView) {
    returnView = createReturnView(thisOp, newGenericOp, builder);
    TT_assertv(returnView,
               "withParallelization expects a return view to the original "
               "generic result");
  }
  return d2m::ParallelizedGeneric{std::move(operandViews), newGenericOp,
                                  returnView};
}

// Re-parallelize a GenericOp by updating grid and/or block factors,
// then rebuild operands/region types to keep the op well-typed.
FailureOr<d2m::ParallelizedGeneric> d2m::GenericOp::withParallelization(
    OpBuilder &builder, std::optional<ttcore::GridAttr> newGrid,
    std::optional<ArrayRef<int64_t>> newBlockFactors, bool generateReturnView) {
  TT_assert((newGrid.has_value() || newBlockFactors.has_value()));
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPoint(*this);

  ttcore::GridAttr normalizedGrid = newGrid.value_or(getGrid());
  SmallVector<int64_t> normalizedBlockFactors =
      newBlockFactors ? llvm::to_vector(*newBlockFactors)
                      : getBlockFactorsValue();
  TT_assertv(normalizedBlockFactors.size() == getNumDims(),
             "withParallelization block factor count {} does not match "
             "generic rank {}",
             normalizedBlockFactors.size(), getNumDims());

  auto computeReblockedTypes =
      [&](ArrayRef<int64_t> opGridShape) -> FailureOr<SmallVector<Type>> {
    SmallVector<Type> reblockedTypes;
    reblockedTypes.reserve(getInputsAndOutputs().size());
    for (auto [operandIndex, operand] :
         llvm::enumerate(getInputsAndOutputsMutable())) {
      // Get the new grid shape for the operand.
      FailureOr<SmallVector<int64_t>> reblockedGridShape =
          computeReblockedOperandGridShape(*this, operandIndex, opGridShape,
                                           normalizedBlockFactors);
      if (failed(reblockedGridShape)) {
        this->emitOpError()
            << "withParallelization failed to compute reblocked grid shape "
            << "for operand " << operandIndex;
        return failure();
      }
      auto operandType = mlir::cast<ShapedType>(operand.get().getType());
      reblockedTypes.push_back(
          d2m::utils::reblockShapedType(operandType, *reblockedGridShape));
    }
    return reblockedTypes;
  };

  // Compute the operand reblocked types from the requested newGrid.
  FailureOr<SmallVector<Type>> reblockedTypes =
      computeReblockedTypes(normalizedGrid.getShape());
  if (failed(reblockedTypes)) {
    this->emitOpError()
        << "withParallelization failed to derive reblocked types";
    return failure();
  }

  // If the derived grid shape is different from the requested newGrid,
  // compute the reblocked types again with the adjusted grid.
  const std::size_t numInputs = getInputs().size();
  const std::size_t numOutputs = getOutputs().size();
  if (numOutputs > 0) {
    // derive grid from first output index
    auto [derivedGridShape, _] = getGridAndShardFromShapedType(
        mlir::cast<ShapedType>((*reblockedTypes)[numInputs]));
    if (derivedGridShape.size() == normalizedGrid.getShape().size() &&
        !llvm::equal(derivedGridShape, normalizedGrid.getShape())) {
      normalizedGrid =
          ttcore::GridAttr::get(builder.getContext(), derivedGridShape,
                                normalizedGrid.getVirtToPhysicalMap(),
                                normalizedGrid.getPhysicalToVirtMap());
      reblockedTypes = computeReblockedTypes(normalizedGrid.getShape());
      if (failed(reblockedTypes)) {
        this->emitOpError()
            << "withParallelization failed to derive reblocked types after "
            << "adjusting grid to " << normalizedGrid;
        return failure();
      }
    }
  }

  // Actually perform the IR rewrite.
  return withParallelizationImpl(*this, builder, *reblockedTypes,
                                 normalizedGrid, normalizedBlockFactors,
                                 generateReturnView);
}

/// Returns true if the generic op has non-empty block_factors, indexing_maps,
/// and iterator_types attributes, and a single unified region.
static bool hasBlockingAttributes(d2m::GenericOp genericOp) {
  if (genericOp.getBlockFactors().empty() ||
      genericOp.getIndexingMaps().empty() ||
      genericOp.getIteratorTypes().empty()) {
    return false;
  }

  return genericOp.getNumRegions() == 1 &&
         genericOp.getRegionThreadType(0) == ThreadType::Unified;
}

bool d2m::GenericOp::isImplicitBlockedForm() {
  if (!hasBlockingAttributes(*this)) {
    return false;
  }

  // No affine blocking loops must be present.
  bool hasBlockingLoop = false;
  getRegion(0).walk([&](affine::AffineForOp forOp) {
    if (forOp->hasAttr("d2m.blocking_loop")) {
      hasBlockingLoop = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return !hasBlockingLoop;
}

bool d2m::GenericOp::isAffineBlockedForm() {
  if (!hasBlockingAttributes(*this)) {
    return false;
  }

  // Each block factor dimension must have a corresponding affine.for loop
  // whose upper bound is defined by a get_block_factor() op.
  unsigned numBlockFactors = static_cast<unsigned>(getBlockFactors().size());
  SmallVector<bool> factorUsed(numBlockFactors, false);
  unsigned loopCount = 0;

  getRegion(0).walk([&](affine::AffineForOp forOp) {
    if (!forOp->hasAttr("d2m.blocking_loop")) {
      return;
    }
    ++loopCount;

    auto ubOperands = forOp.getUpperBoundOperands();
    if (ubOperands.size() != 1) {
      return;
    }

    auto getBlockFactorOp =
        dyn_cast_or_null<GetBlockFactorOp>(ubOperands[0].getDefiningOp());
    if (!getBlockFactorOp) {
      return;
    }

    int64_t dim = getBlockFactorOp.getDim();
    if (dim >= 0 && static_cast<unsigned>(dim) < numBlockFactors) {
      factorUsed[static_cast<unsigned>(dim)] = true;
    }
  });

  return loopCount == numBlockFactors &&
         llvm::all_of(factorUsed, [](bool used) { return used; });
}

mlir::SmallVector<int64_t> d2m::GenericOp::getFullBlockFactors() {
  auto maps = getIndexingMapsValue();
  // Priority doesn't matter here, so reverse can be false.
  auto flatInverseMap =
      ttmlir::utils::concatInversePermutationMap(maps, /*reverse=*/false);

  SmallVector<int64_t> flattenedOperandShardShapes;
  for (Value v : getInputsAndOutputs()) {
    auto [_, shardShape] = getGridAndShardFromValue(v);
    flattenedOperandShardShapes.append(shardShape.begin(), shardShape.end());
  }

  auto currentBlockFactors = getBlockFactorsValue();
  auto factorizations = flatInverseMap.compose(flattenedOperandShardShapes);
  TT_assertv(currentBlockFactors.size() == factorizations.size(),
             "full block factor count {} does not match factorization count {}",
             currentBlockFactors.size(), factorizations.size());

  // Multiply back in the current block factors to normalize the result.
  for (std::size_t i = 0; i < currentBlockFactors.size(); ++i) {
    factorizations[i] *= currentBlockFactors[i];
  }

  return factorizations;
}

bool d2m::GenericOp::isOutputOperandIdx(unsigned int operandIndex) {
  return operandIndex >= getOutputs().getBeginOperandIndex() &&
         operandIndex <
             getOutputs().getBeginOperandIndex() + getOutputs().size();
}

mlir::SmallVector<mlir::SmallVector<int64_t>>
d2m::GenericOp::getInputOutputOperandGridShapes() {
  SmallVector<SmallVector<int64_t>> gridShapes;
  gridShapes.reserve(getInputsAndOutputs().size());
  for (auto operand : this->getInputsAndOutputs()) {
    auto [gridShape, _] = getGridAndShardFromValue(operand);
    gridShapes.emplace_back(std::move(gridShape));
  }
  return gridShapes;
}

mlir::SmallVector<mlir::SmallVector<int64_t>>
d2m::GenericOp::getInputOutputOperandShardShapes(bool convertTileToScalar) {
  SmallVector<SmallVector<int64_t>> shardShapes;
  shardShapes.reserve(getInputsAndOutputs().size());

  for (auto operand : this->getInputsAndOutputs()) {
    auto shapedType = mlir::cast<ShapedType>(operand.getType());
    Type elementType;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(shapedType)) {
      elementType = memrefType.getElementType();
    } else {
      elementType = mlir::cast<RankedTensorType>(shapedType).getElementType();
    }

    auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
    auto [_, shardShape] = getGridAndShardFromValue(operand);
    shardShapes.emplace_back(
        (convertTileToScalar && tileType)
            ? tileType.getScalarShape(SmallVector<int64_t>(shardShape))
            : shardShape);
  }

  return shardShapes;
}

mlir::SmallVector<int64_t> d2m::GenericOp::getPhysicalGridShape() {
  if (getGrid().getVirtToPhysicalMap().isEmpty()) {
    return llvm::to_vector(getGrid().getShape());
  }

  // NOTE: This assumes the virtual grid maps to a rectangular physical grid
  // (which does not necessarily have to be the case. For example, a 1x37
  // virtual grid would map to cores [(0, 0), (3, 7)] and [(4, 0), (4, 4)]).
  auto map = getGrid().getVirtToPhysicalMap();
  // Drop d0 from (d0, d1, d2) -> (...) by replacing d0 with 0.
  auto newMap = ttmlir::utils::dropDim(map, 0);
  auto shape =
      llvm::to_vector(ttmlir::utils::evalShape(newMap, getGrid().getShape()));
  return shape;
}

mlir::SmallVector<int64_t> d2m::GenericOp::getLoopBounds() {
  return getBlockFactorsValue();
}

mlir::SmallVector<int64_t>
d2m::GenericOp::getParticipatingLoopDims(int64_t operandIndex) {
  TT_debugv(!isExplicitDatamovementForm(),
            "getParticipatingLoopDims should not be called on explicit "
            "data movement form operations.");
  AffineMap indexingMap = getIndexingMap(operandIndex);
  auto dimExprs =
      llvm::make_filter_range(indexingMap.getResults(), [](AffineExpr expr) {
        return mlir::isa<AffineDimExpr>(expr);
      });
  return llvm::map_to_vector(dimExprs, [](AffineExpr expr) {
    return static_cast<int64_t>(mlir::cast<AffineDimExpr>(expr).getPosition());
  });
}

mlir::SmallVector<int64_t>
d2m::GenericOp::getNonParticipatingLoopDims(int64_t operandIndex) {
  TT_debugv(!isExplicitDatamovementForm(),
            "getNonParticipatingLoopDims should not be called on explicit "
            "data movement form operations.");
  AffineMap indexingMap = getIndexingMap(operandIndex);
  SmallVector<int64_t> participatingDims =
      getParticipatingLoopDims(operandIndex);
  llvm::BitVector nonParticipatingDims(indexingMap.getNumDims(), true);
  llvm::for_each(participatingDims, [&nonParticipatingDims](int64_t dim) {
    nonParticipatingDims.reset(dim);
  });
  return llvm::SmallVector<int64_t>(nonParticipatingDims.set_bits());
}

std::optional<SmallVector<int64_t>> d2m::GenericOp::computeGridDimConstraints(
    std::function<bool(ttcore::MetalLayoutAttr, bool)> operandFilterPredicate) {
  auto indexingMaps = getIndexingMapsValue();
  auto shapes = getInputOutputOperandGridShapes();

  // Filter shape/map pairs that form constraints based on the operand filter
  // predicate.
  SmallVector<SmallVector<int64_t>> filteredShapes;
  SmallVector<AffineMap> filteredIndexingMaps;
  for (auto [operandIdx, operand] : llvm::enumerate(getInputsAndOutputs())) {
    auto metalTensor = mlir::cast<mlir::RankedTensorType>(operand.getType());
    auto baseMetalLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(metalTensor.getEncoding());
    bool isOutputOperand = operandIdx >= getOutputs().getBeginOperandIndex();

    if (operandFilterPredicate(baseMetalLayout, isOutputOperand)) {
      filteredShapes.push_back(shapes[operandIdx]);
      filteredIndexingMaps.push_back(indexingMaps[operandIdx]);
    }
  }

  if (filteredIndexingMaps.empty()) {
    return SmallVector<int64_t>(indexingMaps.front().getNumDims(), 0);
  }

  return d2m::utils::computeDimConstraints(filteredIndexingMaps,
                                           filteredShapes);
}

void d2m::GenericOp::getAsmBlockArgumentNames(
    Region &region, function_ref<void(Value, StringRef)> setNameFn) {
  int semIndex = 0;
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::isa<SemaphoreType>(arg.getType())) {
      setNameFn(arg, "sem" + std::to_string(semIndex++));
    }
  }
}

void d2m::GenericOp::getAsmBlockNames(
    function_ref<void(Block *, StringRef)> setNameFn) {
  std::array<int, getMaxEnumValForThreadType() + 1> threadTypeCounts{};
  for (Region &region : getRegions()) {
    auto type = getRegionThreadType(region.getRegionNumber());
    setNameFn(&region.front(),
              stringifyEnum(type).str() +
                  Twine(threadTypeCounts[llvm::to_underlying(type)]++).str());
  }
}

mlir::LogicalResult d2m::GenericOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  if (getNumResults() == 0) {
    return failure();
  }

  assert(getNumResults() == 1 && "GenericOp should have exactly one result");
  assert(getOutputs().size() == 1 &&
         "GenericOp should have exactly one output");

  if (!mlir::isa<mlir::RankedTensorType>(getResult(0).getType())) {
    return failure();
  }
  mlir::SmallVector<mlir::Value> bufferInputs;
  bufferInputs.reserve(getInputs().size());
  for (auto input : getInputs()) {
    auto maybeValue = bufferization::getBuffer(rewriter, input, options, state);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferInputs.push_back(*maybeValue);
  }
  mlir::SmallVector<mlir::Value> bufferOutputs;
  bufferOutputs.reserve(getOutputs().size());
  for (auto output : getOutputs()) {
    auto maybeValue =
        bufferization::getBuffer(rewriter, output, options, state);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferOutputs.push_back(*maybeValue);
  }
  auto bufferGeneric = rewriter.create<d2m::GenericOp>(
      getLoc(), ValueRange(), bufferInputs, bufferOutputs, getAdditionalArgs(),
      getGrid(), getBlockFactors(), getIndexingMaps(), getIteratorTypes(),
      getThreads(), getScratchInputsAttr(), getNumRegions());
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    region.takeBody(getRegion(region.getRegionNumber()));
  }

  // Bufferize get_cb ops: convert from cb<tensor<...>> to cb<memref<...>>.
  for (Region &region : bufferGeneric.getRegions()) {
    region.walk([&](d2m::GetCBOp getCbOp) {
      auto cbType = mlir::dyn_cast<CBType>(getCbOp.getResult().getType());
      if (!cbType || !cbType.hasTensorType()) {
        return;
      }
      auto bufferType =
          cbType.getBufferType(options, [&]() { return this->emitError(); });
      if (failed(bufferType)) {
        return;
      }
      getCbOp.getResult().setType(mlir::cast<CBType>(*bufferType));
    });
  }

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     bufferOutputs);
  return success();
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::GenericOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &options,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  // Handle CB types - delegate to CBType::getBufferType.
  if (auto cbType = mlir::dyn_cast<d2m::CBType>(value.getType())) {
    return cbType.getBufferType(options, [&]() { return this->emitError(); });
  }

  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  // tensor.empty ops inside the region (replacing old CB block args) get L1
  // memory space.
  if (!tensorType.getEncoding()) {
    return mlir::cast<bufferization::BufferLikeType>(MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(), nullptr,
        ttcore::MemorySpaceAttr::get(tensorType.getContext(),
                                     ttcore::MemorySpace::DeviceL1)));
  }
  return ttcore::getBufferType(tensorType, /*isView=*/false);
}

bool d2m::GenericOp::hasCompatibleBlocking(GenericOp b) {
  return this->getGrid() == b.getGrid() &&
         this->getBlockFactors() == b.getBlockFactors();
}

/// Returns true if op or any of the operations nested within its regions have
/// the D2MSkipOpEltwiseFusionTrait.
bool d2m::GenericOp::hasSkipOpEltwiseFusionTrait() {
  bool skipFusion = false;

  walk([&](Operation *op) {
    skipFusion |= op->hasTrait<D2MSkipOpEltwiseFusionTrait>();
    return skipFusion ? WalkResult::interrupt() : WalkResult::advance();
  });

  return skipFusion;
}

bool d2m::GenericOp::hasReduction() {
  SmallVector<Attribute> iters = llvm::to_vector(getIteratorTypes());
  return llvm::any_of(iters, [](Attribute it) {
    auto itAttr = cast<mlir::tt::ttcore::IteratorTypeAttr>(it);
    return itAttr.getValue() == mlir::tt::ttcore::IteratorType::Reduction;
  });
}

bool d2m::GenericOp::hasMultiUseInputOperand() {
  for (OpOperand *input : getDpsInputOperands()) {
    // Count users that are outside this generic operation
    unsigned numExternalUsers = 0;
    for (auto *user : input->get().getUsers()) {
      // Skip if the user is this operation itself
      if (user == this->getOperation()) {
        continue;
      }
      // Skip users that are nested inside this generic operation's regions
      if (this->getOperation()->isProperAncestor(user)) {
        continue;
      }
      // This is an external user
      numExternalUsers++;
    }

    if (numExternalUsers > 1) {
      return true;
    }
  }
  return false;
}

bool d2m::GenericOp::isAllParallel() {
  return llvm::all_of(getIteratorTypes(), [](Attribute it) {
    auto itAttr = mlir::cast<mlir::tt::ttcore::IteratorTypeAttr>(it);
    return itAttr.getValue() == mlir::tt::ttcore::IteratorType::Parallel;
  });
}

bool d2m::GenericOp::hasComputeOpsInRegion(unsigned regionIndex) {
  if (regionIndex >= getNumRegions()) {
    return false;
  }

  bool hasCompute = false;
  getRegion(regionIndex).walk([&](Operation *op) {
    if (op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
      hasCompute = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return hasCompute;
}

bool d2m::GenericOp::isNontriviallyEltwiseFused() {
  // if all parallel --> no reductions --> eltwise only
  if (!this->isAllParallel()) {
    return false;
  }

  // doesn't contain skippable eltwise ops (i.e. went through eltwise fusion)
  if (this->hasSkipOpEltwiseFusionTrait()) {
    return false;
  }

  // op is ternary or higher degree
  if (!(this->getNumOperands() > 3)) {
    return false;
  }

  return true;
}

Value d2m::GenericOp::findAssocOperand(memref::AllocOp allocOp) {
  // First check that the memref.alloc is within a generic op
  GenericOp genericOp = allocOp->getParentOfType<GenericOp>();
  if (!genericOp) {
    return Value();
  }

  // The alloc result should be used directly by a remote_load or remote_store
  // op as its localBuffer parameter. The associated operand is the memref
  // parameter of that remote_load/remote_store op.
  Value allocResult = allocOp.getResult();
  for (Operation *userOp : allocResult.getUsers()) {
    if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(userOp)) {
      Value localBuffer = loadOp.getLocalBuffer();
      if (localBuffer && localBuffer == allocResult) {
        return loadOp.getMemref();
      }
    }
    if (auto storeOp = mlir::dyn_cast<RemoteStoreOp>(userOp)) {
      Value localBuffer = storeOp.getLocalBuffer();
      if (localBuffer && localBuffer == allocResult) {
        return storeOp.getMemref();
      }
    }
  }

  return Value();
}

Value d2m::GenericOp::findAssocOperand(mlir::tensor::EmptyOp emptyOp) {
  // First check that the tensor.empty is within a generic op
  GenericOp genericOp = emptyOp->getParentOfType<GenericOp>();
  if (!genericOp) {
    return Value();
  }

  // Assert that the parent GenericOp has a single output
  int64_t numOutputs = static_cast<int64_t>(genericOp.getOutputs().size());
  TT_assertv(numOutputs == 1,
             "tensor.empty within generic op with multiple outputs - "
             "cannot determine associated operand");

  // By default, assume the associated operand is the sole output operand
  Value associatedOperand = genericOp.getOutputs()[0];

  // If one of the uses is a RemoteLoadOp or RemoteStoreOp, the associated
  // operand is the memref of that load/store op
  Value emptyResult = emptyOp.getResult();
  for (Operation *userOp : emptyResult.getUsers()) {
    if (auto loadOp = mlir::dyn_cast<RemoteLoadOp>(userOp)) {
      Value localBuffer = loadOp.getLocalBuffer();
      if (localBuffer && localBuffer == emptyResult) {
        associatedOperand = loadOp.getMemref();
        break;
      }
    }
    if (auto storeOp = mlir::dyn_cast<RemoteStoreOp>(userOp)) {
      associatedOperand = storeOp.getMemref();
      break;
    }
  }

  return associatedOperand;
}

Value d2m::GenericOp::findAssocCBByOperandIndex(Operation *op,
                                                unsigned operandIndex) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find the generic op's thread region that contains this operation
  Region *genericRegion = nullptr;
  if (generic.getNumRegions() == 1) {
    genericRegion = &generic.getRegion(0);
  } else {
    genericRegion = ttmlir::utils::getRegionWithParentOfType<GenericOp>(op);
  }

  if (!genericRegion || genericRegion->empty()) {
    return Value();
  }

  return getOperandAlloc(*genericRegion, operandIndex);
}

Value d2m::GenericOp::findAssocCBByOperand(Operation *op, Value operand) {
  GenericOp generic = op->getParentOfType<GenericOp>();
  if (!generic) {
    return Value();
  }

  // Find which operand index this corresponds to.
  unsigned operandIndex = UINT_MAX;
  for (unsigned i = 0; i < generic->getNumOperands(); ++i) {
    if (generic->getOperand(i) == operand) {
      operandIndex = i;
      break;
    }
  }

  if (operandIndex == UINT_MAX) {
    return Value();
  }

  return findAssocCBByOperandIndex(op, operandIndex);
}

Value d2m::GenericOp::getOperandAlloc(Region &region, unsigned operandIndex) {
  if (region.empty()) {
    return Value();
  }

  // Walk the region looking for tensor.empty/memref.alloc/d2m.get_cb ops,
  // stepping into blocking loops only. Do NOT walk into compute loops
  // (scf.for without d2m.blocking_loop) — allocs inside those are local
  // working buffers, not operand allocations.
  //
  // d2m.get_cb ops are matched by operand_index when present, otherwise by
  // their remote_load/remote_store binding. tensor.empty/memref.alloc ops are
  // matched by positional order.
  Value result;
  unsigned idx = 0;
  std::function<void(Block &)> scanBlock = [&](Block &block) {
    for (Operation &op : block) {
      if (result) {
        return;
      }
      if (auto getCbOp = mlir::dyn_cast<d2m::GetCBOp>(&op)) {
        if (std::optional<int64_t> assocOperandIndex =
                getCbOp.getOperandIndex()) {
          if (*assocOperandIndex >= 0 &&
              static_cast<unsigned>(*assocOperandIndex) == operandIndex) {
            result = getCbOp.getResult();
            return;
          }
        } else if (Value associatedOperand =
                       findAssocOperandForGetCB(getCbOp)) {
          GenericOp generic = getCbOp->getParentOfType<GenericOp>();
          if (generic && generic.getOperandIndex(associatedOperand) ==
                             static_cast<int64_t>(operandIndex)) {
            result = getCbOp.getResult();
            return;
          }
        }
      } else if (mlir::isa<mlir::tensor::EmptyOp, memref::AllocOp>(&op)) {
        if (idx == operandIndex) {
          result = op.getResult(0);
          return;
        }
        ++idx;
      } else if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(&op)) {
        if (forOp->hasAttr("d2m.blocking_loop")) {
          scanBlock(*forOp.getBody());
        }
      } else if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
        if (forOp->hasAttr("d2m.blocking_loop")) {
          scanBlock(*forOp.getBody());
        }
      }
    }
  };
  scanBlock(region.front());

  return result;
}

bool d2m::SpatialOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an input, it is bufferized to a memory read.
  return isDpsInput(&operand);
}

bool d2m::SpatialOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an output, it is bufferized to a memory write.
  return isDpsInit(&operand);
}

mlir::LogicalResult d2m::SpatialOp::bufferize(
    // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {

  for (mlir::OpResult result : getResults()) {
    if (!mlir::isa<mlir::RankedTensorType>(result.getType())) {
      return failure();
    }
  }
  if (getNumResults() != 0 && getNumResults() != getOutputs().size()) {
    return failure();
  }
  mlir::SmallVector<mlir::Value> bufferInputs;
  bufferInputs.reserve(getInputs().size());
  for (auto input : getInputs()) {
    auto maybeValue = bufferization::getBuffer(rewriter, input, options, state);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferInputs.push_back(*maybeValue);
  }
  mlir::SmallVector<mlir::Value> bufferOutputs;
  bufferOutputs.reserve(getOutputs().size());
  for (auto output : getOutputs()) {
    auto maybeValue =
        bufferization::getBuffer(rewriter, output, options, state);
    if (failed(maybeValue)) {
      return maybeValue;
    }
    bufferOutputs.push_back(*maybeValue);
  }
  auto bufferSpatial = rewriter.create<d2m::SpatialOp>(
      getLoc(), ValueRange(), bufferInputs, bufferOutputs, getGridRanges(),
      getNumRegions());
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
  for (mlir::Region &region : bufferSpatial.getRegions()) {
    region.takeBody(getRegion(region.getRegionNumber()));
  }

  // One-shot bufferization uses BottomUp: inner GenericOps are bufferized
  // first, so region bodies already use buffer SSA. We only replace the
  // boundary (operands/results) with buffers and move the body.
  if (getNumResults() == 0) {
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this, {});
  } else {
    mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                       bufferOutputs);
  }
  return success();
}

mlir::bufferization::AliasingValueList
d2m::SpatialOp::getAliasingValues(mlir::OpOperand &,
                                  const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}
mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::SpatialOp::getBufferType(mlir::Value value,
                              const mlir::bufferization::BufferizationOptions &,
                              const mlir::bufferization::BufferizationState &,
                              ::llvm::SmallVector<mlir::Value> &) {
  // SpatialOp operands/results are tensors from outside or results; no block
  // arguments on SpatialOp's regions.
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  return ttcore::getBufferType(tensorType, /*isView=*/false);
}

bool d2m::SpatialOp::isWritable(mlir::Value value,
                                const mlir::bufferization::AnalysisState &) {
  // SpatialOp has no region block arguments; only outputs (OpResults) are
  // writable.
  return mlir::isa<mlir::OpResult>(value);
}

bool d2m::SpatialOp::hasTensorSemantics() {
  auto isaTensor = [](Type t) { return isa<bufferization::TensorLikeType>(t); };
  if (any_of(getResultTypes(), isaTensor)) {
    return true;
  }
  return any_of(getOperandTypes(), isaTensor);
}

void d2m::SpatialOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}
} // namespace mlir::tt::d2m
