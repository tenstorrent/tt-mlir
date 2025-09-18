// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

namespace mlir::tt::d2m {
// Convert TensorType + MetalLayout into a memref including a
// Shard/View/HostAttr.
MemRefType getBufferType(Type type, bool isView,
                         std::optional<ttcore::MetalLayoutAttr> hostInfo) {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(type);
  MLIRContext *ctx = tensorType.getContext();

  if (!tensorType.getEncoding()) {
    // Calculate host layout and attach, for I/O with (potentially) unaligned
    // host memref.
    ttcore::HostLayoutAttr hostLayout = nullptr;
    if (hostInfo.has_value()) {
      hostLayout = ttcore::HostLayoutAttr::get(ctx, tensorType.getShape(),
                                               hostInfo->getHostStride(),
                                               hostInfo->getHostVolume());
    }
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                           hostLayout);
  }

  auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());

  auto gridShape = layout.getGridShape(tensorType);
  auto shardShape = layout.getShardShape(tensorType);
  SmallVector<int64_t> fullMemrefShape;
  fullMemrefShape.append(gridShape.begin(), gridShape.end());
  fullMemrefShape.append(shardShape.begin(), shardShape.end());

  MemRefLayoutAttrInterface layoutAttr;
  if (isView) {
    const unsigned rank = static_cast<unsigned>(fullMemrefShape.size());
    mlir::AffineMap map = layout.getIndexAffineMap();
    assert(map && map.getNumResults() == rank && map.getNumDims() == rank &&
           "expected tensor encoding to provide a concrete index_map for view");
    layoutAttr = ttcore::ViewLayoutAttr::get(ctx, map);
  } else {
    SmallVector<int64_t> shardStride = layout.getShardStride(tensorType);
    layoutAttr = ttcore::ShardLayoutAttr::get(ctx, shardStride, /*buffered=*/1);
  }

  return MemRefType::get(
      fullMemrefShape, tensorType.getElementType(), layoutAttr,
      ttcore::MemorySpaceAttr::get(ctx, layout.getMemorySpace()));
}

void d2m::GenericOp::getAsmBlockArgumentNames(
    Region &region, function_ref<void(Value, StringRef)> setNameFn) {
  int cbIndex = 0;
  int semIndex = 0;
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::isa<MemRefType>(arg.getType())) {
      setNameFn(arg, "cb" + std::to_string(cbIndex++));
    } else if (mlir::isa<RankedTensorType>(arg.getType())) {
      setNameFn(arg, "t" + std::to_string(cbIndex++));
    } else if (mlir::isa<SemaphoreType>(arg.getType())) {
      setNameFn(arg, "sem" + std::to_string(semIndex++));
    } else {
      llvm_unreachable("Unexpected region argument type");
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

void d2m::GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}

SmallVector<AffineMap> d2m::GenericOp::getIndexingMapsValue() {
  return llvm::to_vector(llvm::map_range(getIndexingMaps(), [](Attribute a) {
    return mlir::cast<AffineMapAttr>(a).getValue();
  }));
}

SmallVector<int64_t> d2m::GenericOp::getBlockFactorsValue() {
  return llvm::map_to_vector(getBlockFactors(), [](Attribute a) {
    return mlir::cast<IntegerAttr>(a).getInt();
  });
}

SmallVector<mlir::tt::ttcore::IteratorType>
d2m::GenericOp::getIteratorTypesValue() {
  return llvm::to_vector(llvm::map_range(getIteratorTypes(), [](Attribute a) {
    return mlir::cast<mlir::tt::ttcore::IteratorTypeAttr>(a).getValue();
  }));
}

mlir::tt::ttcore::DeviceAttr d2m::GenericOp::getDevice() {
  return ttcore::lookupDevice(*this);
}

SmallVector<SmallVector<int64_t>> d2m::GenericOp::getOperandGridShapes() {
  SmallVector<SmallVector<int64_t>> gridShapes;
  gridShapes.reserve(getOperands().size());

  for (auto operand : this->getOperands()) {
    auto shapedType = mlir::cast<ShapedType>(operand.getType());
    mlir::tt::ttcore::DeviceLayoutInterface layout;

    if (auto memrefType = mlir::dyn_cast<MemRefType>(shapedType)) {
      layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
          memrefType.getLayout());
    } else {
      auto tensorType = mlir::cast<RankedTensorType>(shapedType);
      layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
          tensorType.getEncoding());
    }

    gridShapes.emplace_back(layout.getGridShape(shapedType));
  }

  return gridShapes;
}

SmallVector<int64_t>
d2m::GenericOp::getParticipatingLoopDims(int64_t operandIndex) {
  AffineMap indexingMap =
      mlir::cast<AffineMapAttr>(getIndexingMaps()[operandIndex]).getValue();
  auto dimExprs =
      llvm::make_filter_range(indexingMap.getResults(), [](AffineExpr expr) {
        return mlir::isa<AffineDimExpr>(expr);
      });
  return llvm::map_to_vector(dimExprs, [](AffineExpr expr) {
    return static_cast<int64_t>(mlir::cast<AffineDimExpr>(expr).getPosition());
  });
}

SmallVector<int64_t>
d2m::GenericOp::getNonParticipatingLoopDims(int64_t operandIndex) {
  AffineMap indexingMap =
      mlir::cast<AffineMapAttr>(getIndexingMaps()[operandIndex]).getValue();
  SmallVector<int64_t> participatingDims =
      getParticipatingLoopDims(operandIndex);
  llvm::BitVector nonParticipatingDims(indexingMap.getNumDims(), true);
  llvm::for_each(participatingDims, [&nonParticipatingDims](int64_t dim) {
    nonParticipatingDims.reset(dim);
  });
  return llvm::SmallVector<int64_t>(nonParticipatingDims.set_bits());
}

SmallVector<int64_t> d2m::GenericOp::getLoopBounds() {
  assert(!getIndexingMaps().empty() && "GenericOp must be pre-loop generated "
                                       "with indexing maps to use this method");
  assert(getOutputs().size() == 1);

  SmallVector<AffineMap> affineMaps = getIndexingMapsValue();
  SmallVector<AffineMap> affineMapsReversed =
      llvm::to_vector(llvm::reverse(affineMaps));
  AffineMap concat = concatAffineMaps(affineMapsReversed, getContext());
  AffineMap inverse = inversePermutation(concat);

  SmallVector<SmallVector<int64_t>> operandGridShapes = getOperandGridShapes();
  SmallVector<int64_t> flattenedGridShapes(
      ttmlir::utils::flatten(llvm::reverse(operandGridShapes)));

  ArrayRef<int64_t> computeGrid = getGrid().getShape();
  for (size_t i = 0; i < computeGrid.size(); ++i) {
    assert(flattenedGridShapes[i] % computeGrid[i] == 0 &&
           "Output grid shape must be divisible by compute grid shape");
    flattenedGridShapes[i] /= computeGrid[i];
  }

  return inverse.compose(flattenedGridShapes);
}

namespace {
static mlir::tt::ttcore::MetalLayoutAttr
createDefaultLayout(mlir::MLIRContext *ctx,
                    mlir::ArrayRef<int64_t> workerGridShape,
                    mlir::RankedTensorType tensorType) {
  llvm::SmallVector<int64_t> logicalShape(tensorType.getShape());

  // TODO (#4820): Remove this during cleanup.
  SmallVector<int64_t> squareGridShape =
      d2m::utils::getSquareTargetGrid(workerGridShape);

  return mlir::tt::ttcore::MetalLayoutAttr::get(
      ctx, logicalShape, squareGridShape, mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::System);
}
} // namespace

mlir::tt::ttcore::MetalLayoutAttr d2m::ToLayoutOp::getOrCreateInputLayout() {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(getInput().getType());
  auto inputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
      tensorType.getEncoding());
  if (inputLayout) {
    return inputLayout;
  }

  ArrayRef<int64_t> workerGridShape =
      ttcore::lookupDevice(*this).getWorkerGrid().getShape();

  return createDefaultLayout(getContext(), workerGridShape, tensorType);
}

mlir::tt::ttcore::MetalLayoutAttr d2m::ToLayoutOp::getOrCreateOutputLayout() {
  auto tensorType = mlir::cast<mlir::RankedTensorType>(getOutput().getType());
  auto outputLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
      tensorType.getEncoding());
  if (outputLayout) {
    return outputLayout;
  }

  ArrayRef<int64_t> workerGridShape =
      ttcore::lookupDevice(*this).getWorkerGrid().getShape();

  return createDefaultLayout(getContext(), workerGridShape, tensorType);
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

  ::llvm::SmallVector<mlir::Value> invocationStack;
  mlir::bufferization::replaceOpWithNewBufferizedOp<memref::AllocOp>(
      rewriter, *this,
      mlir::cast<MemRefType>(
          *getBufferType(getResult(), options, state, invocationStack)));
  return mlir::success();
}

mlir::bufferization::AliasingValueList
d2m::EmptyOp::getAliasingValues(mlir::OpOperand &,
                                const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType>
d2m::EmptyOp::getBufferType(mlir::Value value,
                            const mlir::bufferization::BufferizationOptions &,
                            const mlir::bufferization::BufferizationState &,
                            ::llvm::SmallVector<mlir::Value> &) {
  return d2m::getBufferType(value.getType(), /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// ToLayoutOp
//===----------------------------------------------------------------------===//

struct ToLayoutFoldRedundantPattern : public OpRewritePattern<ToLayoutOp> {
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  ToLayoutFoldRedundantPattern(MLIRContext *context)
      : OpRewritePattern<ToLayoutOp>(context) {
    setDebugName("ttir.ToLayoutFoldRedundantPattern");
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    ToLayoutOp producerLayoutOp = op.getInput().getDefiningOp<ToLayoutOp>();
    if (!producerLayoutOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ToLayoutOp>(op, producerLayoutOp.getInput(),
                                            op.getOutput());
    return success();
  }
};

static ::mlir::LogicalResult
verifyLayoutOp(mlir::Operation *op, mlir::Type inputTensorOrMemrefTy,
               mlir::Type outputTensorOrMemrefTy, bool allowFormatChange,
               bool allowMemorySpaceChange, bool checkMemrefRank = false,
               bool checkMemrefGridShardForm = false,
               bool checkMemrefShardShape = false) {
  if (mlir::RankedTensorType inputTy =
          mlir::dyn_cast<mlir::RankedTensorType>(inputTensorOrMemrefTy)) {
    mlir::RankedTensorType outputTy =
        mlir::dyn_cast<mlir::RankedTensorType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    auto inputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            inputTy.getEncoding());
    auto outputLayout =
        mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
            outputTy.getEncoding());

    if (!inputLayout || !outputLayout) {
      // If the input/output tensor does not have a layout, we can early exit.
      return mlir::success();
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (isFormatChange && !allowFormatChange) {
      return op->emitOpError(
          "Input and output tensor element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output layout memory spaces must be the same");
    }
    return mlir::success();
  }

  if (mlir::MemRefType inputTy =
          mlir::dyn_cast<mlir::MemRefType>(inputTensorOrMemrefTy)) {
    mlir::MemRefType outputTy =
        mlir::dyn_cast<mlir::MemRefType>(outputTensorOrMemrefTy);
    if (!outputTy) {
      return op->emitOpError("Input and output types must be the same");
    }

    const bool isFormatChange =
        inputTy.getElementType() != outputTy.getElementType();
    if (!allowFormatChange && isFormatChange) {
      return op->emitOpError(
          "Input and output layout element types must be the same");
    }

    const bool isMemorySpaceChange =
        inputTy.getMemorySpace() != outputTy.getMemorySpace();
    if (!allowMemorySpaceChange && isMemorySpaceChange) {
      return op->emitOpError(
          "Input and output memref memory spaces must be the same");
    }

    const bool sameRank = inputTy.getRank() == outputTy.getRank();
    if (checkMemrefRank && !sameRank) {
      return op->emitOpError("Input and output memref ranks must be the same");
    }

    auto inputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            inputTy.getLayout());
    if (checkMemrefGridShardForm && !inputDeviceLayout) {
      return op->emitOpError(
          "input memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    auto outputDeviceLayout =
        mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
            outputTy.getLayout());
    if (checkMemrefGridShardForm && !outputDeviceLayout) {
      return op->emitOpError(
          "output memref must have device layout, i.e. have even rank, grid "
          "shape followed by shard shape of equal rank, e.g. GxGxSxS");
    }

    return mlir::success();
  }

  return op->emitOpError("Unsupported input type for view");
}

// ToLayoutOp verification
::mlir::LogicalResult ToLayoutOp::verify() {
  return verifyLayoutOp(*this, getInput().getType(), getOutput().getType(),
                        /*allowFormatChange*/ true,
                        /*allowMemorySpaceChange*/ true);
}

// ToLayoutOp utility methods
ToLayoutOp::CompoundComponents ToLayoutOp::compoundComponents() {
  CompoundComponents components;

  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  TT_assertv(mlir::isa<mlir::RankedTensorType>(inputType),
             "ToLayoutOp::compoundComponents() is only supported on tensors.");

  auto inputTensor = mlir::cast<mlir::RankedTensorType>(inputType);
  auto outputTensor = mlir::cast<mlir::RankedTensorType>(outputType);

  const bool hasInputLayout = inputTensor.getEncoding() != nullptr;
  const bool hasOutputLayout = outputTensor.getEncoding() != nullptr;

  // Layout versus no layout special case.
  if (hasInputLayout != hasOutputLayout) {
    // Always treat this as purely a host <-> device transition.
    components.isMemorySpaceChange = true;
    components.isGridChange = false;
    components.isFormatChange =
        inputTensor.getElementType() != outputTensor.getElementType();
    components.isLayoutChange = false;
    return components;
  }

  // Both lack layouts special case--purely host-side operation.
  if (!hasInputLayout && !hasOutputLayout) {
    components.isMemorySpaceChange = false;
    components.isGridChange = false;
    components.isLayoutChange = false;
    components.isFormatChange =
        inputTensor.getElementType() != outputTensor.getElementType();
    return components;
  }

  // Both have layouts--do a full comparison.
  ttcore::MetalLayoutAttr inputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(inputTensor.getEncoding());
  ttcore::MetalLayoutAttr outputLayout =
      mlir::cast<ttcore::MetalLayoutAttr>(outputTensor.getEncoding());

  components.isMemorySpaceChange =
      inputLayout.getMemorySpace() != outputLayout.getMemorySpace();

  auto inputGrid = inputLayout.getGridShape(inputTensor);
  auto outputGrid = outputLayout.getGridShape(outputTensor);
  components.isGridChange = inputGrid != outputGrid;

  components.isFormatChange =
      inputTensor.getElementType() != outputTensor.getElementType();

  // Check layout (collapsed intervals and alignments).
  components.isLayoutChange =
      inputLayout.getNormalizedIntervals() !=
          outputLayout.getNormalizedIntervals() ||
      inputLayout.getDimAlignments() != outputLayout.getDimAlignments();

  return components;
}

mlir::LogicalResult
ToLayoutOp::fold(FoldAdaptor,
                 llvm::SmallVectorImpl<::mlir::OpFoldResult> &results) {
  mlir::RankedTensorType inputType =
      dyn_cast<mlir::RankedTensorType>(getInput().getType());
  mlir::RankedTensorType outputType =
      dyn_cast<mlir::RankedTensorType>(getOutput().getType());
  if (inputType && outputType && inputType == outputType) {
    results.push_back(getInput());
    return mlir::success();
  }
  return mlir::failure();
}

bool ToLayoutOp::isHostToDevice() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
      nullptr;
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
      nullptr;
  return hostInput && !hostOutput;
}

bool ToLayoutOp::isDeviceToHost() {
  const bool hostInput =
      mlir::cast<mlir::RankedTensorType>(getInput().getType()).getEncoding() ==
      nullptr;
  const bool hostOutput =
      mlir::cast<mlir::RankedTensorType>(getOutput().getType()).getEncoding() ==
      nullptr;
  return !hostInput && hostOutput;
}

void ToLayoutOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *context) {
  // Fold into d2m.empty w/ desired layout
  patterns.add(+[](ToLayoutOp op, mlir::PatternRewriter &rewriter) {
    EmptyOp emptyOp = op.getInput().getDefiningOp<EmptyOp>();
    if (!emptyOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType());
    return success();
  });

  patterns.add(std::make_unique<ToLayoutFoldRedundantPattern>(context));
}

bool mlir::tt::d2m::ToLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 0; // Input operand
}

bool mlir::tt::d2m::ToLayoutOp::bufferizesToMemoryWrite(
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

  // For unaligned H2D, copy the unaligned host tensor to an aligned & padded
  // bounce buffer, then write to the device.
  if (isHostToDevice()) {
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
  }

  auto toLayoutOp =
      rewriter.create<ToLayoutOp>(getLoc(), TypeRange(), *maybeInput,
                                  *maybeOutput, getLayout().value_or(nullptr));

  // For unaligned D2H, read the device tensor to an aligned & padded bounce
  // buffer, then do strided memcpy to copy the data into the unaligned tensor.
  if (isDeviceToHost()) {
    llvm::SmallVector<mlir::Value> invocationStack;
    MemRefType alignedHostMemref = mlir::cast<MemRefType>(
        *getBufferType(getOutput(), options, state, invocationStack));

    if (mlir::cast<ttcore::HostLayoutAttr>(alignedHostMemref.getLayout())
            .isPadded()) {
      rewriter.setInsertionPoint(toLayoutOp);
      auto alignedHostTensor =
          rewriter.create<memref::AllocOp>(getLoc(), alignedHostMemref);

      rewriter.setInsertionPointAfter(toLayoutOp);
      rewriter.create<memref::CopyOp>(getLoc(), alignedHostTensor,
                                      *maybeOutput);
      toLayoutOp.getOutputMutable().assign(alignedHostTensor);
    }
  }

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

mlir::FailureOr<mlir::BaseMemRefType>
ToLayoutOp::getBufferType(mlir::Value value,
                          const mlir::bufferization::BufferizationOptions &,
                          const mlir::bufferization::BufferizationState &,
                          ::llvm::SmallVector<mlir::Value> &) {
  return d2m::getBufferType(value.getType(), /*isView=*/false, getLayout());
}

//===----------------------------------------------------------------------===//
// GenericOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

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
      getLoc(), ValueRange(), bufferInputs, bufferOutputs, getGrid(),
      getBlockFactors(), getIndexingMaps(), getIteratorTypes(), getThreads(),
      getNumRegions());
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    region.takeBody(getRegion(region.getRegionNumber()));
  }

  // Bufferize region block arguments.
  ::llvm::SmallVector<mlir::Value> invocationStack;
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    mlir::Block &block = region.front();
    for (unsigned argNumber = 0; argNumber < block.getNumArguments();
         ++argNumber) {
      mlir::BlockArgument oldArg = block.getArgument(argNumber);
      if (!mlir::isa<mlir::RankedTensorType>(oldArg.getType())) {
        continue;
      }
      auto newArgType = getBufferType(oldArg, options, state, invocationStack);
      mlir::BlockArgument newArg =
          block.insertArgument(argNumber, *newArgType, oldArg.getLoc());
      auto toTensor = rewriter.create<bufferization::ToTensorOp>(
          bufferGeneric.getLoc(), oldArg.getType(), newArg);
      oldArg.replaceAllUsesWith(toTensor);
      block.eraseArgument(argNumber + 1);
    }
  }

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     bufferOutputs);
  return success();
}

mlir::FailureOr<mlir::BaseMemRefType>
d2m::GenericOp::getBufferType(mlir::Value value,
                              const mlir::bufferization::BufferizationOptions &,
                              const mlir::bufferization::BufferizationState &,
                              ::llvm::SmallVector<mlir::Value> &) {
  auto tensorType = mlir::cast<RankedTensorType>(value.getType());
  if (mlir::isa<mlir::BlockArgument>(value)) {
    assert(!tensorType.getEncoding());
    return MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(), nullptr,
        ttcore::MemorySpaceAttr::get(tensorType.getContext(),
                                     ttcore::MemorySpace::DeviceL1));
  }
  return d2m::getBufferType(tensorType, /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// StreamLayoutOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool d2m::StreamLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool d2m::StreamLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::LogicalResult d2m::StreamLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {
  if (!mlir::isa<::mlir::RankedTensorType>(getResult().getType())) {
    return failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  auto maybeStorage =
      mlir::bufferization::getBuffer(rewriter, getStorage(), options, state);
  if (failed(maybeStorage)) {
    return maybeStorage;
  }

  ::llvm::SmallVector<mlir::Value> invocationStack;
  Value result = rewriter.create<d2m::StreamLayoutOp>(
      getLoc(), *getBufferType(getResult(), options, state, invocationStack),
      *maybeInput, *maybeStorage);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this, result);
  return success();
}

mlir::bufferization::AliasingValueList d2m::StreamLayoutOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType> d2m::StreamLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return d2m::getBufferType(value.getType(), /*isView=*/true);
}
} // namespace mlir::tt::d2m
