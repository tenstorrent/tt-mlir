// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include <functional>

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

namespace mlir::tt::d2m {

void d2m::GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}

mlir::tt::ttcore::DeviceAttr d2m::GenericOp::getDevice() {
  return ttcore::lookupDevice(*this);
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
      mlir::tt::ttcore::MemorySpace::System,
      mlir::tt::ttcore::TensorMemoryLayout::Sharded);
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

  // Don't bufferize if tensor has a ttnn_layout; lowering to ttnn generic.
  if (options.allowUnknownOps &&
      mlir::isa<ttnn::TTNNLayoutAttr>(getResult().getType().getEncoding())) {
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
  return ttcore::getBufferType(value.getType(), /*isView=*/false);
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
  auto memrefType = mlir::cast<mlir::MemRefType>(
      getBufferType(getResult(), options, state, invocationStack).value());

  auto denseAttr =
      mlir::DenseElementsAttr::get(getResult().getType(), getFillValueAttr());

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

mlir::FailureOr<mlir::BaseMemRefType>
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
}

mlir::FailureOr<mlir::BaseMemRefType> d2m::MeshShardOp::getBufferType(
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

// ToLayoutOp utility methods
ToLayoutOp::CompoundComponents ToLayoutOp::compoundComponents() {
  CompoundComponents components;

  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();

  TT_assertv(mlir::isa<mlir::RankedTensorType>(inputType),
             "ToLayoutOp::compoundComponents() is only supported on tensors.");

  auto inputTensor = mlir::cast<mlir::RankedTensorType>(inputType);
  auto outputTensor = mlir::cast<mlir::RankedTensorType>(outputType);

  ttcore::MetalLayoutAttr inputLayout =
      mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
          inputTensor.getEncoding());
  ttcore::MetalLayoutAttr outputLayout =
      mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
          outputTensor.getEncoding());

  const bool hasInputLayout = inputLayout != nullptr;
  const bool hasOutputLayout = outputLayout != nullptr;

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
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType());
    return success();
  });

  patterns.add(std::make_unique<ToLayoutFoldRedundantPattern>(context));
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

  // For unaligned H2D, copy the unaligned host tensor to an aligned & padded
  // bounce buffer, then write to the device.
  if (isHostToDevice()) {
    llvm::SmallVector<mlir::Value> invocationStack;
    MemRefType alignedHostMemref = mlir::cast<MemRefType>(
        *getBufferType(getInput(), options, state, invocationStack));

    auto hostLayoutAttr = mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
        alignedHostMemref.getLayout());
    if (hostLayoutAttr && hostLayoutAttr.isPadded()) {
      auto alignedHostTensor =
          rewriter.create<memref::AllocOp>(getLoc(), alignedHostMemref);
      rewriter.create<memref::CopyOp>(getLoc(), *maybeInput, alignedHostTensor);
      maybeInput = alignedHostTensor.getResult();
    }
  }

  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  auto toLayoutOp =
      rewriter.create<ToLayoutOp>(getLoc(), TypeRange(), *maybeInput,
                                  *maybeOutput, getLayout().value_or(nullptr));
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)

  // For unaligned D2H, read the device tensor to an aligned & padded bounce
  // buffer, then do strided memcpy to copy the data into the unaligned tensor.
  if (isDeviceToHost()) {
    llvm::SmallVector<mlir::Value> invocationStack;
    MemRefType alignedHostMemref = mlir::cast<MemRefType>(
        *getBufferType(getOutput(), options, state, invocationStack));

    auto hostLayoutAttr = mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
        alignedHostMemref.getLayout());
    if (hostLayoutAttr && hostLayoutAttr.isPadded()) {
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
  return ttcore::getBufferType(value.getType(), /*isView=*/false, getLayout());
}

//===----------------------------------------------------------------------===//
// StreamLayoutOp
//===----------------------------------------------------------------------===//

void d2m::StreamLayoutOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "stream");
}

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
  return ttcore::getBufferType(value.getType(), /*isView=*/true);
}

void d2m::StreamLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  patterns.add(+[](StreamLayoutOp op, mlir::PatternRewriter &rewriter) {
    ViewLayoutOp viewOp = op.getInput().getDefiningOp<ViewLayoutOp>();
    if (!viewOp) {
      return failure();
    }

    auto viewMemref = mlir::dyn_cast<MemRefType>(viewOp.getResult().getType());
    if (!viewMemref) {
      return failure();
    }

    auto currentResultMemref = mlir::cast<MemRefType>(op.getResult().getType());
    auto streamAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
        viewMemref.getLayout().getAffineMap().compose(
            currentResultMemref.getLayout().getAffineMap()));
    auto newMemref = MemRefType::get(
        currentResultMemref.getShape(), currentResultMemref.getElementType(),
        streamAttr, currentResultMemref.getMemorySpace());
    rewriter.replaceOpWithNewOp<StreamLayoutOp>(
        op, newMemref, viewOp.getInput(), op.getStorage());
    return success();
  });
}

mlir::LogicalResult StreamLayoutOp::verify() {
  auto inputStorageVerification = verifyLayoutOp(
      *this, "input", "storage", getInput().getType(), getStorage().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ false,
      /*checkSameRank*/ true,
      /*checkSameGridShape*/ false,
      /*checkSameShardShape*/ false);
  if (failed(inputStorageVerification)) {
    return inputStorageVerification;
  }

  auto storageResultVerification = verifyLayoutOp(
      *this, "storage", "result", getStorage().getType(), getResult().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ false,
      /*checkSameRank*/ true,
      /*checkSameGridShape*/ false,
      /*checkSameShardShape*/ true);
  if (failed(storageResultVerification)) {
    return storageResultVerification;
  }

  auto inputResultVerification = verifyLayoutOp(
      *this, "input", "result", getInput().getType(), getResult().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ true,
      /*checkSameRank*/ true,
      /*checkSameGridShape*/ false,
      /*checkSameShardShape*/ false);
  if (failed(inputResultVerification)) {
    return inputResultVerification;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ViewLayoutOp
//===----------------------------------------------------------------------===//

void d2m::ViewLayoutOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "view");
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
    // For regular reblocking, verify it's valid; total elements must match.
    int64_t inputElements = 1, outputElements = 1;
    for (auto d : inputType.getShape()) {
      inputElements *= d;
    }
    for (auto d : resultType.getShape()) {
      outputElements *= d;
    }
    if (inputElements != outputElements) {
      return emitOpError("view must preserve total number of elements");
    }

    // We also should not change element type unless reinterpretting.
    if (inputType.getElementType() != resultType.getElementType()) {
      return emitOpError("view must not change dtype");
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
  if (mlir::isa<mlir::MemRefType>(getInput().getType())) {
    return mlir::failure();
  }

  auto maybeInput =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInput)) {
    return maybeInput;
  }

  // Build the memref result type from the tensor result encoding so that any
  // index_map on the encoding is honored when creating the view layout.
  ::llvm::SmallVector<mlir::Value> dummy;
  auto outMemrefTypeOr = getBufferType(getResult(), options, state, dummy);
  if (mlir::failed(outMemrefTypeOr)) {
    return outMemrefTypeOr;
  }

  auto outMemrefType = mlir::cast<mlir::MemRefType>(*outMemrefTypeOr);
  auto newOp = rewriter.create<d2m::ViewLayoutOp>(
      getLoc(), outMemrefType, *maybeInput, getReinterpretLayout());

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     newOp.getResult());

  return mlir::success();
}

mlir::bufferization::AliasingValueList d2m::ViewLayoutOp::getAliasingValues(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType> d2m::ViewLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/true);
}

mlir::OpFoldResult d2m::ViewLayoutOp::fold(FoldAdaptor adaptor) {
  ViewLayoutOp consecutiveView = getInput().getDefiningOp<d2m::ViewLayoutOp>();
  if (!consecutiveView) {
    return nullptr;
  }

  // Replace the input through the consecutive view.
  setOperand(consecutiveView.getInput());

  MemRefType inputType = mlir::dyn_cast<MemRefType>(consecutiveView.getType());
  if (!inputType) {
    return getResult();
  }

  // If we're dealing with memrefs, we need to compose the layouts.
  MemRefType resultType = mlir::cast<MemRefType>(getType());
  ttcore::ViewLayoutAttr inputView =
      mlir::cast<ttcore::ViewLayoutAttr>(inputType.getLayout());
  ttcore::ViewLayoutAttr resultView =
      mlir::cast<ttcore::ViewLayoutAttr>(resultType.getLayout());
  ttcore::ViewLayoutAttr newView = inputView.compose(resultView);
  getResult().setType(MemRefType::Builder(resultType).setLayout(newView));

  return getResult();
}

//===----------------------------------------------------------------------===//
// GenericOp
//===----------------------------------------------------------------------===//

void d2m::GenericOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, ValueRange inputs,
                           ValueRange outputs, ArrayAttr indexingMaps,
                           ArrayAttr iteratorTypes, ThreadType singleThreadType,
                           ttcore::GridAttr grid,
                           ArrayRef<int64_t> blockFactors) {
  TT_assertv(!indexingMaps.empty(), "expected non-empty indexing maps");
  TT_assertv(outputs.size() == 1u, "expected single output");

  if (!grid) {
    auto shapedType = mlir::cast<ShapedType>(outputs[0].getType());
    ttcore::DeviceLayoutInterface layout = ttcore::getDeviceLayout(shapedType);
    TT_assertv(
        layout,
        "This generic constructor expects operands to be in device layout");
    grid = builder.getAttr<ttcore::GridAttr>(layout.getGridShape(shapedType));
  }

  ArrayAttr blockFactorsAttr;
  if (blockFactors.empty()) {
    auto maps =
        llvm::to_vector(llvm::map_range(indexingMaps, [](Attribute attr) {
          return cast<AffineMapAttr>(attr).getValue();
        }));
    auto flatInverseMap =
        utils::concatInversePermutationMap(maps, /*reverse=*/true);

    SmallVector<int64_t> flattenedOperandGridShapes;
    for (Value v :
         llvm::reverse(llvm::to_vector(llvm::concat<Value>(inputs, outputs)))) {
      auto shapedType = mlir::cast<ShapedType>(v.getType());
      ttcore::DeviceLayoutInterface layout =
          ttcore::getDeviceLayout(shapedType);
      TT_assertv(
          layout,
          "This generic constructor expects operands to be in device layout");
      auto gridShape = layout.getGridShape(shapedType);
      flattenedOperandGridShapes.append(gridShape.begin(), gridShape.end());
    }

    // Divide out the grid shape, this is safe to do because we reversed the
    // affine map above, so output dims are guaranteed to appear first in the
    // affine map.

    for (std::size_t i = 0; i < grid.getShape().size(); ++i) {
      flattenedOperandGridShapes[i] /= grid.getShape()[i];
    }

    blockFactorsAttr = builder.getI64ArrayAttr(
        flatInverseMap.compose(flattenedOperandGridShapes));
  } else {
    blockFactorsAttr = builder.getI64ArrayAttr(blockFactors);
  }

  auto threads =
      builder.getArrayAttr(builder.getAttr<ThreadAttr>(singleThreadType));

  build(builder, state, TypeRange(outputs), inputs, outputs, grid,
        blockFactorsAttr, indexingMaps, iteratorTypes, threads, 1);
}

void d2m::GenericOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ValueRange inputs,
    ValueRange outputs, ArrayAttr indexingMaps, ArrayAttr iteratorTypes,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)>
        singleThreadRegionBuilder,
    ThreadType singleThreadType, ttcore::GridAttr grid,
    ArrayRef<int64_t> blockFactors) {
  build(builder, state, inputs, outputs, indexingMaps, iteratorTypes,
        singleThreadType, grid, blockFactors);
  llvm::SmallVector<Type> blockTypes =
      llvm::map_to_vector(TypeRange(state.operands), [&](Type t) -> Type {
        mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
        auto layout =
            mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
        auto shardShape = layout.getShardShape(tensorType);
        return mlir::RankedTensorType::get(shardShape,
                                           tensorType.getElementType());
      });
  Region &region = *state.regions.front().get();
  llvm::SmallVector<mlir::Location> locs(state.operands.size(), state.location);
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&region, region.end(), blockTypes, locs);
  singleThreadRegionBuilder(builder, state.location, block->getArguments());
}

void d2m::GenericOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, ValueRange inputs,
    ValueRange outputs,
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
  build(builder, state, inputs, outputs, indexingMaps, iteratorTypes,
        singleThreadRegionBuilder, singleThreadType, grid);
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
    mlir::ArrayRef<int64_t> opGridShape,
    llvm::function_ref<mlir::InFlightDiagnostic()> diagFn) {
  assert(indexingMaps.size() == shapes.size());

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
  mlir::SmallVector<int64_t> factors = inverseOpGridMap.compose(opGridShape);
  assert(factors.size() == blockingFactors.size());
  for (size_t i = 0; i < blockingFactors.size(); ++i) {
    if (factors[i] == 0) {
      // The "Broacast" part of inverseAndBroadcastProjectedPermutation will 0
      // fill unparticipating dims.  Promote these to 1's so that we can
      // multiply by blocking factor.
      factors[i] = 1;
    }
    factors[i] *= blockingFactors[i];
  }

  for (size_t operand = 0; operand < indexingMaps.size(); ++operand) {
    auto shape = shapes[operand];
    auto factor = indexingMaps[operand].compose(factors);
    assert(shape.size() == factor.size());
    if (auto dim = isNotEqualOrBroadcast(shape, factor)) {
      return diagFn() << shapeName << " dim unexpected for operand[" << operand
                      << "] " << shapeName << "_shape=[" << shapes[operand]
                      << "] expected " << shapeName << "_shape=[" << factor
                      << "] at affine dim d" << *dim;
    }
  }

  return mlir::success();
}

// GenericOp verification
::mlir::LogicalResult d2m::GenericOp::verify() {
  if (hasPureTensorSemantics()) {
    if (this->getNumRegions() != 1) {
      return emitOpError(
          "generic op with pure tensor semantics must have exactly 1 region");
    }

    Region &region = this->getRegion(0);
    if (!region.hasOneBlock()) {
      return emitOpError(
          "generic op with pure tensor semantics must have exactly 1 block");
    }

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

  if (!getGrid().getMapping().isEmpty()) {
    return emitOpError("grid mapping is not supported");
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
    Type operandType = output.getType();
    ArrayRef<int64_t> outputGridShape;
    if (RankedTensorType tensorType =
            mlir::dyn_cast<RankedTensorType>(operandType)) {
      if (!tensorType.getEncoding()) {
        // Skip layout checks if the tensor type does not have a layout yet.
        continue;
      }
      ttcore::MetalLayoutAttr layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      outputGridShape = layout.getGridShape(tensorType);
    } else {
      auto memref = mlir::cast<MemRefType>(operandType);
      // If the top level operand is a memref, the front half of its shape
      // is the grid shape, so we cut it off the back to get just the grid
      // shape.
      mlir::tt::ttcore::DeviceLayoutInterface layout =
          mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
              memref.getLayout());
      outputGridShape = layout.getGridShape(memref);
    }
    if (!llvm::all_of(llvm::zip(outputGridShape, opGridShape), [](auto pair) {
          auto [out, op] = pair;
          return out % op == 0;
        })) {
      return emitOpError("output grid shape must be divisible by the generic "
                         "op's grid shape");
    }
  }

  if (!llvm::all_equal(llvm::map_range(getIndexingMapsValue(), [](AffineMap m) {
        return m.getNumDims();
      }))) {
    return emitOpError(
        "all indexing maps must have the same number of dimensions");
  }

  auto numIterators = getIteratorTypes().size();
  auto blockFactors = getBlockFactorsValue();
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
    auto emitDiag = [&]() -> InFlightDiagnostic { return this->emitOpError(); };
    SmallVector<SmallVector<int64_t>> gridShapes = getOperandGridShapes();
    LogicalResult gridResult = verifyAffineShapesPermutation(
        "grid", indexingMaps, gridShapes, emitDiag);

    if (failed(gridResult)) {
      return gridResult;
    }

    SmallVector<SmallVector<int64_t>> scalarShardShapes =
        getOperandShardShapes(/*convertTileToScalar=*/true);
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

  ValueTypeRange<OperandRange> operandTypes = getOperation()->getOperandTypes();
  auto *firstRegion = getRegions().begin();
  for (Region &region : getRegions()) {
    if (!region.hasOneBlock()) {
      return emitOpError("region must have a single block");
    }

    if (region.getNumArguments() < this->getNumOperands()) {
      return emitOpError("region must have at least as many "
                         "arguments as the number of top-level operands");
    }

    // All regions must have the same number of arguments and signature.
    if (region.getNumArguments() != firstRegion->getNumArguments()) {
      return emitOpError("all regions must have the same number of arguments");
    }

    for (BlockArgument arg : region.getArguments()) {
      if (arg.getType() !=
          firstRegion->getArgument(arg.getArgNumber()).getType()) {
        return emitOpError("all regions must have the same argument types");
      }
    }

    if (indexingMaps.empty()) {
      // If there are no indexing maps, then we can no longer validate block
      // argument shapes.
      continue;
    }

    auto valueArguments = region.getArguments().take_front(operandTypes.size());
    for (BlockArgument arg : valueArguments) {
      Type blockArgType = arg.getType();

      Type operandType = operandTypes[arg.getArgNumber()];
      ArrayRef<int64_t> expectedShardShape;
      std::optional<Attribute> expectedMemorySpace;
      if (RankedTensorType tensorType =
              mlir::dyn_cast<RankedTensorType>(operandType)) {
        if (!tensorType.getEncoding()) {
          // Skip layout checks if the tensor type does not have a layout yet
          continue;
        }
        ttcore::MetalLayoutAttr layout =
            mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
        expectedMemorySpace =
            ttcore::MemorySpaceAttr::get(getContext(), layout.getMemorySpace());
        expectedShardShape = layout.getShardShape(tensorType);
      } else {
        auto memref = mlir::cast<MemRefType>(operandType);
        expectedMemorySpace = memref.getMemorySpace();
        // If the top level operand is a memref, the front half of its shape
        // will include the grid shape, so we cut it off to get just the shard
        // shape.
        mlir::tt::ttcore::DeviceLayoutInterface layout =
            mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
                memref.getLayout());
        expectedShardShape = layout.getShardShape(memref);
      }

      if (auto blockMemref = mlir::dyn_cast<MemRefType>(blockArgType)) {
        if (expectedShardShape != blockMemref.getShape()) {
          return emitOpError("region argument shape must match the "
                             "shape of the corresponding operand");
        }
      } else if (auto blockTensor =
                     mlir::dyn_cast<RankedTensorType>(blockArgType)) {
        if (expectedShardShape != blockTensor.getShape()) {
          return emitOpError("region argument shape must match the "
                             "shape of the corresponding operand");
        }
        // Memory space is not encoded in tensor types; skip that check.
      } else {
        return emitOpError(
            "region arguments must be of RankedTensorType or MemRefType");
      }
    }

    auto additionalArguments =
        region.getArguments().drop_front(operandTypes.size());
    for (BlockArgument arg : additionalArguments) {
      bool supportedType = mlir::isa<SemaphoreType>(arg.getType());
      if (!supportedType) {
        return emitOpError(
            "additional region arguments must be of 'semaphore' type");
      }
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

        auto replaceWithOutputCb =
            +[](PatternRewriter &rewriter, Region &region, Operation *regionOp,
                OpOperand &initOperand, int64_t dpsIOBoundary) -> bool {
          BlockArgument blockArg =
              mlir::dyn_cast<BlockArgument>(initOperand.get());
          if (blockArg && blockArg.getArgNumber() >= dpsIOBoundary) {
            return false;
          }

          Operation *origDefiningOp = initOperand.get().getDefiningOp();
          if (origDefiningOp &&
              !mlir::isa<EmptyOp, mlir::tensor::EmptyOp>(origDefiningOp)) {
            return false;
          }

          rewriter.modifyOpInPlace(regionOp, [&]() {
            initOperand.assign(region.getArgument(dpsIOBoundary));
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

                updated |= replaceWithOutputCb(rewriter, region, regionOp,
                                               initOperand, dpsIOBoundary);
              }
            } else if (TileMatmulBlockOp tmb =
                           mlir::dyn_cast<TileMatmulBlockOp>(regionOp);
                       tmb) {
              updated |=
                  replaceWithOutputCb(rewriter, region, regionOp,
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
  return mlir::cast<mlir::AffineMapAttr>(getIndexingMapsAttr()[0])
      .getAffineMap()
      .getNumDims();
}

mlir::AffineMap d2m::GenericOp::getIndexingMap(int64_t operandIndex) {
  return mlir::cast<AffineMapAttr>(getIndexingMaps()[operandIndex]).getValue();
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

mlir::SmallVector<int64_t> d2m::GenericOp::getFullBlockFactors() {
  auto maps = getIndexingMapsValue();
  // Priority doesn't matter here, so reverse can be false.
  auto flatInverseMap =
      utils::concatInversePermutationMap(maps, /*reverse=*/false);

  SmallVector<int64_t> flattenedOperandShardShapes;
  for (Value v : getOperands()) {
    auto shapedType = mlir::cast<ShapedType>(v.getType());
    ttcore::DeviceLayoutInterface layout = ttcore::getDeviceLayout(shapedType);
    TT_assertv(
        layout,
        "This generic constructor expects operands to be in device layout");
    auto shardShape = layout.getShardShape(shapedType);
    flattenedOperandShardShapes.append(shardShape.begin(), shardShape.end());
  }

  auto currentBlockFactors = getBlockFactorsValue();
  auto factorizations = flatInverseMap.compose(flattenedOperandShardShapes);
  TT_assert(currentBlockFactors.size() == factorizations.size());

  // Multiply back in the current block factors to normalize the result.
  for (std::size_t i = 0; i < currentBlockFactors.size(); ++i) {
    factorizations[i] *= currentBlockFactors[i];
  }

  return factorizations;
}

mlir::SmallVector<mlir::SmallVector<int64_t>>
d2m::GenericOp::getOperandGridShapes() {
  SmallVector<SmallVector<int64_t>> gridShapes;
  gridShapes.reserve(getOperands().size());
  for (auto operand : this->getOperands()) {
    auto memrefType = mlir::dyn_cast<MemRefType>(operand.getType());
    if (memrefType) {
      mlir::tt::ttcore::DeviceLayoutInterface layout =
          mlir::dyn_cast<mlir::tt::ttcore::DeviceLayoutInterface>(
              memrefType.getLayout());
      gridShapes.emplace_back(layout.getGridShape(memrefType));
    } else {
      auto tensorType = mlir::cast<RankedTensorType>(operand.getType());
      ttcore::MetalLayoutAttr layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      gridShapes.emplace_back(layout.getGridShape(tensorType));
    }
  }
  return gridShapes;
}

mlir::SmallVector<mlir::SmallVector<int64_t>>
d2m::GenericOp::getOperandShardShapes(bool convertTileToScalar) {
  SmallVector<SmallVector<int64_t>> shardShapes;
  shardShapes.reserve(getOperands().size());

  for (auto operand : this->getOperands()) {
    auto shapedType = mlir::cast<ShapedType>(operand.getType());
    mlir::tt::ttcore::DeviceLayoutInterface layout;
    Type elementType;

    if (auto memrefType = mlir::dyn_cast<MemRefType>(shapedType)) {
      layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
          memrefType.getLayout());
      elementType = memrefType.getElementType();
    } else {
      auto tensorType = mlir::cast<RankedTensorType>(shapedType);
      layout = mlir::cast<mlir::tt::ttcore::DeviceLayoutInterface>(
          tensorType.getEncoding());
      elementType = tensorType.getElementType();
    }

    auto tileType = mlir::dyn_cast<ttcore::TileType>(elementType);
    auto shardShape = layout.getShardShape(shapedType);
    shardShapes.emplace_back(
        (convertTileToScalar && tileType)
            ? tileType.getScalarShape(SmallVector<int64_t>(shardShape))
            : shardShape);
  }

  return shardShapes;
}

mlir::SmallVector<int64_t> d2m::GenericOp::getLoopBounds() {
  return getBlockFactorsValue();
}

mlir::SmallVector<int64_t>
d2m::GenericOp::getParticipatingLoopDims(int64_t operandIndex) {
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
  AffineMap indexingMap = getIndexingMap(operandIndex);
  SmallVector<int64_t> participatingDims =
      getParticipatingLoopDims(operandIndex);
  llvm::BitVector nonParticipatingDims(indexingMap.getNumDims(), true);
  llvm::for_each(participatingDims, [&nonParticipatingDims](int64_t dim) {
    nonParticipatingDims.reset(dim);
  });
  return llvm::SmallVector<int64_t>(nonParticipatingDims.set_bits());
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
  return ttcore::getBufferType(tensorType, /*isView=*/false);
}

bool d2m::GenericOp::hasCompatibleBlocking(GenericOp b) {
  return this->getGrid() == b.getGrid() &&
         this->getBlockFactors() == b.getBlockFactors();
}

/// Returns true if op or any of the operations nested within it's regions have
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
    if (llvm::range_size(input->get().getUsers()) > 1) {
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

} // namespace mlir::tt::d2m
