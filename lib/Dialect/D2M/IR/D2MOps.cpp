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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

namespace mlir::tt::d2m {

static std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
getGridAndShardFromValue(Value v) {
  auto shapedType = mlir::cast<ShapedType>(v.getType());
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

void d2m::GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}

mlir::tt::ttcore::DeviceAttr d2m::GenericOp::getDevice() {
  return ttcore::lookupDevice(*this);
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
      (mlir::isa<ttnn::TTNNLayoutAttr>(getResult().getType().getEncoding()) ||
       mlir::isa<ttnn::TTNNNDLayoutAttr>(
           getResult().getType().getEncoding()))) {
    return success();
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
  if (options.allowUnknownOps &&
      mlir::isa<ttnn::TTNNLayoutAttr>(getResult().getType().getEncoding())) {
    return mlir::success();
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
    // Don't fold if the input is a stream/view — the remapping it carries
    // must be materialized by this to_layout, even if the types match.
    if (getInput().getDefiningOp<StreamLayoutOp>() ||
        getInput().getDefiningOp<ViewLayoutOp>()) {
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
    rewriter.replaceOpWithNewOp<EmptyOp>(op, op.getOutput().getType(),
                                         /*virtualGridInverseMapping=*/nullptr,
                                         /*virtualGridForwardMapping=*/nullptr);
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
      *maybeInput, getRemapping(), *maybeStorage);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this, result);
  return success();
}

mlir::bufferization::AliasingValueList d2m::StreamLayoutOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  result.addAlias({getResult(), bufferization::BufferRelation::Equivalent});
  return result;
}

mlir::FailureOr<mlir::bufferization::BufferLikeType>
d2m::StreamLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return ttcore::getBufferType(value.getType(), /*isView=*/true);
}

void d2m::StreamLayoutOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *) {
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
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
    auto composedMap = viewOp.getRemapping().compose(op.getRemapping());
    auto newMemref = MemRefType::get(
        currentResultMemref.getShape(), currentResultMemref.getElementType(),
        rewriter.getAttr<ttcore::ViewLayoutAttr>(currentResultMemref.getRank()),
        currentResultMemref.getMemorySpace());
    rewriter.replaceOpWithNewOp<StreamLayoutOp>(
        op, newMemref, viewOp.getInput(), AffineMapAttr::get(composedMap),
        op.getStorage());
    return success();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
}

mlir::LogicalResult StreamLayoutOp::verify() {
  auto inputStorageVerification = verifyLayoutOp(
      *this, "input", "storage", getInput().getType(), getStorage().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ false,
      /*checkSameRank*/ false,
      /*checkSameGridShape*/ false,
      /*checkSameShardShape*/ false);
  if (failed(inputStorageVerification)) {
    return inputStorageVerification;
  }

  auto storageResultVerification = verifyLayoutOp(
      *this, "storage", "result", getStorage().getType(), getResult().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ false,
      /*checkSameRank*/ false,
      /*checkSameGridShape*/ false,
      /*checkSameShardShape*/ false);
  if (failed(storageResultVerification)) {
    return storageResultVerification;
  }

  auto inputResultVerification = verifyLayoutOp(
      *this, "input", "result", getInput().getType(), getResult().getType(),
      /*checkSameElementType*/ true,
      /*checkSameMemorySpace*/ true,
      /*checkSameRank*/ false,
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
    // For affine map-based views, verify logical shapes are preserved.
    // Device tensor shapes can differ (grid redistribution, alignment changes),
    // but the underlying logical data must be the same.

    auto inputTensor = mlir::dyn_cast<mlir::RankedTensorType>(inputType);
    auto resultTensor = mlir::dyn_cast<mlir::RankedTensorType>(resultType);

    if (inputTensor && resultTensor) {
      auto inputLayout =
          mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
              inputTensor.getEncoding());
      auto resultLayout =
          mlir::dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
              resultTensor.getEncoding());

      if (inputLayout && resultLayout) {
        // Both have layouts: verify logical shapes match.
        if (inputLayout.getLogicalShape() != resultLayout.getLogicalShape()) {
          return emitOpError("view must preserve logical shape");
        }
      } else if (!inputLayout && !resultLayout) {
        // Neither has layout: verify device tensor shapes match.
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
      if (inputLayout.getLogicalShape() != resultLayout.getLogicalShape()) {
        return emitOpError("view cannot change logical shape");
      }

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
  mlir::AffineMap reblockMap = ttmlir::utils::calculateReblockMap(
      mlir::cast<mlir::ShapedType>(getInput().getType()).getShape(),
      mlir::cast<mlir::ShapedType>(getResult().getType()).getShape(),
      getContext());
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
  // Fold view(stream) -> stream with composed layout.
  // NOLINTBEGIN(clang-analyzer-core.StackAddressEscape)
  patterns.add(+[](ViewLayoutOp op, mlir::PatternRewriter &rewriter) {
    StreamLayoutOp streamOp = op.getInput().getDefiningOp<StreamLayoutOp>();
    if (!streamOp) {
      return failure();
    }

    auto streamMemref =
        mlir::dyn_cast<MemRefType>(streamOp.getResult().getType());
    if (!streamMemref) {
      return failure();
    }

    auto viewResultMemref = mlir::cast<MemRefType>(op.getResult().getType());

    // Compose the stream's remapping with the view's remapping.
    mlir::AffineMap composedRemapping =
        streamOp.getRemapping().compose(op.getRemapping());

    auto composedAttr = rewriter.getAttr<ttcore::ViewLayoutAttr>(
        static_cast<unsigned>(viewResultMemref.getRank()));
    auto newMemref = MemRefType::get(
        viewResultMemref.getShape(), viewResultMemref.getElementType(),
        composedAttr, viewResultMemref.getMemorySpace());
    rewriter.replaceOpWithNewOp<StreamLayoutOp>(
        op, newMemref, streamOp.getInput(),
        AffineMapAttr::get(composedRemapping), streamOp.getStorage());
    return success();
  });
  // NOLINTEND(clang-analyzer-core.StackAddressEscape)
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
        grid = builder.getAttr<ttcore::GridAttr>(gridShape, *invMap);
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
            auto invMap = ttmlir::utils::createGridInverseMapFor2DPermutation(
                indexMap, gridShape.size(), builder.getContext());
            grid = builder.getAttr<ttcore::GridAttr>(gridShape, invMap);
          }
        }
      }

      // 3. Fallback: if the logical grid differs from the physical grid
      //    (e.g. ND grids) and no explicit mapping was found, derive one.
      if (!grid) {
        SmallVector<int64_t> physGridShape =
            d2m::utils::getPhysicalGridShape(output);
        if (!llvm::equal(gridShape, physGridShape)) {
          auto [_, invMap] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
              builder.getContext(), gridShape, physGridShape);
          grid = builder.getAttr<ttcore::GridAttr>(gridShape, invMap);
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
    auto flatInverseMap =
        ttmlir::utils::concatInversePermutationMap(maps, /*reverse=*/true);

    SmallVector<int64_t> flattenedOperandGridShapes;
    const auto values = llvm::to_vector(llvm::concat<Value>(inputs, outputs));
    for (Value v : llvm::reverse(values)) {
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
  llvm::SmallVector<Type> blockTypes =
      llvm::map_to_vector(TypeRange(inputOutputOperands), [&](Type t) -> Type {
        mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
        auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
            tensorType.getEncoding());

        // If the operand is a view/stream, get the layout from its source.
        if (!layout) {
          for (auto operand : inputOutputOperands) {
            if (operand.getType() != t) {
              continue;
            }
            if (auto streamOp = operand.getDefiningOp<d2m::StreamLayoutOp>()) {
              auto storageType =
                  mlir::cast<RankedTensorType>(streamOp.getStorage().getType());
              layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                  storageType.getEncoding());
              break;
            }
            if (auto viewOp = operand.getDefiningOp<d2m::ViewLayoutOp>()) {
              auto inputType =
                  mlir::cast<RankedTensorType>(viewOp.getInput().getType());
              layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                  inputType.getEncoding());
              break;
            }
          }
        }

        assert(
            layout &&
            "Expected MetalLayoutAttr or ViewLayoutAttr with StreamLayoutOp");
        auto shardShape = layout.getShardShape(tensorType);
        return d2m::CBType::get(mlir::RankedTensorType::get(
            shardShape, tensorType.getElementType()));
      });
  Region &region = *state.regions.front().get();
  llvm::SmallVector<mlir::Location> locs(inputOutputOperands.size(),
                                         state.location);
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&region, region.end(), blockTypes, locs);
  singleThreadRegionBuilder(builder, state.location, block->getArguments());
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
    if (!isExplicitDatamovementForm()) {
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

  if (!getGrid().getMapping().isEmpty()) {

    if (getGrid().getMapping().getNumInputs() != 2ul) {
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

    // Verify per-output VGM consistency:
    // 1. The output's inverse VGM must match the GridAttr's inverse map.
    // 2. The inverse map applied to the physical grid shape must produce
    //    a virtual grid shape matching the output's grid shape.
    AffineMap gridInvMap = getGrid().getMapping();
    for (Value output : getOutputs()) {
      auto outputInvMap = utils::getVirtualGridInverseMapping(output);
      if (outputInvMap && *outputInvMap != gridInvMap) {
        return emitOpError("grid inverse map does not match output operand's "
                           "inverse VGM");
      }
      if (!outputInvMap && !gridInvMap.isEmpty()) {
        return emitOpError("grid has an inverse map but output operand "
                           "does not have a VGM");
      }

      SmallVector<int64_t> physicalGridShape =
          d2m::utils::getPhysicalGridShape(output);

      // Drop the deviceID result (first result) from the inverse map.
      AffineMap invMapNoDevice = gridInvMap.dropResult(0);

      SmallVector<int64_t> impliedVirtShape =
          ttmlir::utils::evalShape(invMapNoDevice, physicalGridShape);

      SmallVector<int64_t> outputGridShape;
      if (auto memrefType = mlir::dyn_cast<MemRefType>(output.getType())) {
        auto shape = memrefType.getShape();
        TT_assert((shape.size() % 2) == 0ul);
        outputGridShape.assign(shape.begin(), shape.begin() + shape.size() / 2);
      } else {
        outputGridShape = llvm::to_vector(ttcore::getGridShape(output));
      }

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
  if (isUnifiedForm()) {
    if (failed(utils::checkForIllegalSemaphoreOps(&getRegion(0).front()))) {
      return failure();
    }
  }

  ValueTypeRange<OperandRange> inputOutputOperandTypes =
      getInputsAndOutputs().getTypes();
  auto *firstRegion = getRegions().begin();
  for (Region &region : getRegions()) {
    if (!region.hasOneBlock()) {
      return emitOpError("region must have a single block");
    }

    if (region.getNumArguments() < inputOutputOperandTypes.size()) {
      return emitOpError("region must have at least as many "
                         "arguments as the number of top-level operands");
    }

    // All regions must have the same number of arguments and signature.
    if (region.getNumArguments() != firstRegion->getNumArguments()) {
      return emitOpError("all regions must have the same number of arguments");
    }

    for (BlockArgument arg : region.getArguments()) {
      if (!mlir::isa<d2m::CBType, d2m::SemaphoreType>(arg.getType())) {
        return emitOpError(
            "all regions must either cb or semaphore block argument type");
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

    auto valueArguments =
        region.getArguments().take_front(inputOutputOperandTypes.size());
    for (BlockArgument arg : valueArguments) {
      mlir::ShapedType operandType = mlir::cast<mlir::ShapedType>(
          inputOutputOperandTypes[arg.getArgNumber()]);
      ttcore::DeviceLayoutInterface layout =
          ttcore::getDeviceLayout(operandType);
      if (!layout) {
        continue;
      }

      mlir::ShapedType blockArgType =
          mlir::cast<mlir::ShapedType>(arg.getType());

      ArrayRef<int64_t> expectedShardShape = layout.getShardShape(operandType);
      if (expectedShardShape != blockArgType.getShape()) {
        return emitOpError("region argument shape must match the "
                           "shape of the corresponding operand");
      }
    }

    auto additionalArguments =
        region.getArguments().drop_front(inputOutputOperandTypes.size());
    ;
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

  auto isReductionIterator = [](Attribute iteratorType) {
    return mlir::cast<ttcore::IteratorTypeAttr>(iteratorType).getValue() ==
           ttcore::IteratorType::Reduction;
  };
  auto isStreamingOutput = [](Value output) {
    return output.getDefiningOp() != nullptr &&
           mlir::dyn_cast<d2m::StreamLayoutOp>(output.getDefiningOp()) !=
               nullptr;
  };
  if (llvm::any_of(getIteratorTypes(), isReductionIterator) &&
      llvm::any_of(getOutputs(), isStreamingOutput)) {
    return emitOpError("Streaming outputs are not supported for reduction "
                       "iterators. Issue #5446");
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
          emitOpError("region uses value defined outside that is "
                      "not an operand of this generic op: ")
              << operand;
        }
      }
    }
  });

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
            [op](PatternRewriter &rewriter, Region &region, Operation *regionOp,
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

          blockArg = region.getArgument(dpsIOBoundary);
          assert(blockArg.getNumUses() > 0);

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
          for (Operation *user : blockArg.getUsers()) {
            assert((mlir::isa<d2m::WaitOp, d2m::ReserveOp, d2m::PushOp,
                              d2m::PopOp>(user)) &&
                   "block argument users must be wait/reserve/push/pop "
                   "operations");
            // Check if this wait/reserve dominates the regionOp.
            // Note: push/pop don't have results, so they won't be selected
            // here.
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
  TT_assert(currentBlockFactors.size() == factorizations.size());

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
  TT_assert(getOutputs().size() == 1u);
  return d2m::utils::getPhysicalGridShape(getOutputs().front());
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
  int cbIndex = 0;
  int semIndex = 0;
  for (BlockArgument arg : region.getArguments()) {
    if (mlir::isa<MemRefType>(arg.getType())) {
      setNameFn(arg, "cb" + std::to_string(cbIndex++));
    } else if (mlir::isa<CBType>(arg.getType())) {
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
      getLoc(), ValueRange(), bufferInputs, bufferOutputs, getAdditionalArgs(),
      getGrid(), getBlockFactors(), getIndexingMaps(), getIteratorTypes(),
      getThreads(), getScratchInputsAttr(), getNumRegions());
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    region.takeBody(getRegion(region.getRegionNumber()));
  }

  // Bufferize region block arguments.
  ::llvm::SmallVector<mlir::Value> invocationStack;
  for (mlir::Region &region : bufferGeneric.getRegions()) {
    OpBuilder::InsertionGuard guard(rewriter);
    mlir::Block &block = region.front();
    rewriter.setInsertionPointToStart(&block);
    for (unsigned argNumber = 0; argNumber < block.getNumArguments();
         ++argNumber) {
      mlir::BlockArgument oldArg = block.getArgument(argNumber);
      if (mlir::isa<d2m::SemaphoreType>(oldArg.getType())) {
        continue;
      }
      auto cbType = mlir::cast<d2m::CBType>(oldArg.getType());
      auto newArgType =
          cbType.getBufferType(options, [&]() { return this->emitError(); });
      mlir::BlockArgument newArg =
          block.insertArgument(argNumber, *newArgType, oldArg.getLoc());
      auto toTensor = rewriter.create<bufferization::ToTensorOp>(
          bufferGeneric.getLoc(), oldArg.getType(), newArg);
      rewriter.replaceAllUsesWith(oldArg, toTensor.getResult());
      block.eraseArgument(argNumber + 1);
    }
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
  if (mlir::isa<mlir::BlockArgument>(value)) {
    assert(!tensorType.getEncoding());
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

  Block *threadBlock = &genericRegion->front();

  if (threadBlock->getNumArguments() > operandIndex) {
    return threadBlock->getArgument(operandIndex);
  }

  return Value();
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

} // namespace mlir::tt::d2m
