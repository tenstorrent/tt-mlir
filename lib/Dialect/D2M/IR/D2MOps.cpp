// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "ttmlir/Dialect/D2M/IR/D2MOps.cpp.inc"

using namespace mlir;
using namespace mlir::tt::d2m;

void mlir::tt::d2m::GenericOp::getAsmBlockArgumentNames(
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

void mlir::tt::d2m::GenericOp::getAsmBlockNames(
    function_ref<void(Block *, StringRef)> setNameFn) {
  std::array<int, getMaxEnumValForThreadType() + 1> threadTypeCounts{};
  for (Region &region : getRegions()) {
    auto type = getRegionThreadType(region.getRegionNumber());
    setNameFn(&region.front(),
              stringifyEnum(type).str() +
                  Twine(threadTypeCounts[llvm::to_underlying(type)]++).str());
  }
}

void mlir::tt::d2m::GenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  d2m::getDpsEffects(*this, effects);
}

SmallVector<AffineMap> mlir::tt::d2m::GenericOp::getIndexingMapsValue() {
  return llvm::to_vector(llvm::map_range(getIndexingMaps(), [](Attribute a) {
    return mlir::cast<AffineMapAttr>(a).getValue();
  }));
}

SmallVector<int64_t> mlir::tt::d2m::GenericOp::getBlockFactorsValue() {
  return llvm::map_to_vector(getBlockFactors(), [](Attribute a) {
    return mlir::cast<IntegerAttr>(a).getInt();
  });
}

SmallVector<mlir::tt::ttcore::IteratorType>
mlir::tt::d2m::GenericOp::getIteratorTypesValue() {
  return llvm::to_vector(llvm::map_range(getIteratorTypes(), [](Attribute a) {
    return mlir::cast<mlir::tt::ttcore::IteratorTypeAttr>(a).getValue();
  }));
}

mlir::tt::ttcore::DeviceAttr mlir::tt::d2m::GenericOp::getDevice() {
  return ttcore::lookupDevice(*this);
}

SmallVector<SmallVector<int64_t>>
mlir::tt::d2m::GenericOp::getOperandGridShapes() {
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
mlir::tt::d2m::GenericOp::getParticipatingLoopDims(int64_t operandIndex) {
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
mlir::tt::d2m::GenericOp::getNonParticipatingLoopDims(int64_t operandIndex) {
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

SmallVector<int64_t> mlir::tt::d2m::GenericOp::getLoopBounds() {
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
      mlir::tt::ttir::utils::getSquareTargetGrid(workerGridShape);

  return mlir::tt::ttcore::MetalLayoutAttr::get(
      ctx, logicalShape, squareGridShape, mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::System);
}
} // namespace

mlir::tt::ttcore::MetalLayoutAttr
mlir::tt::d2m::ToLayoutOp::getOrCreateInputLayout() {
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

mlir::tt::ttcore::MetalLayoutAttr
mlir::tt::d2m::ToLayoutOp::getOrCreateOutputLayout() {
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

mlir::tt::d2m::ToLayoutOp::CompoundComponents
mlir::tt::d2m::ToLayoutOp::compoundComponents() {
  CompoundComponents components;
  if (mlir::isa<mlir::RankedTensorType>(getInput().getType())) {
    auto inputTensor = mlir::cast<mlir::RankedTensorType>(getInput().getType());
    auto outputTensor =
        mlir::cast<mlir::RankedTensorType>(getOutput().getType());
    const bool hasInputLayout = inputTensor.getEncoding() != nullptr;
    const bool hasOutputLayout = outputTensor.getEncoding() != nullptr;

    ttcore::MetalLayoutAttr inputLayout = getOrCreateInputLayout();
    ttcore::MetalLayoutAttr outputLayout = getOrCreateOutputLayout();

    // Tensors without grids are assumed to have 1 grids for comparison.
    SmallVector<int64_t> inputGrid =
        (hasInputLayout)
            ? SmallVector<int64_t>{inputLayout.getGridShape(inputTensor)}
            : SmallVector<int64_t>{
                  static_cast<int64_t>(
                      outputLayout.getGridShape(outputTensor).size()),
                  1};
    SmallVector<int64_t> outputGrid =
        (hasOutputLayout)
            ? SmallVector<int64_t>{outputLayout.getGridShape(outputTensor)}
            : SmallVector<int64_t>{
                  static_cast<int64_t>(
                      inputLayout.getGridShape(inputTensor).size()),
                  1};

    components.isGridChange = inputGrid != outputGrid;
    components.isMemorySpaceChange =
        inputLayout.getMemorySpace() != outputLayout.getMemorySpace();
    components.isFormatChange =
        inputTensor.getElementType() != outputTensor.getElementType();
    components.isLayoutChange =
        inputLayout.getNormalizedIntervals() !=
            outputLayout.getNormalizedIntervals() ||
        inputLayout.getDimAlignments() != outputLayout.getDimAlignments();
  } else {
    auto inputMemref = mlir::cast<mlir::MemRefType>(getInput().getType());
    auto outputMemref = mlir::cast<mlir::MemRefType>(getOutput().getType());
    components.isLayoutChange = false;
    bool isShapeChange = inputMemref.getShape() != outputMemref.getShape();
    components.isGridChange = isShapeChange;
    components.isFormatChange =
        inputMemref.getElementType() != outputMemref.getElementType();
    components.isMemorySpaceChange =
        inputMemref.getMemorySpace() != outputMemref.getMemorySpace();
  }
  return components;
}

//===----------------------------------------------------------------------===//
// EmptyOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

// Forward declaration from TTIR
namespace mlir::tt::ttir {
MemRefType getBufferType(Type type, bool isView);
}

bool mlir::tt::d2m::EmptyOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool mlir::tt::d2m::EmptyOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::LogicalResult mlir::tt::d2m::EmptyOp::bufferize(
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
mlir::tt::d2m::EmptyOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType> mlir::tt::d2m::EmptyOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return mlir::tt::ttir::getBufferType(value.getType(), /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// ToLayoutOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool mlir::tt::d2m::ToLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 0; // Input operand
}

bool mlir::tt::d2m::ToLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  return operand.getOperandNumber() == 1; // Output operand
}

mlir::LogicalResult mlir::tt::d2m::ToLayoutOp::bufferize(
    mlir::RewriterBase &rewriter,
    const mlir::bufferization::BufferizationOptions &options,
    mlir::bufferization::BufferizationState &state) {

  auto maybeInputBuffer =
      mlir::bufferization::getBuffer(rewriter, getInput(), options, state);
  if (failed(maybeInputBuffer)) {
    return maybeInputBuffer;
  }

  auto maybeOutputBuffer =
      mlir::bufferization::getBuffer(rewriter, getOutput(), options, state);
  if (failed(maybeOutputBuffer)) {
    return maybeOutputBuffer;
  }

  rewriter.create<mlir::tt::d2m::ToLayoutOp>(
      getLoc(), TypeRange(), *maybeInputBuffer, *maybeOutputBuffer,
      getLayoutAttr());

  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this,
                                                     *maybeOutputBuffer);
  return mlir::success();
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

mlir::FailureOr<mlir::BaseMemRefType> mlir::tt::d2m::ToLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return mlir::tt::ttir::getBufferType(value.getType(), /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// GenericOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool mlir::tt::d2m::GenericOp::bufferizesToMemoryRead(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an input, it is bufferized to a memory read.
  return isDpsInput(&operand);
}

bool mlir::tt::d2m::GenericOp::bufferizesToMemoryWrite(
    mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
  // If the operand is an output, it is bufferized to a memory write.
  return isDpsInit(&operand);
}

mlir::bufferization::AliasingValueList
mlir::tt::d2m::GenericOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

bool mlir::tt::d2m::GenericOp::isWritable(
    mlir::Value value, const mlir::bufferization::AnalysisState &) {
  return mlir::isa<mlir::OpResult>(value) ||
         mlir::isa<mlir::BlockArgument>(value);
}

mlir::LogicalResult mlir::tt::d2m::GenericOp::bufferize(
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
  auto bufferGeneric = rewriter.create<mlir::tt::d2m::GenericOp>(
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

mlir::FailureOr<mlir::BaseMemRefType> mlir::tt::d2m::GenericOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
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
  return mlir::tt::ttir::getBufferType(tensorType, /*isView=*/false);
}

//===----------------------------------------------------------------------===//
// StreamLayoutOp Bufferization Interface Implementation
//===----------------------------------------------------------------------===//

bool mlir::tt::d2m::StreamLayoutOp::bufferizesToMemoryRead(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

bool mlir::tt::d2m::StreamLayoutOp::bufferizesToMemoryWrite(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  return false;
}

mlir::LogicalResult mlir::tt::d2m::StreamLayoutOp::bufferize(
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
  Value result = rewriter.create<mlir::tt::d2m::StreamLayoutOp>(
      getLoc(), *getBufferType(getResult(), options, state, invocationStack),
      *maybeInput, *maybeStorage);
  mlir::bufferization::replaceOpWithBufferizedValues(rewriter, *this, result);
  return success();
}

mlir::bufferization::AliasingValueList
mlir::tt::d2m::StreamLayoutOp::getAliasingValues(
    mlir::OpOperand &, const mlir::bufferization::AnalysisState &) {
  bufferization::AliasingValueList result;
  return result;
}

mlir::FailureOr<mlir::BaseMemRefType>
mlir::tt::d2m::StreamLayoutOp::getBufferType(
    mlir::Value value, const mlir::bufferization::BufferizationOptions &,
    const mlir::bufferization::BufferizationState &,
    ::llvm::SmallVector<mlir::Value> &) {
  return mlir::tt::ttir::getBufferType(value.getType(), /*isView=*/true);
}
