// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir::tt {

namespace detail {

static SmallVector<int64_t> getTensorShape(MemRefType memrefType) {
  auto deviceLayout = ttcore::getDeviceLayout(memrefType);
  ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(memrefType);
  ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(memrefType);

  SmallVector<int64_t> tileDims = {1, 1};
  if (auto tileType =
          mlir::dyn_cast<ttcore::TileType>(memrefType.getElementType())) {
    tileDims[0] = tileType.getHeight();
    tileDims[1] = tileType.getWidth();
  }

  // Compute tensor shape in elements: (grid * shard * tileDims). If dims have
  // been collapsed, we cannot recover the original uncollapsed shape.
  SmallVector<int64_t> tensorShape;
  for (size_t i = 0; i < gridShape.size(); ++i) {
    int64_t tileDim = (i < tileDims.size()) ? tileDims[i] : 1;
    tensorShape.push_back(gridShape[i] * shardShape[i] * tileDim);
  }
  return tensorShape;
}

static Type getScalarElementType(MemRefType memrefType) {
  Type elemType = memrefType.getElementType();
  if (auto tileType = mlir::dyn_cast<ttcore::TileType>(elemType)) {
    return tileType.getElementType();
  }
  return elemType;
}

static ttnn::TTNNLayoutAttr getTTNNLayoutFromDeviceLayout(MLIRContext *ctx,
                                                          Value memrefValue) {
  MemRefType memrefType = mlir::cast<MemRefType>(memrefValue.getType());

  auto bufferType = ttnn::BufferType::DRAM;
  if (auto memSpace = mlir::dyn_cast_if_present<ttcore::MemorySpaceAttr>(
          memrefType.getMemorySpace());
      memSpace && memSpace.getValue() == ttcore::MemorySpace::DeviceL1) {
    bufferType = ttnn::BufferType::L1;
  }

  auto deviceLayout = ttcore::getDeviceLayout(memrefValue);
  ArrayRef<int64_t> shardShape = deviceLayout.getShardShape(memrefType);

  auto shardMemref =
      MemRefType::get(shardShape, memrefType.getElementType(),
                      AffineMap::getMultiDimIdentityMap(shardShape.size(), ctx),
                      ttnn::BufferTypeAttr::get(ctx, bufferType));

  ttcore::GridAttr grid;
  ttnn::TensorMemoryLayout memLayoutEnum;
  ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(memrefType);

  if (mlir::isa<ttcore::InterleavedLayoutAttr>(memrefType.getLayout())) {
    grid = ttcore::GridAttr::get(ctx, SmallVector<int64_t>(2, 1));
    memLayoutEnum = ttnn::TensorMemoryLayout::Interleaved;
  } else {
    auto virtMap = d2m::utils::getVirtualGridForwardMapping(memrefValue);

    if (!virtMap) {
      grid = ttcore::GridAttr::get(ctx, gridShape);
      memLayoutEnum = ttnn::TensorMemoryLayout::BlockSharded;
    } else {
      grid = ttcore::GridAttr::get(
          ctx, d2m::utils::getPhysicalGridShape(memrefValue));
      TT_assertv(gridShape.size() >= 2u,
                 "Expected at least 2 dimensions in grid shape");
      int64_t gridY = gridShape[gridShape.size() - 2];
      memLayoutEnum = (gridY > 1) ? ttnn::TensorMemoryLayout::HeightSharded
                                  : ttnn::TensorMemoryLayout::WidthSharded;
    }
  }

  constexpr size_t kRank = 2;
  // This affine map only describes dim collapsing for rank > 2 tensors. Since
  // we can only recover the collapsed shape here, we can just set it to
  // identity.
  auto linearMap = AffineMap::getMultiDimIdentityMap(kRank, ctx);
  auto memLayout = ttnn::TensorMemoryLayoutAttr::get(ctx, memLayoutEnum);

  return {ttnn::TTNNLayoutAttr::get(
      ctx, linearMap, grid, shardMemref, memLayout, /*tensorMesh=*/nullptr,
      /*ignorePhysicalLayout=*/false, /*exactGrid=*/true)};
}

static RankedTensorType convertMemrefToTTNNTensor(MLIRContext *ctx,
                                                  Value memrefValue) {
  MemRefType memrefType = mlir::cast<MemRefType>(memrefValue.getType());
  TT_assertv(mlir::isa<ttcore::DeviceLayoutInterface>(memrefType.getLayout()),
             "memref must have device layout");

  auto ttnnLayoutAttr = getTTNNLayoutFromDeviceLayout(ctx, memrefValue);

  // Use scalar element type (unwrap tile if present) and logical element shape.
  return RankedTensorType::get(getTensorShape(memrefType),
                               getScalarElementType(memrefType),
                               ttnnLayoutAttr);
}
} // namespace detail

namespace {

// Helper struct to extract and return both IO and CB from a d2m.generic
// operand.
struct IOAndCB {
  Value io;
  Value cb;
};

static ttnn::ComputeKernelMathFidelity
convertMathFidelity(ttmetal::MathFidelity fidelity) {
  switch (fidelity) {
  case ttmetal::MathFidelity::LoFi:
    return ttnn::ComputeKernelMathFidelity::LoFi;
  case ttmetal::MathFidelity::HiFi2:
    return ttnn::ComputeKernelMathFidelity::HiFi2;
  case ttmetal::MathFidelity::HiFi3:
    return ttnn::ComputeKernelMathFidelity::HiFi3;
  case ttmetal::MathFidelity::HiFi4:
    return ttnn::ComputeKernelMathFidelity::HiFi4;
  }
  llvm_unreachable("Invalid MathFidelity");
}

static mlir::Attribute convertKernelArg(
    Builder &builder, const ttkernel::ArgAttr &arg,
    std::optional<ArrayRef<size_t>> operandToGlobalIOIndex = std::nullopt,
    size_t cbOffset = 0, size_t semaphoreOffset = 0) {
  switch (arg.getArgType()) {
  case ttkernel::ArgType::BufferAddress: {
    size_t idx = operandToGlobalIOIndex
                     ? (*operandToGlobalIOIndex)[arg.getOperandIndex()]
                     : arg.getOperandIndex();
    return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(idx);
  }
  case ttkernel::ArgType::CBPort: {
    return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
        arg.getOperandIndex() + cbOffset);
  }
  case ttkernel::ArgType::Semaphore: {
    return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
        arg.getOperandIndex() + semaphoreOffset);
  }
  case ttkernel::ArgType::NamedArgument: {
    return builder.getAttr<ttnn::KernelArgNamedArgAttr>(arg.getArgumentName(),
                                                        arg.getOperandIndex());
  }
  case ttkernel::ArgType::GlobalSemaphore: {
    return builder.getAttr<ttnn::KernelArgGlobalSemaphoreAttr>(
        arg.getOperandIndex());
  }
  }
}

static SmallVector<ttnn::KernelSemaphoreAttr>
createSemaphoreDescriptors(Builder &builder, const ArrayAttr &threads,
                           const ttnn::CoreRangeSetAttr &coreRangeSet,
                           const SymbolTable &symbolTable,
                           size_t semaphoreStartIndex = 0) {
  llvm::DenseSet<size_t> seenSemaphoreIndices;

  for (Attribute threadAttr : threads) {
    auto thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
    auto kernelFunc = symbolTable.lookup<func::FuncOp>(
        thread.getKernelSymbol().getRootReference());
    if (!kernelFunc) {
      continue;
    }

    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);
    if (!kernelSpec) {
      continue;
    }

    for (auto ctArg : kernelSpec.getCtArgs()) {
      if (ctArg.getArgType() == ttkernel::ArgType::Semaphore) {
        seenSemaphoreIndices.insert(ctArg.getOperandIndex());
      }
    }
  }
  size_t numSemaphores = seenSemaphoreIndices.size();
  if (numSemaphores > 0) {
    // Semaphore indices are assigned sequentially in D2MToTTKernel, so they
    // should be dense.
    size_t minIndex = *llvm::min_element(seenSemaphoreIndices);
    size_t maxIndex = *llvm::max_element(seenSemaphoreIndices);
    TT_assertv((minIndex == 0u && maxIndex == numSemaphores - 1),
               "Semaphore indices must be dense (0, 1, 2, ..., n-1)");
  }
  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors(numSemaphores);
  for (size_t i = 0; i < numSemaphores; ++i) {
    semaphoreDescriptors[i] = builder.getAttr<ttnn::KernelSemaphoreAttr>(
        /*id=*/semaphoreStartIndex + i, ttnn::KernelCoreType::Worker,
        coreRangeSet, /*initial_value=*/0);
  }

  return semaphoreDescriptors;
}

static SmallVector<mlir::Attribute> createKernelDescriptors(
    Builder &builder, const ArrayAttr &threads,
    const ttnn::CoreRangeSetAttr &coreRangeSet, const SymbolTable &symbolTable,
    ttmetal::MathFidelity mathFidelity,
    std::optional<ArrayRef<size_t>> operandToGlobalIOIndex = std::nullopt,
    size_t cbOffset = 0, size_t semaphoreOffset = 0) {
  SmallVector<mlir::Attribute> kernelConfigs(threads.size());
  int unassignedNocCounter = 0;
  for (const auto [i, thread] : llvm::enumerate(threads)) {
    const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

    // Get kernel args.
    SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
    auto kernelFunc =
        symbolTable.lookup<mlir::func::FuncOp>(kernelSymbol.getRootReference());
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);

    // Note: D2MToTTKernel will only populate kernelSpec with rtargs in the
    // ttnn-mode, however despite the name, they are actually common runtime
    // args. TTKernel ArgSpec does not have crt field, and the normal tt-metal
    // path doesn't use rt args at all.
    auto crtArgs = kernelSpec.getRtArgs();
    auto ctArgs = kernelSpec.getCtArgs();
    llvm::SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
    llvm::SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
    for (const auto [j, arg] : llvm::enumerate(crtArgs)) {
      kernelCRTArgs[j] = convertKernelArg(builder, arg, operandToGlobalIOIndex,
                                          cbOffset, semaphoreOffset);
    }
    for (const auto [j, arg] : llvm::enumerate(ctArgs)) {
      kernelCTArgs[j] = convertKernelArg(builder, arg, operandToGlobalIOIndex,
                                         cbOffset, semaphoreOffset);
    }

    // Create KernelDescriptor.
    switch (threadAttr.getThreadType()) {
    case d2m::ThreadType::Compute: {
      // TODO (vtangTT) #5032: support lowering to different compute configs.
      kernelConfigs[i] = builder.getAttr<ttnn::ComputeKernelAttr>(
          kernelSymbol, coreRangeSet,
          /*math_fidelity*/ convertMathFidelity(mathFidelity),
          /*fp32DestAccum*/ false,
          /*dst_full_sync_en*/ false,
          /*unpack_to_dest_mode*/
          ArrayRef<ttnn::ComputeKernelUnpackToDestMode>{
              ttnn::ComputeKernelUnpackToDestMode::Default},
          /*bfp8_pack_precise*/ false,
          /*math_approx_mode*/ false, kernelCRTArgs, kernelCTArgs);
      break;
    }
    case d2m::ThreadType::Datamovement: {
      int32_t nocIdx = threadAttr.getNocIndex();
      // For unassigned NOCs, alternate between NOC0 and NOC1.
      if (nocIdx < 0) {
        nocIdx = unassignedNocCounter++ % 2;
      }
      auto nocIndex = nocIdx == 0 ? ttnn::NocIndex::Noc0 : ttnn::NocIndex::Noc1;
      auto processor = nocIdx == 0 ? ttnn::DataMovementProcessor::RiscV1
                                   : ttnn::DataMovementProcessor::RiscV0;
      kernelConfigs[i] = builder.getAttr<ttnn::DataMovementKernelAttr>(
          kernelSymbol, coreRangeSet, processor, nocIndex,
          ttnn::NocMode::DedicatedNoc, kernelCRTArgs, kernelCTArgs);
      break;
    }
    case d2m::ThreadType::Unified: {
      // Unified threads should have been split by SplitUnifiedThread before
      // reaching this pass.
      llvm_unreachable("Unexpected thread type in backend conversion");
    }
    }
  }
  return kernelConfigs;
}

static SmallVector<ttnn::KernelCBAttr> createCBDescriptors(
    Builder &builder, const llvm::SmallVector<Value> &cbs,
    const ttcore::DeviceAttr &device,
    const ttnn::CoreRangeSetAttr &coreRangeSet, size_t cbStartIndex = 0,
    std::optional<ArrayRef<size_t>> globalIOIndexForEachCB = std::nullopt) {
  if (cbs.empty()) {
    llvm_unreachable("Expected circular buffers.");
  }

  MLIRContext *ctx = builder.getContext();
  llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors(cbs.size());

  for (auto [i, cb] : llvm::enumerate(cbs)) {
    auto cb_memref = dyn_cast<MemRefType>(cb.getType());
    TT_assertv(mlir::isa<ttcore::TileType>(cb_memref.getElementType()),
               "Only TileType supported.");
    ttcore::DataType dtype =
        ttcore::elementTypeToDataType(cb_memref.getElementType());
    size_t pageSize = device.getMemrefCBPageSizeBytes(cb_memref);
    size_t totalSize = device.getMemrefSizeBytes(cb_memref, pageSize, true);

    ttnn::KernelCBFormatAttr cbFormat =
        ttnn::KernelCBFormatAttr::get(ctx, cbStartIndex + i, dtype, pageSize);

    ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;

    // TODO (#7158): This is brittle. Ideally we should specifically identify
    // outputs and handle them separately from inputs, but that will require a
    // larger refactor of this pass.
    bool isAliasedOutput =
        mlir::dyn_cast_if_present<memref::AllocOp>(cb.getDefiningOp()) &&
        llvm::none_of(cb.getUsers(), [](Operation *user) {
          return mlir::isa<d2m::StreamLayoutOp>(user);
        });

    if ((mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
             cb.getDefiningOp()) ||
         isAliasedOutput) &&
        ttcore::getMemorySpace(cb_memref) != ttcore::MemorySpace::DeviceDRAM) {
      size_t tensorIndex =
          globalIOIndexForEachCB ? (*globalIOIndexForEachCB)[i] : i;
      globalCBIndexOfTensor =
          ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, tensorIndex);
    }
    cbDescriptors[i] = ttnn::KernelCBAttr::get(
        ctx, totalSize, coreRangeSet, {cbFormat}, globalCBIndexOfTensor);
  }

  return cbDescriptors;
}

// Extract IO and CB from a generic operand.
// - origOperand: the original operand from op->getOperands()
// - convertedOperand: the remapped operand from adaptor.getOperands()
// IO comes from converted operands (e.g., ttnn.empty results).
// CB comes from original operands (memref types for CB descriptors).
static IOAndCB extractIOAndCBFromGenericOperand(Value origOperand,
                                                Value convertedOperand) {
  if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
          origOperand.getDefiningOp())) {
    auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
        streamLayoutOp.getInput().getDefiningOp());
    TT_assertv(castOp,
               "Expected TTNNMetalLayoutCastOp producing stream input.");
    return {castOp.getOperand(), streamLayoutOp.getStorage()};
  }

  if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
          origOperand.getDefiningOp())) {
    return {castOp.getOperand(), origOperand};
  }

  if (auto viewOp = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
          origOperand.getDefiningOp())) {
    if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
            viewOp.getInput().getDefiningOp())) {
      TT_assertv(castOp,
                 "Expected TTNNMetalLayoutCastOp producing view input.");
      return {castOp.getOperand(), origOperand};
    }
    if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
            viewOp.getInput().getDefiningOp())) {
      auto innerCastOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
          streamLayoutOp.getInput().getDefiningOp());
      TT_assertv(innerCastOp,
                 "Expected TTNNMetalLayoutCastOp producing stream input.");
      return {innerCastOp.getOperand(), viewOp.getInput()};
    }

    if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
            viewOp.getInput().getDefiningOp())) {
      // This is a view on top of the output of a previous generic. This case
      // applies to the generic that implements the final ttir.to_layout and
      // converts the output tensor to the user-selected output layout.
      return {convertedOperand, origOperand};
    }
  }

  if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
          origOperand.getDefiningOp())) {
    // There are intermediate tensors that have been bufferized. The operand
    // will have a DeviceLayout, not a TTNNLayout.
    return {convertedOperand, origOperand};
  }

  llvm_unreachable(
      "Expected stream_layout, view_layout, memref.alloc, ttnn.empty, or "
      "cast op as operand.");
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(context),
        mathFidelity(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // The ttnn.generic op requires ttnn tensor operands. Defer rewriting until
    // memref.alloc operands are converted so we have the memref->ttnn
    // tensor translations.
    for (auto [orig, converted] :
         llvm::zip(op->getOperands(), adaptor.getOperands())) {
      if (mlir::isa_and_present<memref::AllocOp>(orig.getDefiningOp()) &&
          orig == converted) {
        return rewriter.notifyMatchFailure(
            op, "waiting for memref.alloc operands to be converted");
      }
    }
    // Generic ops inside spatial ops should be handled by the spatial rewriter
    // first.
    if (op->getParentRegion() &&
        isa<d2m::SpatialOp>(op->getParentRegion()->getParentOp())) {
      return rewriter.notifyMatchFailure(
          op, "generic op is inside a spatial op and should be handled by the "
              "spatial rewriter first");
    }

    MLIRContext *ctx = rewriter.getContext();
    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    ttcore::GridAttr opGrid = op.getGrid();
    llvm::SmallVector<int64_t> endCoreRange;
    if (!opGrid.getMapping().isEmpty()) {
      // The genericOp has a virtual grid. We need to recover the original
      // physical grid.
      auto output = op.getOutputs()[0];

      auto physicalGridShape = d2m::utils::getPhysicalGridShape(output);
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      endCoreRange = {physicalGridShape[1] - 1, physicalGridShape[0] - 1};
    } else {
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      endCoreRange = {opGrid.getShape()[1] - 1, opGrid.getShape()[0] - 1};
    }

    ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx,
        ttnn::CoreRangeAttr::get(
            ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
            ttnn::CoreCoordAttr::get(ctx, endCoreRange[0], endCoreRange[1])));

    llvm::SmallVector<Value> ios(op.getInputsAndOutputs().size());
    llvm::SmallVector<Value> cbs(op.getInputsAndOutputs().size());
    llvm::SmallVector<Value> adaptorInputsAndOutputs(
        adaptor.getOperands().begin(), adaptor.getOperands().begin() +
                                           adaptor.getInputs().size() +
                                           adaptor.getOutputs().size());
    for (auto [i, orig, converted] :
         llvm::enumerate(op.getInputsAndOutputs(), adaptorInputsAndOutputs)) {
      auto [io, cb] = extractIOAndCBFromGenericOperand(orig, converted);
      ios[i] = io;
      cbs[i] = cb;
    }

    llvm::SmallVector<Value> additionalArgs;
    for (auto operand : op.getAdditionalArgs()) {
      if (mlir::isa<ttnn::GlobalSemaphoreType>(operand.getType())) {
        additionalArgs.push_back(operand);
      } else if (mlir::isa<RankedTensorType>(operand.getType())) {
        additionalArgs.push_back(operand);
      } else {
        op.emitOpError(
            "unexpected operand type in d2m.generic's additionalArgs: ")
            << operand.getType();
        return failure();
      }
    }

    // Create CB descriptors.
    llvm::SmallVector<ttnn::KernelCBAttr> cbDescriptors =
        createCBDescriptors(rewriter, cbs, device, coreRangeSet);

    // Create KernelDescriptors.
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    llvm::SmallVector<mlir::Attribute> kernelDescriptors =
        createKernelDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                opSymTable, this->mathFidelity);

    // Extract semaphore descriptors from kernel functions.
    llvm::SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
        createSemaphoreDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                   opSymTable);

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

    rewriter.replaceOpWithNewOp<ttnn::GenericOp>(
        op, ios, additionalArgs, program, ttnn::MemoryConfigAttr());
    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity;
};

static ttnn::CoreRangeSetAttr
ttcoreRangeToTTNNCoreRangeSet(MLIRContext *ctx, ttcore::CoreRangeAttr range) {
  auto start = range.getStartCoord();
  auto end = range.getEndCoord();
  auto startCoord = ttnn::CoreCoordAttr::get(ctx, start.getX(), start.getY());
  auto endCoord = ttnn::CoreCoordAttr::get(ctx, end.getX(), end.getY());
  auto coreRange = ttnn::CoreRangeAttr::get(ctx, startCoord, endCoord);
  return ttnn::CoreRangeSetAttr::get(ctx, coreRange);
}

static std::pair<SmallVector<Value>, ttnn::ProgramAttr>
buildUnifiedProgramFromGenerics(
    Builder &builder, ConversionPatternRewriter &rewriter,
    d2m::SpatialOp spatialOp, ArrayRef<d2m::GenericOp> generics,
    ArrayRef<ttnn::CoreRangeSetAttr> regionCoreRangeSets,
    const ttcore::DeviceAttr &device, SymbolTable &symbolTable,
    ttmetal::MathFidelity mathFidelity) {
  MLIRContext *ctx = builder.getContext();
  llvm::SmallDenseMap<Value, size_t> valueToGlobalIOIndex;
  SmallVector<Value> globalIos;

  for (Value v : spatialOp.getInputs()) {
    Value io =
        extractIOAndCBFromGenericOperand(v, rewriter.getRemappedValue(v)).io;
    if (valueToGlobalIOIndex.try_emplace(io, globalIos.size()).second) {
      globalIos.push_back(io);
    }
  }
  for (Value v : spatialOp.getOutputs()) {
    Value io =
        extractIOAndCBFromGenericOperand(v, rewriter.getRemappedValue(v)).io;
    if (valueToGlobalIOIndex.try_emplace(io, globalIos.size()).second) {
      globalIos.push_back(io);
    }
  }

  SmallVector<SmallVector<size_t>> perGenericIOIndices(generics.size());
  for (const auto [gIdx, g] : llvm::enumerate(generics)) {
    auto &indices = perGenericIOIndices[gIdx];
    auto operands = g->getOperands();
    indices.reserve(operands.size());
    for (Value operand : operands) {
      Value io = extractIOAndCBFromGenericOperand(
                     operand, rewriter.getRemappedValue(operand))
                     .io;
      indices.push_back(valueToGlobalIOIndex.find(io)->second);
    }
  }

  SmallVector<ttnn::KernelCBAttr> allCBs;
  SmallVector<ttnn::KernelSemaphoreAttr> allSemaphores;
  SmallVector<Attribute> allKernels;
  size_t cbNext = 0;
  size_t semNext = 0;

  for (const auto [gIdx, g] : llvm::enumerate(generics)) {
    auto operands = g->getOperands();
    llvm::SmallVector<Value> ios(operands.size());
    llvm::SmallVector<Value> cbs(operands.size());
    for (auto [i, operand] : llvm::enumerate(operands)) {
      auto [io, cb] = extractIOAndCBFromGenericOperand(
          operand, rewriter.getRemappedValue(operand));
      ios[i] = io;
      cbs[i] = cb;
    }

    size_t cbOffset = cbNext;
    ArrayRef<size_t> globalIOForCBs = perGenericIOIndices[gIdx];
    auto cbDescs =
        createCBDescriptors(builder, cbs, device, regionCoreRangeSets[gIdx],
                            cbOffset, globalIOForCBs);
    allCBs.append(cbDescs.begin(), cbDescs.end());
    cbNext = allCBs.size();

    size_t semOffset = semNext;
    auto threads = g->getAttrOfType<ArrayAttr>("threads");
    auto semDescs = createSemaphoreDescriptors(
        builder, threads, regionCoreRangeSets[gIdx], symbolTable, semOffset);
    allSemaphores.append(semDescs.begin(), semDescs.end());
    semNext = allSemaphores.size();

    ArrayRef<size_t> ioMap = perGenericIOIndices[gIdx];
    auto kernelDescs = createKernelDescriptors(
        builder, threads, regionCoreRangeSets[gIdx], symbolTable, mathFidelity,
        ioMap, cbOffset, semOffset);
    allKernels.append(kernelDescs.begin(), kernelDescs.end());
  }

  ttnn::ProgramAttr program =
      ttnn::ProgramAttr::get(ctx, allKernels, allCBs, allSemaphores);
  return {std::move(globalIos), program};
}

class D2MSpatialRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  D2MSpatialRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::SpatialOp>(context),
        mathFidelity(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op, d2m::SpatialOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getNumRegions() == 0) {
      return rewriter.notifyMatchFailure(op, "no regions");
    }

    SmallVector<d2m::GenericOp> generics;
    generics.reserve(op.getNumRegions());
    for (Region &r : op.getRegions()) {
      auto genericOps = r.front().getOps<d2m::GenericOp>();
      if (!llvm::hasSingleElement(genericOps)) {
        return rewriter.notifyMatchFailure(
            op, "each region must contain exactly one d2m.generic op");
      }
      generics.push_back(*genericOps.begin());
    }

    auto device = ttcore::lookupDevice(op->getParentOp());
    if (!device) {
      return rewriter.notifyMatchFailure(op, "no device");
    }

    auto gridRanges = op.getGridRanges().getCoreRanges();
    SmallVector<ttnn::CoreRangeSetAttr> regionCoreRangeSets;
    regionCoreRangeSets.reserve(gridRanges.size());
    for (ttcore::CoreRangeAttr range : gridRanges) {
      regionCoreRangeSets.push_back(
          ttcoreRangeToTTNNCoreRangeSet(rewriter.getContext(), range));
    }

    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto [globalIos, program] = buildUnifiedProgramFromGenerics(
        rewriter, rewriter, op, generics, regionCoreRangeSets, device,
        symbolTable, mathFidelity);

    llvm::SmallVector<Value> additionalArgs;
    for (d2m::GenericOp g : generics) {
      for (Value arg : g.getAdditionalArgs()) {
        // ToDo : Need to handle the case where the additional arg is a global
        // semaphore and they are shared across the generic ops in the spatial
        // op.
        additionalArgs.push_back(rewriter.getRemappedValue(arg));
      }
    }

    rewriter.replaceOpWithNewOp<ttnn::GenericOp>(
        op, globalIos, additionalArgs, program, ttnn::MemoryConfigAttr());
    return success();
  }

private:
  ttmetal::MathFidelity mathFidelity;
};
} // namespace

namespace {
class TTNNMetalLayoutCastRewriter
    : public OpConversionPattern<ttir::TTNNMetalLayoutCastOp> {
public:
  using OpConversionPattern<ttir::TTNNMetalLayoutCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::TTNNMetalLayoutCastOp op,
                  ttir::TTNNMetalLayoutCastOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (auto inner =
            op.getOperand().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
      // At this point (D2M→TTNN conversion), the D2M pipeline has already
      // materialized all data movement for VGMs. Back-to-back casts can
      // be safely collapsed even when they carry VGM attributes.
      rewriter.replaceOp(op, inner.getOperand());
    } else if (auto inner =
                   op.getOperand().getDefiningOp<d2m::StreamLayoutOp>()) {
      // Match the pattern cast(stream(cast(output_tensor))) and rewrite as just
      // output_tensor.
      if (auto inner2 =
              inner.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        rewriter.replaceOp(op, inner2.getOperand());
      }
    } else if (auto inner =
                   op.getOperand().getDefiningOp<d2m::ViewLayoutOp>()) {
      // Match the pattern cast(view(cast(output_tensor))) and rewrite as just
      // output_tensor.
      if (auto inner2 =
              inner.getInput().getDefiningOp<ttir::TTNNMetalLayoutCastOp>()) {
        rewriter.replaceOp(op, inner2.getOperand());
      }
    }
    return success();
  };
};
} // namespace

namespace {
class StreamLayoutRewriter : public OpConversionPattern<d2m::StreamLayoutOp> {
public:
  using OpConversionPattern<d2m::StreamLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::StreamLayoutOp op, d2m::StreamLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
class ViewLayoutRewriter : public OpConversionPattern<d2m::ViewLayoutOp> {
public:
  using OpConversionPattern<d2m::ViewLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ViewLayoutOp op, d2m::ViewLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  };
};
} // namespace

namespace {
class D2MEmptyRewriter : public OpConversionPattern<d2m::EmptyOp> {
public:
  using OpConversionPattern<d2m::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::EmptyOp op, d2m::EmptyOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();
    auto shape = ttnn::ShapeAttr::get(ctx, tensorType.getShape());

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::EmptyOp>(op, tensorType, device, shape,
                                               dtype, layout, memcfg);
    return success();
  };
};
} // namespace

namespace {
class D2MFullRewriter : public OpConversionPattern<d2m::FullOp> {
public:
  using OpConversionPattern<d2m::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::FullOp op, d2m::FullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    auto tensorType = cast<RankedTensorType>(op.getResult().getType());
    auto encoding = tensorType.getEncoding();

    // Convert DenseI32ArrayAttr shape to ttnn::ShapeAttr
    auto shapeI32 = adaptor.getShape();
    SmallVector<int64_t> shapeI64(shapeI32.begin(), shapeI32.end());
    auto shape = ttnn::ShapeAttr::get(ctx, shapeI64);

    ttcore::DataTypeAttr dtype;
    ttnn::LayoutAttr layout;
    ttnn::MemoryConfigAttr memcfg;

    // Reuses the existing ttnn.get_device op if present, else create one.
    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
    auto deviceAttr = ttcore::lookupDevice(op);

    // Handle both TTNNLayoutAttr and TTNNNDLayoutAttr
    if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
      memcfg =
          ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid());
    } else if (auto ndLayoutAttr =
                   mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
      dtype = ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType());
      layout = ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
      auto bufferType =
          ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
      auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
      memcfg = ttnn::MemoryConfigAttr::get(
          ctx, ndLayoutAttr.getMemLayout(), bufferType,
          /*shardSpec=*/std::nullopt, ndShardSpec);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported encoding type");
    }

    rewriter.replaceOpWithNewOp<ttnn::FullOp>(op, tensorType, device, shape,
                                              adaptor.getFillValue(), dtype,
                                              layout, memcfg);
    return success();
  };
};
} // namespace

namespace {

class MemrefAllocRewriter : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, memref::AllocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    MLIRContext *ctx = rewriter.getContext();
    MemRefType memrefType = op.getMemref().getType();
    bool isBackingGlobalSemaphore =
        llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
          return mlir::isa<d2m::CreateGlobalSemaphoreOp>(user);
        });

    if (isBackingGlobalSemaphore) {
      // Check if this is a global semaphore backing buffer (used by
      // d2m.create_global_semaphore). If so, erase the alloc/dealloc since TTNN
      // creates the global semaphore buffer itself.
      for (Operation *user :
           llvm::make_early_inc_range(op.getResult().getUsers())) {
        if (mlir::isa<memref::DeallocOp>(user)) {
          rewriter.eraseOp(user);
        }
      }
      rewriter.eraseOp(op);
    } else if (mlir::isa_and_present<ttcore::DeviceLayoutInterface>(
                   memrefType.getLayout())) {
      auto deviceAttr = ttcore::lookupDevice(op);
      if (!deviceAttr) {
        return rewriter.notifyMatchFailure(op,
                                           "could not find device attribute");
      }

      auto convertedTensorType = detail::convertMemrefToTTNNTensor(
          rewriter.getContext(), op.getMemref());
      auto convertedLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(convertedTensorType.getEncoding());

      // Find and handle users of the alloc result. We need to:
      // 1. Erase any dealloc ops
      // 2. Replace ttnn_metal_layout_cast ops with the empty result directly
      llvm::SmallVector<memref::DeallocOp> deallocsToErase;
      llvm::SmallVector<ttir::TTNNMetalLayoutCastOp> castsToReplace;

      for (Operation *user : op.getMemref().getUsers()) {
        if (auto deallocOp = mlir::dyn_cast<memref::DeallocOp>(user)) {
          deallocsToErase.push_back(deallocOp);
        } else if (auto castOp =
                       mlir::dyn_cast<ttir::TTNNMetalLayoutCastOp>(user)) {
          castsToReplace.push_back(castOp);
        }
      }

      // Determine the tensor type for the ttnn.empty op. If there's a
      // ttnn_metal_layout_cast user, use its result type to preserve the
      // uncollapsed shape. Otherwise, use the converted type.
      RankedTensorType emptyTensorType = convertedTensorType;
      if (!castsToReplace.empty()) {
        auto castResultType = mlir::cast<RankedTensorType>(
            castsToReplace[0].getResult().getType());
        auto castLayoutAttr =
            mlir::cast<ttnn::TTNNLayoutAttr>(castResultType.getEncoding());

        // Assert that the converted type is compatible with the cast result
        // type. Cannot assert on shape because we cannot recover the
        // uncollapsed shape, but we can assert on volume.
        TT_assertv(
            castResultType.getNumElements() ==
                convertedTensorType.getNumElements(),
            "ttnn_metal_layout_cast and converted type must have the same "
            "volume");

        TT_assertv(
            castLayoutAttr.getBufferType() ==
                convertedLayoutAttr.getBufferType(),
            "ttnn_metal_layout_cast and converted type must have the same "
            "buffer type");

        TT_assertv(
            castLayoutAttr.getShardShape() ==
                convertedLayoutAttr.getShardShape(),
            "ttnn_metal_layout_cast and converted type must have the same "
            "shard shape");

        TT_assertv(
            castLayoutAttr.getGrid().getShape() ==
                convertedLayoutAttr.getGrid().getShape(),
            "ttnn_metal_layout_cast and converted type must have the same "
            "grid shape");

        emptyTensorType = castResultType;
      }

      auto emptyLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(emptyTensorType.getEncoding());

      auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
      auto memcfg = ttnn::MemoryConfigAttr::get(emptyLayoutAttr,
                                                deviceAttr.getWorkerGrid());

      auto emptyOp = rewriter.create<ttnn::EmptyOp>(
          op.getLoc(), emptyTensorType, device,
          ttnn::ShapeAttr::get(ctx, emptyTensorType.getShape()),
          ttcore::DataTypeAttr::get(ctx, emptyLayoutAttr.getDataType()),
          ttnn::LayoutAttr::get(ctx, emptyLayoutAttr.getLayout()), memcfg);

      for (auto deallocOp : deallocsToErase) {
        rewriter.eraseOp(deallocOp);
      }

      for (auto castOp : castsToReplace) {
        rewriter.replaceOp(castOp, emptyOp.getResult());
      }

      // Replace the alloc with the empty result. This registers the value
      // mapping so that other patterns (like D2MGenericRewriter) can get the
      // converted value through the adaptor.
      rewriter.replaceOp(op, emptyOp.getResult());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "memref alloc does not correspond to "
                                         "a ttnn tensor or global semaphore");
    }
    return success();
  }
};
} // namespace

namespace {
class D2MCreateGlobalSemaphoreRewriter
    : public OpConversionPattern<d2m::CreateGlobalSemaphoreOp> {
public:
  using OpConversionPattern<d2m::CreateGlobalSemaphoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::CreateGlobalSemaphoreOp op,
                  d2m::CreateGlobalSemaphoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto allocOp = op.getInput().getDefiningOp<memref::AllocOp>();
    assert(
        allocOp &&
        "No memref alloc found for CreateGlobalSemaphoreOp's input, failing.");

    // Get core range from memref shape.
    auto gridShape = ttcore::getGridShape(op.getInput());
    auto coreRange = ttnn::CoreRangeAttr::get(
        rewriter.getContext(),
        ttnn::CoreCoordAttr::get(rewriter.getContext(), 0, 0),
        ttnn::CoreCoordAttr::get(rewriter.getContext(), gridShape[0] - 1,
                                 gridShape[1] - 1));
    rewriter.replaceOpWithNewOp<ttnn::CreateGlobalSemaphoreOp>(
        op, adaptor.getValueAttr(), coreRange);
    return success();
  }
};
} // namespace

namespace {
class D2MResetGlobalSemaphoreRewriter
    : public OpConversionPattern<d2m::ResetGlobalSemaphoreOp> {
public:
  using OpConversionPattern<d2m::ResetGlobalSemaphoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ResetGlobalSemaphoreOp op,
                  d2m::ResetGlobalSemaphoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto createGlobalSemaphoreOp =
        op.getSemaphore().getDefiningOp<d2m::CreateGlobalSemaphoreOp>();
    assert(createGlobalSemaphoreOp &&
           "No create global semaphore op found for ResetGlobalSemaphoreOp's "
           "input, failing.");

    rewriter.replaceOpWithNewOp<ttnn::ResetGlobalSemaphoreOp>(
        op, adaptor.getSemaphore(), adaptor.getValueAttr());
    return success();
  }
};
} // namespace

void populateD2MToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttmetal::MathFidelity mathFidelity) {
  patterns.add<MemrefAllocRewriter>(ctx);
  patterns.add<D2MSpatialRewriter>(ctx, mathFidelity);
  patterns.add<D2MGenericRewriter>(ctx, mathFidelity);
  patterns
      .add<TTNNMetalLayoutCastRewriter, D2MEmptyRewriter, D2MFullRewriter,
           StreamLayoutRewriter, ViewLayoutRewriter,
           D2MCreateGlobalSemaphoreRewriter, D2MResetGlobalSemaphoreRewriter>(
          ctx);
}
} // namespace mlir::tt
