// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt {

namespace {

// Helper struct to extract and return both IO and CB from a d2m.generic
// operand.
struct IOAndCB {
  Value io;
  Value cb;
};
//===----------------------------------------------------------------------===//
// Helpers for SpatialOp: remap kernel/CB buffer indices when merging regions.
// Each region's GenericOp uses CB indices 0,1,...; after merge we assign
// contiguous global indices, so kernel args and CB format buffer_index must
// be updated.
//===----------------------------------------------------------------------===//

static SmallVector<mlir::Attribute>
remapKernelArgsCBIndices(MLIRContext *ctx, ArrayRef<mlir::Attribute> args,
                         size_t cbIndexOffset) {
  SmallVector<mlir::Attribute> remapped;
  for (mlir::Attribute arg : args) {
    if (auto cbArg = mlir::dyn_cast<ttnn::KernelArgCBBufferIndexAttr>(arg)) {
      size_t newIndex = cbArg.getBufferIndex() + cbIndexOffset;
      remapped.push_back(ttnn::KernelArgCBBufferIndexAttr::get(ctx, newIndex));
    } else {
      remapped.push_back(arg);
    }
  }
  return remapped;
}

// Remap a kernel descriptor's CB buffer indices by adding cbIndexOffset.
static mlir::Attribute
remapKernelDescriptorCBIndices(mlir::Attribute kernelAttr,
                               size_t cbIndexOffset) {
  MLIRContext *ctx = kernelAttr.getContext();

  if (auto computeKernel =
          mlir::dyn_cast<ttnn::ComputeKernelAttr>(kernelAttr)) {
    auto ctArgs =
        remapKernelArgsCBIndices(ctx, computeKernel.getCtArgs(), cbIndexOffset);
    auto commonRtArgs = remapKernelArgsCBIndices(
        ctx, computeKernel.getCommonRtArgs(), cbIndexOffset);
    return ttnn::ComputeKernelAttr::get(
        ctx, computeKernel.getSymbolRef(), computeKernel.getCoreRanges(),
        computeKernel.getMathFidelity(), computeKernel.getFp32DestAccEn(),
        computeKernel.getDstFullSyncEn(), computeKernel.getUnpackToDestModes(),
        computeKernel.getBfp8PackPrecise(), computeKernel.getMathApproxMode(),
        commonRtArgs, computeKernel.getRtArgs(), ctArgs);
  }

  if (auto readKernel = mlir::dyn_cast<ttnn::ReadKernelAttr>(kernelAttr)) {
    auto ctArgs =
        remapKernelArgsCBIndices(ctx, readKernel.getCtArgs(), cbIndexOffset);
    auto commonRtArgs = remapKernelArgsCBIndices(
        ctx, readKernel.getCommonRtArgs(), cbIndexOffset);
    return ttnn::ReadKernelAttr::get(ctx, readKernel.getSymbolRef(),
                                     readKernel.getCoreRanges(), commonRtArgs,
                                     readKernel.getRtArgs(), ctArgs);
  }

  if (auto writeKernel = mlir::dyn_cast<ttnn::WriteKernelAttr>(kernelAttr)) {
    auto ctArgs =
        remapKernelArgsCBIndices(ctx, writeKernel.getCtArgs(), cbIndexOffset);
    auto commonRtArgs = remapKernelArgsCBIndices(
        ctx, writeKernel.getCommonRtArgs(), cbIndexOffset);
    return ttnn::WriteKernelAttr::get(ctx, writeKernel.getSymbolRef(),
                                      writeKernel.getCoreRanges(), commonRtArgs,
                                      writeKernel.getRtArgs(), ctArgs);
  }

  return kernelAttr;
}

// Resolve a GenericOp operand (stream_layout or cast) to the TTNN io value and
// the CB storage value. Used by both GenericOp and SpatialOp operand
// extraction.
static void resolveOperandToIoAndCb(Value operand, Value &io, Value &cb) {
  if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
          operand.getDefiningOp())) {
    if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
            streamLayoutOp.getInput().getDefiningOp())) {
      io = castOp.getOperand();
    } else {
      llvm_unreachable(
          "Expected TTNNMetalLayoutCastOp producing stream input.");
    }
    cb = streamLayoutOp.getStorage();
  } else if (auto castOp =
                 mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                     operand.getDefiningOp())) {
    io = castOp.getOperand();
    cb = operand;
  } else {
    llvm_unreachable("Expected stream_layout or cast op as operand.");
  }
}

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

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(context),
        mathFidelity(mathFidelity) {}

  static mlir::Attribute convertKernelArg(Builder &builder,
                                          const ttkernel::ArgAttr &arg) {
    switch (arg.getArgType()) {
    case ttkernel::ArgType::BufferAddress: {
      return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::CBPort: {
      return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::Semaphore: {
      return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(
          arg.getOperandIndex());
    }
    case ttkernel::ArgType::NamedArgument: {
      return builder.getAttr<ttnn::KernelArgNamedArgAttr>(
          arg.getArgumentName(), arg.getOperandIndex());
    }
    }
  }

  static SmallVector<ttnn::KernelSemaphoreAttr>
  createSemaphoreDescriptors(Builder &builder, const ArrayAttr &threads,
                             const ttnn::CoreRangeSetAttr &coreRangeSet,
                             const SymbolTable &symbolTable) {
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
          /*id=*/i, ttnn::KernelCoreType::Worker, coreRangeSet,
          /*initial_value=*/0);
    }

    return semaphoreDescriptors;
  }

  static SmallVector<mlir::Attribute>
  createKernelDescriptors(Builder &builder, const ArrayAttr &threads,
                          const ttnn::CoreRangeSetAttr &coreRangeSet,
                          const SymbolTable &symbolTable,
                          ttmetal::MathFidelity mathFidelity) {
    SmallVector<mlir::Attribute> kernelConfigs(threads.size());
    int nocIndex = 0;
    for (const auto [i, thread] : llvm::enumerate(threads)) {
      const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

      // Get kernel args.
      SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
      auto kernelFunc = symbolTable.lookup<mlir::func::FuncOp>(
          kernelSymbol.getRootReference());
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
      for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
        kernelCRTArgs[i] = convertKernelArg(builder, arg);
      }
      for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
        kernelCTArgs[i] = convertKernelArg(builder, arg);
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
      // TODO (vtangTT) #5033: fix this assumption that order is
      // read->write->compute; nocIndex == 0 for read, nocIndex == 1 for write.
      case d2m::ThreadType::Datamovement: {
        TT_assert(nocIndex < 2);
        if (nocIndex == 0) {
          kernelConfigs[i] = builder.getAttr<ttnn::ReadKernelAttr>(
              kernelSymbol, coreRangeSet, kernelCRTArgs, kernelCTArgs);
        } else {
          kernelConfigs[i] = builder.getAttr<ttnn::WriteKernelAttr>(
              kernelSymbol, coreRangeSet, kernelCRTArgs, kernelCTArgs);
        }
        nocIndex++;
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

  static SmallVector<ttnn::KernelCBAttr>
  createCBDescriptors(Builder &builder, const llvm::SmallVector<Value> &cbs,
                      const ttcore::DeviceAttr &device,
                      const ttnn::CoreRangeSetAttr &coreRangeSet) {
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
      size_t numPages = device.getMemrefCBNumPages(cb_memref);

      ttnn::KernelCBFormatAttr cbFormat =
          ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);

      ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
      if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
              cb.getDefiningOp())) {
        // Input is not streamed, thus buffer must be aliased.
        TT_assertv(ttcore::getMemorySpace(cb_memref) ==
                       ttcore::MemorySpace::DeviceL1,
                   "Can only alias L1 buffers.");
        globalCBIndexOfTensor =
            ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, i);
      }
      cbDescriptors[i] =
          ttnn::KernelCBAttr::get(ctx, numPages * pageSize, coreRangeSet,
                                  {cbFormat}, globalCBIndexOfTensor);
    }

    return cbDescriptors;
  }

  static IOAndCB extractIOAndCBFromGenericOperand(Value operand) {
    if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
            operand.getDefiningOp())) {
      auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
          streamLayoutOp.getInput().getDefiningOp());
      TT_assertv(castOp,
                 "Expected TTNNMetalLayoutCastOp producing stream input.");
      return {castOp.getOperand(), streamLayoutOp.getStorage()};
    }

    if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
            operand.getDefiningOp())) {
      return {castOp.getOperand(), operand};
    }

    if (auto viewOp = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            operand.getDefiningOp())) {
      if (auto castOp = mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
              viewOp.getInput().getDefiningOp())) {
        TT_assertv(castOp,
                   "Expected TTNNMetalLayoutCastOp producing view input.");
        return {castOp.getOperand(), operand};
      }
      if (auto streamLayoutOp = mlir::dyn_cast_if_present<d2m::StreamLayoutOp>(
              viewOp.getInput().getDefiningOp())) {
        auto innerCastOp =
            mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                streamLayoutOp.getInput().getDefiningOp());
        TT_assertv(innerCastOp,
                   "Expected TTNNMetalLayoutCastOp producing stream input.");
        return {innerCastOp.getOperand(), viewOp.getInput()};
      }
    }

    llvm_unreachable(
        "Expected stream_layout, view_layout, or cast op as operand.");
  }
  
  static ttnn::CoreRangeSetAttr createCoreRangeSet(
      mlir::Builder &builder, llvm::ArrayRef<int64_t> gridSize,
      std::optional<llvm::ArrayRef<int64_t>> startCoord = std::nullopt) {

    llvm::SmallVector<int64_t, 4> defaultStart;
    llvm::ArrayRef<int64_t> startCoordRef;

    if (startCoord.has_value()) {
      startCoordRef = *startCoord;
    } else {
      defaultStart.assign(gridSize.size(), 0);
      startCoordRef = defaultStart;
    }

    return ttnn::CoreRangeSetAttr::get(
        builder.getContext(),
        ttnn::CoreRangeAttr::get(
            builder.getContext(),
            ttnn::CoreCoordAttr::get(builder.getContext(), startCoordRef[0],
                                     startCoordRef[1]),
            ttnn::CoreCoordAttr::get(builder.getContext(),
                                     startCoordRef[0] + gridSize[0] - 1,
                                     startCoordRef[1] + gridSize[1] - 1)));
  }

  // Structure to hold all descriptors and I/O values for a GenericOp.
  struct GenericOpDescriptors {
    SmallVector<mlir::Attribute> kernelDescriptors;
    SmallVector<ttnn::KernelCBAttr> cbDescriptors;
    SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors;
    SmallVector<Value> ios;
  };

  // Extract I/O operands and circular buffers from a GenericOp.
  static void extractOperandsFromGenericOp(d2m::GenericOp op,
                                           llvm::SmallVector<Value> &ios,
                                           llvm::SmallVector<Value> &cbs) {
    const size_t size = op.getOperands().size();
    ios.resize(size);
    cbs.resize(size);
    for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
      resolveOperandToIoAndCb(operand, ios[i], cbs[i]);
    }
  }

  // Extract inputs and outputs separately from a GenericOp. If `cbs` is
  // non-null, also fills the CB value for each operand (inputs then outputs).
  static void extractInputsAndOutputsFromGenericOp(
      d2m::GenericOp op, llvm::SmallVector<Value> &inputs,
      llvm::SmallVector<Value> &outputs,
      llvm::SmallVector<Value> *cbs = nullptr) {
    for (Value input : op.getInputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(input, ioValue, cbValue);
      inputs.push_back(ioValue);
      if (cbs) {
        cbs->push_back(cbValue);
      }
    }
    for (Value output : op.getOutputs()) {
      Value ioValue;
      Value cbValue;
      resolveOperandToIoAndCb(output, ioValue, cbValue);
      outputs.push_back(ioValue);
      if (cbs) {
        cbs->push_back(cbValue);
      }
    }
  }

  // Compute grid size from a GenericOp.
  static llvm::SmallVector<int64_t>
  computeGridSizeFromGenericOp(d2m::GenericOp op) {
    ttcore::GridAttr opGrid = op.getGrid();
    llvm::SmallVector<int64_t> gridSize;

    if (!opGrid.getMapping().isEmpty()) {
      // The genericOp has a virtual grid. We need to recover the original
      // physical grid.
      auto output = op.getOutputs()[0];
      mlir::ShapedType outputType =
          mlir::cast<mlir::ShapedType>(output.getType());
      auto shardLayout = mlir::dyn_cast<ttcore::ShardLayoutAttr>(
          ttcore::getDeviceLayout(outputType));
      TT_assertv(shardLayout, "Expected shardLayoutAttr for the output of a "
                              "generic op with a virtual grid.");

      auto physicalGridShape = d2m::utils::getPhysicalGridShape(output);
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      gridSize = {physicalGridShape[1], physicalGridShape[0]};
    } else {
      // TTNN grids are (Width, Height), while D2M grids are (Height, Width).
      gridSize = {opGrid.getShape()[1], opGrid.getShape()[0]};
    }

    return gridSize;
  }

  // Create all descriptors from a GenericOp.
  static GenericOpDescriptors createDescriptorsFromGenericOp(
      Builder &builder, d2m::GenericOp op, const ttcore::DeviceAttr &device,
      const ttnn::CoreRangeSetAttr &coreRangeSet,
      const SymbolTable &symbolTable, ttmetal::MathFidelity mathFidelity) {
    GenericOpDescriptors descriptors;

    // Extract operands (ios and cbs).
    llvm::SmallVector<Value> cbs;
    extractOperandsFromGenericOp(op, descriptors.ios, cbs);

    // Create CB descriptors.
    descriptors.cbDescriptors =
        createCBDescriptors(builder, cbs, device, coreRangeSet);

    // Create KernelDescriptors.
    descriptors.kernelDescriptors = createKernelDescriptors(
        builder, op.getThreads(), coreRangeSet, symbolTable, mathFidelity);

    // Extract semaphore descriptors from kernel functions.
    descriptors.semaphoreDescriptors = createSemaphoreDescriptors(
        builder, op.getThreads(), coreRangeSet, symbolTable);

    return descriptors;
  }

  // Create a ttnn::GenericOp from descriptors.
  static ttnn::GenericOp
  createTTNNGenericOpFromDescriptors(ConversionPatternRewriter &rewriter,
                                     Operation *op,
                                     const GenericOpDescriptors &descriptors) {
    MLIRContext *ctx = rewriter.getContext();

    ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
        ctx, descriptors.kernelDescriptors, descriptors.cbDescriptors,
        descriptors.semaphoreDescriptors);

    return rewriter.create<ttnn::GenericOp>(op->getLoc(), descriptors.ios,
                                            program, ttnn::MemoryConfigAttr());
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    // Compute grid size and create core range set.
    llvm::SmallVector<int64_t> gridSize = computeGridSizeFromGenericOp(op);
    ttnn::CoreRangeSetAttr coreRangeSet =
        createCoreRangeSet(rewriter, gridSize);

    // Create all descriptors from the GenericOp.
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    GenericOpDescriptors descriptors = createDescriptorsFromGenericOp(
        rewriter, op, device, coreRangeSet, opSymTable, this->mathFidelity);

    // Create ttnn::GenericOp and replace.
    auto ttnnGenericOp = createTTNNGenericOpFromDescriptors(
        rewriter, op.getOperation(), descriptors);
    rewriter.replaceOp(op, ttnnGenericOp->getResults());

    return success();
  };

private:
  ttmetal::MathFidelity mathFidelity;
};
} // namespace

namespace {
class D2MSpatialRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  D2MSpatialRewriter(MLIRContext *context, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::SpatialOp>(context),
        mathFidelity(mathFidelity) {}

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op, d2m::SpatialOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto device = ttcore::lookupDevice(op->getParentOp());
    TT_assert(device);

    // Get grid_ranges - each region corresponds to one core range.
    ttcore::CoreRangeSetAttr gridRanges = op.getGridRanges();
    auto coreRanges = gridRanges.getCoreRanges();
    TT_assert(!coreRanges.empty());
    TT_assert((op->getRegions().size() == coreRanges.size() &&
               "SpatialOp region count must match grid_ranges size"));

    // Merge: one ttnn::GenericOp with [unique inputs, unique outputs] as I/O,
    // and concatenated kernels/CBs/semaphores per region with remapped CB
    // indices.
    SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
    SmallVector<mlir::Attribute> allKernelDescriptors;
    SmallVector<ttnn::KernelCBAttr> allCBDescriptors;
    SmallVector<ttnn::KernelSemaphoreAttr> allSemaphoreDescriptors;

    // Collect unique inputs and outputs from all regions.
    llvm::DenseSet<Value> seenInputs;
    llvm::DenseSet<Value> seenOutputs;
    SmallVector<Value> allInputs;
    SmallVector<Value> allOutputs;

    // Process each region.
    size_t regionIndex = 0;
    for (Region &region : op->getRegions()) {
      // Find the GenericOp in this region.
      auto genericOpIt = region.front().getOps<d2m::GenericOp>().begin();
      if (genericOpIt == region.front().getOps<d2m::GenericOp>().end()) {
        return rewriter.notifyMatchFailure(
            op, "Expected a GenericOp in each region of SpatialOp");
      }
      d2m::GenericOp genericOp = *genericOpIt;

      // Compute grid size for this region's GenericOp.
      llvm::SmallVector<int64_t> gridSize =
          D2MGenericRewriter::computeGridSizeFromGenericOp(genericOp);

      // Get the start coordinate from the corresponding core range in
      // grid_ranges.
      TT_assert(regionIndex < coreRanges.size());
      auto coreRange = coreRanges[regionIndex];
      auto startCoord = coreRange.getStartCoord();
      llvm::SmallVector<int64_t> startCoordVec = {startCoord.getX(),
                                                  startCoord.getY()};

      // Create core range set for this region using its grid size and start
      // coord.
      ttnn::CoreRangeSetAttr coreRangeSet =
          D2MGenericRewriter::createCoreRangeSet(rewriter, gridSize,
                                                 startCoordVec);

      // Extract inputs and outputs for merged I/O list (CBs come from
      // createDescriptorsFromGenericOp per region).
      llvm::SmallVector<Value> regionInputs;
      llvm::SmallVector<Value> regionOutputs;
      D2MGenericRewriter::extractInputsAndOutputsFromGenericOp(
          genericOp, regionInputs, regionOutputs);

      // Collect unique inputs.
      for (Value input : regionInputs) {
        if (seenInputs.insert(input).second) {
          allInputs.push_back(input);
        }
      }

      // Collect unique outputs.
      for (Value output : regionOutputs) {
        if (seenOutputs.insert(output).second) {
          allOutputs.push_back(output);
        }
      }

      // Create all descriptors for this region's GenericOp.
      // This includes CB descriptors with the region's own coreRangeSet.
      D2MGenericRewriter::GenericOpDescriptors regionDescriptors =
          D2MGenericRewriter::createDescriptorsFromGenericOp(
              rewriter, genericOp, device, coreRangeSet, opSymTable,
              this->mathFidelity);

      // CB index offset for this region (indices before appending this region's
      // CBs).
      size_t cbIndexOffset = allCBDescriptors.size();

      // Remap this region's kernel descriptors so
      // #ttnn.kernel_arg_cb_buffer_index refers to the merged CB list (offset
      // by previous regions' CB count).
      for (mlir::Attribute kernelAttr : regionDescriptors.kernelDescriptors) {
        allKernelDescriptors.push_back(
            remapKernelDescriptorCBIndices(kernelAttr, cbIndexOffset));
      }

      // Append this region's CB descriptors with remapped buffer_index so
      // global CB indices are contiguous and match kernel args above.
      for (auto cbDesc : regionDescriptors.cbDescriptors) {
        // Remap buffer_index in each CB format.
        SmallVector<ttnn::KernelCBFormatAttr> remappedFormats;
        for (auto format : cbDesc.getFormats()) {
          uint32_t newBufferIndex = format.getBufferIndex() + cbIndexOffset;
          auto remappedFormat = ttnn::KernelCBFormatAttr::get(
              rewriter.getContext(), newBufferIndex, format.getDtype(),
              format.getPageSize());
          remappedFormats.push_back(remappedFormat);
        }

        // Preserve global buffer address attr if present (tensor_operand_index
        // refers to the merged I/O list; no remap needed when indices align).
        ttnn::KernelCBGlobalBufferAddressOfTensorAttr remappedGlobalBuffer;
        if (auto globalBuffer = cbDesc.getBuffer()) {
          remappedGlobalBuffer =
              ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(
                  rewriter.getContext(), globalBuffer.getTensorOperandIndex());
        }

        auto remappedCBDesc = ttnn::KernelCBAttr::get(
            rewriter.getContext(), cbDesc.getTotalSize(),
            cbDesc.getCoreRanges(), remappedFormats, remappedGlobalBuffer);
        allCBDescriptors.push_back(remappedCBDesc);
      }

      allSemaphoreDescriptors.append(
          regionDescriptors.semaphoreDescriptors.begin(),
          regionDescriptors.semaphoreDescriptors.end());

      regionIndex++;
    }

    // Create the final I/O list: inputs first, then outputs.
    SmallVector<Value> allIos;
    allIos.append(allInputs.begin(), allInputs.end());
    allIos.append(allOutputs.begin(), allOutputs.end());

    // Create the merged descriptors structure.
    D2MGenericRewriter::GenericOpDescriptors mergedDescriptors;
    mergedDescriptors.kernelDescriptors = allKernelDescriptors;
    mergedDescriptors.cbDescriptors = allCBDescriptors;
    mergedDescriptors.semaphoreDescriptors = allSemaphoreDescriptors;
    mergedDescriptors.ios = allIos;

    // Create ttnn::GenericOp and replace.
    auto ttnnGenericOp = D2MGenericRewriter::createTTNNGenericOpFromDescriptors(
        rewriter, op.getOperation(), mergedDescriptors);
    rewriter.replaceOp(op, ttnnGenericOp->getResults());

    return success();
  };

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

void populateD2MToTTNNPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttmetal::MathFidelity mathFidelity) {
  patterns.add<D2MGenericRewriter, D2MSpatialRewriter>(ctx, mathFidelity);
  patterns.add<TTNNMetalLayoutCastRewriter, D2MEmptyRewriter, D2MFullRewriter,
               StreamLayoutRewriter, ViewLayoutRewriter>(ctx);
}
} // namespace mlir::tt
