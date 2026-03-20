// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt {

namespace {

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
    return builder.getAttr<ttnn::KernelArgNamedArgAttr>(arg.getArgumentName(),
                                                        arg.getOperandIndex());
  }
  case ttkernel::ArgType::GlobalSemaphore: {
    return builder.getAttr<ttnn::KernelArgGlobalSemaphoreAttr>(
        arg.getOperandIndex());
  }
  }
  llvm_unreachable("Invalid ArgType");
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

enum class OperandLocality {
  L1Local,
  L1Remote,
  DRAM,
};

struct GenericOperandInfo {
  Value ioTensor;
  Value cbMemref;
  bool isOutput;
  OperandLocality locality;
  unsigned operandIdx;
};

static SmallVector<ttnn::KernelCBAttr>
createCBDescriptors(Builder &builder,
                    const SmallVector<GenericOperandInfo> &infos,
                    const ttcore::DeviceAttr &device,
                    const ttnn::CoreRangeSetAttr &coreRangeSet) {
  if (infos.empty()) {
    llvm_unreachable("Expected circular buffers.");
  }

  MLIRContext *ctx = builder.getContext();
  SmallVector<ttnn::KernelCBAttr> cbDescriptors(infos.size());

  for (auto [i, info] : llvm::enumerate(infos)) {
    auto cbMemref = mlir::cast<MemRefType>(info.cbMemref.getType());
    TT_assertv(mlir::isa<ttcore::TileType>(cbMemref.getElementType()),
               "Only TileType supported.");
    ttcore::DataType dtype =
        ttcore::elementTypeToDataType(cbMemref.getElementType());
    size_t pageSize = device.getMemrefCBPageSizeBytes(cbMemref);
    size_t totalSize = device.getMemrefSizeBytes(cbMemref, pageSize, true);

    ttnn::KernelCBFormatAttr cbFormat =
        ttnn::KernelCBFormatAttr::get(ctx, i, dtype, pageSize);

    ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
    if (info.locality == OperandLocality::L1Local) {
      globalCBIndexOfTensor =
          ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, i);
    }

    cbDescriptors[i] = ttnn::KernelCBAttr::get(
        ctx, totalSize, coreRangeSet, {cbFormat}, globalCBIndexOfTensor);
  }

  return cbDescriptors;
}

static LogicalResult
materializeIntermediateTensor(memref::AllocOp op, IRRewriter &rewriter,
                              DenseMap<Value, Value> &valueMapping) {
  MLIRContext *ctx = rewriter.getContext();
  MemRefType memrefType = op.getMemref().getType();
  Location loc = op.getLoc();

  // Global semaphores are processed separately.
  bool isBackingGlobalSemaphore =
      llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
        return mlir::isa<d2m::CreateGlobalSemaphoreOp>(user);
      });
  if (isBackingGlobalSemaphore) {
    return success();
  }

  // Hoisted CBs do not need to be materialized as TTNN tensors.
  if (auto cbLayout = mlir::dyn_cast_if_present<ttcore::CBLayoutAttr>(
          memrefType.getLayout())) {
    return success();
  }

  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    return op.emitOpError("could not find device attribute");
  }

  if (mlir::isa_and_present<ttcore::DeviceLayoutInterface>(
          memrefType.getLayout())) {
    auto convertedTensorType = convertMemrefToTTNNTensor(ctx, op.getMemref());
    auto convertedLayoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(convertedTensorType.getEncoding());

    // Check for ttnn_metal_layout_cast users -- if present, use cast's result
    // type to preserve uncollapsed shape (e.g., 3D -> 2D collapse).
    RankedTensorType emptyTensorType = convertedTensorType;
    SmallVector<ttir::TTNNMetalLayoutCastOp> castsToMap;

    for (Operation *user : op.getMemref().getUsers()) {
      if (auto castOp = dyn_cast<ttir::TTNNMetalLayoutCastOp>(user)) {
        castsToMap.push_back(castOp);
      }
    }

    if (!castsToMap.empty()) {
      auto castResultType =
          mlir::cast<RankedTensorType>(castsToMap[0].getResult().getType());
      auto castLayoutAttr =
          mlir::cast<ttnn::TTNNLayoutAttr>(castResultType.getEncoding());

      TT_assertv(castResultType.getNumElements() ==
                     convertedTensorType.getNumElements(),
                 "ttnn_metal_layout_cast and converted type must have the same "
                 "volume");
      TT_assertv(castLayoutAttr.getBufferType() ==
                     convertedLayoutAttr.getBufferType(),
                 "ttnn_metal_layout_cast and converted type must have the same "
                 "buffer type");
      TT_assertv(castLayoutAttr.getShardShape() ==
                     convertedLayoutAttr.getShardShape(),
                 "ttnn_metal_layout_cast and converted type must have the same "
                 "shard shape");
      TT_assertv(castLayoutAttr.getGrid().getShape() ==
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

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto emptyOp = ttnn::EmptyOp::create(
        rewriter, loc, emptyTensorType, device,
        ttnn::ShapeAttr::get(ctx, emptyTensorType.getShape()),
        ttcore::DataTypeAttr::get(ctx, emptyLayoutAttr.getDataType()),
        ttnn::LayoutAttr::get(ctx, emptyLayoutAttr.getLayout()), memcfg);

    valueMapping[op.getResult()] = emptyOp.getResult();
    for (auto castOp : castsToMap) {
      valueMapping[castOp.getResult()] = emptyOp.getResult();
    }
    return success();
  }

  return op.emitOpError("Unsupported memref.alloc");
}

struct TensorAllocAttrs {
  ttcore::DataTypeAttr dtype;
  ttnn::LayoutAttr layout;
  ttnn::MemoryConfigAttr memcfg;
};

static FailureOr<TensorAllocAttrs>
getTensorAllocAttrs(Operation *op, RankedTensorType tensorType) {
  MLIRContext *ctx = op->getContext();
  auto encoding = tensorType.getEncoding();
  auto deviceAttr = ttcore::lookupDevice(op);

  if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
    return TensorAllocAttrs{
        ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType()),
        ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout()),
        ttnn::MemoryConfigAttr::get(layoutAttr, deviceAttr.getWorkerGrid())};
  }
  if (auto ndLayoutAttr = mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
    auto bufferType =
        ttnn::BufferTypeAttr::get(ctx, ndLayoutAttr.getBufferType());
    auto ndShardSpec = ttnn::NDShardSpecAttr::get(ndLayoutAttr);
    return TensorAllocAttrs{
        ttcore::DataTypeAttr::get(ctx, ndLayoutAttr.getDataType()),
        ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout()),
        ttnn::MemoryConfigAttr::get(ctx, ndLayoutAttr.getMemLayout(),
                                    bufferType, /*shardSpec=*/std::nullopt,
                                    ndShardSpec)};
  }
  return op->emitOpError("unsupported encoding type"), failure();
}

static LogicalResult convertD2MEmpty(d2m::EmptyOp op, IRRewriter &rewriter,
                                     DenseMap<Value, Value> &valueMapping) {
  auto tensorType = cast<RankedTensorType>(op.getResult().getType());
  auto attrs = getTensorAllocAttrs(op, tensorType);
  if (failed(attrs)) {
    return failure();
  }

  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
  auto shape = ttnn::ShapeAttr::get(op.getContext(), tensorType.getShape());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto emptyOp =
      ttnn::EmptyOp::create(rewriter, op.getLoc(), tensorType, device, shape,
                            attrs->dtype, attrs->layout, attrs->memcfg);
  valueMapping[op.getResult()] = emptyOp.getResult();
  return success();
}

static LogicalResult convertD2MFull(d2m::FullOp op, IRRewriter &rewriter,
                                    DenseMap<Value, Value> &valueMapping) {
  auto tensorType = cast<RankedTensorType>(op.getResult().getType());
  auto attrs = getTensorAllocAttrs(op, tensorType);
  if (failed(attrs)) {
    return failure();
  }

  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
  auto shapeI32 = op.getShape();
  SmallVector<int64_t> shapeI64(shapeI32.begin(), shapeI32.end());
  auto shape = ttnn::ShapeAttr::get(op.getContext(), shapeI64);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto fullOp = ttnn::FullOp::create(rewriter, op.getLoc(), tensorType, device,
                                     shape, op.getFillValueAttr(), attrs->dtype,
                                     attrs->layout, attrs->memcfg);
  valueMapping[op.getResult()] = fullOp.getResult();
  return success();
}

static LogicalResult
handleD2MCreateGlobalSemaphore(d2m::CreateGlobalSemaphoreOp op,
                               IRRewriter &rewriter,
                               DenseMap<Value, Value> &valueMapping) {
  auto allocOp = op.getInput().getDefiningOp<memref::AllocOp>();
  TT_assertv(allocOp,
             "No memref alloc found for CreateGlobalSemaphoreOp's input");

  auto gridShape = ttcore::getGridShape(op.getInput());
  auto coreRange = ttnn::CoreRangeAttr::get(
      rewriter.getContext(),
      ttnn::CoreCoordAttr::get(rewriter.getContext(), 0, 0),
      ttnn::CoreCoordAttr::get(rewriter.getContext(), gridShape[0] - 1,
                               gridShape[1] - 1));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto ttnnOp = ttnn::CreateGlobalSemaphoreOp::create(
      rewriter, op.getLoc(), op.getValueAttr(), coreRange);
  valueMapping[op.getResult()] = ttnnOp.getResult();
  return success();
}

static LogicalResult
handleD2MResetGlobalSemaphore(d2m::ResetGlobalSemaphoreOp op,
                              IRRewriter &rewriter,
                              DenseMap<Value, Value> &valueMapping) {
  auto createGlobalSemaphoreOp =
      op.getSemaphore().getDefiningOp<d2m::CreateGlobalSemaphoreOp>();
  TT_assertv(createGlobalSemaphoreOp, "No create global semaphore op found for "
                                      "ResetGlobalSemaphoreOp's input");

  Value mappedSemaphore = valueMapping.count(op.getSemaphore())
                              ? valueMapping[op.getSemaphore()]
                              : op.getSemaphore();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  ttnn::ResetGlobalSemaphoreOp::create(rewriter, op.getLoc(), mappedSemaphore,
                                       op.getValueAttr());
  return success();
}

static LogicalResult
materializeTTNNTensors(ModuleOp moduleOp,
                       DenseMap<Value, Value> &valueMapping) {
  IRRewriter rewriter(moduleOp.getContext());

  // Walk all operations, handling each type.
  // We need to collect ops first because we insert new ops during traversal.
  SmallVector<memref::AllocOp> allocOps;
  SmallVector<d2m::EmptyOp> emptyOps;
  SmallVector<d2m::FullOp> fullOps;

  moduleOp.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(allocOp);
    } else if (auto emptyOp = dyn_cast<d2m::EmptyOp>(op)) {
      emptyOps.push_back(emptyOp);
    } else if (auto fullOp = dyn_cast<d2m::FullOp>(op)) {
      fullOps.push_back(fullOp);
    }
  });

  for (auto op : allocOps) {
    if (failed(materializeIntermediateTensor(op, rewriter, valueMapping))) {
      return failure();
    }
  }
  for (auto op : emptyOps) {
    if (failed(convertD2MEmpty(op, rewriter, valueMapping))) {
      return failure();
    }
  }
  for (auto op : fullOps) {
    if (failed(convertD2MFull(op, rewriter, valueMapping))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult convertSemaphores(ModuleOp moduleOp,
                                       DenseMap<Value, Value> &valueMapping) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<d2m::CreateGlobalSemaphoreOp> createSemOps;
  SmallVector<d2m::ResetGlobalSemaphoreOp> resetSemOps;

  moduleOp.walk([&](Operation *op) {
    if (auto createOp = dyn_cast<d2m::CreateGlobalSemaphoreOp>(op)) {
      createSemOps.push_back(createOp);
    } else if (auto resetOp = dyn_cast<d2m::ResetGlobalSemaphoreOp>(op)) {
      resetSemOps.push_back(resetOp);
    }
  });

  for (auto op : createSemOps) {
    if (failed(handleD2MCreateGlobalSemaphore(op, rewriter, valueMapping))) {
      return failure();
    }
  }
  for (auto op : resetSemOps) {
    if (failed(handleD2MResetGlobalSemaphore(op, rewriter, valueMapping))) {
      return failure();
    }
  }

  return success();
}

static std::pair<Value, OperandLocality>
findIOTensor(Value operand, DenseMap<Value, Value> &valueMapping,
             OperandLocality currentLocality = OperandLocality::L1Local) {
  auto iter = valueMapping.find(operand);
  if (iter != valueMapping.end()) {
    auto tensorType = dyn_cast<RankedTensorType>(iter->second.getType());
    TT_assertv(tensorType, "expected mapped value to be a ranked tensor");
    TT_assertv(isa<ttnn::TTNNLayoutAttr>(tensorType.getEncoding()),
               "expected mapped value to be a TTNN tensor");
    return {iter->second, currentLocality};
  }

  auto *def = operand.getDefiningOp();
  if (auto view = dyn_cast<d2m::ViewLayoutOp>(def)) {
    return findIOTensor(view.getInput(), valueMapping,
                        OperandLocality::L1Remote);
  }
  if (auto cast = dyn_cast<ttir::TTNNMetalLayoutCastOp>(def)) {
    // Input is already a TTNN tensor.
    auto tensorType = dyn_cast<RankedTensorType>(cast.getInput().getType());
    TT_assertv(tensorType, "expected input to be a ranked tensor");
    TT_assertv(isa<ttnn::TTNNLayoutAttr>(tensorType.getEncoding()),
               "expected input to be a ranked tensor");
    return {cast.getInput(), currentLocality};
  }

  llvm_unreachable("unexpected operand def chain");
}

static Value findCBMemref(Value operand) {
  Operation *def = operand.getDefiningOp();
  if (!def) {
    TT_assertv(isa<MemRefType>(operand.getType()),
               "expected operand to be a memref");
    return operand;
  }

  TT_assertv(isa<MemRefType>(operand.getType()),
             "expected operand to be a memref");
  return operand;
}

static SmallVector<GenericOperandInfo>
analyzeGenericOperands(d2m::GenericOp op,
                       DenseMap<Value, Value> &valueMapping) {
  SmallVector<GenericOperandInfo> infos;
  unsigned idx = 0;

  for (Value input : op.getInputs()) {
    auto [tensor, locality] = findIOTensor(input, valueMapping);
    infos.push_back({tensor, findCBMemref(input),
                     /*isOutput=*/false, locality, idx++});
  }
  for (Value output : op.getOutputs()) {
    auto [tensor, locality] = findIOTensor(output, valueMapping);
    infos.push_back({tensor, findCBMemref(output),
                     /*isOutput=*/true, locality, idx++});
  }

  for (auto &info : infos) {
    auto ttnnTensor = mlir::cast<RankedTensorType>(info.ioTensor.getType());
    auto ttnnLayout =
        mlir::cast<ttnn::TTNNLayoutAttr>(ttnnTensor.getEncoding());
    if (ttnnLayout.getBufferType() == ttnn::BufferType::DRAM) {
      info.locality = OperandLocality::DRAM;
    }
  }
  return infos;
}

static LogicalResult convertSingleGeneric(d2m::GenericOp op,
                                          IRRewriter &rewriter,
                                          DenseMap<Value, Value> &valueMapping,
                                          ttmetal::MathFidelity mathFidelity) {
  MLIRContext *ctx = rewriter.getContext();
  auto device = ttcore::lookupDevice(op->getParentOp());
  TT_assert(device);

  // Compute core range set.
  // Note: TTNN grids are (Width, Height), while D2M grids are (Height, Width).
  auto physicalGridShape = op.getPhysicalGridShape();
  llvm::SmallVector<int64_t> endCoreRange = {physicalGridShape[1] - 1,
                                             physicalGridShape[0] - 1};

  ttnn::CoreRangeSetAttr coreRangeSet = ttnn::CoreRangeSetAttr::get(
      ctx,
      ttnn::CoreRangeAttr::get(
          ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
          ttnn::CoreCoordAttr::get(ctx, endCoreRange[0], endCoreRange[1])));

  SmallVector<GenericOperandInfo> infos =
      analyzeGenericOperands(op, valueMapping);

  SmallVector<Value> additionalArgs;
  for (Value arg : op.getAdditionalArgs()) {
    if (auto cbForOp = arg.getDefiningOp()
                           ? arg.getDefiningOp()->getAttrOfType<IntegerAttr>(
                                 "d2m.cb_for_operand")
                           : IntegerAttr()) {
      unsigned targetIdx = static_cast<unsigned>(cbForOp.getInt());
      TT_assertv(targetIdx < infos.size(), "d2m.cb_for_operand out of range");
      infos[targetIdx].cbMemref = arg;
      // The CB has its own L1 allocation (not backed by the input tensor).
      infos[targetIdx].locality = OperandLocality::L1Remote;
    } else if (isa<ttnn::GlobalSemaphoreType>(arg.getType()) ||
               isa<RankedTensorType>(arg.getType())) {
      Value mapped = valueMapping.count(arg) ? valueMapping[arg] : arg;
      additionalArgs.push_back(mapped);
    } else if (isa<d2m::GlobalSemaphoreType>(arg.getType())) {
      Value mapped = valueMapping.count(arg) ? valueMapping[arg] : arg;
      additionalArgs.push_back(mapped);
    } else {
      return op.emitOpError(
                 "unexpected operand type in d2m.generic's additionalArgs: ")
             << arg.getType();
    }
  }

  SmallVector<ttnn::KernelCBAttr> cbDescriptors =
      createCBDescriptors(rewriter, infos, device, coreRangeSet);

  SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
  SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
      rewriter, op.getThreads(), coreRangeSet, opSymTable, mathFidelity);

  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
      createSemaphoreDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                 opSymTable);

  ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
      ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

  SmallVector<Value> ios;
  for (auto &info : infos) {
    ios.push_back(info.ioTensor);
  }
  // Runtime ttnn::generic_op currently requires at least one input and one
  // output tensor in io_tensors. Some generator-style generics (e.g. fill ->
  // unary) have only a single output tensor. Mirror that tensor as both input
  // and output in the IO list so lowering stays within the runtime contract.
  if (op.getInputs().empty() && op.getOutputs().size() == 1 &&
      ios.size() == 1) {
    ios.push_back(ios.front());
  }

  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<ttnn::GenericOp>(op, ios, additionalArgs, program,
                                               ttnn::MemoryConfigAttr());
  return success();
}

static LogicalResult convertGenerics(ModuleOp moduleOp,
                                     DenseMap<Value, Value> &valueMapping,
                                     ttmetal::MathFidelity mathFidelity) {
  IRRewriter rewriter(moduleOp.getContext());

  SmallVector<d2m::GenericOp> genericOps;
  moduleOp.walk([&](d2m::GenericOp op) { genericOps.push_back(op); });

  for (auto op : genericOps) {
    if (failed(
            convertSingleGeneric(op, rewriter, valueMapping, mathFidelity))) {
      return failure();
    }
  }

  return success();
}

// ===----------------------------------------------------------------------===//
// Spatial: merge each region's ttnn.generic into one ttnn.generic. Unified IO
// is built by walking regions in order and appending each input/output Value
// the first time it is seen (dedupe by SSA Value identity). Additional operands
// are concatenated per region. Kernel/CB attrs remap via SpatialRemapTable;
// kernels, CBs, and semaphores use each region's spatial grid core_ranges.
// ===----------------------------------------------------------------------===//

static ttnn::CoreRangeSetAttr
ttCoreRangeToTtnnCoreRangeSet(MLIRContext *ctx, ttcore::CoreRangeAttr cr) {
  auto sc = cr.getStartCoord();
  auto ec = cr.getEndCoord();
  auto tts = ttnn::CoreCoordAttr::get(ctx, static_cast<uint64_t>(sc.getX()),
                                      static_cast<uint64_t>(sc.getY()));
  auto tte = ttnn::CoreCoordAttr::get(ctx, static_cast<uint64_t>(ec.getX()),
                                      static_cast<uint64_t>(ec.getY()));
  return ttnn::CoreRangeSetAttr::get(
      ctx, llvm::ArrayRef{ttnn::CoreRangeAttr::get(ctx, tts, tte)});
}

class SpatialRemapTable {
  using LocalKey = std::pair<ttnn::GenericOp, size_t>;

  SmallVector<Value> unifiedIO_;
  SmallVector<Value> unifiedAdditionalArgs_;
  DenseMap<Value, size_t> tensorToUnifiedIdx_;
  DenseMap<LocalKey, size_t> ioMap_;
  DenseMap<LocalKey, size_t> additionalArgMap_;
  size_t nextAdditionalUnified_ = 0;

public:
  void addOperands(ttnn::GenericOp generic) {
    for (const auto [idx, opnd] :
         llvm::enumerate(generic.getInputsAndOutputs())) {
      size_t uidx;
      auto it = tensorToUnifiedIdx_.find(opnd);
      if (it == tensorToUnifiedIdx_.end()) {
        uidx = unifiedIO_.size();
        tensorToUnifiedIdx_.insert({opnd, uidx});
        unifiedIO_.push_back(opnd);
      } else {
        uidx = it->second;
      }
      ioMap_.insert({{generic, static_cast<size_t>(idx)}, uidx});
    }
    // For now, additional args are only appended in order without extra policy.
    // If we need special handling later (e.g., dedup, ordering, or mapping
    // rules), this path will need to be extended.
    // Keys match flat operand indices on ttnn.generic (IO operands first, then
    // additional_args), same as ttkernel ArgAttr operand_index for globals.
    const size_t ioOperandCount = generic.getInputsAndOutputs().size();
    for (const auto [idx, _] : llvm::enumerate(generic.getAdditionalArgs())) {
      size_t localIdx = static_cast<size_t>(idx);
      size_t flatOperandIdx = ioOperandCount + localIdx;
      additionalArgMap_.insert(
          {{generic, flatOperandIdx}, nextAdditionalUnified_ + localIdx});
    }
    nextAdditionalUnified_ += generic.getAdditionalArgs().size();
    for (Value additionalArg : generic.getAdditionalArgs()) {
      unifiedAdditionalArgs_.push_back(additionalArg);
    }
  }

  ArrayRef<Value> getUnifiedIO() const { return unifiedIO_; }
  ArrayRef<Value> getUnifiedAdditionalArgs() const {
    return unifiedAdditionalArgs_;
  }

  std::optional<size_t> lookupIO(ttnn::GenericOp generic,
                                 size_t localIdx) const {
    auto it = ioMap_.find({generic, localIdx});
    if (it != ioMap_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Key is flat operand index on the pre-merge generic (IO operands then
  // additional_args). Returns flat merged operand index (unified IO count +
  // additional slot).
  std::optional<size_t> lookupAdditional(ttnn::GenericOp generic,
                                         size_t flatOperandIndex) const {
    auto it = additionalArgMap_.find({generic, flatOperandIndex});
    if (it != additionalArgMap_.end()) {
      return unifiedIO_.size() + it->second;
    }
    return std::nullopt;
  }
};

static SmallVector<Attribute>
remapSpatialKernelArgs(MLIRContext *ctx, ArrayRef<Attribute> args,
                       ttnn::GenericOp generic,
                       const SpatialRemapTable &remapTable) {
  SmallVector<Attribute> out;
  for (Attribute arg : args) {
    Attribute mapped = arg;
    if (auto tensor = mlir::dyn_cast<ttnn::KernelArgAddressOfTensorAttr>(arg)) {
      if (auto unified =
              remapTable.lookupIO(generic, tensor.getTensorIndex())) {
        mapped = ttnn::KernelArgAddressOfTensorAttr::get(ctx, *unified);
      }
    } else if (auto globalSem =
                   mlir::dyn_cast<ttnn::KernelArgGlobalSemaphoreAttr>(arg)) {
      if (auto unified = remapTable.lookupAdditional(
              generic, globalSem.getGlobalSemaphoreIndex())) {
        mapped = ttnn::KernelArgGlobalSemaphoreAttr::get(ctx, *unified);
      }
    }
    out.push_back(mapped);
  }
  return out;
}

static Attribute
remapSpatialKernelDescriptor(MLIRContext *ctx, Attribute kernelAttr,
                             ttnn::GenericOp generic,
                             const SpatialRemapTable &remapTable,
                             ttnn::CoreRangeSetAttr gridCoreRanges) {
  auto iface = mlir::dyn_cast<ttnn::KernelInterface>(kernelAttr);
  if (!iface) {
    return kernelAttr;
  }

  SmallVector<Attribute> crt =
      remapSpatialKernelArgs(ctx, iface.getCommonRtArgs(), generic, remapTable);
  SmallVector<Attribute> ct =
      remapSpatialKernelArgs(ctx, iface.getCtArgs(), generic, remapTable);

  return llvm::TypeSwitch<Attribute, Attribute>(kernelAttr)
      .Case<ttnn::ComputeKernelAttr>([&](ttnn::ComputeKernelAttr compute) {
        return ttnn::ComputeKernelAttr::get(
            ctx, compute.getSymbolRef(), gridCoreRanges,
            compute.getMathFidelity(), compute.getFp32DestAccEn(),
            compute.getDstFullSyncEn(), compute.getUnpackToDestModes(),
            compute.getBfp8PackPrecise(), compute.getMathApproxMode(), crt, ct);
      })
      .Case<ttnn::DataMovementKernelAttr>([&](ttnn::DataMovementKernelAttr dm) {
        return ttnn::DataMovementKernelAttr::get(
            ctx, dm.getSymbolRef(), gridCoreRanges, dm.getProcessor(),
            dm.getNocIndex(), dm.getNocMode(), crt, ct);
      })
      .Case<ttnn::ReadKernelAttr>([&](ttnn::ReadKernelAttr read) {
        return ttnn::ReadKernelAttr::get(ctx, read.getSymbolRef(),
                                         gridCoreRanges, crt, ct);
      })
      .Case<ttnn::WriteKernelAttr>([&](ttnn::WriteKernelAttr write) {
        return ttnn::WriteKernelAttr::get(ctx, write.getSymbolRef(),
                                          gridCoreRanges, crt, ct);
      })
      .Default([](Attribute attr) { return attr; });
}

static ttnn::KernelCBAttr
adjustSpatialCBDescriptor(MLIRContext *ctx, ttnn::KernelCBAttr cb,
                          ttnn::GenericOp generic,
                          const SpatialRemapTable &remapTable,
                          ttnn::CoreRangeSetAttr gridCoreRanges) {
  SmallVector<ttnn::KernelCBFormatAttr> formats;
  formats.append(cb.getFormats().begin(), cb.getFormats().end());
  ttnn::KernelCBGlobalBufferAddressOfTensorAttr bufferToUse = cb.getBuffer();
  if (bufferToUse) {
    if (auto unified =
            remapTable.lookupIO(generic, bufferToUse.getTensorOperandIndex())) {
      bufferToUse =
          ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, *unified);
    }
  }
  return ttnn::KernelCBAttr::get(ctx, cb.getTotalSize(), gridCoreRanges,
                                 formats, bufferToUse);
}

// Move every op except ttnn.generic out of each spatial region into the parent
// block immediately before the spatial.
static void hoistNonGenericOpsBeforeSpatial(d2m::SpatialOp spatialOp) {
  SmallVector<Operation *> toHoist;
  for (Region &region : spatialOp.getRegions()) {
    for (Operation &op : region.front()) {
      if (mlir::isa<ttnn::GenericOp>(&op)) {
        continue;
      }
      // skip d2m.yield / d2m.spatial_yield ops
      if (op.mightHaveTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      toHoist.push_back(&op);
    }
  }
  for (Operation *op : toHoist) {
    op->moveBefore(spatialOp);
  }
}

static LogicalResult convertSingleSpatial(d2m::SpatialOp spatialOp,
                                          IRRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();

  // ttnn.generic has no op results; replaceOpWithNewOp does not wire spatial
  // results. An earlier pipeline stage is expected to leave spatial with zero
  // results (outs remain as operands only).
  if (spatialOp.getNumResults() != 0) {
    return spatialOp.emitOpError(
        "expected no results on d2m.spatial before merge to ttnn.generic");
  }

  SmallVector<ttnn::GenericOp> regionGenerics;
  SpatialRemapTable remapTable;

  for (Region &region : spatialOp.getRegions()) {
    auto generics = llvm::to_vector(region.front().getOps<ttnn::GenericOp>());
    if (generics.size() != 1) {
      return spatialOp.emitOpError(
                 "each region must contain exactly one ttnn.generic, got ")
             << generics.size();
    }
    ttnn::GenericOp generic = generics.front();
    if (llvm::isa<ttnn::MeshProgramDescriptorAttr>(generic.getProgram())) {
      return spatialOp.emitOpError(
          "d2m.spatial with MeshProgramDescriptor not supported");
    }
    regionGenerics.push_back(generic);
    remapTable.addOperands(generic);
  }

  mlir::ArrayAttr spatialGridRanges = spatialOp.getGridRanges();
  SmallVector<Attribute> mergedKernels;
  SmallVector<ttnn::KernelCBAttr> mergedCBs;
  SmallVector<ttnn::KernelSemaphoreAttr> mergedSemaphores;
  for (const auto [regionIdx, generic] : llvm::enumerate(regionGenerics)) {
    auto regionCoreRange =
        mlir::cast<ttcore::CoreRangeAttr>(spatialGridRanges[regionIdx]);
    ttnn::CoreRangeSetAttr gridCoreRanges =
        ttCoreRangeToTtnnCoreRangeSet(ctx, regionCoreRange);
    auto program = llvm::cast<ttnn::ProgramAttr>(generic.getProgram());
    for (Attribute kernelAttr : program.getKernels()) {
      mergedKernels.push_back(remapSpatialKernelDescriptor(
          ctx, kernelAttr, generic, remapTable, gridCoreRanges));
    }
    for (ttnn::KernelCBAttr cb : program.getCbs()) {
      mergedCBs.push_back(adjustSpatialCBDescriptor(
          ctx, cb, generic, remapTable, gridCoreRanges));
    }
    for (ttnn::KernelSemaphoreAttr sem : program.getSemaphores()) {
      mergedSemaphores.push_back(ttnn::KernelSemaphoreAttr::get(
          ctx, sem.getId(), sem.getCoreType(), gridCoreRanges,
          sem.getInitialValue()));
    }
  }

  ttnn::ProgramAttr mergedProgram =
      ttnn::ProgramAttr::get(ctx, mergedKernels, mergedCBs, mergedSemaphores);

  // Move every op except ttnn.generic out of each spatial region into the
  // parent
  hoistNonGenericOpsBeforeSpatial(spatialOp);

  rewriter.setInsertionPoint(spatialOp);
  rewriter.replaceOpWithNewOp<ttnn::GenericOp>(
      spatialOp, remapTable.getUnifiedIO(),
      remapTable.getUnifiedAdditionalArgs(), mergedProgram,
      ttnn::MemoryConfigAttr());
  return success();
}

static LogicalResult convertSpatials(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());
  SmallVector<d2m::SpatialOp> spatialOps;
  moduleOp.walk([&](d2m::SpatialOp op) { spatialOps.push_back(op); });
  for (d2m::SpatialOp op : spatialOps) {
    if (failed(convertSingleSpatial(op, rewriter))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult cleanupAndVerify(ModuleOp moduleOp,
                                      DenseMap<Value, Value> &valueMapping) {
  // Replace all mapped values' uses with their TTNN equivalents.
  // This handles d2m.empty/full results used directly by func.return.
  for (auto &[oldVal, newVal] : valueMapping) {
    oldVal.replaceAllUsesWith(newVal);
  }

  // Resolve "exit" casts -- TTNNMetalLayoutCastOps whose result type
  // is a tensor (used by surviving TTNN ops like ttnn.to_memory_config or
  // function returns). Replace their uses with the resolved TTNN tensor.
  moduleOp.walk([&](ttir::TTNNMetalLayoutCastOp op) {
    if (!isa<RankedTensorType>(op.getResult().getType())) {
      return;
    }
    if (op.use_empty()) {
      return;
    }
    Value resolved = findIOTensor(op.getOperand(), valueMapping).first;
    op.getResult().replaceAllUsesWith(resolved);
  });

  // Collect and erase all remaining D2M/memref/cast ops. Generics are
  // already erased. External uses are resolved by the remapping.
  SmallVector<Operation *> opsToErase;
  moduleOp.walk([&](Operation *op) {
    if (isa<ttir::TTNNMetalLayoutCastOp, d2m::ViewLayoutOp, d2m::EmptyOp,
            d2m::FullOp, d2m::ResetGlobalSemaphoreOp,
            d2m::CreateGlobalSemaphoreOp, memref::DeallocOp, memref::AllocOp>(
            op)) {
      opsToErase.push_back(op);
    }
  });
  for (auto *op : llvm::reverse(opsToErase)) {
    op->erase();
  }

  WalkResult result = moduleOp.walk([](Operation *op) -> WalkResult {
    if (isa<d2m::D2MDialect>(op->getDialect())) {
      return op->emitError("unexpected D2M op after conversion"),
             WalkResult::interrupt();
    }
    if (isa<memref::AllocOp, memref::DeallocOp>(op)) {
      return op->emitError("unexpected memref op after conversion"),
             WalkResult::interrupt();
    }
    if (isa<ttir::TTNNMetalLayoutCastOp>(op)) {
      return op->emitError("unexpected layout cast after conversion"),
             WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

} // namespace

// ===----------------------------------------------------------------------===//
// Entry Point
// ===----------------------------------------------------------------------===//

LogicalResult runD2MToTTNNConversion(ModuleOp moduleOp,
                                     ttmetal::MathFidelity mathFidelity) {
  DenseMap<Value, Value> valueMapping;
  if (failed(materializeTTNNTensors(moduleOp, valueMapping))) {
    return failure();
  }
  if (failed(convertSemaphores(moduleOp, valueMapping))) {
    return failure();
  }
  if (failed(convertGenerics(moduleOp, valueMapping, mathFidelity))) {
    return failure();
  }
  if (failed(convertSpatials(moduleOp))) {
    return failure();
  }
  if (failed(cleanupAndVerify(moduleOp, valueMapping))) {
    return failure();
  }
  return success();
}

} // namespace mlir::tt
