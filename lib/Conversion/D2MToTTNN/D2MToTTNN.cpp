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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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

struct GenericOperandInfo {
  Value ioTensor;
  Value cbMemref;
  bool isOutput;
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

    if (ttcore::getMemorySpace(cbMemref) != ttcore::MemorySpace::DeviceDRAM) {
      if (info.isOutput) {
        // L1 outputs are always aliased.
        globalCBIndexOfTensor =
            ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, i);
      } else {
        bool isLocalFuncArg =
            mlir::dyn_cast_if_present<ttir::TTNNMetalLayoutCastOp>(
                info.cbMemref.getDefiningOp());
        bool isLocalIntermediate =
            mlir::dyn_cast_if_present<memref::AllocOp>(
                info.cbMemref.getDefiningOp()) &&
            !mlir::isa_and_present<ttcore::CBLayoutAttr>(
                  cbMemref.getLayout()) &&
            llvm::none_of(info.cbMemref.getUsers(), [](Operation *user) {
              return mlir::isa<d2m::StreamLayoutOp>(user);
            });
        if (isLocalFuncArg || isLocalIntermediate) {
          globalCBIndexOfTensor =
              ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(ctx, i);
        }
      }
    }

    cbDescriptors[i] = ttnn::KernelCBAttr::get(
        ctx, totalSize, coreRangeSet, {cbFormat}, globalCBIndexOfTensor);
  }

  return cbDescriptors;
}

static LogicalResult handleMemrefAlloc(memref::AllocOp op, IRRewriter &rewriter,
                                       DenseMap<Value, Value> &valueMapping) {
  MLIRContext *ctx = rewriter.getContext();
  MemRefType memrefType = op.getMemref().getType();
  Location loc = op.getLoc();

  // Case 1: Backing a global semaphore -- skip.
  bool isBackingGlobalSemaphore =
      llvm::any_of(op.getResult().getUsers(), [](Operation *user) {
        return mlir::isa<d2m::CreateGlobalSemaphoreOp>(user);
      });
  if (isBackingGlobalSemaphore) {
    return success();
  }

  // Case 2: Hoisted CB alloc (CBLayoutAttr) -- skip.
  if (auto cbLayout = mlir::dyn_cast_if_present<ttcore::CBLayoutAttr>(
          memrefType.getLayout())) {
    return success();
  }

  auto deviceAttr = ttcore::lookupDevice(op);
  if (!deviceAttr) {
    return op.emitOpError("could not find device attribute");
  }

  // Case 3: Device layout (ShardLayoutAttr or InterleavedLayoutAttr).
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
    auto emptyOp = rewriter.create<ttnn::EmptyOp>(
        loc, emptyTensorType, device,
        ttnn::ShapeAttr::get(ctx, emptyTensorType.getShape()),
        ttcore::DataTypeAttr::get(ctx, emptyLayoutAttr.getDataType()),
        ttnn::LayoutAttr::get(ctx, emptyLayoutAttr.getLayout()), memcfg);

    valueMapping[op.getResult()] = emptyOp.getResult();
    for (auto castOp : castsToMap) {
      valueMapping[castOp.getResult()] = emptyOp.getResult();
    }
    return success();
  }

  // Case 4: Unknown layout.
  return op.emitOpError(
      "memref alloc does not correspond to a ttnn tensor or global semaphore");
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

static LogicalResult handleD2MEmpty(d2m::EmptyOp op, IRRewriter &rewriter,
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
  auto emptyOp = rewriter.create<ttnn::EmptyOp>(op.getLoc(), tensorType, device,
                                                shape, attrs->dtype,
                                                attrs->layout, attrs->memcfg);
  valueMapping[op.getResult()] = emptyOp.getResult();
  return success();
}

static LogicalResult handleD2MFull(d2m::FullOp op, IRRewriter &rewriter,
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
  auto fullOp = rewriter.create<ttnn::FullOp>(
      op.getLoc(), tensorType, device, shape, op.getFillValueAttr(),
      attrs->dtype, attrs->layout, attrs->memcfg);
  valueMapping[op.getResult()] = fullOp.getResult();
  return success();
}

static LogicalResult
handleD2MCreateGlobalSemaphore(d2m::CreateGlobalSemaphoreOp op,
                               IRRewriter &rewriter,
                               DenseMap<Value, Value> &valueMapping) {
  auto allocOp = op.getInput().getDefiningOp<memref::AllocOp>();
  assert(allocOp &&
         "No memref alloc found for CreateGlobalSemaphoreOp's input, failing.");

  auto gridShape = ttcore::getGridShape(op.getInput());
  auto coreRange = ttnn::CoreRangeAttr::get(
      rewriter.getContext(),
      ttnn::CoreCoordAttr::get(rewriter.getContext(), 0, 0),
      ttnn::CoreCoordAttr::get(rewriter.getContext(), gridShape[0] - 1,
                               gridShape[1] - 1));

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto ttnnOp = rewriter.create<ttnn::CreateGlobalSemaphoreOp>(
      op.getLoc(), op.getValueAttr(), coreRange);
  valueMapping[op.getResult()] = ttnnOp.getResult();
  return success();
}

static LogicalResult
handleD2MResetGlobalSemaphore(d2m::ResetGlobalSemaphoreOp op,
                              IRRewriter &rewriter,
                              DenseMap<Value, Value> &valueMapping) {
  auto createGlobalSemaphoreOp =
      op.getSemaphore().getDefiningOp<d2m::CreateGlobalSemaphoreOp>();
  assert(createGlobalSemaphoreOp &&
         "No create global semaphore op found for ResetGlobalSemaphoreOp's "
         "input, failing.");

  Value mappedSemaphore = valueMapping.count(op.getSemaphore())
                              ? valueMapping[op.getSemaphore()]
                              : op.getSemaphore();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  rewriter.create<ttnn::ResetGlobalSemaphoreOp>(op.getLoc(), mappedSemaphore,
                                                op.getValueAttr());
  return success();
}

static LogicalResult
materializeTTNNTensors(ModuleOp module, DenseMap<Value, Value> &valueMapping) {
  IRRewriter rewriter(module.getContext());

  // Walk all operations, handling each type.
  // We need to collect ops first because we insert new ops during traversal.
  SmallVector<memref::AllocOp> allocOps;
  SmallVector<d2m::EmptyOp> emptyOps;
  SmallVector<d2m::FullOp> fullOps;

  module.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(allocOp);
    } else if (auto emptyOp = dyn_cast<d2m::EmptyOp>(op)) {
      emptyOps.push_back(emptyOp);
    } else if (auto fullOp = dyn_cast<d2m::FullOp>(op)) {
      fullOps.push_back(fullOp);
    }
  });

  for (auto op : allocOps) {
    if (failed(handleMemrefAlloc(op, rewriter, valueMapping))) {
      return failure();
    }
  }
  for (auto op : emptyOps) {
    if (failed(handleD2MEmpty(op, rewriter, valueMapping))) {
      return failure();
    }
  }
  for (auto op : fullOps) {
    if (failed(handleD2MFull(op, rewriter, valueMapping))) {
      return failure();
    }
  }

  return success();
}

LogicalResult convertSemaphores(ModuleOp module,
                                DenseMap<Value, Value> &valueMapping) {
  IRRewriter rewriter(module.getContext());

  SmallVector<d2m::CreateGlobalSemaphoreOp> createSemOps;
  SmallVector<d2m::ResetGlobalSemaphoreOp> resetSemOps;

  module.walk([&](Operation *op) {
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

static Value findIOTensor(Value operand, DenseMap<Value, Value> &valueMapping) {
  if (valueMapping.count(operand)) {
    return valueMapping[operand];
  }

  Operation *def = operand.getDefiningOp();
  if (!def) {
    // Function args (tensor<..., #ttnn_layout>) are already TTNN-typed.
    return operand;
  }

  if (auto stream = dyn_cast<d2m::StreamLayoutOp>(def)) {
    return findIOTensor(stream.getInput(), valueMapping);
  }
  if (auto view = dyn_cast<d2m::ViewLayoutOp>(def)) {
    return findIOTensor(view.getInput(), valueMapping);
  }
  if (auto cast = dyn_cast<ttir::TTNNMetalLayoutCastOp>(def)) {
    return findIOTensor(cast.getOperand(), valueMapping);
  }

  // If we've walked past all D2M/cast ops and reached something else (e.g.,
  // ttnn.to_memory_config), the operand is already a TTNN tensor.
  return operand;
}

static Value findCBMemref(Value operand) {
  Operation *def = operand.getDefiningOp();
  if (!def) {
    return operand;
  }

  if (auto stream = dyn_cast<d2m::StreamLayoutOp>(def)) {
    return stream.getStorage();
  }
  if (auto view = dyn_cast<d2m::ViewLayoutOp>(def)) {
    if (isa_and_present<d2m::StreamLayoutOp>(view.getInput().getDefiningOp())) {
      return view.getInput();
    }
  }
  return operand;
}

static SmallVector<GenericOperandInfo>
analyzeGenericOperands(d2m::GenericOp op,
                       DenseMap<Value, Value> &valueMapping) {
  SmallVector<GenericOperandInfo> infos;
  unsigned idx = 0;

  for (Value input : op.getInputs()) {
    infos.push_back({findIOTensor(input, valueMapping), findCBMemref(input),
                     /*isOutput=*/false, idx++});
  }
  for (Value output : op.getOutputs()) {
    infos.push_back({findIOTensor(output, valueMapping), findCBMemref(output),
                     /*isOutput=*/true, idx++});
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
  ttcore::GridAttr opGrid = op.getGrid();
  SmallVector<int64_t> endCoreRange;
  if (!opGrid.getMapping().isEmpty()) {
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

  // Analyze generic operands.
  SmallVector<GenericOperandInfo> infos =
      analyzeGenericOperands(op, valueMapping);

  // Process additionalArgs.
  SmallVector<Value> additionalArgs;

  for (Value arg : op.getAdditionalArgs()) {
    if (auto cbForOp = arg.getDefiningOp()
                           ? arg.getDefiningOp()->getAttrOfType<IntegerAttr>(
                                 "d2m.cb_for_operand")
                           : IntegerAttr()) {
      unsigned targetIdx = static_cast<unsigned>(cbForOp.getInt());
      TT_assertv(targetIdx < infos.size(), "d2m.cb_for_operand out of range");
      infos[targetIdx].cbMemref = arg;
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

  // Create CB descriptors.
  SmallVector<ttnn::KernelCBAttr> cbDescriptors =
      createCBDescriptors(rewriter, infos, device, coreRangeSet);

  // Create Kernel descriptors.
  SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
  SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
      rewriter, op.getThreads(), coreRangeSet, opSymTable, mathFidelity);

  // Extract semaphore descriptors from kernel functions.
  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
      createSemaphoreDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                 opSymTable);

  ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
      ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

  // Collect IO tensors.
  SmallVector<Value> ios;
  for (auto &info : infos) {
    ios.push_back(info.ioTensor);
  }

  rewriter.setInsertionPoint(op);
  rewriter.replaceOpWithNewOp<ttnn::GenericOp>(op, ios, additionalArgs, program,
                                               ttnn::MemoryConfigAttr());
  return success();
}

static LogicalResult convertGenerics(ModuleOp module,
                                     DenseMap<Value, Value> &valueMapping,
                                     ttmetal::MathFidelity mathFidelity) {
  IRRewriter rewriter(module.getContext());

  SmallVector<d2m::GenericOp> genericOps;
  module.walk([&](d2m::GenericOp op) { genericOps.push_back(op); });

  for (auto op : genericOps) {
    if (failed(
            convertSingleGeneric(op, rewriter, valueMapping, mathFidelity))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult cleanupAndVerify(ModuleOp module,
                                      DenseMap<Value, Value> &valueMapping) {
  // Replace all mapped values' uses with their TTNN equivalents.
  // This handles d2m.empty/full results used directly by func.return, etc.
  for (auto &[oldVal, newVal] : valueMapping) {
    oldVal.replaceAllUsesWith(newVal);
  }

  // Resolve "exit" casts -- TTNNMetalLayoutCastOps whose result type
  // is a tensor (used by surviving TTNN ops like ttnn.to_memory_config or
  // function returns). Replace their uses with the resolved TTNN tensor.
  module.walk([&](ttir::TTNNMetalLayoutCastOp op) {
    if (!isa<RankedTensorType>(op.getResult().getType())) {
      return;
    }
    if (op.use_empty()) {
      return;
    }
    Value resolved = findIOTensor(op.getOperand(), valueMapping);
    op.getResult().replaceAllUsesWith(resolved);
  });

  // Collect and erase all remaining D2M/memref/cast ops. Generics are
  // already erased by replaceOpWithNewOp in Phase 2. External uses were
  // resolved in Step 1; remaining uses are internal among ops being erased, so
  // dropAllUses is safe.
  SmallVector<Operation *> opsToErase;
  module.walk([&](Operation *op) {
    if (isa<ttir::TTNNMetalLayoutCastOp, d2m::StreamLayoutOp, d2m::ViewLayoutOp,
            d2m::EmptyOp, d2m::FullOp, d2m::ResetGlobalSemaphoreOp,
            d2m::CreateGlobalSemaphoreOp, memref::DeallocOp, memref::AllocOp>(
            op)) {
      opsToErase.push_back(op);
    }
  });
  for (auto *op : llvm::reverse(opsToErase)) {
    op->erase();
  }

  // Verification walk.
  WalkResult result = module.walk([](Operation *op) -> WalkResult {
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

LogicalResult runD2MToTTNNConversion(ModuleOp module,
                                     ttmetal::MathFidelity mathFidelity) {
  DenseMap<Value, Value> valueMapping;
  if (failed(materializeTTNNTensors(module, valueMapping))) {
    return failure();
  }
  if (failed(convertSemaphores(module, valueMapping))) {
    return failure();
  }
  if (failed(convertGenerics(module, valueMapping, mathFidelity))) {
    return failure();
  }
  if (failed(cleanupAndVerify(module, valueMapping))) {
    return failure();
  }
  return success();
}

} // namespace mlir::tt
