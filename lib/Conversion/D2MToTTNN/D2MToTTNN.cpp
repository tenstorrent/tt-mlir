// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTNN/D2MToTTNN.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
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

  llvm::SmallVector<int64_t> ttnnGridShape;
  ttnn::TensorMemoryLayout memLayoutEnum;
  ArrayRef<int64_t> gridShape = deviceLayout.getGridShape(memrefType);

  if (mlir::isa<ttcore::InterleavedLayoutAttr>(memrefType.getLayout())) {
    ttnnGridShape = {1, 1};
    memLayoutEnum = ttnn::TensorMemoryLayout::Interleaved;
  } else {
    auto virtMap = d2m::utils::getVirtualGridForwardMapping(memrefValue);

    if (!virtMap) {
      ttnnGridShape.assign(gridShape.begin(), gridShape.end());
      memLayoutEnum = ttnn::TensorMemoryLayout::BlockSharded;
    } else {
      auto physicalGrid = d2m::utils::getPhysicalGridShape(memrefValue);
      ttnnGridShape.assign(physicalGrid.begin(), physicalGrid.end());
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

  // The gridShape is already in physical core coords; emit a single rectangle
  // at the origin.
  ttnn::CoreRangeSetAttr coreRangeSet{};
  if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
    coreRangeSet = ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, ttnnGridShape[1] - 1,
                                          ttnnGridShape[0] - 1)));
  }

  return {ttnn::TTNNLayoutAttr::get(
      ctx, linearMap, ttnnGridShape, shardMemref, memLayout,
      /*tensorMesh=*/nullptr,
      /*ignorePhysicalLayout=*/false, coreRangeSet)};
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

static ttnn::CoreRangeSetAttr coreRangeSetFromGeneric(MLIRContext *ctx,
                                                      d2m::GenericOp op) {
  ttcore::GridAttr grid = op.getGrid();
  AffineMap virtToPhysMap = grid.getVirtToPhysicalMap();
  ArrayRef<int64_t> gridShape = grid.getShape();
  if (virtToPhysMap.isEmpty()) {
    auto physicalGridShape = op.getPhysicalGridShape();
    return ttnn::CoreRangeSetAttr::get(
        ctx, ttnn::CoreRangeAttr::get(
                 ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                 ttnn::CoreCoordAttr::get(ctx, physicalGridShape[1] - 1,
                                          physicalGridShape[0] - 1)));
  }

  TT_assertv(gridShape.size() == virtToPhysMap.getNumDims(),
             "virt_to_physical num dims {} must match grid rank {}",
             virtToPhysMap.getNumDims(), gridShape.size());
  TT_assertv((virtToPhysMap.getNumResults() == 2u ||
              virtToPhysMap.getNumResults() == 3u),
             "virt_to_physical must have 2 or 3 results (got {})",
             virtToPhysMap.getNumResults());

  if (virtToPhysMap.getNumResults() == 3u) {
    virtToPhysMap = virtToPhysMap.getSubMap({1, 2});
  }

  llvm::SmallVector<int64_t> start(gridShape.size(), 0);
  llvm::SmallVector<int64_t> end;
  end.reserve(gridShape.size());
  for (int64_t dim : gridShape) {
    end.push_back(dim - 1);
  }
  d2m::utils::BoundingBox physBox = d2m::utils::getProjectedBoundingBox(
      d2m::utils::BoundingBox{start, end}, virtToPhysMap);
  auto coreStart =
      ttnn::CoreCoordAttr::get(ctx, static_cast<uint64_t>(physBox.start[1]),
                               static_cast<uint64_t>(physBox.start[0]));
  auto coreEnd =
      ttnn::CoreCoordAttr::get(ctx, static_cast<uint64_t>(physBox.end[1]),
                               static_cast<uint64_t>(physBox.end[0]));
  return ttnn::CoreRangeSetAttr::get(
      ctx, llvm::ArrayRef{ttnn::CoreRangeAttr::get(ctx, coreStart, coreEnd)});
}

static mlir::Attribute
convertKernelArg(Builder &builder, const ttkernel::ArgAttr &arg,
                 const llvm::DenseMap<size_t, size_t> &semIndexMap,
                 const DenseMap<uint32_t, uint32_t> &additionalArgMapping,
                 const DenseMap<size_t, size_t> &cbOperandIndexToPortMapping) {
  switch (arg.getArgType()) {
  case ttkernel::ArgType::BufferAddress: {
    return builder.getAttr<ttnn::KernelArgAddressOfTensorAttr>(
        arg.getOperandIndex());
  }
  case ttkernel::ArgType::CBPort: {
    return builder.getAttr<ttnn::KernelArgCBBufferIndexAttr>(
        cbOperandIndexToPortMapping.at(arg.getOperandIndex()));
  }
  case ttkernel::ArgType::LocalSemaphore: {
    auto it = semIndexMap.find(arg.getOperandIndex());
    assert(it != semIndexMap.end() &&
           "local semaphore operand index not in map");
    return builder.getAttr<ttnn::KernelArgSemaphoreAtAttr>(it->second);
  }
  case ttkernel::ArgType::NamedArgument: {
    return builder.getAttr<ttnn::KernelArgNamedArgAttr>(arg.getArgumentName(),
                                                        arg.getOperandIndex());
  }
  case ttkernel::ArgType::GlobalSemaphore: {
    return builder.getAttr<ttnn::KernelArgGlobalSemaphoreAttr>(
        additionalArgMapping.at(arg.getOperandIndex()));
  }
  case ttkernel::ArgType::Scalar: {
    return builder.getAttr<ttnn::KernelArgScalarAttr>(
        additionalArgMapping.at(arg.getOperandIndex()));
  }
  }
  llvm_unreachable("Invalid ArgType");
}

static SmallVector<ttnn::KernelSemaphoreAttr>
createSemaphoreDescriptors(Builder &builder, const ArrayAttr &threads,
                           const ttnn::CoreRangeSetAttr &coreRangeSet,
                           const SymbolTable &symbolTable,
                           llvm::DenseMap<size_t, size_t> &semIndexMap) {
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

    auto collectLocalSemaphore = [&](ttkernel::ArgAttr arg) {
      if (arg.getArgType() == ttkernel::ArgType::LocalSemaphore) {
        seenSemaphoreIndices.insert(arg.getOperandIndex());
      }
    };
    for (auto rtArg : kernelSpec.getRtArgs()) {
      collectLocalSemaphore(rtArg);
    }
    for (auto ctArg : kernelSpec.getCtArgs()) {
      collectLocalSemaphore(ctArg);
    }
  }

  // Sort collected operand indices and build a mapping to 0-based semaphore
  // descriptor ids. The operand indices are positions in the d2m.generic's
  // full operand list (which includes buffers), so they are not necessarily
  // 0-based.
  SmallVector<size_t> sortedIndices(seenSemaphoreIndices.begin(),
                                    seenSemaphoreIndices.end());
  llvm::sort(sortedIndices);
  for (auto [id, idx] : llvm::enumerate(sortedIndices)) {
    semIndexMap[idx] = id;
  }

  size_t numSemaphores = sortedIndices.size();
  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors(numSemaphores);
  for (size_t i = 0; i < numSemaphores; ++i) {
    semaphoreDescriptors[i] = builder.getAttr<ttnn::KernelSemaphoreAttr>(
        /*id=*/i, ttnn::KernelCoreType::Worker, coreRangeSet,
        /*initial_value=*/0);
  }

  return semaphoreDescriptors;
}

static SmallVector<mlir::Attribute> createKernelDescriptors(
    Builder &builder, const ArrayAttr &threads,
    const ttnn::CoreRangeSetAttr &coreRangeSet, const SymbolTable &symbolTable,
    const ttmetal::MathFidelity mathFidelity,
    const llvm::DenseMap<size_t, size_t> &semIndexMap,
    const DenseMap<uint32_t, uint32_t> &additionalArgMapping,
    const DenseMap<size_t, size_t> &cbOperandIndexToPortMapping,
    const ttcore::Arch arch) {
  SmallVector<mlir::Attribute> kernelConfigs(threads.size());
  for (const auto [i, thread] : llvm::enumerate(threads)) {
    const d2m::ThreadAttr threadAttr = mlir::cast<d2m::ThreadAttr>(thread);

    // Get kernel args.
    SymbolRefAttr kernelSymbol = threadAttr.getKernelSymbol();
    auto kernelFunc =
        symbolTable.lookup<mlir::func::FuncOp>(kernelSymbol.getRootReference());
    auto kernelSpec = kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
        ttkernel::ArgSpecAttr::name);

    // Uniform TTKernel runtime args are modeled as common runtime args in
    // TTNN generic descriptors.
    auto crtArgs = kernelSpec.getRtArgs();
    auto ctArgs = kernelSpec.getCtArgs();
    llvm::SmallVector<mlir::Attribute> kernelCTArgs(ctArgs.size());
    llvm::SmallVector<mlir::Attribute> kernelCRTArgs(crtArgs.size());
    for (const auto [i, arg] : llvm::enumerate(crtArgs)) {
      kernelCRTArgs[i] =
          convertKernelArg(builder, arg, semIndexMap, additionalArgMapping,
                           cbOperandIndexToPortMapping);
    }
    for (const auto [i, arg] : llvm::enumerate(ctArgs)) {
      kernelCTArgs[i] =
          convertKernelArg(builder, arg, semIndexMap, additionalArgMapping,
                           cbOperandIndexToPortMapping);
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
      const int32_t dmCoreIndex = threadAttr.getDmCoreIndex();
      TT_assert(dmCoreIndex >= 0);
      const auto nocIndex = ttcore::getDmCoreDefaultNoc(arch, dmCoreIndex);
      auto processor = dmCoreIndex == 0 ? ttnn::DataMovementProcessor::RiscV0
                                        : ttnn::DataMovementProcessor::RiscV1;
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

static std::pair<SmallVector<ttnn::KernelCBAttr>, DenseMap<size_t, size_t>>
createCBDescriptors(Builder &builder, d2m::GenericOp op,
                    const ttcore::DeviceAttr &device,
                    const ttnn::CoreRangeSetAttr &coreRangeSet) {
  MLIRContext *ctx = builder.getContext();
  SmallVector<ttnn::KernelCBAttr> cbDescriptors;
  DenseMap<size_t, size_t> cbOperandIndexToPort;
  size_t cbPorts = 0;
  // Add additional args.
  unsigned ioSize = op.getInputsAndOutputs().size();
  for (unsigned i = 0; i < op.getAdditionalArgs().size(); ++i) {
    auto operandIndex = ioSize + i;
    auto operand = op.getOperands()[operandIndex];
    // Check for hoisted CB buffer
    auto cbMemref = mlir::dyn_cast_if_present<MemRefType>(operand.getType());
    if (!cbMemref) {
      continue;
    }

    TT_assertv(mlir::isa<ttcore::TileType>(cbMemref.getElementType()),
               "Only TileType supported.");
    ttcore::DataType dtype =
        ttcore::elementTypeToDataType(cbMemref.getElementType());
    size_t pageSize = device.getMemrefCBPageSizeBytes(cbMemref);
    size_t totalSize = device.getMemrefSizeBytes(cbMemref, pageSize, true);

    ttnn::KernelCBFormatAttr cbFormat =
        ttnn::KernelCBFormatAttr::get(ctx, cbPorts, dtype, pageSize);

    ttnn::KernelCBGlobalBufferAddressOfTensorAttr globalCBIndexOfTensor;
    if (auto aliasOp =
            mlir::dyn_cast<d2m::OperandAliasOp>(operand.getDefiningOp())) {
      // OperandAliasOp's input is the generic's operand that this CB aliases.
      globalCBIndexOfTensor =
          ttnn::KernelCBGlobalBufferAddressOfTensorAttr::get(
              ctx, op.getOperandIndex(aliasOp.getMemref()));
    } else {
      assert(mlir::isa<memref::AllocOp>(operand.getDefiningOp()) &&
             "expected alloc or alias op for cb memref");
    }

    cbDescriptors.push_back(ttnn::KernelCBAttr::get(
        ctx, totalSize, coreRangeSet, {cbFormat}, globalCBIndexOfTensor));
    cbOperandIndexToPort[operandIndex] = cbPorts;
    cbPorts++;
  }

  return {cbDescriptors, cbOperandIndexToPort};
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
      TT_assertv(castLayoutAttr.getGridShape() ==
                     convertedLayoutAttr.getGridShape(),
                 "ttnn_metal_layout_cast and converted type must have the same "
                 "grid shape");

      emptyTensorType = castResultType;
    }

    auto device = ttnn::utils::getOrInsertDevice(rewriter, op);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto emptyOp = rewriter.create<ttnn::EmptyOp>(
        loc, emptyTensorType, device,
        ttnn::ShapeAttr::get(ctx, emptyTensorType.getShape()));

    valueMapping[op.getResult()] = emptyOp.getResult();
    for (auto castOp : castsToMap) {
      valueMapping[castOp.getResult()] = emptyOp.getResult();
    }
    return success();
  }

  return op.emitOpError("Unsupported memref.alloc");
}

static FailureOr<ttnn::LayoutAttr>
getTensorAllocLayout(Operation *op, RankedTensorType tensorType) {
  MLIRContext *ctx = op->getContext();
  auto encoding = tensorType.getEncoding();

  if (auto layoutAttr = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(encoding)) {
    return ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
  }
  if (auto ndLayoutAttr = mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(encoding)) {
    return ttnn::LayoutAttr::get(ctx, ndLayoutAttr.getLayout());
  }
  return op->emitOpError("unsupported encoding type"), failure();
}

static LogicalResult convertD2MEmpty(d2m::EmptyOp op, IRRewriter &rewriter,
                                     DenseMap<Value, Value> &valueMapping) {
  auto tensorType = cast<RankedTensorType>(op.getResult().getType());
  auto layout = getTensorAllocLayout(op, tensorType);
  if (failed(layout)) {
    return failure();
  }

  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);
  auto shape = ttnn::ShapeAttr::get(op.getContext(), tensorType.getShape());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto emptyOp =
      rewriter.create<ttnn::EmptyOp>(op.getLoc(), tensorType, device, shape);
  valueMapping[op.getResult()] = emptyOp.getResult();
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
  auto coreRangeSet = ttnn::CoreRangeSetAttr::get(
      rewriter.getContext(), llvm::ArrayRef<ttnn::CoreRangeAttr>{coreRange});

  auto device = ttnn::utils::getOrInsertDevice(rewriter, op);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);
  auto ttnnOp = rewriter.create<ttnn::CreateGlobalSemaphoreOp>(
      op.getLoc(), device, op.getValueAttr(), coreRangeSet);
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
  rewriter.create<ttnn::ResetGlobalSemaphoreOp>(op.getLoc(), mappedSemaphore,
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

  moduleOp.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      allocOps.push_back(allocOp);
    } else if (auto emptyOp = dyn_cast<d2m::EmptyOp>(op)) {
      emptyOps.push_back(emptyOp);
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

static Value findIOTensor(Value operand, DenseMap<Value, Value> &valueMapping) {
  if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
    if (isa<ttnn::TTNNLayoutAttr>(tensorType.getEncoding())) {
      return operand;
    }
  }

  auto iter = valueMapping.find(operand);
  if (iter != valueMapping.end()) {
    auto tensorType = dyn_cast<RankedTensorType>(iter->second.getType());
    TT_assertv(tensorType, "expected mapped value to be a ranked tensor");
    TT_assertv(isa<ttnn::TTNNLayoutAttr>(tensorType.getEncoding()),
               "expected mapped value to be a TTNN tensor");
    return iter->second;
  }

  auto *def = operand.getDefiningOp();
  if (auto view = dyn_cast<d2m::ViewLayoutOp>(def)) {
    return findIOTensor(view.getInput(), valueMapping);
  }
  if (auto cast = dyn_cast<ttir::TTNNMetalLayoutCastOp>(def)) {
    // Input is already a TTNN tensor.
    auto tensorType = dyn_cast<RankedTensorType>(cast.getInput().getType());
    TT_assertv(tensorType, "expected input to be a ranked tensor");
    TT_assertv(isa<ttnn::TTNNLayoutAttr>(tensorType.getEncoding()),
               "expected input to be a ranked tensor");
    return cast.getInput();
  }

  llvm_unreachable("unexpected operand def chain");
}

static LogicalResult convertSingleGeneric(d2m::GenericOp op,
                                          IRRewriter &rewriter,
                                          DenseMap<Value, Value> &valueMapping,
                                          ttmetal::MathFidelity mathFidelity) {
  MLIRContext *ctx = rewriter.getContext();
  auto device = ttcore::lookupDevice(op->getParentOp());
  TT_assert(device);

  ttnn::CoreRangeSetAttr coreRangeSet = coreRangeSetFromGeneric(ctx, op);

  SmallVector<Value> ttnnGenericAdditionalArgs;
  // Additional args in the d2m.generic don't map 1:1 to the ttnn.generic's
  // additional args (d2m generic's additional args has cbs whereas ttnn
  // generic's additional args does not) so we need to create mapping to adjust
  // the kernel descriptors later.
  DenseMap<uint32_t, uint32_t> additionalArgMapping;
  for (const auto [idx, arg] : llvm::enumerate(op.getAdditionalArgs())) {
    if (isa<d2m::GlobalSemaphoreType>(arg.getType())) {
      Value mapped = valueMapping.count(arg) ? valueMapping[arg] : arg;
      additionalArgMapping[op.getInputsAndOutputs().size() + idx] =
          op.getInputsAndOutputs().size() + ttnnGenericAdditionalArgs.size();
      ttnnGenericAdditionalArgs.push_back(mapped);
    } else if (isa<d2m::LocalSemaphoreType>(arg.getType())) {
      // Local semaphores are described via createSemaphoreDescriptors; skip.
    } else if (isa<MemRefType>(arg.getType())) {
      // CBs are described via createCBDescriptors; skip.
    } else if (mlir::isa<IntegerType, IndexType, FloatType>(arg.getType())) {
      additionalArgMapping[op.getInputsAndOutputs().size() + idx] =
          op.getInputsAndOutputs().size() + ttnnGenericAdditionalArgs.size();
      ttnnGenericAdditionalArgs.push_back(arg);
    } else {
      return op.emitOpError(
                 "unexpected operand type in d2m.generic's additionalArgs: ")
             << arg.getType();
    }
  }

  auto [cbDescriptors, cbOperandIndexToPortMapping] =
      createCBDescriptors(rewriter, op, device, coreRangeSet);

  SymbolTable opSymTable(op->getParentOfType<ModuleOp>());
  // Build the semaphore mapping first so createKernelDescriptors can use it.
  llvm::DenseMap<size_t, size_t> semIndexMap;
  SmallVector<ttnn::KernelSemaphoreAttr> semaphoreDescriptors =
      createSemaphoreDescriptors(rewriter, op.getThreads(), coreRangeSet,
                                 opSymTable, semIndexMap);

  const auto arch = ttcore::getOpChipDescAttr(op).getArch().getValue();
  SmallVector<mlir::Attribute> kernelDescriptors = createKernelDescriptors(
      rewriter, op.getThreads(), coreRangeSet, opSymTable, mathFidelity,
      semIndexMap, additionalArgMapping, cbOperandIndexToPortMapping, arch);

  ttnn::ProgramAttr program = ttnn::ProgramAttr::get(
      ctx, kernelDescriptors, cbDescriptors, semaphoreDescriptors);

  SmallVector<Value> ios;
  for (Value input : op.getInputsAndOutputs()) {
    auto tensor = findIOTensor(input, valueMapping);
    ios.push_back(tensor);
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
  rewriter.replaceOpWithNewOp<ttnn::GenericOp>(
      op, ios, ttnnGenericAdditionalArgs, program);
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
// kernels, CBs, and semaphores preserve per-region core_ranges from converted
// ttnn.generics.
// ===----------------------------------------------------------------------===//

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
    } else if (auto scalar = mlir::dyn_cast<ttnn::KernelArgScalarAttr>(arg)) {
      if (auto unified =
              remapTable.lookupAdditional(generic, scalar.getOperandIndex())) {
        mapped = ttnn::KernelArgScalarAttr::get(ctx, *unified);
      }
    }
    out.push_back(mapped);
  }
  return out;
}

static Attribute
remapSpatialKernelDescriptor(MLIRContext *ctx, Attribute kernelAttr,
                             ttnn::GenericOp generic,
                             const SpatialRemapTable &remapTable) {
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
            ctx, compute.getSymbolRef(), compute.getCoreRanges(),
            compute.getMathFidelity(), compute.getFp32DestAccEn(),
            compute.getDstFullSyncEn(), compute.getUnpackToDestModes(),
            compute.getBfp8PackPrecise(), compute.getMathApproxMode(), crt, ct);
      })
      .Case<ttnn::DataMovementKernelAttr>([&](ttnn::DataMovementKernelAttr dm) {
        return ttnn::DataMovementKernelAttr::get(
            ctx, dm.getSymbolRef(), dm.getCoreRanges(), dm.getProcessor(),
            dm.getNocIndex(), dm.getNocMode(), crt, ct);
      })
      .Case<ttnn::ReadKernelAttr>([&](ttnn::ReadKernelAttr read) {
        return ttnn::ReadKernelAttr::get(ctx, read.getSymbolRef(),
                                         read.getCoreRanges(), crt, ct);
      })
      .Case<ttnn::WriteKernelAttr>([&](ttnn::WriteKernelAttr write) {
        return ttnn::WriteKernelAttr::get(ctx, write.getSymbolRef(),
                                          write.getCoreRanges(), crt, ct);
      })
      .Default([](Attribute attr) { return attr; });
}

static ttnn::KernelCBAttr
adjustSpatialCBDescriptor(MLIRContext *ctx, ttnn::KernelCBAttr cb,
                          ttnn::GenericOp generic,
                          const SpatialRemapTable &remapTable) {
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
  return ttnn::KernelCBAttr::get(ctx, cb.getTotalSize(), cb.getCoreRanges(),
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

  SmallVector<Attribute> mergedKernels;
  SmallVector<ttnn::KernelCBAttr> mergedCBs;
  SmallVector<ttnn::KernelSemaphoreAttr> mergedSemaphores;
  for (ttnn::GenericOp generic : regionGenerics) {
    auto program = llvm::cast<ttnn::ProgramAttr>(generic.getProgram());
    for (Attribute kernelAttr : program.getKernels()) {
      mergedKernels.push_back(
          remapSpatialKernelDescriptor(ctx, kernelAttr, generic, remapTable));
    }
    for (ttnn::KernelCBAttr cb : program.getCbs()) {
      mergedCBs.push_back(
          adjustSpatialCBDescriptor(ctx, cb, generic, remapTable));
    }
    for (ttnn::KernelSemaphoreAttr sem : program.getSemaphores()) {
      mergedSemaphores.push_back(ttnn::KernelSemaphoreAttr::get(
          ctx, sem.getId(), sem.getCoreType(), sem.getCoreRanges(),
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
      remapTable.getUnifiedAdditionalArgs(), mergedProgram);
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
  // This handles d2m.empty results used directly by func.return.
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
    Value resolved = findIOTensor(op.getOperand(), valueMapping);
    op.getResult().replaceAllUsesWith(resolved);
  });

  // Collect and erase all remaining D2M/memref/cast ops. Generics are
  // already erased. External uses are resolved by the remapping.
  SmallVector<Operation *> opsToErase;
  moduleOp.walk([&](Operation *op) {
    if (isa<ttir::TTNNMetalLayoutCastOp, d2m::ViewLayoutOp, d2m::EmptyOp,
            d2m::ResetGlobalSemaphoreOp, d2m::CreateGlobalSemaphoreOp,
            d2m::CreateLocalSemaphoreOp, memref::DeallocOp, memref::AllocOp,
            d2m::OperandAliasOp>(op)) {
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
