// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTMetal/D2MToTTMetal.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstdint>
#include <mlir-c/IR.h>
#include <optional>

namespace mlir::tt::ttmetal {

namespace {

// Returns true if the kernel function contains an op of type OpT.
template <typename OpT>
static bool kernelContainsOp(const SymbolTable &symbolTable,
                             SymbolRefAttr kernelSymbol) {
  auto kernelFunc =
      symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
  return kernelFunc.walk([](OpT) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

class D2MGenericRewriter : public OpConversionPattern<d2m::GenericOp> {
public:
  D2MGenericRewriter(MLIRContext *ctx, ttmetal::MathFidelity mathFidelity)
      : OpConversionPattern<d2m::GenericOp>(ctx), mathFidelity_(mathFidelity) {}

  static KernelArgsAttr evalKernelArgsFromSpec(Builder &builder,
                                               const SymbolTable &symbolTable,
                                               SymbolRefAttr kernelSymbol) {
    auto kernelFunc =
        symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
    ttkernel::ArgSpecAttr kernelSpec =
        kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
            ttkernel::ArgSpecAttr::name);
    SmallVector<ttmetal::KernelArgAttr> rtArgs;
    SmallVector<ttmetal::KernelArgAttr> ctArgs;
    for (ttkernel::ArgAttr arg : kernelSpec.getRtArgs()) {
      rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getArgType(), arg.getOperandIndex()));
    }
    for (ttkernel::ArgAttr arg : kernelSpec.getCtArgs()) {
      ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
          arg.getArgType(), arg.getOperandIndex()));
    }
    return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
  }

  static ArrayAttr convertThreadsToKernelConfigs(
      Builder &builder, mlir::ValueRange inputOutputOperands, ArrayAttr threads,
      CoreRangeAttr coreRange, const SymbolTable &symbolTable,
      ttmetal::MathFidelity mathFidelity) {
    SmallVector<Attribute> kernelConfigs;
    int unassignedNocCounter = 0;

    for (Attribute threadAttr : threads) {
      d2m::ThreadAttr thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, symbolTable, thread.getKernelSymbol());
      Attribute kernelConfig = nullptr;
      switch (thread.getThreadType()) {
      case d2m::ThreadType::Compute: {
        bool fp32DestAccum = false;
        for (size_t i = 0; i < inputOutputOperands.size(); ++i) {
          ttcore::DataType dataType = ttcore::elementTypeToDataType(
              ttcore::getOperandInnerElementType(inputOutputOperands[i]));

          if (getNumberOfBits(dataType) == 32) {
            fp32DestAccum = true;
          }
        }
        // This must stay in-sync with ChipDescAttr::getDstLogicalSizeTiles().
        constexpr bool dstFullSyncEn = false;
        // Enable fp32 unpack mode for typecast kernels.
        // TODO(ckaravasilisTT): Enable fp32 unpack mode in the general case.
        bool isTypecast = kernelContainsOp<ttkernel::TypecastTileOp>(
            symbolTable, thread.getKernelSymbol());
        UnpackToDestMode mode = (fp32DestAccum && isTypecast)
                                    ? UnpackToDestMode::Fp32
                                    : UnpackToDestMode::Default;
        std::vector<UnpackToDestMode> unpackModes{mode};
        kernelConfig = builder.getAttr<ttmetal::ComputeConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs, mathFidelity,
            fp32DestAccum, dstFullSyncEn, unpackModes);
        break;
      }
      case d2m::ThreadType::Datamovement: {
        int32_t nocIdx = thread.getNocIndex();
        if (nocIdx < 0) {
          nocIdx = unassignedNocCounter++ % 2;
        }
        kernelConfig = builder.getAttr<ttmetal::NocConfigAttr>(
            thread.getKernelSymbol(), coreRange, kernelArgs,
            *ttcore::symbolizeNocIndex(nocIdx));
        break;
      }
      case d2m::ThreadType::Unified: {
        // Unified threads should have been split by SplitUnifiedThread before
        // reaching this pass.
        llvm_unreachable(
            "Unified threads are not supported for TTMetal backend");
      }
      }
      assert(kernelConfig != nullptr);
      kernelConfigs.push_back(kernelConfig);
    }
    return builder.getArrayAttr(kernelConfigs);
  }

  LogicalResult
  matchAndRewrite(d2m::GenericOp op, d2m::GenericOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());

    llvm::SmallVector<Value> remappedBuffers;
    llvm::SmallVector<Value> cbs;
    llvm::SmallVector<int64_t> cbPorts;
    llvm::SmallVector<Value> args;
    int64_t cbPort = 0;
    for (unsigned i = 0; i < op.getInputsAndOutputs().size(); ++i) {
      auto operand = adaptor.getOperands()[i];

      if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
              operand.getDefiningOp());
          view) {
        args.push_back(view.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(view.getInput()));
        cbs.push_back(view.getInput());
      } else {
        args.push_back(operand);
        remappedBuffers.push_back(rewriter.getRemappedValue(operand));
        cbs.push_back(operand);
      }

      cbPorts.push_back(cbPort++);
    }

    // Add additional args.
    unsigned ioSize = op.getInputsAndOutputs().size();
    for (unsigned i = 0; i < op.getAdditionalArgs().size(); ++i) {
      auto operand = adaptor.getOperands()[ioSize + i];
      if (mlir::isa<ttmetal::GlobalSemaphoreType>(operand.getType())) {
        args.push_back(operand);
      } else if (mlir::isa<MemRefType>(operand.getType())) {
        // Hoisted CB buffer (already converted to CreateBufferOp by
        // MemrefAllocRewriter).  If it backs a regular operand, override
        // that operand's CB; otherwise add as a new CB entry.
        if (auto cbForOp =
                operand.getDefiningOp()
                    ? operand.getDefiningOp()->getAttrOfType<IntegerAttr>(
                          "d2m.cb_for_operand")
                    : IntegerAttr()) {
          unsigned idx = static_cast<unsigned>(cbForOp.getInt());
          assert(idx < cbs.size() && "d2m.cb_for_operand out of range");
          cbs[idx] = operand;
        } else {
          cbs.push_back(operand);
          cbPorts.push_back(cbPort++);
        }
      } else {
        op.emitOpError(
            "unexpected operand type in d2m.generic's additionalArgs: ")
            << operand.getType();
        return failure();
      }
    }

    ArrayAttr threads = op.getThreads();
    CoreRangeAttr coreRange = coreRangeAttrFromOp(rewriter, op);
    auto kernelConfigs = convertThreadsToKernelConfigs(
        rewriter, op.getInputsAndOutputs(), threads, coreRange, symbolTable,
        mathFidelity_);
    rewriter.replaceOpWithNewOp<ttmetal::EnqueueProgramOp>(
        op, args, cbs, cbPorts, kernelConfigs,
        op.getFabricConnectionConfigAttr());
    return success();
  };

private:
  static CoreRangeAttr coreRangeAttrFromOp(Builder &builder, d2m::GenericOp op);

  ttmetal::MathFidelity mathFidelity_;
};

// CoreRange from d2m.generic grid: when virt_to_physical is present, project
// the virtual bounding box through it (rank must match map dims; 2 or 3
// results). When the map is empty, skip that projection and return the core
// range implied by the op's physical grid alone.
CoreRangeAttr D2MGenericRewriter::coreRangeAttrFromOp(Builder &builder,
                                                      d2m::GenericOp op) {
  MLIRContext *ctx = builder.getContext();
  ttcore::GridAttr grid = op.getGrid();
  AffineMap virtToPhysMap = grid.getVirtToPhysicalMap();
  ArrayRef<int64_t> gridShape = grid.getShape();
  if (virtToPhysMap.isEmpty()) {
    return CoreRangeAttr::getPhysicalCoreRange(ctx, op.getPhysicalGridShape());
  }

  TT_assertv(gridShape.size() == virtToPhysMap.getNumDims(),
             "virt_to_physical num dims {} must match grid rank {}",
             virtToPhysMap.getNumDims(), gridShape.size());
  TT_assertv((virtToPhysMap.getNumResults() == 2u ||
              virtToPhysMap.getNumResults() == 3u),
             "virt_to_physical must have 2 or 3 results (got {})",
             virtToPhysMap.getNumResults());

  if (virtToPhysMap.getNumResults() == 3) {
    virtToPhysMap = virtToPhysMap.getSubMap({1, 2});
  }

  llvm::SmallVector<int64_t> start(gridShape.size(), 0);
  llvm::SmallVector<int64_t> end;
  for (auto dim : gridShape) {
    end.push_back(dim - 1);
  }
  d2m::utils::BoundingBox physBox = d2m::utils::getProjectedBoundingBox(
      d2m::utils::BoundingBox{start, end}, virtToPhysMap);

  const int64_t y0 = physBox.start[0];
  const int64_t x0 = physBox.start[1];
  const int64_t y1 = physBox.end[0];
  const int64_t x1 = physBox.end[1];
  int64_t offset[] = {y0, x0};
  int64_t size[] = {y1 - y0 + 1, x1 - x0 + 1};
  return CoreRangeAttr::get(ctx, offset, size);
}

} // namespace

namespace {
class MemrefAllocRewriter : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, memref::AllocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto address = op->getAttrOfType<IntegerAttr>("address");
    if (!address) {
      return failure();
    }

    assert(op.getMemref().getType().getMemorySpace() &&
           "No memref memory space found, failing.");
    auto memrefType = op.getMemref().getType();

    auto vgm = op->getAttrOfType<AffineMapAttr>(
        d2m::utils::kVirtualGridInverseMappingAttr);
    auto fwd = op->getAttrOfType<AffineMapAttr>(
        d2m::utils::kVirtualGridForwardMappingAttr);

    // Hoisted CB allocs carry CBLayoutAttr (per-core local shape).
    // Keep the original type on CreateBufferOp so the dialect conversion
    // framework doesn't see a type mismatch.
    if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
      auto cbForOperandAttr =
          op->getAttrOfType<IntegerAttr>("d2m.cb_for_operand");
      auto cbOp = rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(
          op, memrefType, address, /*virtualGridInverseMapping=*/vgm,
          /*virtualGridForwardMapping=*/fwd,
          /*cb_core_range=*/CoreRangeAttr());
      if (cbForOperandAttr) {
        cbOp->setAttr("d2m.cb_for_operand", cbForOperandAttr);
      }
      return success();
    }

    assert((mlir::isa<ttcore::ShardLayoutAttr, ttcore::InterleavedLayoutAttr>(
               memrefType.getLayout())) &&
           "expected physical device layout (shard or interleaved)");

    rewriter.replaceOpWithNewOp<ttmetal::CreateBufferOp>(
        op, memrefType, address, /*virtualGridInverseMapping=*/vgm,
        /*virtualGridForwardMapping=*/fwd,
        /*cb_core_range=*/CoreRangeAttr());

    return success();
  };
};
} // namespace

namespace {
class MemrefDeallocRewriter : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, memref::DeallocOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocateBufferOp>(op,
                                                             op.getMemref());

    return success();
  };
};
} // namespace

//===----------------------------------------------------------------------===//
// Host Transfer Ops: ToDeviceOp -> EnqueueWriteBuffer, ToHostOp ->
// EnqueueReadBuffer
//===----------------------------------------------------------------------===//

namespace {
class D2MToDeviceRewriter : public OpConversionPattern<d2m::ToDeviceOp> {
public:
  using OpConversionPattern<d2m::ToDeviceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ToDeviceOp op, d2m::ToDeviceOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            adaptor.getInput().getDefiningOp())) {
      input = view.getInput();
    } else {
      input = adaptor.getInput();
    }
    Value output;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            adaptor.getOutput().getDefiningOp())) {
      output = view.getInput();
    } else {
      output = adaptor.getOutput();
    }

    [[maybe_unused]] MemRefType outputTy =
        mlir::cast<MemRefType>(output.getType());
    assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
               outputTy.getLayout()) &&
           "output should be device memory");

    rewriter.replaceOpWithNewOp<ttmetal::EnqueueWriteBufferOp>(op, input,
                                                               output);
    return success();
  }
};
} // namespace

namespace {
class D2MToHostRewriter : public OpConversionPattern<d2m::ToHostOp> {
public:
  using OpConversionPattern<d2m::ToHostOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ToHostOp op, d2m::ToHostOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            adaptor.getInput().getDefiningOp())) {
      input = view.getInput();
    } else {
      input = adaptor.getInput();
    }
    Value output;
    if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
            adaptor.getOutput().getDefiningOp())) {
      output = view.getInput();
    } else {
      output = adaptor.getOutput();
    }

    [[maybe_unused]] MemRefType inputTy =
        mlir::cast<MemRefType>(input.getType());
    assert(!mlir::dyn_cast_if_present<ttcore::HostLayoutAttr>(
               inputTy.getLayout()) &&
           "input should be device memory");

    rewriter.replaceOpWithNewOp<ttmetal::EnqueueReadBufferOp>(op, input,
                                                              output);
    // Insert global barrier to ensure the read completes before subsequent
    // ops use it.
    rewriter.create<ttmetal::FinishOp>(op->getLoc());
    return success();
  }
};
} // namespace

namespace {
class D2MMeshShardRewriter : public OpConversionPattern<d2m::MeshShardOp> {
public:
  using OpConversionPattern<d2m::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::MeshShardOp op, d2m::MeshShardOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value input = adaptor.getInput();

    // Use the op's result type directly because bufferization has already
    // determined the correct output shape. For shard_to_full operations, the
    // result shape differs from the input (e.g., concatenating shards), while
    // for full_to_shard it's the opposite. The bufferization pass correctly
    // handles these transformations.
    Type resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<ttmetal::MeshShardOp>(
        op, resultType, input, op.getShardType(), op.getShardDirection(),
        op.getShardShape(), op.getShardDims());
    return success();
  };
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

    // Get address from memref consumed and pass it to create global semaphore
    // op.
    auto address = allocOp->getAttrOfType<IntegerAttr>("address");
    // Get core range from memref shape.
    auto coreRange = ttmetal::CoreRangeAttr::getPhysicalCoreRange(
        rewriter.getContext(), ttcore::getGridShape(op.getInput()));
    rewriter.replaceOpWithNewOp<ttmetal::CreateGlobalSemaphoreOp>(
        op, ttmetal::GlobalSemaphoreType::get(rewriter.getContext()), address,
        adaptor.getValueAttr(), coreRange);
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

    rewriter.replaceOpWithNewOp<ttmetal::ResetGlobalSemaphoreOp>(
        op, adaptor.getSemaphore(), adaptor.getValueAttr());
    return success();
  }
};
} // namespace

namespace {
class D2MViewLayoutRewriter : public OpConversionPattern<d2m::ViewLayoutOp> {
public:
  using OpConversionPattern<d2m::ViewLayoutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::ViewLayoutOp op, d2m::ViewLayoutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Erase views.
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

/// Peel view_layout so we reach the buffer producer (e.g.
/// ttmetal.create_buffer) used as an enqueue_program CB operand.
static Value stripEnqueueCbProducer(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto view = dyn_cast<d2m::ViewLayoutOp>(def)) {
      v = view.getInput();
      continue;
    }
    break;
  }
  return v;
}

/// For each CB operand of this region's enqueue_program, if the backing buffer
/// is defined by ttmetal.create_buffer with CBLayoutAttr or ShardLayoutAttr,
/// set cb_core_range to this spatial region's tensix rectangle. Shard tensors
/// that are not compiler-owned buffers (e.g. function block arguments) are
/// skipped. Returns failure if a CBLayout buffer is not defined by
/// ttmetal.create_buffer or cb_core_range would conflict.
static LogicalResult
stampSpatialRegionCbCoreRanges(ttmetal::EnqueueProgramOp enqueueProgram,
                               CoreRangeAttr spatialMetalRange) {
  for (Value cbVal : enqueueProgram.getCbs()) {
    Value root = stripEnqueueCbProducer(cbVal);
    auto memrefTy = dyn_cast<MemRefType>(root.getType());
    if (!memrefTy) {
      continue;
    }
    const bool isCB = mlir::isa<ttcore::CBLayoutAttr>(memrefTy.getLayout());
    const bool isShard =
        mlir::isa<ttcore::ShardLayoutAttr>(memrefTy.getLayout());
    if (!isCB && !isShard) {
      continue;
    }
    auto createBuffer = root.getDefiningOp<ttmetal::CreateBufferOp>();
    if (!createBuffer) {
      if (isCB) {
        return failure();
      }
      continue;
    }
    if (auto existing = createBuffer.getCbCoreRange()) {
      if (*existing != spatialMetalRange) {
        return failure();
      }
      continue;
    }
    createBuffer.setCbCoreRangeAttr(spatialMetalRange);
  }
  return success();
}

/// Second-phase lowering for d2m.spatial:
/// - Merge all nested ttmetal.enqueue_program ops into one enqueue_program.
/// - Keep cb_ports unchanged (concatenated per region). CB hardware ids are
/// per-core; spatial regions use disjoint core ranges, so CBPort kernel-arg
/// indices are not shifted across regions.
/// - Remap enqueue args indices in kernel args when args lists are
/// concatenated.
class SpatialOpRewriter : public OpConversionPattern<d2m::SpatialOp> {
public:
  using OpConversionPattern<d2m::SpatialOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::SpatialOp op,
                  [[maybe_unused]] d2m::SpatialOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.getNumResults() != 0) {
      return rewriter.notifyMatchFailure(
          op, "SpatialOp with results is not supported in TTMetal lowering");
    }

    ArrayAttr gridRanges = op.getGridRanges();
    SpatialRemapTable remapTable;
    SmallVector<Value> mergedCbs;
    SmallVector<int64_t> mergedCbPorts;
    SmallVector<Attribute> mergedKernelConfigs;
    ttcore::FabricConnectionConfigAttr mergedFabricConfig = nullptr;
    SmallVector<Operation *> preEnqueueOps;
    SmallVector<Operation *> postEnqueueOps;

    for (auto [regionIndex, region] : llvm::enumerate(op.getRegions())) {
      auto spatialCoreRange =
          mlir::cast<ttcore::CoreRangeAttr>(gridRanges.getValue()[regionIndex]);
      CoreRangeAttr spatialMetalRange =
          ttCoreSpatialRangeToTtmetalCoreRange(rewriter, spatialCoreRange);
      FailureOr<ttmetal::EnqueueProgramOp> maybeEnqueue =
          collectRegionOps(region, preEnqueueOps, postEnqueueOps);
      if (failed(maybeEnqueue)) {
        return rewriter.notifyMatchFailure(
            op, "each spatial region must contain exactly one enqueue_program");
      }
      ttmetal::EnqueueProgramOp enqueueProgram = *maybeEnqueue;
      remapTable.addEnqueueArgs(enqueueProgram);

      if (failed(stampSpatialRegionCbCoreRanges(enqueueProgram,
                                                spatialMetalRange))) {
        return rewriter.notifyMatchFailure(
            op,
            "failed to stamp spatial CB core ranges (CBLayout buffers must be "
            "defined by ttmetal.create_buffer with consistent cb_core_range "
            "across spatial merge)");
      }

      llvm::append_range(mergedCbs, enqueueProgram.getCbs());
      llvm::append_range(mergedCbPorts, enqueueProgram.getCbPorts());
      for (Attribute kernelConfig : enqueueProgram.getKernelConfigs()) {
        mergedKernelConfigs.push_back(remapKernelConfig(
            kernelConfig, spatialMetalRange, enqueueProgram, remapTable));
      }

      auto enqueueFabricConfig = enqueueProgram.getFabricConnectionConfigAttr();
      if (hasConflictingFabricConfig(mergedFabricConfig, enqueueFabricConfig)) {
        return rewriter.notifyMatchFailure(
            op, "failed to merge region enqueue_program ops due to fabric "
                "config conflict");
      }
      if (enqueueFabricConfig) {
        mergedFabricConfig = enqueueFabricConfig;
      }
    }

    if (mergedKernelConfigs.empty()) {
      return rewriter.notifyMatchFailure(
          op, "SpatialOp has no nested enqueue_program");
    }

    for (Operation *operation : preEnqueueOps) {
      rewriter.moveOpBefore(operation, op);
    }

    rewriter.create<ttmetal::EnqueueProgramOp>(
        op.getLoc(), remapTable.getUnifiedArgs(), mergedCbs, mergedCbPorts,
        rewriter.getArrayAttr(mergedKernelConfigs), mergedFabricConfig);

    for (Operation *operation : postEnqueueOps) {
      rewriter.moveOpBefore(operation, op);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  class SpatialRemapTable {
    using LocalKey = std::pair<Operation *, size_t>;

    SmallVector<Value> unifiedArgs_;
    DenseMap<Value, size_t> ioToUnifiedIdx_;
    DenseMap<LocalKey, size_t> ioArgMap_;
    DenseMap<LocalKey, size_t> globalSemaphoreArgMap_;

  public:
    void addEnqueueArgs(ttmetal::EnqueueProgramOp enqueueProgram) {
      Operation *op = enqueueProgram.getOperation();
      for (const auto [idx, arg] : llvm::enumerate(enqueueProgram.getArgs())) {
        size_t localIdx = static_cast<size_t>(idx);
        if (mlir::isa<ttmetal::GlobalSemaphoreType>(arg.getType())) {
          size_t unifiedIdx = unifiedArgs_.size();
          unifiedArgs_.push_back(arg);
          globalSemaphoreArgMap_.insert({{op, localIdx}, unifiedIdx});
          continue;
        }

        auto it = ioToUnifiedIdx_.find(arg);
        size_t unifiedIdx;
        if (it == ioToUnifiedIdx_.end()) {
          unifiedIdx = unifiedArgs_.size();
          ioToUnifiedIdx_.insert({arg, unifiedIdx});
          unifiedArgs_.push_back(arg);
        } else {
          unifiedIdx = it->second;
        }
        ioArgMap_.insert({{op, localIdx}, unifiedIdx});
      }
    }

    ArrayRef<Value> getUnifiedArgs() const { return unifiedArgs_; }

    std::optional<size_t> lookupIO(ttmetal::EnqueueProgramOp enqueueProgram,
                                   size_t localIdx) const {
      auto it = ioArgMap_.find({enqueueProgram.getOperation(), localIdx});
      if (it != ioArgMap_.end()) {
        return it->second;
      }
      return std::nullopt;
    }

    std::optional<size_t>
    lookupGlobalSemaphore(ttmetal::EnqueueProgramOp enqueueProgram,
                          size_t localIdx) const {
      auto it = globalSemaphoreArgMap_.find(
          {enqueueProgram.getOperation(), localIdx});
      if (it != globalSemaphoreArgMap_.end()) {
        return it->second;
      }
      return std::nullopt;
    }
  };

  static KernelArgAttr remapKernelArg(Builder &builder, KernelArgAttr kernelArg,
                                      ttmetal::EnqueueProgramOp enqueueProgram,
                                      const SpatialRemapTable &remapTable) {
    size_t operandIndex = kernelArg.getOperandIndex();
    if (kernelArg.getType() == ttkernel::ArgType::BufferAddress) {
      if (auto unified = remapTable.lookupIO(enqueueProgram, operandIndex)) {
        operandIndex = *unified;
      }
    } else if (kernelArg.getType() == ttkernel::ArgType::GlobalSemaphore) {
      if (auto unified =
              remapTable.lookupGlobalSemaphore(enqueueProgram, operandIndex)) {
        operandIndex = *unified;
      }
    }
    return builder.getAttr<KernelArgAttr>(kernelArg.getType(), operandIndex);
  }

  static KernelArgsAttr
  remapKernelArgs(Builder &builder, KernelArgsAttr kernelArgs,
                  ttmetal::EnqueueProgramOp enqueueProgram,
                  const SpatialRemapTable &remapTable) {
    SmallVector<KernelArgAttr> remappedRuntimeArgs;
    SmallVector<KernelArgAttr> remappedCompileTimeArgs;
    remappedRuntimeArgs.reserve(kernelArgs.getRtArgs().size());
    remappedCompileTimeArgs.reserve(kernelArgs.getCtArgs().size());

    for (KernelArgAttr runtimeArg : kernelArgs.getRtArgs()) {
      remappedRuntimeArgs.push_back(
          remapKernelArg(builder, runtimeArg, enqueueProgram, remapTable));
    }
    for (KernelArgAttr compileTimeArg : kernelArgs.getCtArgs()) {
      remappedCompileTimeArgs.push_back(
          remapKernelArg(builder, compileTimeArg, enqueueProgram, remapTable));
    }

    return builder.getAttr<KernelArgsAttr>(remappedRuntimeArgs,
                                           remappedCompileTimeArgs);
  }

  static Attribute remapKernelConfig(Attribute kernelConfig,
                                     CoreRangeAttr spatialCoreRange,
                                     ttmetal::EnqueueProgramOp enqueueProgram,
                                     const SpatialRemapTable &remapTable) {
    Builder builder(kernelConfig.getContext());
    return TypeSwitch<Attribute, Attribute>(kernelConfig)
        .Case<ComputeConfigAttr>([&](ComputeConfigAttr computeConfig) {
          return ComputeConfigAttr::get(
              computeConfig.getContext(), computeConfig.getKernelSymbol(),
              spatialCoreRange,
              remapKernelArgs(builder, computeConfig.getKernelArgs(),
                              enqueueProgram, remapTable),
              computeConfig.getMathFidelity(), computeConfig.getFp32DestAccEn(),
              computeConfig.getDstFullSyncEn(),
              computeConfig.getMathApproxMode(),
              computeConfig.getUnpackToDestMode());
        })
        .Case<NocConfigAttr>([&](NocConfigAttr nocConfig) {
          return NocConfigAttr::get(
              nocConfig.getContext(), nocConfig.getKernelSymbol(),
              spatialCoreRange,
              remapKernelArgs(builder, nocConfig.getKernelArgs(),
                              enqueueProgram, remapTable),
              nocConfig.getNocIndex());
        })
        .Case<EthernetConfigAttr>([&](EthernetConfigAttr ethernetConfig) {
          return EthernetConfigAttr::get(
              ethernetConfig.getContext(), ethernetConfig.getKernelSymbol(),
              spatialCoreRange,
              remapKernelArgs(builder, ethernetConfig.getKernelArgs(),
                              enqueueProgram, remapTable),
              ethernetConfig.getEthType(), ethernetConfig.getNocIndex());
        })
        .Default([](Attribute) -> Attribute {
          llvm_unreachable(
              "unexpected kernel config attribute kind in spatial merge");
        });
  }

  static bool hasConflictingFabricConfig(
      ttcore::FabricConnectionConfigAttr mergedFabricConfig,
      ttcore::FabricConnectionConfigAttr enqueueFabricConfig) {
    return mergedFabricConfig && enqueueFabricConfig &&
           mergedFabricConfig != enqueueFabricConfig;
  }

  static FailureOr<ttmetal::EnqueueProgramOp>
  collectRegionOps(Region &region, SmallVector<Operation *> &preEnqueueOps,
                   SmallVector<Operation *> &postEnqueueOps) {
    Block &block = region.front();
    ttmetal::EnqueueProgramOp onlyEnqueue = nullptr;

    for (Operation &innerOperation : block) {
      if (isa<d2m::SpatialYieldOp>(innerOperation)) {
        continue;
      }

      if (auto enqueueProgram =
              dyn_cast<ttmetal::EnqueueProgramOp>(&innerOperation)) {
        if (onlyEnqueue) {
          return failure();
        }
        onlyEnqueue = enqueueProgram;
        continue;
      }

      if (!onlyEnqueue) {
        preEnqueueOps.push_back(&innerOperation);
      } else {
        postEnqueueOps.push_back(&innerOperation);
      }
    }
    if (!onlyEnqueue) {
      return failure();
    }
    return onlyEnqueue;
  }

  static CoreRangeAttr
  ttCoreSpatialRangeToTtmetalCoreRange(Builder &builder,
                                       ttcore::CoreRangeAttr cr);
};

CoreRangeAttr SpatialOpRewriter::ttCoreSpatialRangeToTtmetalCoreRange(
    Builder &builder, ttcore::CoreRangeAttr cr) {
  auto sc = cr.getStartCoord();
  auto ec = cr.getEndCoord();
  SmallVector<int64_t> offset = {sc.getY(), sc.getX()};
  SmallVector<int64_t> size = {ec.getY() - sc.getY() + 1,
                               ec.getX() - sc.getX() + 1};
  return CoreRangeAttr::get(builder.getContext(), offset, size);
}

} // namespace

} // namespace mlir::tt::ttmetal

namespace mlir::tt {

void populateD2MToTTMetalPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                                  TypeConverter & /*typeConverter*/,
                                  ttmetal::MathFidelity mathFidelity) {
  patterns.add<
      ttmetal::MemrefAllocRewriter, ttmetal::MemrefDeallocRewriter,
      ttmetal::D2MToDeviceRewriter, ttmetal::D2MToHostRewriter,
      ttmetal::D2MMeshShardRewriter, ttmetal::D2MCreateGlobalSemaphoreRewriter,
      ttmetal::D2MResetGlobalSemaphoreRewriter, ttmetal::D2MViewLayoutRewriter>(
      ctx);
  patterns.add<ttmetal::D2MGenericRewriter>(ctx, mathFidelity);
}

void populateD2MToTTMetalSpatialOpPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns) {
  patterns.add<ttmetal::SpatialOpRewriter>(ctx);
}

} // namespace mlir::tt
