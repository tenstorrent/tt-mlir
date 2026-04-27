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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
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

  static KernelArgsAttr
  evalKernelArgsFromSpec(Builder &builder, const SymbolTable &symbolTable,
                         SymbolRefAttr kernelSymbol,
                         const DenseMap<size_t, size_t> &cbOperandIndexToPort) {
    auto kernelFunc =
        symbolTable.lookup<func::FuncOp>(kernelSymbol.getRootReference());
    ttkernel::ArgSpecAttr kernelSpec =
        kernelFunc->getAttrOfType<ttkernel::ArgSpecAttr>(
            ttkernel::ArgSpecAttr::name);
    SmallVector<ttmetal::KernelArgAttr> rtArgs;
    SmallVector<ttmetal::KernelArgAttr> ctArgs;
    for (ttkernel::ArgAttr arg : kernelSpec.getRtArgs()) {
      if (arg.getArgType() == ttkernel::ArgType::CB) {
        rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), cbOperandIndexToPort.at(arg.getOperandIndex())));
      } else {
        rtArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), arg.getOperandIndex()));
      }
    }
    for (ttkernel::ArgAttr arg : kernelSpec.getCtArgs()) {
      if (arg.getArgType() == ttkernel::ArgType::CB) {
        ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), cbOperandIndexToPort.at(arg.getOperandIndex())));
      } else {
        ctArgs.push_back(builder.getAttr<ttmetal::KernelArgAttr>(
            arg.getArgType(), arg.getOperandIndex()));
      }
    }
    return builder.getAttr<ttmetal::KernelArgsAttr>(rtArgs, ctArgs);
  }

  static ArrayAttr convertThreadsToKernelConfigs(
      Builder &builder, mlir::ValueRange inputOutputOperands, ArrayAttr threads,
      CoreRangeAttr coreRange, const SymbolTable &symbolTable,
      ttmetal::MathFidelity mathFidelity,
      const DenseMap<size_t, size_t> &cbOperandIndexToPort) {
    SmallVector<Attribute> kernelConfigs;
    int unassignedNocCounter = 0;

    for (Attribute threadAttr : threads) {
      d2m::ThreadAttr thread = mlir::cast<d2m::ThreadAttr>(threadAttr);
      KernelArgsAttr kernelArgs = evalKernelArgsFromSpec(
          builder, symbolTable, thread.getKernelSymbol(), cbOperandIndexToPort);
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
    llvm::SmallVector<Value> args;
    for (unsigned i = 0; i < op.getInputsAndOutputs().size(); ++i) {
      auto operand = adaptor.getOperands()[i];

      if (auto view = mlir::dyn_cast_if_present<d2m::ViewLayoutOp>(
              operand.getDefiningOp());
          view) {
        args.push_back(view.getInput());
        remappedBuffers.push_back(rewriter.getRemappedValue(view.getInput()));
      } else {
        args.push_back(operand);
        remappedBuffers.push_back(rewriter.getRemappedValue(operand));
      }
    }

    // Add additional args.
    llvm::SmallVector<Value> cbs;
    llvm::SmallVector<int64_t> cbPorts;
    int64_t cbPort = 0;
    DenseMap<size_t, size_t> cbOperandIndexToPort;
    unsigned ioSize = op.getInputsAndOutputs().size();
    for (unsigned i = 0; i < op.getAdditionalArgs().size(); ++i) {
      auto operandIndex = ioSize + i;
      auto operand = adaptor.getOperands()[operandIndex];
      if (mlir::isa<ttmetal::GlobalSemaphoreType>(operand.getType())) {
        args.push_back(operand);
      } else if (mlir::isa<ttmetal::LocalSemaphoreType>(operand.getType())) {
        args.push_back(operand);
      } else if (auto memrefType =
                     mlir::dyn_cast_if_present<MemRefType>(operand.getType());
                 memrefType) {
        assert(mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout()) &&
               "expected cb layout");
        // Hoisted CB buffer (already converted to CreateBufferOp by
        // MemrefAllocRewriter).
        if (auto aliasOp = mlir::dyn_cast<d2m::OperandAliasOp>(
                op.getOperands()[operandIndex].getDefiningOp())) {
          // OperandAliasOp's input is the generic's operand that this CB
          // aliases. It could be a function argument of the parent func or an
          // AllocOp.
          Value aliasedMemref = aliasOp.getMemref();
          auto parentFunc = op->getParentOfType<func::FuncOp>();
          bool isFuncArg =
              mlir::isa<BlockArgument>(aliasedMemref) &&
              mlir::cast<BlockArgument>(aliasedMemref).getOwner() ==
                  &parentFunc.getBody().front();
          assert((isFuncArg ||
                  mlir::isa<memref::AllocOp>(aliasedMemref.getDefiningOp())) &&
                 "expected OperandAliasOp input to be a func argument or "
                 "memref::AllocOp");
          cbs.push_back(aliasedMemref);
          cbOperandIndexToPort[operandIndex] = cbPort;
          cbPorts.push_back(cbPort++);
        } else if (auto allocOp = mlir::dyn_cast_if_present<memref::AllocOp>(
                       op.getOperands()[operandIndex].getDefiningOp());
                   allocOp) {
          cbs.push_back(operand);
          cbOperandIndexToPort[operandIndex] = cbPort;
          cbPorts.push_back(cbPort++);
        } else {
          llvm_unreachable("expected alloc or aliad op for cb memref");
        }
      } else if (mlir::isa<IntegerType, IndexType, FloatType>(
                     operand.getType())) {
        args.push_back(operand);
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
        mathFidelity_, cbOperandIndexToPort);
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
          /*virtualGridForwardMapping=*/fwd);
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
        /*virtualGridForwardMapping=*/fwd);

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
class D2MCreateLocalSemaphoreRewriter
    : public OpConversionPattern<d2m::CreateLocalSemaphoreOp> {
public:
  using OpConversionPattern<d2m::CreateLocalSemaphoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::CreateLocalSemaphoreOp op,
                  d2m::CreateLocalSemaphoreOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::CreateLocalSemaphoreOp>(
        op, ttmetal::LocalSemaphoreType::get(rewriter.getContext()),
        adaptor.getInitialValueAttr());
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

/// Second-phase lowering for d2m.spatial:
/// - Merge all nested ttmetal.enqueue_program ops into one enqueue_program.
/// - Concatenate per-region `cbs`. Temporary workaround: remap hardware
/// `cb_ports` to globally unique ids (sequential 0..N-1) so merged regions do
/// not reuse the same CB port id when circular-buffer placement is
/// over-approximated to the full worker grid (see TTMetalToFlatbuffer
/// circular_buffer_config).
/// - Remap kernel-arg indices for merged enqueue: BufferAddress,
/// GlobalSemaphore, and CBPort. Runtime resolves CBPort by indexing the merged
/// `cbs` list, then reads that entry's hardware port.
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
    unsigned chipNumCbs = 0;
    bool foundSystemDesc = false;
    for (Operation *walk = op->getParentOp(); walk;
         walk = walk->getParentOp()) {
      if (auto mod = dyn_cast<mlir::ModuleOp>(walk)) {
        if (auto systemDesc = mod->getAttrOfType<ttcore::SystemDescAttr>(
                ttcore::SystemDescAttr::name)) {
          chipNumCbs = systemDesc.getChipDesc(0).getNumCBs();
          foundSystemDesc = true;
          break;
        }
      }
    }
    if (!foundSystemDesc) {
      return rewriter.notifyMatchFailure(
          op,
          "missing ttcore.system_desc on an enclosing module for spatial CB "
          "port remap");
    }

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

      const size_t mergedCbSlotBase = mergedCbs.size();
      llvm::append_range(mergedCbs, enqueueProgram.getCbs());
      // Workaround: rebuild cb_ports as a contiguous global range when merging.
      const size_t regionCbPortCount = enqueueProgram.getCbPorts().size();
      if (mergedCbPorts.size() + regionCbPortCount > chipNumCbs) {
        return rewriter.notifyMatchFailure(
            op, "merged spatial enqueue_program cb_ports exceed chip num_cbs");
      }
      const int64_t portBase = static_cast<int64_t>(mergedCbPorts.size());
      llvm::append_range(
          mergedCbPorts,
          llvm::seq<int64_t>(
              portBase, portBase + static_cast<int64_t>(regionCbPortCount)));
      for (Attribute kernelConfig : enqueueProgram.getKernelConfigs()) {
        mergedKernelConfigs.push_back(
            remapKernelConfig(kernelConfig, spatialMetalRange, enqueueProgram,
                              remapTable, mergedCbSlotBase));
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
                                      const SpatialRemapTable &remapTable,
                                      size_t mergedCbSlotBase) {
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
    } else if (kernelArg.getType() == ttkernel::ArgType::CB) {
      operandIndex += mergedCbSlotBase;
    }
    return builder.getAttr<KernelArgAttr>(kernelArg.getType(), operandIndex);
  }

  static KernelArgsAttr
  remapKernelArgs(Builder &builder, KernelArgsAttr kernelArgs,
                  ttmetal::EnqueueProgramOp enqueueProgram,
                  const SpatialRemapTable &remapTable,
                  size_t mergedCbSlotBase) {
    SmallVector<KernelArgAttr> remappedRuntimeArgs;
    SmallVector<KernelArgAttr> remappedCompileTimeArgs;
    remappedRuntimeArgs.reserve(kernelArgs.getRtArgs().size());
    remappedCompileTimeArgs.reserve(kernelArgs.getCtArgs().size());

    for (KernelArgAttr runtimeArg : kernelArgs.getRtArgs()) {
      remappedRuntimeArgs.push_back(remapKernelArg(
          builder, runtimeArg, enqueueProgram, remapTable, mergedCbSlotBase));
    }
    for (KernelArgAttr compileTimeArg : kernelArgs.getCtArgs()) {
      remappedCompileTimeArgs.push_back(
          remapKernelArg(builder, compileTimeArg, enqueueProgram, remapTable,
                         mergedCbSlotBase));
    }

    return builder.getAttr<KernelArgsAttr>(remappedRuntimeArgs,
                                           remappedCompileTimeArgs);
  }

  static Attribute remapKernelConfig(Attribute kernelConfig,
                                     CoreRangeAttr spatialCoreRange,
                                     ttmetal::EnqueueProgramOp enqueueProgram,
                                     const SpatialRemapTable &remapTable,
                                     size_t mergedCbSlotBase) {
    Builder builder(kernelConfig.getContext());
    return TypeSwitch<Attribute, Attribute>(kernelConfig)
        .Case<ComputeConfigAttr>([&](ComputeConfigAttr computeConfig) {
          return ComputeConfigAttr::get(
              computeConfig.getContext(), computeConfig.getKernelSymbol(),
              spatialCoreRange,
              remapKernelArgs(builder, computeConfig.getKernelArgs(),
                              enqueueProgram, remapTable, mergedCbSlotBase),
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
                              enqueueProgram, remapTable, mergedCbSlotBase),
              nocConfig.getNocIndex());
        })
        .Case<EthernetConfigAttr>([&](EthernetConfigAttr ethernetConfig) {
          return EthernetConfigAttr::get(
              ethernetConfig.getContext(), ethernetConfig.getKernelSymbol(),
              spatialCoreRange,
              remapKernelArgs(builder, ethernetConfig.getKernelArgs(),
                              enqueueProgram, remapTable, mergedCbSlotBase),
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

namespace {
class D2MOperandAliasRewriter
    : public OpConversionPattern<d2m::OperandAliasOp> {
public:
  using OpConversionPattern<d2m::OperandAliasOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(d2m::OperandAliasOp op, d2m::OperandAliasOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::OperandAliasOp>(
        op, op.getResult().getType(), adaptor.getMemref());
    return success();
  }
};
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
      ttmetal::D2MResetGlobalSemaphoreRewriter,
      ttmetal::D2MCreateLocalSemaphoreRewriter, ttmetal::D2MViewLayoutRewriter>(
      ctx);
  patterns.add<ttmetal::D2MGenericRewriter>(ctx, mathFidelity);

  // remove alias op after generic conversion
  patterns.add<ttmetal::D2MOperandAliasRewriter>(ctx);
}

void populateD2MToTTMetalSpatialOpPattern(MLIRContext *ctx,
                                          RewritePatternSet &patterns) {
  patterns.add<ttmetal::SpatialOpRewriter>(ctx);
}

} // namespace mlir::tt
