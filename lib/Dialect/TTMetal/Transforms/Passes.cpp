// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"

#include "ttmlir/Dialect/TT/Utils/PhysicalCoreCoord.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

// This routine walks the SSA value chain to find the address of the value.
// It runs into and gets the address from one of the following:
//   - ttir::AllocOp: The address is the address of the allocation is embedded
//   in the op attributes.
//   - func::FuncOp Argument: The address is taken from ArgumentAllocationAttr.
//
// This routine is used to lookup the address of tensors for address generation
// and for CB creation.
static uint64_t lookupAddress(Value value) {
  auto blockArg = mlir::dyn_cast<BlockArgument>(value);
  if (blockArg) {
    auto funcOp = blockArg.getOwner()->getParentOp();
    if (mlir::isa<func::FuncOp>(funcOp)) {
      auto argAlloc = mlir::cast<ArgumentAllocationAttr>(
          mlir::cast<ArrayAttr>(funcOp->getDiscardableAttr(
              ArgumentAllocationAttr::name))[blockArg.getArgNumber()]);
      return argAlloc.getAddress();
    }
  }

  auto *op = value.getDefiningOp();
  if (!op) {
    return 0;
  }

  while (isa<DestinationStyleOpInterface>(op)) {
    assert(op->getResults().size() == 1);
    auto dps = cast<DestinationStyleOpInterface>(op);
    assert(dps.getNumDpsInits() == 1);
    auto *opOperand = dps.getDpsInitOperand(0);
    value = opOperand->get();
    op = value.getDefiningOp();
  }

  auto allocOp = dyn_cast<ttir::AllocOp>(op);
  if (!allocOp) {
    return 0;
  }
  return allocOp.getAddress();
}

class TTIRToTTMetalLayoutRewriter : public OpRewritePattern<ttir::ToLayoutOp> {
public:
  using OpRewritePattern<ttir::ToLayoutOp>::OpRewritePattern;

  struct NocRead {
    PhysicalCoreCoord srcCoord;
    std::int64_t srcOffset = 0;
    std::int64_t dstOffset = 0;
    std::int64_t size = 0;

    NocRead(PhysicalCoreCoord srcCoord, std::int64_t srcOffset,
            std::int64_t dstOffset, std::int64_t size)
        : srcCoord(srcCoord), srcOffset(srcOffset), dstOffset(dstOffset),
          size(size) {}

    bool isContiguous(PhysicalCoreCoord nextCoord, std::int64_t nextSrcOffset,
                      std::int64_t nextDstOffset) const {
      return (nextCoord == srcCoord) && (nextSrcOffset == srcOffset + size) &&
             (nextDstOffset == dstOffset + size);
    }
  };

  // This routine calculates the data movement for a tensor layout change by
  // tracing the walk order of the src and dst affine maps.  The sample routine
  // is just a helper function that iterates over the tensor shape and calls the
  // lambda with the current index.  It walks the shape in innermost-major
  // order. It also coalesces the noc transactions.
  //
  // The return value is a map of destination physical cores where each core has
  // an associated list of noc reads to be performed.
  llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocRead>>
  calculateDataMovement(ArrayRef<std::int64_t> tensorShape,
                        std::int64_t elemSize, AffineMap src,
                        AffineMap dst) const {
    // For now it's just a simple pull model, but eventually we want to leverage
    // both NoCs and the both read and write
    llvm::MapVector<PhysicalCoreCoord, mlir::SmallVector<NocRead>> dst2srcMap;
    assert(src.getNumResults() == 4);
    assert(dst.getNumResults() == 4);

    ::ttmlir::utils::sample(tensorShape, [&dst2srcMap, src, dst, elemSize](
                                             ArrayRef<std::int64_t> index) {
      auto srcResults = src.compose(index);
      auto dstResults = dst.compose(index);
      assert(srcResults.size() == src.getNumResults());
      assert(dstResults.size() == dst.getNumResults());
      PhysicalCoreCoord srcCoord(srcResults);
      PhysicalCoreCoord dstCoord(dstResults);
      std::int64_t srcOffset = srcResults.back() * elemSize;
      std::int64_t dstOffset = dstResults.back() * elemSize;
      mlir::SmallVector<NocRead> &srcs = dst2srcMap[dstCoord];
      if (not srcs.empty() &&
          srcs.back().isContiguous(srcCoord, srcOffset, dstOffset)) {
        srcs.back().size += elemSize;
      } else {
        srcs.push_back(NocRead(srcCoord, srcOffset, dstOffset, elemSize));
      }
    });

    return dst2srcMap;
  }

  void buildNocAsyncRead(mlir::Location loc, std::int64_t inputBaseAddress,
                         std::int64_t outputBaseAddress,
                         std::int64_t addressAlignment, NocRead read,
                         PhysicalCoreCoordMapping const &physicalCoordMapping,
                         mlir::OpBuilder &nocBuilder) const {
    assert(read.srcOffset % addressAlignment == 0);
    assert(read.dstOffset % addressAlignment == 0);
    assert(read.size % addressAlignment == 0);
    auto [yPhys, xPhys] = physicalCoordMapping[read.srcCoord];
    auto y = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(yPhys));
    auto x = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(xPhys));
    auto srcOffset = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(inputBaseAddress + read.srcOffset));
    auto srcRemoteNocAddr =
        nocBuilder.create<ttkernel::GetNocAddrOp>(loc, x, y, srcOffset);
    auto dstLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(),
        nocBuilder.getI32IntegerAttr(outputBaseAddress + read.dstOffset));
    auto size = nocBuilder.create<arith::ConstantOp>(
        loc, nocBuilder.getI32Type(), nocBuilder.getI32IntegerAttr(read.size));
    nocBuilder.create<ttkernel::NocAsyncReadOp>(loc, srcRemoteNocAddr,
                                                dstLocalL1Addr, size);
  }

  LogicalResult relayout(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());
    assert(inputLayout.getMemorySpace() == outputLayout.getMemorySpace());
    tt::DeviceAttr device = op.getDevice();
    assert(device);
    tt::SystemDescAttr systemDesc = op.getSystemDesc();
    assert(systemDesc);
    auto addressAlignment = systemDesc.getAddressAlignBytes(
        /*inputLayout.getMemorySpace() issue #407*/);
    assert(inputLayout.getPhysicalShape(inputTy.getShape()) ==
               outputLayout.getPhysicalShape(outputTy.getShape()) &&
           "Physical shapes must match for now");
    AffineMap src =
        inputLayout.projectOnto(inputTy.getShape(), device.getGrid());
    AffineMap dst =
        outputLayout.projectOnto(outputTy.getShape(), device.getGrid());
    auto dm = calculateDataMovement(
        inputTy.getShape(), inputLayout.getElementSizeBytes(), src, dst);

    auto noc0Attr =
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc0);
    SmallVector<Attribute> threadTypes(dm.size(), noc0Attr);
    SmallVector<Attribute> operand_cb_port_mapping;
    SmallVector<Attribute> coreRanges;
    coreRanges.reserve(dm.size());
    for (auto [dstCoord, srcs] : dm) {
      SmallVector<int64_t> offset = {dstCoord.y, dstCoord.x};
      SmallVector<int64_t> size = {1, 1};
      coreRanges.push_back(
          rewriter.getAttr<ttmetal::CoreRangeAttr>(offset, size));
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes),
        rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

    int i = 0;
    PhysicalCoreCoordMapping physicalCoordMapping(systemDesc.getChipDescs());
    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
    assert(inputBaseAddress);
    assert(outputBaseAddress);
    assert(inputBaseAddress % addressAlignment == 0);
    assert(outputBaseAddress % addressAlignment == 0);
    for (auto [dstCoord, srcs] : dm) {
      Block *nocBlock = rewriter.createBlock(&metalDispatch.getRegion(i++));
      OpBuilder nocBuilder(nocBlock, nocBlock->begin());
      for (auto s : srcs) {
        buildNocAsyncRead(op.getLoc(), inputBaseAddress, outputBaseAddress,
                          addressAlignment, s, physicalCoordMapping,
                          nocBuilder);
      }
      nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
      nocBuilder.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    }

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }

  LogicalResult reformat(ttir::ToLayoutOp op, PatternRewriter &rewriter) const {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());
    bool shouldTilize = not inputLayout.isTiled() && outputLayout.isTiled();
    bool shouldUntilize = inputLayout.isTiled() && not outputLayout.isTiled();
    assert(shouldTilize ^ shouldUntilize);
    assert(inputLayout.getGrid() == outputLayout.getGrid());

    auto tensixAttr = rewriter.getAttr<ttkernel::ThreadTypeAttr>(
        ttkernel::ThreadType::Tensix);
    SmallVector<Attribute> threadTypes = {tensixAttr};
    SmallVector<Attribute> operand_cb_port_mapping = {
        rewriter.getI64IntegerAttr(0),
        rewriter.getI64IntegerAttr(16),
    };

    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(inputLayout.getGrid()),
    };

    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), SmallVector<Type>({outputTy}),
        SmallVector<Value>({op.getInput()}),
        SmallVector<Value>({op.getOutput()}), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes),
        rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

    std::int64_t inputBaseAddress = lookupAddress(op.getInput());
    std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
    Block *tensixBlock = rewriter.createBlock(&metalDispatch.getRegion(0));
    OpBuilder tensixBuilder(tensixBlock, tensixBlock->begin());
    uint64_t pageSize = inputLayout.isTiled()
                            ? inputLayout.getElementSizeBytes()
                            : outputLayout.getElementSizeBytes();
    Type inputCBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::In0, inputBaseAddress,
        mlir::cast<MemRefType>(inputLayout.getMemref()), pageSize,
        /*num_buffers*/ 1);
    Type outputCBTy = rewriter.getType<ttkernel::CBType>(
        ttkernel::CBPort::Out0, outputBaseAddress,
        mlir::cast<MemRefType>(outputLayout.getMemref()), pageSize,
        /*num_buffers*/ 1);
    tensixBlock->addArgument(inputCBTy, op.getLoc());
    tensixBlock->addArgument(outputCBTy, op.getLoc());

    int shardTileVolume = 1;
    for (auto dim : (shouldTilize ? outputLayout.getMemref().getShape()
                                  : inputLayout.getMemref().getShape())) {
      shardTileVolume *= dim;
    }

    auto numTiles = tensixBuilder.create<arith::ConstantOp>(
        op.getLoc(), tensixBuilder.getI32Type(),
        tensixBuilder.getI32IntegerAttr(shardTileVolume));

    if (shouldTilize) {
      tensixBuilder.create<ttkernel::TilizeInitOp>(
          op.getLoc(), tensixBlock->getArgument(0), numTiles,
          tensixBlock->getArgument(1));
    } else {
      tensixBuilder.create<ttkernel::UntilizeInitOp>(
          op.getLoc(), tensixBlock->getArgument(0),
          tensixBlock->getArgument(1));
    }

    if (shouldTilize) {
      tensixBuilder.create<ttkernel::TilizeBlockOp>(
          op.getLoc(), tensixBlock->getArgument(0), numTiles,
          tensixBlock->getArgument(1));
    } else {
      tensixBuilder.create<ttkernel::UntilizeBlockOp>(
          op.getLoc(), tensixBlock->getArgument(0), numTiles,
          tensixBlock->getArgument(1));
    }

    tensixBuilder.create<ttkernel::ReturnOp>(op.getLoc());

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }

  LogicalResult matchAndRewrite(ttir::ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    if (not inputTy.getEncoding() || not outputTy.getEncoding()) {
      return failure();
    }
    assert(inputTy.getShape() == outputTy.getShape());
    assert(mlir::isa<tt::LayoutAttr>(inputTy.getEncoding()));
    assert(mlir::isa<tt::LayoutAttr>(outputTy.getEncoding()));
    auto inputLayout = mlir::cast<tt::LayoutAttr>(inputTy.getEncoding());
    auto outputLayout = mlir::cast<tt::LayoutAttr>(outputTy.getEncoding());

    auto [isLayoutChange, isGridChange, isFormatChange, isMemorySpaceChange] =
        op.compoundComponents();
    bool isCompound =
        (static_cast<int>(isLayoutChange) + static_cast<int>(isGridChange) +
         static_cast<int>(isFormatChange) +
         static_cast<int>(isMemorySpaceChange)) > 1;
    assert(!isCompound && "Only one change is allowed");

    if (isMemorySpaceChange) {
      if (inputLayout.isSystemMemorySpace()) {
        assert(outputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
            op, outputTy, op.getInput(), op.getOutput());
      } else if (outputLayout.isSystemMemorySpace()) {
        assert(inputLayout.isDeviceMemorySpace());
        rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
            op, outputTy, op.getInput(), op.getOutput());
      } else {
        assert(false && "L1 <-> DRAM not supported yet");
      }
    } else if (isLayoutChange || isGridChange) {
      return relayout(op, rewriter);
    } else {
      assert(isFormatChange);
      return reformat(op, rewriter);
    }
    return failure();
  }
};

class TTIRToTTMetalKernelRewriter : public OpRewritePattern<ttir::KernelOp> {
public:
  using OpRewritePattern<ttir::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::KernelOp op,
                                PatternRewriter &rewriter) const final {
    if (not op->use_empty()) {
      return failure();
    }
    rewriter.create<ttkernel::BuiltinOp>(op.getLoc(), op.getOpAttr(),
                                         op.getKindAttr(), op.getOperands());
    op->dropAllUses();
    rewriter.eraseOp(op);
    return success();
  }
};

class TTIRToTTMetalReturnRewriter : public OpRewritePattern<ttir::YieldOp> {
public:
  using OpRewritePattern<ttir::YieldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::YieldOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttkernel::ReturnOp>(op);
    return success();
  }
};

class TTIRToTTMetalDispatchRewriter : public OpRewritePattern<ttir::GenericOp> {
public:
  using OpRewritePattern<ttir::GenericOp>::OpRewritePattern;

  bool hasUnloweredTTIRKernel(ttir::GenericOp op) const {
    bool exists = false;
    op->getRegion(0).walk([&exists](Operation *op) {
      if (isa<ttir::KernelOp>(op)) {
        exists = true;
      }
    });
    return exists;
  }

  SmallVector<Type> getBlockArgumentTypesAsCBs(
      mlir::Block::BlockArgListType blockArguments,
      SmallVector<Attribute> const &operand_cb_port_mapping,
      PatternRewriter &rewriter) const {
    SmallVector<Type> rewrittenBlockArgumentTypes;
    for (auto arg : blockArguments) {
      auto address = lookupAddress(arg);
      auto port =
          mlir::cast<IntegerAttr>(operand_cb_port_mapping[arg.getArgNumber()])
              .getInt();
      auto tensor = mlir::cast<RankedTensorType>(arg.getType());
      auto buffer = mlir::cast<BufferAttr>(tensor.getEncoding());
      auto memref = buffer.getMemref();
      rewrittenBlockArgumentTypes.push_back(rewriter.getType<ttkernel::CBType>(
          ttkernel::symbolizeCBPort(port).value(), address, memref));
    }
    return rewrittenBlockArgumentTypes;
  }

  LogicalResult matchAndRewrite(ttir::GenericOp op,
                                PatternRewriter &rewriter) const final {
    if (hasUnloweredTTIRKernel(op)) {
      return failure();
    }

    SmallVector<Attribute> threadTypes = {
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc0),
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(ttkernel::ThreadType::Noc1),
        rewriter.getAttr<ttkernel::ThreadTypeAttr>(
            ttkernel::ThreadType::Tensix),
    };
    SmallVector<Attribute> coreRanges = {
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
        rewriter.getAttr<ttmetal::CoreRangeAttr>(op.getGrid()),
    };
    SmallVector<Attribute> operand_cb_port_mapping;
    for (auto &operand : op->getOpOperands()) {
      operand_cb_port_mapping.push_back(
          rewriter.getI64IntegerAttr(operand.getOperandNumber()));
    }
    auto metalDispatch = rewriter.create<ttmetal::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), rewriter.getArrayAttr(coreRanges),
        rewriter.getArrayAttr(threadTypes),
        rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

    auto rewrittenBlockArgumentTypes = getBlockArgumentTypesAsCBs(
        op->getRegion(0).getArguments(), operand_cb_port_mapping, rewriter);

    metalDispatch.getRegion(2).takeBody(op->getRegion(0));
    Block *tensixBlock = &metalDispatch.getRegion(2).front();
    Block *noc0Block = rewriter.createBlock(&metalDispatch.getRegion(0));
    Block *noc1Block = rewriter.createBlock(&metalDispatch.getRegion(1));

    int i = 0;
    for (auto ty : rewrittenBlockArgumentTypes) {
      noc0Block->addArgument(ty, op.getLoc());
      noc1Block->addArgument(ty, op.getLoc());
      auto arg = tensixBlock->getArgument(i++);
      arg.setType(ty);
    }

    {
      OpBuilder noc0Builder(noc0Block, noc0Block->begin());
      auto one = noc0Builder.create<arith::ConstantOp>(
          op.getLoc(), noc0Builder.getI32Type(),
          noc0Builder.getI32IntegerAttr(1));
      noc0Builder.create<ttkernel::CBReserveBackOp>(
          op.getLoc(), noc0Block->getArgument(0), one);
      noc0Builder.create<ttkernel::CBPushBackOp>(
          op.getLoc(), noc0Block->getArgument(0), one);
      noc0Builder.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    }

    {
      OpBuilder noc1Builder(noc1Block, noc1Block->begin());
      auto one = noc1Builder.create<arith::ConstantOp>(
          op.getLoc(), noc1Builder.getI32Type(),
          noc1Builder.getI32IntegerAttr(1));
      noc1Builder.create<ttkernel::CBReserveBackOp>(
          op.getLoc(), noc1Block->getArgument(0), one);
      noc1Builder.create<ttkernel::CBPushBackOp>(
          op.getLoc(), noc1Block->getArgument(0), one);
      noc1Builder.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    }

    rewriter.replaceOp(op, metalDispatch);

    return success();
  }
};

class TTIRToTTMetalAllocRewriter : public OpRewritePattern<ttir::AllocOp> {
public:
  using OpRewritePattern<ttir::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::AllocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::AllocOp>(
        op, op.getType(), op.getAddress(), op.getSize(), op.getMemorySpace());
    return success();
  }
};

class TTIRToTTMetalDeallocRewriter : public OpRewritePattern<ttir::DeallocOp> {
public:
  using OpRewritePattern<ttir::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ttmetal::DeallocOp>(op, op.getResult());
    return success();
  }
};

class ConvertTTIRToTTMetal
    : public impl::ConvertTTIRToTTMetalBase<ConvertTTIRToTTMetal> {
public:
  using impl::ConvertTTIRToTTMetalBase<
      ConvertTTIRToTTMetal>::ConvertTTIRToTTMetalBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRToTTMetalLayoutRewriter, TTIRToTTMetalKernelRewriter,
                 TTIRToTTMetalReturnRewriter, TTIRToTTMetalDispatchRewriter,
                 TTIRToTTMetalAllocRewriter, TTIRToTTMetalDeallocRewriter>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttmetal::TTMetalDialect>();
    registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};

void createTTIRToTTMetalBackendPipeline(OpPassManager &pm) {
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice());
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegion());
  mlir::tt::ttir::TTIRLayoutOptions layoutOptions;
  layoutOptions.initMemorySpace = mlir::tt::MemorySpace::DeviceL1;
  pm.addPass(mlir::tt::ttir::createTTIRLayout(layoutOptions));
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  pm.addPass(mlir::tt::ttir::createTTIRAllocate());
  pm.addPass(createConvertTTIRToTTMetal());
}

} // namespace mlir::tt::ttmetal
