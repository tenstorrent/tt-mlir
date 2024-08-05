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

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttmetal {
struct CoreCoord {
  std::int64_t d = 0;
  std::int64_t y = 0;
  std::int64_t x = 0;

  CoreCoord() = default;
  CoreCoord(std::int64_t d, std::int64_t y, std::int64_t x)
      : d(d), y(y), x(x) {}
  CoreCoord(ArrayRef<std::int64_t> coord)
      : d(coord[0]), y(coord[1]), x(coord[2]) {}

  std::int64_t &operator[](std::size_t i) {
    assert(i < 3);
    return i == 0 ? d : i == 1 ? y : x;
  }

  std::int64_t operator[](std::size_t i) const {
    assert(i < 3);
    return i == 0 ? d : i == 1 ? y : x;
  }

  bool operator==(CoreCoord const &other) const {
    return d == other.d && y == other.y && x == other.x;
  }
};
} // namespace mlir::tt::ttmetal

namespace llvm {
template <> struct DenseMapInfo<mlir::tt::ttmetal::CoreCoord> {
  static mlir::tt::ttmetal::CoreCoord getEmptyKey() {
    return mlir::tt::ttmetal::CoreCoord{-1, -1, -1};
  }

  static mlir::tt::ttmetal::CoreCoord getTombstoneKey() {
    return mlir::tt::ttmetal::CoreCoord{-2, -2, -2};
  }

  static unsigned getHashValue(mlir::tt::ttmetal::CoreCoord coord) {
    return llvm::hash_combine(coord.d, coord.y, coord.x);
  }

  static bool isEqual(mlir::tt::ttmetal::CoreCoord lhs,
                      mlir::tt::ttmetal::CoreCoord rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

static std::array<int64_t, 2> coordMapping[8][8] = {
    {{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 6}, {2, 7}, {2, 8}, {2, 9}},
    {{3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 6}, {3, 7}, {3, 8}, {3, 9}},
    {{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 6}, {4, 7}, {4, 8}, {4, 9}},
    {{5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 6}, {5, 7}, {5, 8}, {5, 9}},
    {{7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 6}, {7, 7}, {7, 8}, {7, 9}},
    {{8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 6}, {8, 7}, {8, 8}, {8, 9}},
    {{9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 6}, {9, 7}, {9, 8}, {9, 9}},
    {{10, 1}, {10, 2}, {10, 3}, {10, 4}, {10, 6}, {10, 7}, {10, 8}, {10, 9}},
};

static uint64_t lookupAddress(Value value) {
  auto blockArg = mlir::dyn_cast<BlockArgument>(value);
  if (blockArg) {
    auto funcOp = blockArg.getOwner()->getParentOp();
    auto argAlloc = mlir::cast<ArgumentAllocationAttr>(
        mlir::cast<ArrayAttr>(funcOp->getDiscardableAttr(
            "argument_allocations"))[blockArg.getArgNumber()]);
    return argAlloc.getAddress();
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
    CoreCoord srcCoord;
    std::int64_t srcOffset = 0;
    std::int64_t dstOffset = 0;
    std::int64_t size = 0;

    NocRead(CoreCoord srcCoord, std::int64_t srcOffset, std::int64_t dstOffset,
            std::int64_t size)
        : srcCoord(srcCoord), srcOffset(srcOffset), dstOffset(dstOffset),
          size(size) {}

    bool isContiguous(CoreCoord nextCoord, std::int64_t nextSrcOffset,
                      std::int64_t nextDstOffset) const {
      return (nextCoord == srcCoord) && (nextSrcOffset == srcOffset + size) &&
             (nextDstOffset == dstOffset + size);
    }
  };

  llvm::MapVector<CoreCoord, mlir::SmallVector<NocRead>>
  calculateDataMovement(ArrayRef<std::int64_t> tensorShape,
                        std::int64_t elemSize, AffineMap src,
                        AffineMap dst) const {
    // For now it's just a simple pull model, but eventually we want to leverage
    // both NoCs and the both read and write
    llvm::MapVector<CoreCoord, mlir::SmallVector<NocRead>> dst2srcMap;
    assert(src.getNumResults() == 4);
    assert(dst.getNumResults() == 4);

    ::ttmlir::utils::sample(tensorShape, [&dst2srcMap, src, dst, elemSize](
                                             ArrayRef<std::int64_t> index) {
      auto srcResults = src.compose(index);
      auto dstResults = dst.compose(index);
      assert(srcResults.size() == src.getNumResults());
      assert(dstResults.size() == dst.getNumResults());
      CoreCoord srcCoord(srcResults);
      CoreCoord dstCoord(dstResults);
      std::int64_t srcOffset = srcResults[3] * elemSize;
      std::int64_t dstOffset = dstResults[3] * elemSize;
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
    if (inputLayout.isSystemMemorySpace()) {
      assert(outputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttmetal::HostWriteOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else if (outputLayout.isSystemMemorySpace()) {
      assert(inputLayout.isDeviceMemorySpace());
      rewriter.replaceOpWithNewOp<ttmetal::HostReadOp>(
          op, outputTy, op.getInput(), op.getOutput());
    } else {
      tt::DeviceAttr device = op.getDevice();
      assert(inputLayout.getPhysicalShape(inputTy.getShape()) ==
                 outputLayout.getPhysicalShape(outputTy.getShape()) &&
             "Physical shapes must match for now");
      assert(device);
      AffineMap src =
          inputLayout.projectOnto(inputTy.getShape(), device.getGrid());
      AffineMap dst =
          outputLayout.projectOnto(outputTy.getShape(), device.getGrid());
      auto dm = calculateDataMovement(
          inputTy.getShape(), inputLayout.getElementSizeBytes(), src, dst);

      auto noc0Attr = rewriter.getAttr<ttkernel::ThreadTypeAttr>(
          ttkernel::ThreadType::Noc0);
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
          SmallVector<Value>({op.getOutput()}),
          rewriter.getArrayAttr(coreRanges), rewriter.getArrayAttr(threadTypes),
          rewriter.getArrayAttr(operand_cb_port_mapping), threadTypes.size());

      int i = 0;
      std::int64_t inputBaseAddress = lookupAddress(op.getInput());
      std::int64_t outputBaseAddress = lookupAddress(op.getOutput());
      assert(inputBaseAddress);
      assert(outputBaseAddress);
      for (auto [dstCoord, srcs] : dm) {
        Block *nocBlock = rewriter.createBlock(&metalDispatch.getRegion(i++));
        OpBuilder nocBuilder(nocBlock, nocBlock->begin());
        for (auto s : srcs) {
          // TODO: ASSERT NOC ALIGNMENT
          // TODO: ASSERT PCIE ALIGNMENT
          // TODO: ASSERT DRAM ALIGNMENT
          auto [yPhys, xPhys] = coordMapping[s.srcCoord.y][s.srcCoord.x];
          auto y = nocBuilder.create<arith::ConstantOp>(
              op.getLoc(), nocBuilder.getI32Type(),
              nocBuilder.getI32IntegerAttr(yPhys));
          auto x = nocBuilder.create<arith::ConstantOp>(
              op.getLoc(), nocBuilder.getI32Type(),
              nocBuilder.getI32IntegerAttr(xPhys));
          auto srcOffset = nocBuilder.create<arith::ConstantOp>(
              op.getLoc(), nocBuilder.getI32Type(),
              nocBuilder.getI32IntegerAttr(inputBaseAddress + s.srcOffset));
          auto srcRemoteNocAddr = nocBuilder.create<ttkernel::GetNocAddrOp>(
              op.getLoc(), x, y, srcOffset);
          auto dstLocalL1Addr = nocBuilder.create<arith::ConstantOp>(
              op.getLoc(), nocBuilder.getI32Type(),
              nocBuilder.getI32IntegerAttr(outputBaseAddress + s.dstOffset));
          auto size = nocBuilder.create<arith::ConstantOp>(
              op.getLoc(), nocBuilder.getI32Type(),
              nocBuilder.getI32IntegerAttr(s.size));
          nocBuilder.create<ttkernel::NocAsyncReadOp>(
              op.getLoc(), srcRemoteNocAddr, dstLocalL1Addr, size);
        }
        nocBuilder.create<ttkernel::NocAsyncReadBarrierOp>(op.getLoc());
        nocBuilder.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
      }

      rewriter.replaceOp(op, metalDispatch);
    }
    return success();
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
      auto memref = mlir::cast<MemRefType>(arg.getType());
      rewrittenBlockArgumentTypes.push_back(
          rewriter.getType<ttkernel::CBType>(address, port, memref));
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
  pm.addPass(mlir::tt::ttir::createTTIRGeneric());
  pm.addPass(mlir::tt::ttir::createTTIRLayout());
  pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  pm.addPass(mlir::tt::ttir::createTTIRAllocate());
  pm.addPass(createConvertTTIRToTTMetal());
}

} // namespace mlir::tt::ttmetal
