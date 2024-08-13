// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetalOpsTypes.h"
#include <llvm/Support/raw_ostream.h>

namespace mlir::tt::ttmetal {

#define GEN_PASS_DEF_CONVERTTTIRTOTTMETAL
#include "ttmlir/Dialect/TTMetal/Transforms/Passes.h.inc"

class TTIRToTTMetalLayoutRewriter : public OpRewritePattern<ttir::ToLayoutOp> {
public:
  using OpRewritePattern<ttir::ToLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto inputTy = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputTy = mlir::cast<RankedTensorType>(op.getType());
    if (not inputTy.getEncoding() || not outputTy.getEncoding()) {
      return failure();
    }
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
      return failure();
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

  uint64_t lookupAddress(Value value) const {
    auto *op = value.getDefiningOp();
    if (!op) {
      return 0;
    }
    auto allocOp = dyn_cast<ttir::AllocOp>(op);
    if (!allocOp) {
      return 0;
    }
    return allocOp.getAddress();
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

    rewriter.setInsertionPointToStart(noc0Block);
    auto push0 = rewriter.create<ttkernel::CBPushBackOp>(
        op.getLoc(), noc0Block->getArgument(0));
    push0->remove();
    noc0Block->push_back(push0);
    auto return0 =
        rewriter.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    return0->remove();
    noc0Block->push_back(return0);

    rewriter.setInsertionPointToStart(noc1Block);
    auto push1 = rewriter.create<ttkernel::CBPushBackOp>(
        op.getLoc(), noc1Block->getArgument(1));
    push1->remove();
    noc1Block->push_back(push1);
    auto return1 =
        rewriter.create<ttkernel::ReturnOp>(op.getLoc(), ValueRange());
    return1->remove();
    noc1Block->push_back(return1);

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
  }
};

void createTTIRToTTMetalBackendPipeline(OpPassManager &pm) {
  pm.addPass(mlir::tt::ttir::createTTIRLoadSystemDesc());
  pm.addPass(mlir::tt::ttir::createTTIRImplicitDevice());

  // insert pass that expands broadcasted tensor shapes
  // need to check if layout is created normally afterwards as Nick explained
  // - whether affine map drops some of dimensions...

  // onnx-mlir does following:
  //     pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  //     -> return return std::make_unique<ShapeInferencePass>();
  //     -> void ShapeInferencePass::runOnOperation() {
  //   func::FuncOp f = getOperation();
  //   GreedyRewriteConfig config;
  //   config.useTopDownTraversal = true;
  //   (void)applyPatternsAndFoldGreedily(f.getBody(), patterns, config);
  //   inferFunctionReturnShapes(f);
  // }

  //   LogicalResult initialize(MLIRContext *context) override {
  //   RewritePatternSet cumulativePatterns(context);
  //   getShapeInferencePatterns(cumulativePatterns);
  //   patterns = FrozenRewritePatternSet(std::move(cumulativePatterns));
  //   return success();
  // }

  // getShapeInferencePatterns:
  //   void getShapeInferencePatterns(RewritePatternSet &set) {
  //   // Bump up the pattern benefit of the shape inference patterns to run
  //   them
  //   // before other patterns, because most other patterns (e.g.
  //   canonicalization)
  //   // work best after shapes are inferred.
  //   PatternBenefit highPriority(10000);
  //   set.insert<InferShapesPattern>(set.getContext(), highPriority);
  //   set.insert<YieldShapesPattern>(set.getContext(), highPriority);
  // }

  // we can do as above, or try with pass that uses
  // canScheduleOn and hasInterface

  // the below compiles but does not actually run the pass, at least doesn't
  // print anything
  OpPassManager &nestedModulePM = pm.nest<func::FuncOp>();
  nestedModulePM.addPass(mlir::tt::ttir::createTTIRInferBroadcastedShapes());
  // OpPassManager &nestedPM = nestedModulePM.nestAny();

  // pm.addNestedPass<ttir::>(mlir::tt::ttir::createTTIRInferBroadcastedShapes());
  // pm.addNestedPass<mlir::tt::ttir::TTIROp>(mlir::tt::ttir::createTTIRInferBroadcastedShapes());
  // nestedPM.addPass(mlir::tt::ttir::createTTIRInferBroadcastedShapes());

  // llvm::outs() << "added pass nestedPm.size=" << nestedPM.size() << "\n";

  llvm::outs() << "added pass pm.size=" << pm.size() << "\n";
  llvm::outs() << "added pass nestedModulePM.size=" << nestedModulePM.size()
               << "\n";

  // pm.addPass(mlir::tt::ttir::createTTIRGeneric());
  // pm.addPass(mlir::tt::ttir::createTTIRLayout());
  // pm.addPass(mlir::tt::ttir::createTTIRGenericRegionOperandsToMemref());
  // pm.addPass(mlir::tt::ttir::createTTIRAllocate());
  // pm.addPass(createConvertTTIRToTTMetal());
}

} // namespace mlir::tt::ttmetal
