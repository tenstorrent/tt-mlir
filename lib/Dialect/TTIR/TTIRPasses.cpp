// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "ttmlir/Dialect/TTIR/TTIRPasses.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CONVERTTOSATOTTIR
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRLOWER
#include "ttmlir/Dialect/TTIR/TTIRPasses.h.inc"

template <typename TosaOp>
class TosaToTTIRKernelRewriter
    : public OpRewritePattern<TosaOp> {
public:
  using OpRewritePattern<TosaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    if constexpr (std::is_same<TosaOp, tosa::MulOp>::value) {
      assert(op.getShift() == 0);
      kernelName = "mulitply";
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Tosa operation for TTIR");
    }
    assert(kernelName.size() > 0);

    // Create empty output tensor for destination passing style (DPS)
    auto outputType = op.getOutput().getType();
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType());

    auto kernel = rewriter.create<ttir::KernelOp>(
        op.getLoc(), TypeRange(output.getType()), kernelName, op.getOperands(),
        ValueRange(output));

    rewriter.replaceOp(op, kernel);

    return success();
  }
};

class ConvertTosaToTTIR : public impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR> {
public:
  using impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR>::ConvertTosaToTTIRBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TosaToTTIRKernelRewriter<tosa::MulOp>>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

class TTIRLinalgGenericToDispatchRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

class TTIRKernelDispatchRewriter : public OpRewritePattern<KernelOp> {
public:
  using OpRewritePattern<KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const final {
    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("ttir.dispatch", rewriter.getContext()))
      return failure();

    // Create a dispatch op
    auto dispatch = rewriter.create<ttir::DispatchOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(),
        GridAttr::get(rewriter.getContext(),
                      GridType::unit(rewriter.getContext())));

    // Create a new basic block for the dispatch op and create block arguments
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    SmallVector<Location> blockArgumentLocs(dispatch.getOperands().size(),
                                            dispatch.getLoc());
    block->addArguments(TypeRange(dispatch.getOperandTypes()),
                        blockArgumentLocs);

    // Update the operands of the original op to use the block arguments
    op.getOperation()->setOperands(block->getArguments());

    // Move the original op into the dispatch block
    Operation *operation = op.getOperation()->clone();
    block->push_back(operation);
    rewriter.setInsertionPoint(block, block->end());
    rewriter.create<ttir::YieldOp>(dispatch.getLoc(),
                                   ValueRange({operation->getResult(0)}));
    rewriter.replaceOp(op, dispatch);
    return success();
  }
};

class TTIRDispatch : public impl::TTIRDispatchBase<TTIRDispatch> {
public:
  using impl::TTIRDispatchBase<TTIRDispatch>::TTIRDispatchBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<TTIRLinalgGenericToDispatchRewriter, TTIRKernelDispatchRewriter>(
            &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final { assert(false); }
};

class TTIRShard : public impl::TTIRShardBase<TTIRShard> {
public:
  using impl::TTIRShardBase<TTIRShard>::TTIRShardBase;

  void runOnOperation() final { assert(false); }
};

class TTIRLower : public impl::TTIRLowerBase<TTIRLower> {
public:
  using impl::TTIRLowerBase<TTIRLower>::TTIRLowerBase;

  void runOnOperation() final { assert(false); }
};

} // namespace mlir::tt::ttir
