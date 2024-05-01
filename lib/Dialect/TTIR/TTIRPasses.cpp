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

template <typename T> T div_up(T n, T d) { return (n + d - 1) / d; }

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRLOWER
#include "ttmlir/Dialect/TTIR/TTIRPasses.h.inc"

class TTIRLinalgGenericToDispatchRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  template <typename TensorTyT>
  static TensorTyT unparallelize(TensorTyT tensorTy) {
    SmallVector<int64_t> shape(tensorTy.getShape());
    assert(shape.size() > 2);
    SmallVector<int64_t> newShape(shape.begin() + 2, shape.end());
    return tensorTy.clone(newShape);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

template <typename TosaEltwiseOp>
class TTIRTosaElementwiseToDispatchRewriter
    : public OpRewritePattern<TosaEltwiseOp> {
public:
  using OpRewritePattern<TosaEltwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaEltwiseOp op,
                                PatternRewriter &rewriter) const final {
    if constexpr (std::is_same<TosaEltwiseOp, tosa::MulOp>::value) {
      assert(op.getShift() == 0);
    }

    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("ttir.dispatch", rewriter.getContext()))
      return failure();

    // Create empty output tensor for destination passing style (DPS)
    auto outputType = op.getOutput().getType();
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType());

    // Create a dispatch op
    auto dispatch = rewriter.create<ttir::DispatchOp>(
        op.getLoc(), TypeRange(output.getType()), op.getOperands(),
        ValueRange(output),
        GridAttr::get(rewriter.getContext(),
                      GridType::unit(rewriter.getContext())));

    // Create a new basic block for the dispatch op, and plumb the operands
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    SmallVector<Location> blockArgumentLocs(dispatch.getOperands().size(),
                                            dispatch.getLoc());
    block->addArguments(TypeRange(dispatch.getOperandTypes()),
                        blockArgumentLocs);
    auto blockArgs = block->getArguments();
    op.getOperation()->setOperands(blockArgs.slice(0, op.getNumOperands()));

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
    patterns.add<TTIRLinalgGenericToDispatchRewriter,
                 TTIRTosaElementwiseToDispatchRewriter<tosa::MulOp>>(
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
