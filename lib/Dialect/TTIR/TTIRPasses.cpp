// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
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

static GridAttr unitGridAttr(mlir::PatternRewriter &rewriter) {
  return rewriter.getAttr<GridAttr>(SmallVector<int64_t>{1, 1},
                                    rewriter.getMultiDimIdentityMap(2));
}

template <typename TosaOp>
class TosaToTTIRKernelRewriter
    : public OpRewritePattern<TosaOp> {
public:
  using OpRewritePattern<TosaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    StringRef kernelKind;
    if constexpr (std::is_same<TosaOp, tosa::MulOp>::value) {
      assert(op.getShift() == 0);
      kernelName = "mulitply";
      kernelKind = "eltwise";
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
        op.getLoc(), TypeRange(output.getType()), kernelName, kernelKind,
        op.getOperands(), ValueRange(output));

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
        op.getOutputs(), unitGridAttr(rewriter));

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

class TTIRLayoutRewriter : public OpRewritePattern<DispatchOp> {
public:
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  SmallVector<int64_t> canonicalStride(::llvm::ArrayRef<int64_t> shape) const {
    SmallVector<int64_t> stride;
    stride.push_back(1);
    for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter) {
      stride.insert(stride.begin(),
                    stride.front() * *iter); // TODO: alignup 16B
    }
    return stride;
  }

  std::optional<ttir::LayoutOp> createLayout(PatternRewriter &rewriter,
                                             Location loc, Value input) const {
    auto ty = input.getType().cast<RankedTensorType>();
    if (ty.getEncoding()) {
      assert(ty.getEncoding().isa<LayoutAttr>());
      return std::nullopt;
    }
    auto map = rewriter.getMultiDimIdentityMap(2);
    auto memref =
        MemRefType::get(ty.getShape(), ty.getElementType(), map,
                        rewriter.getAttr<MemorySpaceAttr>(MemorySpace::System));
    LayoutAttr layoutEncoding = rewriter.getAttr<LayoutAttr>(
        canonicalStride(ty.getShape()), OOBVal::Undef, unitGridAttr(rewriter),
        memref);
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, ty.getShape(), ty.getElementType(), layoutEncoding);

    tensor::EmptyOp exising_empty = input.getDefiningOp<tensor::EmptyOp>();
    if (exising_empty) {
      rewriter.replaceOp(exising_empty, output);
      return std::nullopt;
    } else {
      return rewriter.create<ttir::LayoutOp>(loc, output.getType(), input,
                                             output, true);
    }
  }

  LogicalResult matchAndRewrite(DispatchOp op,
                                PatternRewriter &rewriter) const final {
    int operandIndex = 0;
    bool modified = false;
    for (auto operand : op.getOperands()) {
      if (auto layout = createLayout(rewriter, op.getLoc(), operand); layout) {
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(operandIndex, *layout);
          // This is kind of hacky, the last operand is the output so it'll be
          // last in setting the result type
          op.getResult(0).setType(layout->getType());
        });
        modified = true;
      }
      ++operandIndex;
    }

    // Update the region arguments to use the memref type
    if (modified) {
      operandIndex = 0;
      for (auto operand : op.getOperands()) {
        rewriter.modifyOpInPlace(op, [&]() {
          auto memref = operand.getType()
                            .cast<RankedTensorType>()
                            .getEncoding()
                            .cast<LayoutAttr>()
                            .getMemref();
          auto blockArg = op.getRegion().getArgument(operandIndex);
          blockArg.setType(memref);
          for (auto user : blockArg.getUsers()) {
            // This is kind of hacky, the last operand is the output so it'll be
            // last in setting the result type
            dyn_cast<KernelOp>(user).getResult(0).setType(memref);
          }
        });
        ++operandIndex;
      }
    }

    return modified ? success() : failure();
  }
};

class TTIRUnlayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  using OpRewritePattern<mlir::func::ReturnOp>::OpRewritePattern;

  std::optional<Value> createUnlayout(PatternRewriter &rewriter, Location loc,
                                      Value input) const {
    auto ty = input.getType().cast<RankedTensorType>();
    if (not ty.getEncoding()) {
      return std::nullopt;
    }
    assert(ty.getEncoding().isa<LayoutAttr>());
    auto output = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                   ty.getElementType());
    return rewriter.create<ttir::LayoutOp>(loc, output.getType(), input, output,
                                           true);
  }

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    int operandIndex = 0;
    bool modified = false;
    for (auto operand : op.getOperands()) {
      if (auto layout = createUnlayout(rewriter, op.getLoc(), operand);
          layout) {
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(operandIndex, *layout);
        });
        modified = true;
      }
      ++operandIndex;
    }
    return modified ? success() : failure();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLayoutRewriter, TTIRUnlayoutFuncReturnRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
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
