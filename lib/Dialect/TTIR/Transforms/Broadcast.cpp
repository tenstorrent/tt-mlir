// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITBROADCASTFOLD
#define GEN_PASS_DEF_TTIRFOLDFULLTOSCALAR
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

// Compute the implicit broadcast shape from `op`'s current operand types.
// If it differs from the result shape, update the result type and add a
// BroadcastOp on the output to restore the original shape.
// Returns true when the output was modified.
static bool addOutputBroadcastIfNeeded(Operation *op,
                                       PatternRewriter &rewriter) {
  llvm::SmallVector<int64_t> implicitShape;
  for (Value operand : op->getOperands()) {
    auto shape = mlir::cast<RankedTensorType>(operand.getType()).getShape();
    llvm::SmallVector<int64_t> prev = implicitShape;
    assert(
        mlir::OpTrait::util::getBroadcastedShape(prev, shape, implicitShape) &&
        "Operands must be broadcast-compatible");
  }

  auto resultType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();

  if (implicitShape == resultShape) {
    return false;
  }

  auto newResultType =
      mlir::RankedTensorType::get(implicitShape, resultType.getElementType());
  rewriter.modifyOpInPlace(op,
                           [&]() { op->getResult(0).setType(newResultType); });

  rewriter.setInsertionPointAfter(op);
  auto broadcastDimensions = ttmlir::utils::getBroadcastDimensions<int64_t>(
      implicitShape, resultShape);
  auto broadcastOp = rewriter.create<ttir::BroadcastOp>(
      op->getLoc(),
      RankedTensorType::get(resultShape, newResultType.getElementType(),
                            newResultType.getEncoding()),
      op->getResult(0), broadcastDimensions);
  rewriter.replaceAllUsesExcept(op->getResult(0), broadcastOp.getResult(),
                                broadcastOp);
  return true;
}

//===----------------------------------------------------------------------===//
// TTIRImplicitBroadcastFold
//===----------------------------------------------------------------------===//

// Remove explicit broadcasts from the operands of Broadcastable ops.
// If the resulting implicit broadcast shape differs from the result shape,
// add a broadcast on the output to compensate.
class TTIRImplicitBroadcastFoldRewriter : public RewritePattern {
public:
  TTIRImplicitBroadcastFoldRewriter(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<ttir::Broadcastable>()) {
      return llvm::failure();
    }

    bool operandsChanged = false;

    // Remove all explicit broadcasts from the operands.
    for (int64_t i = 0; i < op->getNumOperands(); ++i) {
      if (auto broadcastOp = mlir::dyn_cast_if_present<ttir::BroadcastOp>(
              op->getOperand(i).getDefiningOp())) {
        rewriter.modifyOpInPlace(
            op, [&]() { op->setOperand(i, broadcastOp.getInput()); });
        operandsChanged = true;
      }
    }

    if (addOutputBroadcastIfNeeded(op, rewriter)) {
      return llvm::success();
    }

    return llvm::success(operandsChanged);
  }
};

class TTIRImplicitBroadcastFold
    : public impl::TTIRImplicitBroadcastFoldBase<TTIRImplicitBroadcastFold> {
public:
  using impl::TTIRImplicitBroadcastFoldBase<
      TTIRImplicitBroadcastFold>::TTIRImplicitBroadcastFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRImplicitBroadcastFoldRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};

//===----------------------------------------------------------------------===//
// TTIRFoldFullToScalar
//===----------------------------------------------------------------------===//

// Fold creation ops (full, zeros, ones) to volume-1 tensors when all users
// are Broadcastable. If the creation op was the shape carrier for a consumer,
// a broadcast is added on the consumer's output to restore the original shape.
template <typename OpTy>
class CreationToScalarRewriter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (ttmlir::utils::volume(op.getType().getShape()) == 1) {
      return failure();
    }

    if (!llvm::all_of(op.getResult().getUsers(), [](Operation *user) {
          return user->hasTrait<ttir::Broadcastable>();
        })) {
      return failure();
    }

    llvm::SmallVector<int64_t> onesShape(op.getType().getRank(), 1);
    auto scalarType =
        RankedTensorType::get(onesShape, op.getType().getElementType());
    Value scalar = convertToScalar(op, scalarType, rewriter);
    rewriter.replaceOp(op, scalar);

    // If the shrunk creation op was the shape carrier for a consumer,
    // the consumer's result shape may no longer match the implicit
    // broadcast of its operands.
    llvm::SetVector<Operation *> users(scalar.getUsers().begin(),
                                       scalar.getUsers().end());
    for (auto *user : users) {
      addOutputBroadcastIfNeeded(user, rewriter);
    }

    return success();
  }

protected:
  virtual Value convertToScalar(OpTy op, RankedTensorType scalarType,
                                PatternRewriter &rewriter) const = 0;
};

class FullToScalarRewriter : public CreationToScalarRewriter<ttir::FullOp> {
  using CreationToScalarRewriter::CreationToScalarRewriter;

  Value convertToScalar(ttir::FullOp op, RankedTensorType scalarType,
                        PatternRewriter &rewriter) const override {
    return rewriter
        .create<ttir::FullOp>(op.getLoc(), scalarType, op.getFillValueAttr())
        .getResult();
  }
};

template <typename OpTy>
class NamedFullToScalarRewriter : public CreationToScalarRewriter<OpTy> {
  using CreationToScalarRewriter<OpTy>::CreationToScalarRewriter;

  Value convertToScalar(OpTy op, RankedTensorType scalarType,
                        PatternRewriter &rewriter) const override {
    return rewriter
        .create<OpTy>(op.getLoc(), scalarType,
                      SmallVector<int32_t>(scalarType.getRank(), 1))
        .getResult();
  }
};

class TTIRFoldFullToScalar
    : public impl::TTIRFoldFullToScalarBase<TTIRFoldFullToScalar> {
public:
  using impl::TTIRFoldFullToScalarBase<
      TTIRFoldFullToScalar>::TTIRFoldFullToScalarBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FullToScalarRewriter, NamedFullToScalarRewriter<ttir::ZerosOp>,
                 NamedFullToScalarRewriter<ttir::OnesOp>>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

} // namespace mlir::tt::ttir
