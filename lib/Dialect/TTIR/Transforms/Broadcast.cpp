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
        if (!ttir::utils::isImplicitBroadcastSupported(broadcastOp)) {
          continue;
        }
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

// Drop a `ttir.broadcast` feeding a `ttir.matmul` operand when it only
// expands batch (leading) dims and the other operand already supplies the
// expanded size at those positions. matmul's verifier broadcasts batch dims
// natively (see MatmulOp::verify), so removing the explicit broadcast keeps
// the result shape and computation. Avoids materializing the expanded
// operand and prevents ConstEvalHoist from baking it into HBM (AOTAutograd
// einsum tracing emits this on small constant matmul operands; the broadcast
// lands on matmul after dot_general decomposition + canonicalization).
class FoldBroadcastIntoMatmul : public OpRewritePattern<ttir::MatmulOp> {
public:
  using OpRewritePattern<ttir::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value newA = op.getA();
    Value newB = op.getB();
    bool changed = false;
    if (Value dropped = tryDropBroadcast(newA, newB)) {
      newA = dropped;
      changed = true;
    }
    if (Value dropped = tryDropBroadcast(newB, newA)) {
      newB = dropped;
      changed = true;
    }
    if (!changed) {
      return failure();
    }
    rewriter.modifyOpInPlace(op, [&]() {
      op.setOperand(0, newA);
      op.setOperand(1, newB);
    });
    return success();
  }

private:
  // If `operand` is a ttir.broadcast that expanded only batch (leading-but-
  // not-last-two) dims, and `other` already has the expanded size at every
  // such position, return the broadcast's input. Otherwise return a null
  // Value.
  static Value tryDropBroadcast(Value operand, Value other) {
    auto bcast = operand.getDefiningOp<ttir::BroadcastOp>();
    if (!bcast || !ttir::utils::isImplicitBroadcastSupported(bcast)) {
      return {};
    }
    auto inShape =
        mlir::cast<RankedTensorType>(bcast.getInput().getType()).getShape();
    auto outShape = mlir::cast<RankedTensorType>(operand.getType()).getShape();
    auto otherShape = mlir::cast<RankedTensorType>(other.getType()).getShape();
    int64_t rank = outShape.size();
    // Require ≥3D and equal ranks: matmul aligns batch dims by position when
    // ranks match; differing ranks complicate the safety check.
    if (rank < 3 || static_cast<int64_t>(otherShape.size()) != rank) {
      return {};
    }
    // Inner two dims (M/K or K/N) are matmul-inner; broadcasting them would
    // change semantics — in particular expanding K changes the contract size.
    for (int64_t i = rank - 2; i < rank; ++i) {
      if (inShape[i] != outShape[i]) {
        return {};
      }
    }
    // For each expanded batch dim, the other operand must already supply the
    // expanded size — otherwise dropping the broadcast shrinks the result.
    for (int64_t i = 0; i < rank - 2; ++i) {
      if (inShape[i] != outShape[i] && otherShape[i] != outShape[i]) {
        return {};
      }
    }
    return bcast.getInput();
  }
};

class TTIRImplicitBroadcastFold
    : public impl::TTIRImplicitBroadcastFoldBase<TTIRImplicitBroadcastFold> {
public:
  using impl::TTIRImplicitBroadcastFoldBase<
      TTIRImplicitBroadcastFold>::TTIRImplicitBroadcastFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRImplicitBroadcastFoldRewriter, FoldBroadcastIntoMatmul>(
        &getContext());
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
