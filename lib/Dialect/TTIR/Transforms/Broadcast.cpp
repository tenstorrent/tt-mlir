// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

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

// Fold `dot_general(lhs, broadcast(rhs))` (and the symmetric lhs case) by
// dropping batch dims that the broadcast expanded — the dim survives on the
// other operand alone. Avoids materializing the expanded tensor.
//
// Only safe when every expanded dim is a batch dim of the dot_general:
// expanding a contract dim would change semantics, and expanding a non-batch
// non-contract dim would shrink the output. This is why it is done as a 
// a separate pattern rather than using the Broadcastable trait.
class FoldBroadcastIntoDotGeneral
    : public OpRewritePattern<ttir::DotGeneralOp> {
public:
  using OpRewritePattern<ttir::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttir::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    if (succeeded(tryFoldOperand(op, /*isRhs=*/true, rewriter))) {
      return success();
    }
    return tryFoldOperand(op, /*isRhs=*/false, rewriter);
  }

private:
  static LogicalResult tryFoldOperand(ttir::DotGeneralOp op, bool isRhs,
                                      PatternRewriter &rewriter) {
    Value foldOperand = isRhs ? op.getRhs() : op.getLhs();
    auto bcast = foldOperand.getDefiningOp<ttir::BroadcastOp>();
    if (!bcast) {
      return failure();
    }

    auto outType = mlir::cast<RankedTensorType>(foldOperand.getType());
    auto inType = mlir::cast<RankedTensorType>(bcast.getInput().getType());
    if (!inType.hasStaticShape() || !outType.hasStaticShape()) {
      return failure();
    }

    llvm::SmallSet<int64_t, 4> expanded;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (inType.getShape()[i] != outType.getShape()[i]) {
        expanded.insert(i);
      }
    }
    if (expanded.empty()) {
      return failure();
    }

    auto foldBatch = isRhs ? op.getBatchDimsRhs() : op.getBatchDimsLhs();
    auto otherBatch = isRhs ? op.getBatchDimsLhs() : op.getBatchDimsRhs();
    auto foldContract =
        isRhs ? op.getContractDimsRhs() : op.getContractDimsLhs();
    auto otherContract =
        isRhs ? op.getContractDimsLhs() : op.getContractDimsRhs();

    if (!llvm::all_of(expanded, [&](int64_t d) {
          return llvm::is_contained(foldBatch, d);
        })) {
      return failure();
    }

    // Drop expanded positions and build an old→new index remap.
    llvm::SmallVector<int64_t> newShape;
    llvm::SmallVector<int64_t> remap(outType.getRank(), -1);
    for (int64_t i = 0, j = 0; i < outType.getRank(); ++i) {
      if (!expanded.contains(i)) {
        newShape.push_back(inType.getShape()[i]);
        remap[i] = j++;
      }
    }

    llvm::SmallVector<int64_t> newFoldBatch, newOtherBatch;
    for (auto [f, o] : llvm::zip_equal(foldBatch, otherBatch)) {
      if (!expanded.contains(f)) {
        newFoldBatch.push_back(remap[f]);
        newOtherBatch.push_back(o);
      }
    }

    llvm::SmallVector<int64_t> newFoldContract;
    for (int64_t d : foldContract) {
      newFoldContract.push_back(remap[d]);
    }

    Value newOperand = ttir::utils::createReshapeOp(
        rewriter, op.getLoc(), bcast.getInput(), newShape);

    Value newLhs = isRhs ? op.getLhs() : newOperand;
    Value newRhs = isRhs ? newOperand : op.getRhs();
    llvm::ArrayRef<int64_t> newBatchLhs = isRhs ? newOtherBatch : newFoldBatch;
    llvm::ArrayRef<int64_t> newBatchRhs = isRhs ? newFoldBatch : newOtherBatch;
    llvm::ArrayRef<int64_t> newContractLhs =
        isRhs ? otherContract : llvm::ArrayRef<int64_t>(newFoldContract);
    llvm::ArrayRef<int64_t> newContractRhs =
        isRhs ? llvm::ArrayRef<int64_t>(newFoldContract) : otherContract;

    rewriter.replaceOpWithNewOp<ttir::DotGeneralOp>(
        op, op.getResult().getType(), newLhs, newRhs,
        rewriter.getDenseI64ArrayAttr(newBatchLhs),
        rewriter.getDenseI64ArrayAttr(newContractLhs),
        rewriter.getDenseI64ArrayAttr(newBatchRhs),
        rewriter.getDenseI64ArrayAttr(newContractRhs));
    return success();
  }
};

class TTIRImplicitBroadcastFold
    : public impl::TTIRImplicitBroadcastFoldBase<TTIRImplicitBroadcastFold> {
public:
  using impl::TTIRImplicitBroadcastFoldBase<
      TTIRImplicitBroadcastFold>::TTIRImplicitBroadcastFoldBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRImplicitBroadcastFoldRewriter, FoldBroadcastIntoDotGeneral>(
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
