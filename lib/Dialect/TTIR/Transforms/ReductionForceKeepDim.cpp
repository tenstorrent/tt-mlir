// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRREDUCTIONFORCEKEEPDIM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// For any reduction with keep_dim=false, set keep_dim=true and insert a reshape
// after to squeeze the reduced dimensions back out, preserving the original
// output shape.
template <typename ReductionOpTy>
class ForceKeepDimPattern : public mlir::OpRewritePattern<ReductionOpTy> {
  using mlir::OpRewritePattern<ReductionOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReductionOpTy reductionOp,
                  mlir::PatternRewriter &rewriter) const final {
    if (reductionOp.getKeepDim()) {
      return mlir::failure();
    }

    ArrayRef<int64_t> inputShape = reductionOp.getInput().getType().getShape();

    // Compute the intermediate output shape for keep_dim=true: start from the
    // input shape and set each reduced dimension to 1 (instead of removing it).
    // E.g. input <2x3x4>, reduce dim 1 → keep_dim shape <2x1x4>.
    llvm::SmallVector<int64_t> keepDimShape(inputShape);

    if (!reductionOp.getDimArg()) {
      // No dim_arg means reduce over all dimensions (per TTIR_ReductionOp
      // semantics), so every dimension becomes 1.
      keepDimShape.assign(inputShape.size(), 1);
    } else {
      mlir::ArrayAttr reduceDims = *reductionOp.getDimArg();
      for (mlir::Attribute reduceDim : reduceDims) {
        int64_t reduceDimInt =
            mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
        reduceDimInt = (reduceDimInt + inputShape.size()) % inputShape.size();
        keepDimShape[reduceDimInt] = 1;
      }
    }

    auto originalType = reductionOp.getResult().getType();
    auto decomposedType =
        RankedTensorType::get(keepDimShape, originalType.getElementType(),
                              originalType.getEncoding());

    auto newReduction = rewriter.create<ReductionOpTy>(
        reductionOp.getLoc(), decomposedType, reductionOp.getInput(),
        /*keep_dim=*/rewriter.getBoolAttr(true), reductionOp.getDimArgAttr());

    llvm::SmallVector<int32_t> outputShapeI32(originalType.getShape().begin(),
                                              originalType.getShape().end());
    rewriter.replaceOpWithNewOp<ReshapeOp>(
        reductionOp, originalType, newReduction.getResult(),
        rewriter.getI32ArrayAttr(outputShapeI32));

    return mlir::success();
  }
};

class TTIRReductionForceKeepDim
    : public impl::TTIRReductionForceKeepDimBase<TTIRReductionForceKeepDim> {
public:
  using impl::TTIRReductionForceKeepDimBase<
      TTIRReductionForceKeepDim>::TTIRReductionForceKeepDimBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    patterns
        .add<ForceKeepDimPattern<SumOp>, ForceKeepDimPattern<MeanOp>,
             ForceKeepDimPattern<MaxOp>, ForceKeepDimPattern<MinOp>,
             ForceKeepDimPattern<ProdOp>, ForceKeepDimPattern<ReduceAndOp>,
             ForceKeepDimPattern<ReduceOrOp>, ForceKeepDimPattern<ArgMaxOp>>(
            &getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
