// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRREDUCTIONKEEPDIM
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// This pattern detects when a reduction operation is followed by a reshape
// operation that simply adds back dimensions that were reduced. In such cases,
// we can fuse the reshape into the reduction operation by setting
// keep_dim=true.
template <typename ReductionOpTy>
class ReductionFollowedByReshapePattern
    : public mlir::OpRewritePattern<ReductionOpTy> {
  using mlir::OpRewritePattern<ReductionOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReductionOpTy reductionOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if the reduction op has exactly one use and it's a reshape op.
    if (!isFusable(reductionOp)) {
      return mlir::failure();
    }

    // Get the reshape op that follows the reduction op.
    auto reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Replace old reduction with new reduction with keep_dim=true.
    rewriter.replaceOpWithNewOp<ReductionOpTy>(
        reductionOp, reshapeOp.getResult().getType(), reductionOp.getInput(),
        /*keep_dim=*/rewriter.getBoolAttr(true), reductionOp.getDimArgAttr());

    // Reshape will be folded into the new reduction op.
    return mlir::success();
  }

private:
  bool isFusable(ReductionOpTy reductionOp) const {
    // Reduction should only have one use and that use should be a reshape op.
    if (!reductionOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsersOfType<ReshapeOp>(reductionOp)) {
      return false;
    }

    // If keep dim is already set, we cannot fuse.
    if (reductionOp.getKeepDim()) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Get the input and output shapes.
    ArrayRef<int64_t> inputShape = reductionOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();

    // Reshape output shape should have the same rank as the input shape.
    if (reshapeOutputShape.size() != inputShape.size()) {
      return false;
    }

    // Calculate the expected shape after reduction with keep_dim=true.
    llvm::SmallVector<int64_t> expectedShape(inputShape);

    if (!reductionOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced to 1.
      expectedShape = llvm::SmallVector<int64_t>(inputShape.size(), 1);
    } else {
      // Only specified dimensions are reduced to 1.
      mlir::ArrayAttr reduceDims = *reductionOp.getDimArg();
      for (mlir::Attribute reduceDim : reduceDims) {
        int64_t reduceDimInt =
            mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
        // Handle negative indices.
        reduceDimInt = (reduceDimInt + inputShape.size()) % inputShape.size();
        expectedShape[reduceDimInt] = 1;
      }
    }

    // Check if the reshape output matches the expected shape.
    return expectedShape == reshapeOutputShape;
  }
};

// This pattern detects when a reshape operation that adds dimensions of size 1
// is followed by a reduction with keep_dim=false. In such cases, we can
// simplify by removing the reshape and using keep_dim=true with adjusted
// dimension arguments.
//
// Example:
//   %reshaped = reshape(%input) : tensor<2x3x4> -> tensor<2x1x3x4>
//   %reduced = sum(%reshaped, dim=2, keep_dim=false) : tensor<2x1x3x4> ->
//   tensor<2x1x4>
// Becomes:
//   %reduced = sum(%input, dim=1, keep_dim=true) : tensor<2x3x4> ->
//   tensor<2x1x4>
template <typename ReductionOpTy>
class ReshapeBeforeReductionPattern
    : public mlir::OpRewritePattern<ReductionOpTy> {
  using mlir::OpRewritePattern<ReductionOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReductionOpTy reductionOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if the input comes from a reshape op.
    auto reshapeOp = reductionOp.getInput().template getDefiningOp<ReshapeOp>();
    if (!reshapeOp) {
      return mlir::failure();
    }

    // If keep_dim is already true, this pattern doesn't apply.
    if (reductionOp.getKeepDim()) {
      return mlir::failure();
    }

    // Must have dim_arg specified for this pattern.
    if (!reductionOp.getDimArg()) {
      return mlir::failure();
    }

    ArrayRef<int64_t> originalShape = reshapeOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapedShape =
        reshapeOp.getResult().getType().getShape();

    // Check if reshape only adds dimensions of size 1.
    // Build a mapping from reshaped dimensions to original dimensions.
    llvm::SmallVector<int64_t> reshapedToOriginal;
    size_t origIdx = 0;

    for (size_t reshapedIdx = 0; reshapedIdx < reshapedShape.size();
         ++reshapedIdx) {
      if (reshapedShape[reshapedIdx] == 1 &&
          (origIdx >= originalShape.size() || originalShape[origIdx] != 1)) {
        // This is an inserted dimension of size 1.
        reshapedToOriginal.push_back(-1); // -1 indicates inserted dim.
      } else {
        // This dimension maps to an original dimension.
        if (origIdx >= originalShape.size() ||
            reshapedShape[reshapedIdx] != originalShape[origIdx]) {
          // Reshape is not just inserting 1s, bail out.
          return mlir::failure();
        }
        reshapedToOriginal.push_back(origIdx);
        ++origIdx;
      }
    }

    // All original dimensions should be consumed.
    if (origIdx != originalShape.size()) {
      return mlir::failure();
    }

    // Now check if all reduced dimensions map to original dimensions.
    // Also verify the output shape matches what we expect.
    mlir::ArrayAttr reduceDims = *reductionOp.getDimArg();
    llvm::SmallVector<int64_t> newReduceDims;

    for (mlir::Attribute reduceDim : reduceDims) {
      int64_t reduceDimInt = mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
      // Handle negative indices.
      reduceDimInt =
          (reduceDimInt + reshapedShape.size()) % reshapedShape.size();

      int64_t origDim = reshapedToOriginal[reduceDimInt];
      if (origDim == -1) {
        // Reducing an inserted dimension of size 1 - this is a no-op for the
        // reduction. We need the output shape to still contain this 1.
        // This case is complex, bail out for now.
        return mlir::failure();
      }
      newReduceDims.push_back(origDim);
    }

    // Calculate expected output shape for reduction on original with
    // keep_dim=true.
    llvm::SmallVector<int64_t> expectedOutputShape(originalShape);
    for (int64_t dim : newReduceDims) {
      expectedOutputShape[dim] = 1;
    }

    // Check if this matches the actual reduction output shape.
    ArrayRef<int64_t> actualOutputShape =
        reductionOp.getResult().getType().getShape();
    if (expectedOutputShape != actualOutputShape) {
      return mlir::failure();
    }

    // Create new dim_arg attribute.
    llvm::SmallVector<mlir::Attribute> newDimAttrs;
    for (int64_t dim : newReduceDims) {
      newDimAttrs.push_back(rewriter.getI32IntegerAttr(dim));
    }
    mlir::ArrayAttr newDimArg = rewriter.getArrayAttr(newDimAttrs);

    // Replace with reduction on original input with keep_dim=true.
    rewriter.replaceOpWithNewOp<ReductionOpTy>(
        reductionOp, reductionOp.getResult().getType(), reshapeOp.getInput(),
        /*keep_dim=*/rewriter.getBoolAttr(true), newDimArg);

    return mlir::success();
  }
};

class TTIRReductionKeepDim
    : public impl::TTIRReductionKeepDimBase<TTIRReductionKeepDim> {
public:
  using impl::TTIRReductionKeepDimBase<
      TTIRReductionKeepDim>::TTIRReductionKeepDimBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());

    // Pattern 1: Reduction followed by reshape that adds back reduced dims.
    patterns.add<ReductionFollowedByReshapePattern<SumOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<MeanOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<MaxOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<MinOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<ProdOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<ReduceAndOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<ReduceOrOp>>(&getContext());
    patterns.add<ReductionFollowedByReshapePattern<ArgMaxOp>>(&getContext());

    // Pattern 2: Reshape (adding dims) followed by reduction with
    // keep_dim=false.
    patterns.add<ReshapeBeforeReductionPattern<SumOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<MeanOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<MaxOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<MinOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<ProdOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<ReduceAndOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<ReduceOrOp>>(&getContext());
    patterns.add<ReshapeBeforeReductionPattern<ArgMaxOp>>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

} // namespace
} // namespace mlir::tt::ttir
