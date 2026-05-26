// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fold permute(NHWC→NCHW) → concat(dim=1) → permute(NCHW→NHWC) into
// concat(dim=3) with a single new permute on the non-NHWC input.
//
// Pattern (per FPN inner scale):
//   pre_perm  = permute(nhwc_input, [0,3,1,2])   // NHWC→NCHW
//   concat    = concat(pre_perm, backbone, dim=1)
//   post_perm = permute(concat, [0,2,3,1])        // NCHW→NHWC
//
// Becomes:
//   bb_nhwc   = permute(backbone, [0,2,3,1])      // NCHW→NHWC (new)
//   new_concat = concat(nhwc_input, bb_nhwc, dim=3)
//   (post_perm replaced by new_concat)
//
// Net effect: -2 permutes per matched site + 1 new backbone permute.
// For 4 FPN inner scales this is -8 + 4 = -4 permutes, eliminating all
// Group D (pre-concat) and the corresponding Group G (post-concat) permutes.
//
// Safety: the fold is NOT applied when the concat result has no permute
// consumer (e.g. the H=96 outermost FPN scale whose concat is a terminal
// NCHW output), which would add a net +1 permute.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFOLDNHWCCONCAT
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Match permute([0,3,1,2]) — NHWC→NCHW.
static bool isNHWCtoNCHW(ArrayRef<int64_t> perm) {
  return perm.size() == 4 && perm[0] == 0 && perm[1] == 3 && perm[2] == 1 &&
         perm[3] == 2;
}

// Match permute([0,2,3,1]) — NCHW→NHWC.
static bool isNCHWtoNHWC(ArrayRef<int64_t> perm) {
  return perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 3 &&
         perm[3] == 1;
}

// Fold permute(NHWC→NCHW) → concat(dim=1) → permute(NCHW→NHWC) into a
// single concat(dim=3) with the non-NHWC operands permuted to NHWC.
class FoldNHWCConcatPattern : public mlir::OpRewritePattern<ConcatOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(ConcatOp concatOp,
                                      mlir::PatternRewriter &rewriter) const final {
    // Require channel-dim concat (NCHW layout, dim=1).
    if (concatOp.getDim() != 1) {
      return failure();
    }

    // Require exactly one consumer of the concat result, and that consumer
    // must be a permute([0,2,3,1]) (NCHW→NHWC).
    if (!concatOp.getResult().hasOneUse()) {
      return failure();
    }
    auto postPermOp =
        mlir::dyn_cast<PermuteOp>(*concatOp.getResult().getUsers().begin());
    if (!postPermOp) {
      return failure();
    }
    if (!isNCHWtoNHWC(postPermOp.getPermutation())) {
      return failure();
    }

    // Scan concat inputs: collect NHWC→NCHW pre-permutes (single-use) and
    // plain (non-permuted) operands. We require at least one pre-permuted
    // operand; all others become the "backbone" operands.
    ValueRange inputs = concatOp.getInputs();
    llvm::SmallVector<Value> nhwcInputs;       // NHWC sources (bypass pre-perms)
    llvm::SmallVector<Value> ncHWInputs;       // non-permuted operands (backbone)
    llvm::SmallVector<size_t> nhwcIdxInConcat; // positions of permuted args
    llvm::SmallVector<size_t> ncHWIdxInConcat; // positions of plain args
    llvm::SmallVector<PermuteOp> prePermOps;   // pre-perm ops to erase later

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto prePermOp = inputs[i].getDefiningOp<PermuteOp>();
      if (prePermOp && prePermOp.getResult().hasOneUse() &&
          isNHWCtoNCHW(prePermOp.getPermutation())) {
        nhwcInputs.push_back(prePermOp.getInput());
        nhwcIdxInConcat.push_back(i);
        prePermOps.push_back(prePermOp);
      } else {
        ncHWInputs.push_back(inputs[i]);
        ncHWIdxInConcat.push_back(i);
      }
    }

    if (nhwcInputs.empty()) {
      return failure();
    }

    // Build the new operand list in the original order, replacing each
    // pre-permuted operand with its NHWC source, and inserting a new
    // NCHW→NHWC permute for each plain (backbone) operand.
    llvm::SmallVector<Value> newInputs(inputs.size());

    // Place NHWC inputs at their original positions.
    for (size_t k = 0; k < nhwcInputs.size(); ++k) {
      newInputs[nhwcIdxInConcat[k]] = nhwcInputs[k];
    }

    // Permute each backbone operand to NHWC and place at its original position.
    constexpr int64_t kNHWCPerm[] = {0, 2, 3, 1};
    for (size_t k = 0; k < ncHWInputs.size(); ++k) {
      Value bbVal = ncHWInputs[k];
      auto bbType = mlir::cast<RankedTensorType>(bbVal.getType());
      llvm::SmallVector<int64_t> nhwcShape =
          ttmlir::utils::applyPermutation(bbType.getShape(), kNHWCPerm);
      // Use RankedTensorTypeFactory only when the type already has a
      // TTNNLayoutAttr encoding (it would crash on unencoded tensors).
      RankedTensorType nhwcType;
      if (mlir::isa_and_nonnull<TTNNLayoutAttr>(bbType.getEncoding())) {
        nhwcType = utils::RankedTensorTypeFactory::create(bbType, nhwcShape);
      } else {
        nhwcType =
            RankedTensorType::get(nhwcShape, bbType.getElementType());
      }
      auto bbPerm = rewriter.create<PermuteOp>(
          concatOp.getLoc(), nhwcType, bbVal,
          rewriter.getDenseI64ArrayAttr(kNHWCPerm),
          /*memory_config=*/nullptr, /*pad_value=*/mlir::FloatAttr());
      newInputs[ncHWIdxInConcat[k]] = bbPerm.getResult();
    }

    // The new concat produces the same NHWC shape and element type as the
    // post-permute result. Reuse the post-permute result type directly.
    auto newConcatType =
        mlir::cast<RankedTensorType>(postPermOp.getResult().getType());

    auto newConcatOp =
        rewriter.create<ConcatOp>(concatOp.getLoc(), newConcatType, newInputs,
                                  static_cast<int32_t>(3),
                                  /*memory_config=*/MemoryConfigAttr());

    // Replace post-permute with the new concat; this drops all uses of the
    // original concat, making it safe to erase. Then erase the now-dead
    // pre-permutes.
    rewriter.replaceOp(postPermOp, newConcatOp.getResult());
    rewriter.eraseOp(concatOp);
    for (PermuteOp prePermOp : prePermOps) {
      rewriter.eraseOp(prePermOp);
    }

    return mlir::success();
  }
};

class TTNNFoldNHWCConcatPass
    : public impl::TTNNFoldNHWCConcatBase<TTNNFoldNHWCConcatPass> {
public:
  using impl::TTNNFoldNHWCConcatBase<
      TTNNFoldNHWCConcatPass>::TTNNFoldNHWCConcatBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldNHWCConcatPattern>(&getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::tt::ttnn
