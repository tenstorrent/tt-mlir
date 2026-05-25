// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir {

namespace {

// Commute a PermuteOp through a BatchNormInferenceOp by remapping the BN's
// feature `dimension` attribute through the permutation.
//
// BN normalizes along all dims except `dimension`, with scale/offset/mean/
// variance aligned to that axis. Permuting the operand merely shuffles axes —
// it doesn't change the math — so a permute can move past BN if we update
// `dimension` to point at the same logical axis in the new layout and apply
// the same permute to any per-feature parameters that share the operand's
// rank. (Rank-1 parameters index the feature axis directly and don't need
// touching; rank-N parameters broadcast against the operand and must follow
// it.) Parameter permutes land on const-eval'd paths, so the extra TMs are
// folded away at compile time.
//
// Upwards:
//   %y = batch_norm_inference(%x, %scale, %offset, %mean, %var, dim = D)
//   %z = permute(%y, P)
//   ⟹
//   %x' = permute(%x, P)
//   %scale', %offset', %mean', %var' = permute(...)  // only if rank-N
//   %z = batch_norm_inference(%x', %scale', %offset', %mean', %var', dim = D')
//   where D' = inverse(P)[D] = index in P at which D appears.
//
// Downwards:
//   %x' = permute(%x, P)
//   %y  = batch_norm_inference(%x', %scale, %offset, %mean, %var, dim = D)
//   ⟹
//   %scale', ... = inverse_permute(...)  // only if rank-N
//   %y_pre = batch_norm_inference(%x, %scale', ..., dim = D')
//   %y     = permute(%y_pre, P)
//   where D' = P[D].
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughBatchNormInference
    : public TTIRCommuteOpRewritePattern<PermuteOp, BatchNormInferenceOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, BatchNormInferenceOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  // Permute `param` with `permutation` if its rank matches the operand's
  // (e.g. broadcast-shaped `[1, C, 1, 1]`); leave rank-1 vectors alone (they
  // index the feature axis directly).
  static Value maybePermuteParam(Value param, int64_t operandRank,
                                 ArrayRef<int64_t> permutation, Location loc,
                                 PatternRewriter &rewriter) {
    auto type = cast<RankedTensorType>(param.getType());
    if (type.getRank() == 1) {
      return param;
    }
    assert(type.getRank() == operandRank &&
           "BN parameter must be rank-1 or share rank with the operand");
    auto newShape =
        ttmlir::utils::applyPermutation(type.getShape(), permutation);
    auto newType = RankedTensorType::get(newShape, type.getElementType(),
                                         type.getEncoding());
    return rewriter.create<PermuteOp>(loc, newType, param, permutation)
        .getResult();
  }

  void
  performCommuteUpwardsRewrite(BatchNormInferenceOp op, PermuteOp permuteUser,
                               PatternRewriter &rewriter) const override {
    auto permutation = permuteUser.getPermutation();
    auto inversePerm = ttmlir::utils::inversePermutation(permutation);

    auto operandType = cast<RankedTensorType>(op.getOperand().getType());
    int64_t operandRank = operandType.getRank();

    // Move the permute onto the BN's operand.
    auto newOperandShape =
        ttmlir::utils::applyPermutation(operandType.getShape(), permutation);
    auto newOperandType = RankedTensorType::get(newOperandShape,
                                                operandType.getElementType(),
                                                operandType.getEncoding());
    auto newPermute = rewriter.create<PermuteOp>(
        permuteUser->getLoc(), newOperandType, op.getOperand(), permutation);

    // Permute any rank-N per-feature parameters along with the operand.
    Value newScale = maybePermuteParam(op.getScale(), operandRank, permutation,
                                       permuteUser->getLoc(), rewriter);
    Value newOffset = maybePermuteParam(op.getOffset(), operandRank,
                                        permutation, permuteUser->getLoc(),
                                        rewriter);
    Value newMean = maybePermuteParam(op.getMean(), operandRank, permutation,
                                      permuteUser->getLoc(), rewriter);
    Value newVariance = maybePermuteParam(op.getVariance(), operandRank,
                                          permutation, permuteUser->getLoc(),
                                          rewriter);

    // Remap the feature dimension through the inverse permutation: the value
    // formerly at position D is now at position inversePerm[D].
    int64_t newDim =
        inversePerm[static_cast<size_t>(op.getDimension())];
    auto newDimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(newDim));

    auto newBnType = cast<RankedTensorType>(permuteUser.getResult().getType());
    auto newBn = rewriter.create<BatchNormInferenceOp>(
        op->getLoc(), newBnType, newPermute.getResult(), newScale, newOffset,
        newMean, newVariance, op.getEpsilonAttr(), newDimAttr);

    // All users of the BN are identical permutes (guaranteed by viability /
    // favorability checks). Replace each with the new BN.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(permuteUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");
    for (auto *user : users) {
      rewriter.replaceOp(user, newBn.getResult());
    }
  }

  void
  performCommuteDownwardsRewrite(BatchNormInferenceOp op,
                                 PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    auto permutation = permuteOperand.getPermutation();
    auto inversePerm = ttmlir::utils::inversePermutation(permutation);

    auto preBnInput = permuteOperand.getInput();
    auto preBnType = cast<RankedTensorType>(preBnInput.getType());
    int64_t operandRank = preBnType.getRank();

    // Strip the permute from per-feature parameters (rank-N) so their layout
    // matches the un-permuted input.
    Value newScale = maybePermuteParam(op.getScale(), operandRank, inversePerm,
                                       permuteOperand->getLoc(), rewriter);
    Value newOffset = maybePermuteParam(op.getOffset(), operandRank,
                                        inversePerm, permuteOperand->getLoc(),
                                        rewriter);
    Value newMean = maybePermuteParam(op.getMean(), operandRank, inversePerm,
                                      permuteOperand->getLoc(), rewriter);
    Value newVariance = maybePermuteParam(op.getVariance(), operandRank,
                                          inversePerm,
                                          permuteOperand->getLoc(), rewriter);

    // Remap the feature dimension by following the permutation: the position D
    // in the post-permute world corresponds to perm[D] in the pre-permute one.
    int64_t newDim = permutation[static_cast<size_t>(op.getDimension())];
    auto newDimAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(newDim));

    auto newBn = rewriter.create<BatchNormInferenceOp>(
        op->getLoc(), preBnType, preBnInput, newScale, newOffset, newMean,
        newVariance, op.getEpsilonAttr(), newDimAttr);

    auto newPermute = rewriter.create<PermuteOp>(
        permuteOperand->getLoc(), op.getResult().getType(), newBn.getResult(),
        permutation);

    rewriter.replaceOp(op, newPermute.getResult());
  }

private:
  // We accept either rank-1 parameters or rank-N broadcast-shaped parameters
  // that share the operand's rank.
  static bool paramsAreCommutable(BatchNormInferenceOp op) {
    int64_t operandRank =
        cast<RankedTensorType>(op.getOperand().getType()).getRank();
    auto ok = [operandRank](Value v) {
      int64_t r = cast<RankedTensorType>(v.getType()).getRank();
      return r == 1 || r == operandRank;
    };
    return ok(op.getScale()) && ok(op.getOffset()) && ok(op.getMean()) &&
           ok(op.getVariance());
  }

  bool isCommuteUpwardsViable(BatchNormInferenceOp op,
                              PermuteOp) const override {
    return paramsAreCommutable(op);
  }

  bool isCommuteUpwardsFavorable(BatchNormInferenceOp op,
                                 PermuteOp) const override {
    // Commute upwards when all BN users are the same TM — that's the only way
    // moving the permute up reduces TM count instead of duplicating it.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(BatchNormInferenceOp op,
                                PermuteOp) const override {
    return paramsAreCommutable(op);
  }

  bool isCommuteDownwardsFavorable(BatchNormInferenceOp,
                                   PermuteOp) const override {
    // BN has a single normalizable operand; if that's where the permute sits
    // and the BN has one use, pushing the permute below it is TM-count
    // neutral and brings the permute closer to downstream inverses.
    return true;
  }
};

} // namespace

template <CommuteDirection commuteDirection>
void populateBatchNormCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns,
                                      ConstevalForwardAnalysis *analysis) {
  patterns.add<TTIRCommutePermuteThroughBatchNormInference<commuteDirection>>(
      ctx, analysis);
}

template void populateBatchNormCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns,
    ConstevalForwardAnalysis *analysis);
template void populateBatchNormCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns,
    ConstevalForwardAnalysis *analysis);

} // namespace mlir::tt::ttir
