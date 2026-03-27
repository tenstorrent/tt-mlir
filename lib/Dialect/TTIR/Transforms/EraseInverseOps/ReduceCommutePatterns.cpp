// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {

namespace {
template <typename ReduceOpType, CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughReduce
    : public TTIRCommuteOpRewritePattern<PermuteOp, ReduceOpType,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, ReduceOpType, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Commute upwards: reduce(D, kd) → permute(P) ⟹ permute(P') → reduce(D', kd)
  // Example:
  // %0 = reduce(%arg0) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = reduce(%0) {keep_dim = true, dim_arg = [1: i32, 2: i32]}
  //
  // When keep_dim=true, P' = P and D' = P_inv(D) (rank is preserved).
  // When keep_dim=false, P operates on a lower rank than the reduce input.
  // We expand P back to full rank by keeping reduced dims in place, then
  // apply the same inverse-permutation logic to transform D.
  void performCommuteUpwardsRewrite(ReduceOpType op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    auto userPerm = permuteUser.getPermutation();

    auto inputPerm = op.getKeepDim()
                         ? SmallVector<int64_t>(userPerm)
                         : expandPermutation(userPerm, getReduceDimValues(op),
                                             op.getInput().getType().getRank());

    PermuteOp newPerm = createNewPermuteOp(op.getInput(), inputPerm, rewriter,
                                           permuteUser->getLoc());

    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, newPerm.getResult(), inputPerm, rewriter,
        /*inverseDimPermute=*/true);

    // All users must be identical TMs. We must not reference `permuteUser`
    // during/after replacements, as it will be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(permuteUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    for (auto *user : users) {
      rewriter.replaceOp(user, newReduce);
    }
  }

  // Commute downwards: permute(P) → reduce(D, kd) ⟹ reduce(D', kd) →
  // permute(P')
  // Example:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reduce(%0) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  //
  // This method will transform this into:
  // %0 = reduce(%arg0) {keep_dim = true, dim_arg = [1: i32, 2: i32]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 3, 1, 2>}>
  //
  // D' = P(D) in both cases.
  // When keep_dim=true, P' = P (rank is preserved).
  // When keep_dim=false, P' is the contraction of P: remove the reduced
  // positions and renumber the surviving dims contiguously.
  void
  performCommuteDownwardsRewrite(ReduceOpType op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    auto permutation = permuteOperand.getPermutation();

    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, permuteOperand.getInput(), permutation, rewriter);

    auto outputPerm =
        op.getKeepDim()
            ? SmallVector<int64_t>(permutation)
            : contractPermutation(permutation, getReduceDimValues(op));

    PermuteOp newPerm = createNewPermuteOp(newReduce, outputPerm, rewriter,
                                           permuteOperand->getLoc());
    rewriter.replaceOp(op, newPerm);
  }

private:
  PermuteOp createNewPermuteOp(Value input, ArrayRef<int64_t> permutation,
                               PatternRewriter &rewriter, Location loc) const {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto outputShape = ttmlir::utils::applyPermutation(inputShape, permutation);

    auto outputType = RankedTensorType::get(
        outputShape, inputType.getElementType(), inputType.getEncoding());

    return rewriter.create<PermuteOp>(loc, outputType, input, permutation);
  }

  ArrayAttr permuteDims(std::optional<ArrayAttr> dimArg,
                        ArrayRef<int64_t> permutation,
                        PatternRewriter &rewriter) const {
    int64_t rank = permutation.size();
    ArrayAttr dims = dimArg.value_or(rewriter.getI32ArrayAttr(
        llvm::to_vector(llvm::seq<int32_t>(0, static_cast<int32_t>(rank)))));
    auto permutedDims = llvm::to_vector(
        llvm::map_range(dims.getValue(), [&](Attribute attr) -> Attribute {
          int64_t d = cast<IntegerAttr>(attr).getInt();
          d = d < 0 ? d + rank : d;
          return rewriter.getI32IntegerAttr(permutation[d]);
        }));
    return ArrayAttr::get(rewriter.getContext(), permutedDims);
  }

  // Extract the reduce dimensions as normalized (non-negative) int64 values.
  SmallVector<int64_t> getReduceDimValues(ReduceOpType op) const {
    int64_t rank = op.getInput().getType().getRank();
    if (!op.getDimArg()) {
      return llvm::to_vector(llvm::seq<int64_t>(0, rank));
    }
    return llvm::to_vector(llvm::map_range(
        op.getDimArg()->getValue(), [rank](Attribute attr) -> int64_t {
          int64_t d = cast<IntegerAttr>(attr).getInt();
          return d < 0 ? d + rank : d;
        }));
  }

  // Contract a rank-N permutation to rank-(N-|removed|) by dropping the given
  // positions and renumbering the surviving values contiguously.
  //
  // Example: perm=(0,2,3,1), remove positions {1}
  //   surviving positions {0,2,3} with values {0,3,1}
  //   removed values {2}, renumber: 0→0, 1→1, 3→2 → result (0,2,1)
  SmallVector<int64_t>
  contractPermutation(ArrayRef<int64_t> perm,
                      ArrayRef<int64_t> removedPositions) const {
    int64_t n = perm.size();
    llvm::SmallDenseSet<int64_t> removedPosSet(removedPositions.begin(),
                                               removedPositions.end());
    llvm::SmallDenseSet<int64_t> removedValSet;
    for (int64_t pos : removedPositions) {
      removedValSet.insert(perm[pos]);
    }

    SmallVector<int64_t> renumber(n, -1);
    int64_t idx = 0;
    for (int64_t v = 0; v < n; v++) {
      if (!removedValSet.count(v)) {
        renumber[v] = idx++;
      }
    }

    SmallVector<int64_t> result;
    for (auto [i, val] : llvm::enumerate(perm)) {
      if (!removedPosSet.count(i)) {
        result.push_back(renumber[val]);
      }
    }
    return result;
  }

  // Expand a rank-M permutation to rank-N by inserting identity mappings for
  // the reduced dims (which stay in place).
  //
  // Example: smallPerm=(0,2,1), reducedDims={2}, fullRank=4
  //   surviving dims {0,1,3}
  //   result: P[0]=0, P[1]=3, P[2]=2, P[3]=1 → (0,3,2,1)
  SmallVector<int64_t> expandPermutation(ArrayRef<int64_t> smallPerm,
                                         ArrayRef<int64_t> reducedDims,
                                         int64_t fullRank) const {
    llvm::SmallDenseSet<int64_t> reducedSet(reducedDims.begin(),
                                            reducedDims.end());
    auto surviving = llvm::to_vector(llvm::make_filter_range(
        llvm::seq<int64_t>(0, fullRank),
        [&](int64_t i) { return !reducedSet.count(i); }));

    SmallVector<int64_t> result(fullRank);
    for (int64_t d : reducedDims) {
      result[d] = d;
    }
    for (auto [j, survPos] : llvm::enumerate(surviving)) {
      result[survPos] = surviving[smallPerm[j]];
    }
    return result;
  }

  // Create a new reduce op with permuted dimensions. The output shape is
  // derived from the new input shape and permuted reduce dims, handling both
  // keep_dim=true and keep_dim=false uniformly.
  ReduceOpType createNewReduceOpWithPermutedDims(
      ReduceOpType op, Value newInput, ArrayRef<int64_t> permutation,
      PatternRewriter &rewriter, bool inverseDimPermute = false) const {

    auto inversePermutation = ttmlir::utils::inversePermutation(permutation);
    auto dimPermutation = inverseDimPermute ? inversePermutation : permutation;

    ArrayAttr newDimArgAttrs =
        permuteDims(op.getDimArg(), dimPermutation, rewriter);

    auto inputShape = cast<RankedTensorType>(newInput.getType()).getShape();
    llvm::SmallDenseSet<int64_t> reducedDimSet;
    for (Attribute attr : newDimArgAttrs.getValue()) {
      reducedDimSet.insert(cast<IntegerAttr>(attr).getInt());
    }

    SmallVector<int64_t> newReduceShape;
    for (auto [i, dim] : llvm::enumerate(inputShape)) {
      if (!reducedDimSet.count(i)) {
        newReduceShape.push_back(dim);
      } else if (op.getKeepDim()) {
        newReduceShape.push_back(1);
      }
    }

    auto newReduceType =
        RankedTensorType::get(newReduceShape, op.getType().getElementType(),
                              op.getType().getEncoding());

    return rewriter.create<ReduceOpType>(op->getLoc(), newReduceType, newInput,
                                         op.getKeepDimAttr(), newDimArgAttrs);
  }

  bool isCommuteUpwardsViable(ReduceOpType op, PermuteOp) const override {
    return true;
  }

  bool isCommuteUpwardsFavorable(ReduceOpType op, PermuteOp) const override {
    // We should always commute a permute above a reduce if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ReduceOpType op, PermuteOp) const override {
    return true;
  }

  bool isCommuteDownwardsFavorable(ReduceOpType op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    for (auto operandValue : op->getOperands()) {
      if (checkIdenticalTms(operandValue.getDefiningOp(), permuteOperand) ||
          ttcore::valueTracesToConstantArgs(operandValue)) {
        continue;
      }
      return false;
    }
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateReduceCommutePatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns) {
  patterns.add<TTIRCommutePermuteThroughReduce<SumOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<MeanOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<MaxOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<MinOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<ProdOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<ReduceAndOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<ReduceOrOp, commuteDirection>,
               TTIRCommutePermuteThroughReduce<ArgMaxOp, commuteDirection>>(
      ctx);
}

template void populateReduceCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateReduceCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
