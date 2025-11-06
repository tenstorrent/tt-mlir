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

  // Consider the following IR pseudocode:
  // %0 = reduce(%arg0) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = reduce(%0) {keep_dim = true, dim_arg = [1: i32, 2: i32]}
  //
  // The reduce dimensions are transformed by applying the inverse permutation
  // Input and output shapes of reduce are permuted since permute op now come
  // before reduce

  void performCommuteUpwardsRewrite(ReduceOpType op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    auto permutation = permuteUser.getPermutation();

    PermuteOp newPerm = createNewPermuteOp(op.getInput(), permutation, rewriter,
                                           permuteUser->getLoc());

    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, newPerm.getResult(), permutation, rewriter,
        /*inverseDimPermute=*/true);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "shouldCommute should have ensured this is true");
      rewriter.replaceOp(user, newReduce);
    }
  }
  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reduce(%0) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  //
  // This method will transform this into:
  // %0 = reduce(%arg0) {keep_dim = true, dim_arg = [1: i32, 2: i32]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 3, 1, 2>}>
  //
  // The reduce dimensions are transformed by applying the permutation
  // Input and output shapes of reduce are permuted with the inverse permutation
  // since we moved the permute that was before reduce
  void
  performCommuteDownwardsRewrite(ReduceOpType op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    auto permutation = permuteOperand.getPermutation();

    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, permuteOperand.getInput(), permutation, rewriter);

    PermuteOp newPerm = createNewPermuteOp(newReduce, permutation, rewriter,
                                           permuteOperand->getLoc());

    rewriter.replaceOp(op, newPerm);
  }

private:
  PermuteOp createNewPermuteOp(Value input, ArrayRef<int64_t> permutation,
                               PatternRewriter &rewriter, Location loc) const {
    // Apply permutation on input type to create output type
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    auto outputShape = ttmlir::utils::applyPermutation(inputShape, permutation);

    auto outputType = RankedTensorType::get(
        outputShape, inputType.getElementType(), inputType.getEncoding());

    // Create and return the new PermuteOp
    return ttir::utils::createDPSOp<PermuteOp>(rewriter, loc, outputType, input,
                                               permutation);
  }

  ArrayAttr permuteDims(ArrayAttr dimArg, ArrayRef<int64_t> permutation,
                        PatternRewriter &rewriter) const {
    // Apply permutation on each reduce dimension
    SmallVector<Attribute> permutedDims;
    for (Attribute dimAttr : dimArg.getValue()) {
      int64_t dim = cast<IntegerAttr>(dimAttr).getInt();
      if (dim < 0) {
        dim += permutation.size();
      }
      int64_t permutedDim = permutation[dim];
      permutedDims.push_back(rewriter.getI32IntegerAttr(permutedDim));
    }
    return ArrayAttr::get(dimArg.getContext(), permutedDims);
  }

  ReduceOpType createNewReduceOpWithPermutedDims(
      ReduceOpType op, Value newInput, ArrayRef<int64_t> permutation,
      PatternRewriter &rewriter, bool inverseDimPermute = false) const {

    auto inversePermutation = ttmlir::utils::inversePermutation(permutation);

    auto dimPermutation = inverseDimPermute ? inversePermutation : permutation;
    auto shapePermutation =
        inverseDimPermute ? permutation : inversePermutation;

    ArrayAttr newDimArgAttrs =
        permuteDims(*op.getDimArg(), dimPermutation, rewriter);
    auto oldReduceShape = op.getType().getShape();
    auto newReduceShape =
        ttmlir::utils::applyPermutation(oldReduceShape, shapePermutation);
    auto newReduceType =
        RankedTensorType::get(newReduceShape, op.getType().getElementType(),
                              op.getType().getEncoding());

    return utils::createDPSOp<ReduceOpType>(
        rewriter, op->getLoc(), newReduceType, newInput, op.getKeepDimAttr(),
        newDimArgAttrs);
  }

  bool isCommuteUpwardsViable(ReduceOpType op, PermuteOp) const override {
    // Commute when reduce has keepdim = false is not currently supported
    return op.getKeepDim();
  }

  bool isCommuteUpwardsFavorable(ReduceOpType op, PermuteOp) const override {
    // We should always commute a permute above a reduce if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ReduceOpType op, PermuteOp) const override {
    // Commute when reduce has keepdim = false is not currently supported
    return op.getKeepDim();
  }

  bool isCommuteDownwardsFavorable(ReduceOpType op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(op.getOperation());
    for (mlir::OpOperand *operand : dps.getDpsInputOperands()) {
      auto operandValue = operand->get();
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
