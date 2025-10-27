// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include <flatbuffers/array.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinAttributes.h>

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
  // %0 = reduce(%arg0, %output) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  // %1 = permute(%0, %output1) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0, %output0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reduce(%0, %output) {keep_dim = true, dim_arg = [1: i32, 2: i32]}>
  //
  // The reduce dimensions are transformed by applying the inverse permutation
  // Input and output shapes of reduce are permuted since permute op now come
  // before reduce

  void performCommuteUpwardsRewrite(ReduceOpType op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> permutation = permuteUser.getPermutation();

    PermuteOp newPerm =
        createNewPermuteOp(op.getInput(), permutation, rewriter, op->getLoc());
    llvm::outs() << "Creating new reduce op with upward commute:\n";

    llvm::outs() << "  Original reduce args: ";
    for (Attribute dimAttr : op.getDimArg()->getValue()) {
      llvm::outs() << cast<IntegerAttr>(dimAttr).getInt() << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "  Original permutation: ";
    for (int64_t dim : permutation) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, newPerm->getResult(0), permutation, rewriter,
        /*inverseDimPermute=*/true);

    rewriter.replaceAllOpUsesWith(permuteUser, newReduce);
  }
  // Consider the following IR pseudocode:
  // %0 = permute(%arg0, %output0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reduce(%0, %output) {keep_dim = true, dim_arg = [2: i32, 3: i32]}
  //
  // This method will transform this into:
  // %0 = reduce(%arg0, %output) {keep_dim = true, dim_arg = [1: i32, 2: i32]}
  // %1 = permute(%0, %output1) <{permutation = array<i64: 0, 3, 1, 2>}>
  //
  // The reduce dimensions are transformed by applying the permutation
  // Input and output shapes of reduce are permuted with the inverse permutation
  // since we moved the permute that was before reduce
  void
  performCommuteDownwardsRewrite(ReduceOpType op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> permutation = permuteOperand.getPermutation();
    llvm::outs() << "Creating new reduce op with downward commute\n";
    llvm::outs() << "  Original permutation: ";
    for (int64_t dim : permutation) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "  Original reduce args: ";
    for (Attribute dimAttr : op.getDimArg()->getValue()) {
      llvm::outs() << cast<IntegerAttr>(dimAttr).getInt() << " ";
    }
    llvm::outs() << "\n";
    ReduceOpType newReduce = createNewReduceOpWithPermutedDims(
        op, permuteOperand.getInput(), permutation, rewriter);

    PermuteOp newPerm =
        createNewPermuteOp(newReduce, permutation, rewriter, op->getLoc());

    rewriter.replaceAllOpUsesWith(op, newPerm);
  }

private:
  PermuteOp createNewPermuteOp(Value input, ArrayRef<int64_t> permutation,
                               PatternRewriter &rewriter, Location loc) const {
    // Apply permutation on input type to create output type
    auto inputType = cast<RankedTensorType>(input.getType());
    SmallVector<int64_t> inputShape(inputType.getShape());
    SmallVector<int64_t> outputShape = ttmlir::utils::applyPermutation(
        ArrayRef<int64_t>(inputShape), permutation);

    RankedTensorType outputType = RankedTensorType::get(
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
      int64_t permutedDim = permutation[dim];
      permutedDims.push_back(rewriter.getI32IntegerAttr(permutedDim));
    }
    llvm::outs() << "Permuted dims: ";
    for (auto attr : permutedDims) {
      llvm::outs() << cast<IntegerAttr>(attr).getInt() << " ";
    }
    llvm::outs() << "\n";

    return ArrayAttr::get(dimArg.getContext(), permutedDims);
  }

  ReduceOpType createNewReduceOpWithPermutedDims(
      ReduceOpType op, Value newInput, ArrayRef<int64_t> permutation,
      PatternRewriter &rewriter, bool inverseDimPermute = false) const {

    llvm::SmallVector<int64_t> inversePermutation =
        ttmlir::utils::inversePermutation(permutation);

    auto dimPermutation = inverseDimPermute ? inversePermutation : permutation;
    auto shapePermutation =
        inverseDimPermute ? permutation : inversePermutation;

    ArrayAttr newDimArgAttrs =
        permuteDims(*op.getDimArg(), dimPermutation, rewriter);
    SmallVector<int64_t> shapeVec(op.getType().getShape());
    SmallVector<int64_t> newReduceShape = ttmlir::utils::applyPermutation(
        ArrayRef<int64_t>(shapeVec), shapePermutation);
    RankedTensorType newReduceType =
        RankedTensorType::get(newReduceShape, op.getType().getElementType(),
                              op.getType().getEncoding());

    return utils::createDPSOp<ReduceOpType>(
        rewriter, op->getLoc(), newReduceType, newInput, op.getKeepDimAttr(),
        newDimArgAttrs);
  }

  bool isCommuteUpwardsViable(ReduceOpType op, PermuteOp) const override {
    // Commuting a permute through a reduce is always viable
    return true;
  }

  bool isCommuteUpwardsFavorable(ReduceOpType op, PermuteOp) const override {
    // We should always commute a permute above a reduce if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ReduceOpType op, PermuteOp) const override {
    // Commuting a permute through a reduce is always viable
    return true;
  }

  bool isCommuteDownwardsFavorable(ReduceOpType op,
                                   PermuteOp permuteOperand) const override {
    // Commuting downwards is favorable if all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path

    for (uint32_t i = 0; i < op->getNumOperands(); i++) {
      if (op.isDpsInit(&op->getOpOperand(i))) {
        continue;
      }
      if (checkIdenticalTms(op->getOperand(i).getDefiningOp(),
                            permuteOperand) ||
          ttcore::valueTracesToConstantArgs(op->getOperand(i))) {
        continue;
      }
      return false;
    }
    return true;
  }
};
} // namespace

template <typename ReduceOpType, CommuteDirection commuteDirection>
void populateReduceCommutePatterns(MLIRContext *ctx,
                                   RewritePatternSet &patterns) {
  patterns
      .insert<TTIRCommutePermuteThroughReduce<ReduceOpType, commuteDirection>>(
          ctx);
}

// Explicit instantiations for SumOp and MeanOp
template void populateReduceCommutePatterns<SumOp, CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateReduceCommutePatterns<SumOp, CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateReduceCommutePatterns<MeanOp, CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void
populateReduceCommutePatterns<MeanOp, CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
