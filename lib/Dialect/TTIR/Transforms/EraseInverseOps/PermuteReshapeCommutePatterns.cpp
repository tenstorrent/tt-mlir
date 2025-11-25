// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughReshape
    : public TTIRCommuteOpRewritePattern<PermuteOp, ReshapeOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, ReshapeOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // Consider the following IR pseudocode:
  // %0 = reshape(%arg0) {shape = [a, b, c, d]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 2, 3, 1>}>
  //
  // This method will transform this into:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
  // %1 = reshape(%0) {shape = [permuted_shape]}
  //
  // Note: Details about how reshape shape is transformed will be filled in
  // later

  void performCommuteUpwardsRewrite(ReshapeOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    // auto permutation = permuteUser.getPermutation();

    // Create new permute op on reshape input
    // PermuteOp newPerm = createNewPermuteOp(op.getInput(), permutation,
    // rewriter,
    //                                        permuteUser->getLoc());

    // Create new reshape op with transformed shape
    // ReshapeOp newReshape = createNewReshapeOpWithPermutedShape(
    //     op, newPerm.getResult(), permutation, rewriter);

    // All users must be identical TMs. We must not reference `permuteUser`
    // during/after replacements, as it will be erased on its turn.
    SmallVector<Operation *> users(op->getUsers());
    assert(llvm::all_of(users,
                        [&](Operation *user) {
                          return checkIdenticalTms(permuteUser, user);
                        }) &&
           "isCommuteUpwardsViable/Favorable should have ensured all users "
           "are identical TMs");

    // Replace users with new reshape
    // for (auto *user : users) {
    //   rewriter.replaceOp(user, newReshape);
    // }
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reshape(%0) {shape = [a, b, c, d]}
  //
  // This method will transform this into:
  // %0 = reshape(%arg0) {shape = [transformed_shape]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 3, 1, 2>}>
  //
  // The reshape shape is transformed by grouping axes and applying the inverse
  // permutation to determine the new shape.
  void
  performCommuteDownwardsRewrite(ReshapeOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    llvm::errs() << "performCommuteDownwardsRewrite\n";
    auto permutation = permuteOperand.getPermutation();

    // Get shapes: permuted input (after permute, which is reshape input), and
    // reshape output
    auto reshapeInputShape =
        mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
    auto reshapeOutputShape =
        mlir::cast<RankedTensorType>(op.getType()).getShape();

    // PermuteOp can't change the rank; check if the ReshapeOp changes the rank.
    if (reshapeInputShape.size() != reshapeOutputShape.size()) {
      llvm::errs() << "ReshapeOp input and output shapes have different ranks: "
                   << reshapeInputShape.size()
                   << " != " << reshapeOutputShape.size() << "\n";
      return;
    }
    const int64_t rank = reshapeInputShape.size();

    // Group the axes of the ReshapeOp into groups of consecutive axes
    auto axesGroups =
        groupAxes(reshapeInputShape, reshapeOutputShape, permutation);
    // If the axes cannot be grouped, the pattern is not applicable.
    if (!axesGroups) {
      llvm::errs() << "Axes cannot be grouped.\n";
      return;
    }

    // Debug: print the axes groups
    llvm::errs() << "Axes groups: ";
    for (const auto &group : *axesGroups) {
      llvm::errs() << "[";
      for (size_t i = 0; i < group.size(); ++i) {
        llvm::errs() << group[i];
        if (i < group.size() - 1) {
          llvm::errs() << ", ";
        }
      }
      llvm::errs() << "] ";
    }
    llvm::errs() << "\n";

    // Apply original permutation to the axes groups
    auto permutedAxesGroups = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(*axesGroups), permutation);

    // Check if the flattened axes groups are the same as the original axes
    if (!llvm::equal(ttmlir::utils::flatten(permutedAxesGroups),
                     llvm::seq(rank))) {
      llvm::errs()
          << "Flattened axes groups are not the same as the original axes.\n";
      llvm::errs() << "Flattened axes groups: ";
      for (const auto &group : permutedAxesGroups) {
        llvm::errs() << "[";
        for (size_t i = 0; i < group.size(); ++i) {
          llvm::errs() << group[i];
          if (i < group.size() - 1) {
            llvm::errs() << ", ";
          }
        }
        llvm::errs() << "] ";
      }
      llvm::errs() << "\n";
      return;
    }

    // Pattern is applicable; create new reshape op with transformed shape
    // The new reshape should produce the same output when followed by the
    // permute We need to compute the new reshape output shape by applying the
    // inverse permutation to the current reshape output shape
    auto inversePermutation = ttmlir::utils::inversePermutation(permutation);
    auto newReshapeOutputShape =
        ttmlir::utils::applyPermutation(reshapeOutputShape, inversePermutation);

    // Create the new reshape op
    auto newReshapeInputType =
        mlir::cast<RankedTensorType>(permuteOperand.getInput().getType());
    auto newReshapeOutputType = RankedTensorType::get(
        newReshapeOutputShape, newReshapeInputType.getElementType(),
        newReshapeInputType.getEncoding());

    llvm::SmallVector<int32_t> newReshapeShapeAttr(
        newReshapeOutputShape.begin(), newReshapeOutputShape.end());
    ReshapeOp newReshape = ttir::utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), newReshapeOutputType, permuteOperand.getInput(),
        rewriter.getI32ArrayAttr(newReshapeShapeAttr));

    // Create new permute op on reshape output
    PermuteOp newPerm = createNewPermuteOp(newReshape.getResult(), permutation,
                                           rewriter, permuteOperand->getLoc());

    // Replace op with new permute
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

  // Group the axes of the tensor into groups of consecutive axes that are
  // either:
  // - equal to the original axes;
  // - or a multiple of the consecutive original axes (possibly none).
  //
  // Returns the groups of axes (identified by the axes IDs), or std::nullopt if
  // the axes cannot be grouped.
  std::optional<llvm::SmallVector<llvm::SmallVector<int64_t>>>
  groupAxes(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape,
            ArrayRef<int64_t> axesIds) const {
    TT_assertv(inputShape.size() == outputShape.size(),
               "input and output shapes must have the same rank; "
               "inputShape={}, outputShape={}",
               inputShape.size(), outputShape.size());

    llvm::SmallVector<llvm::SmallVector<int64_t>> axesGroups;
    const int64_t rank = inputShape.size();
    int64_t inputIndex = 0;
    for (int64_t outputIndex = 0; outputIndex < rank; ++outputIndex) {
      if (inputIndex < rank &&
          inputShape[inputIndex] == outputShape[outputIndex]) {
        axesGroups.emplace_back(1, axesIds[inputIndex]);
        ++inputIndex;
      } else if (outputShape[outputIndex] == 1) {
        axesGroups.emplace_back();
      } else {
        llvm::SmallVector<int64_t> group;
        int64_t consumed = 1;
        while (inputIndex < rank &&
               outputShape[outputIndex] % (consumed * inputShape[inputIndex]) ==
                   0) {
          group.push_back(axesIds[inputIndex]);
          consumed *= inputShape[inputIndex];
          ++inputIndex;
        }

        if (consumed != outputShape[outputIndex]) {
          llvm::errs() << "Consumed is not equal to output shape: " << consumed
                       << " != " << outputShape[outputIndex] << "\n";
          return {};
        }

        axesGroups.push_back(std::move(group));
      }
    }
    TT_assertv(inputIndex == rank,
               "input is not fully consumed: input_index{}, rank={}",
               inputIndex, rank);

    return axesGroups;
  }

  bool isCommuteUpwardsViable(ReshapeOp op, PermuteOp) const override {
    // Add viability checks - to be filled in later
    return true;
  }

  bool isCommuteUpwardsFavorable(ReshapeOp op, PermuteOp) const override {
    // We should always commute a permute above a reshape if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ReshapeOp op, PermuteOp) const override {
    // Add viability checks - to be filled in later
    return true;
  }

  bool isCommuteDownwardsFavorable(ReshapeOp op,
                                   PermuteOp permuteOperand) const override {
    // Add favorability checks - to be filled in later
    // Commuting downwards is favorable if all other operands satisfy one
    // of the following:
    // - Are an identical TM
    // - Are on a consteval-able path
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populatePermuteReshapeCommutePatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns) {
  patterns.add<TTIRCommutePermuteThroughReshape<commuteDirection>>(ctx);
}

template void populatePermuteReshapeCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void
populatePermuteReshapeCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
