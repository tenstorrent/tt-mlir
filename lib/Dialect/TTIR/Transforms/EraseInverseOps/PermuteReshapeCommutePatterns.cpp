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
#include <llvm/Support/raw_ostream.h>

namespace mlir::tt::ttir {

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughReshape
    : public TTIRCommuteOpRewritePattern<PermuteOp, ReshapeOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      PermuteOp, ReshapeOp, commuteDirection>::TTIRCommuteOpRewritePattern;

  // NOT IMPLEMENTED: Upwards commutation of permute through reshape is not
  // implemented.
  void performCommuteUpwardsRewrite(ReshapeOp op, PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    llvm_unreachable("Not implemented, this should not be called.");
  }

  // Consider the following IR pseudocode:
  // %0 = permute(%arg0) <{permutation = array<i64: 0, 3, 1, 2>}>
  // %1 = reshape(%0) {shape = [a, b, c, d]}
  //
  // This method will transform this into:
  // %0 = reshape(%arg0) {shape = [transformed_shape]}
  // %1 = permute(%0) <{permutation = array<i64: 0, 3, 1, 2>}>
  //
  // The reshape shape is transformed by applying the inverse permutation to the
  // current reshape output shape.
  void
  performCommuteDownwardsRewrite(ReshapeOp op, PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    auto permutation = permuteOperand.getPermutation();
    llvm::outs() << "permutation: ";
    for (const auto &p : permutation) {
      llvm::outs() << p << " ";
    }
    llvm::outs() << "\n";
    ReshapeOp newReshape = createNewReshapeOpWithPermutedShape(
        op, permuteOperand.getInput(), permutation, rewriter);
    llvm::outs() << "newReshape output shape: ";
    for (const auto &dim : newReshape.getType().getShape()) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    PermuteOp newPerm = createNewPermuteOp(newReshape.getResult(), permutation,
                                           rewriter, permuteOperand.getLoc());
    llvm::outs() << "newPerm output shape: ";
    for (const auto &dim : newPerm.getType().getShape()) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";
    rewriter.replaceOp(op, newPerm);
  }

private:
  PermuteOp createNewPermuteOp(Value input, ArrayRef<int64_t> permutation,
                               PatternRewriter &rewriter, Location loc) const {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();

    auto inversePermutation = ttmlir::utils::inversePermutation(permutation);
    auto outputShape =
        ttmlir::utils::applyPermutation(inputShape, inversePermutation);
    auto outputType = RankedTensorType::get(
        outputShape, inputType.getElementType(), inputType.getEncoding());

    return ttir::utils::createDPSOp<PermuteOp>(rewriter, loc, outputType, input,
                                               inversePermutation);
  }

  ReshapeOp
  createNewReshapeOpWithPermutedShape(ReshapeOp op, Value newInput,
                                      ArrayRef<int64_t> permutation,
                                      PatternRewriter &rewriter) const {
    auto oldReshapeShape = op.getType().getShape();
    auto newReshapeOutputShape =
        ttmlir::utils::applyPermutation(oldReshapeShape, permutation);
    auto newReshapeType = RankedTensorType::get(newReshapeOutputShape,
                                                op.getType().getElementType(),
                                                op.getType().getEncoding());

    llvm::SmallVector<int32_t> newReshapeShapeAttr(
        newReshapeOutputShape.begin(), newReshapeOutputShape.end());
    return ttir::utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), newReshapeType, newInput,
        rewriter.getI32ArrayAttr(newReshapeShapeAttr));
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

  // NOT IMPLEMENTED: Upwards commutation of permute through reshape is not
  // implemented.
  bool isCommuteUpwardsViable(ReshapeOp op, PermuteOp) const override {
    return false;
  }

  // NOT IMPLEMENTED: Upwards commutation of permute through reshape is not
  // implemented.
  bool isCommuteUpwardsFavorable(ReshapeOp op, PermuteOp) const override {
    return false;
  }

  bool isCommuteDownwardsViable(ReshapeOp op,
                                PermuteOp permuteOperand) const override {
    auto permutation = permuteOperand.getPermutation();
    auto reshapeInputShape =
        mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
    auto reshapeOutputShape =
        mlir::cast<RankedTensorType>(op.getType()).getShape();

    if (reshapeInputShape.size() != reshapeOutputShape.size()) {
      return false;
    }
    const int64_t rank = reshapeInputShape.size();

    auto axesGroups =
        groupAxes(reshapeInputShape, reshapeOutputShape, permutation);
    if (!axesGroups) {
      return false;
    }
    llvm::outs() << "axesGroups: ";
    for (const auto &group : *axesGroups) {
      llvm::outs() << "[";
      for (const auto &axis : group) {
        llvm::outs() << axis << " ";
      }
      llvm::outs() << "] ";
    }
    llvm::outs() << "\n";

    auto permutedAxesGroups = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(*axesGroups), permutation);
    llvm::outs() << "permutedAxesGroups: ";
    for (const auto &group : permutedAxesGroups) {
      llvm::outs() << "[";
      for (const auto &axis : group) {
        llvm::outs() << axis << " ";
      }
      llvm::outs() << "] ";
    }
    llvm::outs() << "\n";
    return llvm::equal(ttmlir::utils::flatten(permutedAxesGroups),
                       llvm::seq(rank));
  }

  bool isCommuteDownwardsFavorable(ReshapeOp op,
                                   PermuteOp permuteOperand) const override {
    // Favorable if all reshape users are permutes (reshape-permute-reshape) or
    // input of permute is reshape (permute-reshape-permute). Reduces TMs.
    auto reshapeUsers = op.getResult().getUsers();
    bool allReshapeUsersArePermute =
        !reshapeUsers.empty() &&
        llvm::all_of(reshapeUsers,
                     [](Operation *user) { return isa<PermuteOp>(user); });
    llvm::outs() << "allReshapeUsersArePermute: " << allReshapeUsersArePermute
                 << "\n";
    llvm::outs() << permuteOperand.getInput().getDefiningOp() << "\n";
    bool permuteInputIsReshape =
        permuteOperand.getInput().getDefiningOp() &&
        dyn_cast<ReshapeOp>(permuteOperand.getInput().getDefiningOp()) !=
            nullptr;
    llvm::outs() << "permuteInputIsReshape: " << permuteInputIsReshape << "\n";
    return allReshapeUsersArePermute || permuteInputIsReshape;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populatePermuteReshapeCommutePatterns(MLIRContext *ctx,
                                           RewritePatternSet &patterns) {
  patterns.add<TTIRCommutePermuteThroughReshape<commuteDirection>>(ctx);
}

// NOT IMPLEMENTED: Upwards commutation of permute through reshape is not
// implemented. The template is instantiated to allow compilation, but the
// methods return false/do nothing, so the pattern will never match.
template void populatePermuteReshapeCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void
populatePermuteReshapeCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::tt::ttir
