// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
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

    // Get reshape groups and missing axes
    auto reshapeInputShape =
        mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
    auto reshapeOutputShape =
        mlir::cast<RankedTensorType>(op.getType()).getShape();
    // Input of new reshape is the input of original permute
    auto newReshapeInputShape =
        mlir::cast<RankedTensorType>(permuteOperand.getInput().getType())
            .getShape();
    auto axesGroupsResult = ttmlir::utils::getReshapeAxesMapping(
        reshapeInputShape, reshapeOutputShape, permutation);
    if (!axesGroupsResult) {
      return;
    }
    llvm::outs() << "Reshape axes groups found.\n";
    auto axesGroups = axesGroupsResult->first;
    auto missingAxes = axesGroupsResult->second;

    // Check that output size + number of missing axes = input size
    if (reshapeOutputShape.size() + missingAxes.size() !=
        reshapeInputShape.size()) {
      llvm::outs() << "Cannot commute permute through reshape: "
                      "output size + missing axes size != input size\n";
      llvm::outs() << "reshapeInputShape size: " << reshapeInputShape.size()
                   << "\n";
      llvm::outs() << "reshapeOutputShape size: " << reshapeOutputShape.size()
                   << "\n";
      llvm::outs() << "missingAxes size: " << missingAxes.size() << "\n";
      return;
    }

    llvm::outs() << "missingAxes: ";
    for (const auto &axis : missingAxes) {
      llvm::outs() << axis << " ";
    }
    llvm::outs() << "\n";

    llvm::SmallSet<int64_t, 8> missingAxesSet(missingAxes.begin(),
                                              missingAxes.end());

    // Create new permutation that removes values corresponding to missing axes
    // and remaps remaining values to be consecutive indices into
    // filteredAxesGroups
    llvm::SmallVector<int64_t> filteredPermutation;
    llvm::SmallVector<int64_t> valueToNewIndex(newReshapeInputShape.size(), -1);
    int64_t newIndex = 0;
    for (size_t i = 0; i < newReshapeInputShape.size(); ++i) {
      if (!missingAxesSet.contains(static_cast<int64_t>(i))) {
        valueToNewIndex[i] = newIndex++;
      }
    }

    for (size_t i = 0; i < permutation.size(); ++i) {
      int64_t value = permutation[i];
      // Check by value: if the value is not in missing axes, include it
      if (!missingAxesSet.contains(value)) {
        if (valueToNewIndex[value] != -1) {
          filteredPermutation.push_back(valueToNewIndex[value]);
        }
      }
    }

    llvm::outs() << "filteredPermutation: ";
    for (const auto &p : filteredPermutation) {
      llvm::outs() << p << " ";
    }
    llvm::outs() << "\n";

    // Use filtered permutation to permute groups of reshape to get groups of
    // new reshape
    auto permutedAxesGroups = ttmlir::utils::applyPermutation(
        llvm::ArrayRef(axesGroups), filteredPermutation);

    llvm::outs() << "permutedAxesGroups: ";
    for (const auto &group : permutedAxesGroups) {
      llvm::outs() << "[";
      for (const auto &axis : group) {
        llvm::outs() << axis << " ";
      }
      llvm::outs() << "] ";
    }
    llvm::outs() << "\n";

    // Based on those groups, make output shape of new reshape
    llvm::SmallVector<int64_t> newReshapeOutputShape;
    for (const auto &group : permutedAxesGroups) {
      if (group.empty()) {
        newReshapeOutputShape.push_back(1);
      } else {
        int64_t product = 1;
        for (int64_t axis : group) {
          product *= newReshapeInputShape[axis];
        }
        newReshapeOutputShape.push_back(product);
      }
    }

    llvm::outs() << "newReshapeOutputShape: ";
    for (const auto &dim : newReshapeOutputShape) {
      llvm::outs() << dim << " ";
    }
    llvm::outs() << "\n";

    // Create new reshape with the computed output shape
    auto newReshapeType = RankedTensorType::get(newReshapeOutputShape,
                                                op.getType().getElementType(),
                                                op.getType().getEncoding());
    llvm::SmallVector<int32_t> newReshapeShapeAttr(
        newReshapeOutputShape.begin(), newReshapeOutputShape.end());
    ReshapeOp newReshape = ttir::utils::createDPSOp<ReshapeOp>(
        rewriter, op->getLoc(), newReshapeType, permuteOperand.getInput(),
        rewriter.getI32ArrayAttr(newReshapeShapeAttr));

    // Use inverse of filtered permutation in createNewPermuteOp
    // Use output of original reshape as output shape for new permute op
    PermuteOp newPerm = createNewPermuteOp(
        newReshape.getResult(), filteredPermutation, op.getType().getShape(),
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
                               ArrayRef<int64_t> outputShape,
                               PatternRewriter &rewriter, Location loc) const {
    auto inputType = cast<RankedTensorType>(input.getType());
    auto inversePermutation = ttmlir::utils::inversePermutation(permutation);
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
    // auto permutation = permuteOperand.getPermutation();
    // auto reshapeInputShape =
    //     mlir::cast<RankedTensorType>(op.getInput().getType()).getShape();
    // auto reshapeOutputShape =
    //     mlir::cast<RankedTensorType>(op.getType()).getShape();

    // if (reshapeInputShape.size() != reshapeOutputShape.size()) {
    //   return false;
    // }

    // auto axesGroupsResult = ttmlir::utils::getReshapeAxesMapping(
    //     reshapeInputShape, reshapeOutputShape, permutation);
    // if (!axesGroupsResult) {
    //   return false;
    // }
    // auto axesGroups = axesGroupsResult->first;
    // llvm::outs() << "axesGroups: ";
    // for (const auto &group : axesGroups) {
    //   llvm::outs() << "[";
    //   for (const auto &axis : group) {
    //     llvm::outs() << axis << " ";
    //   }
    //   llvm::outs() << "] ";
    // }
    // llvm::outs() << "\n";

    // auto permutedAxesGroups = ttmlir::utils::applyPermutation(
    //     llvm::ArrayRef(axesGroups), permutation);
    // llvm::outs() << "permutedAxesGroups: ";
    // for (const auto &group : permutedAxesGroups) {
    //   llvm::outs() << "[";
    //   for (const auto &axis : group) {
    //     llvm::outs() << axis << " ";
    //   }
    //   llvm::outs() << "] ";
    // }
    // llvm::outs() << "\n";
    // auto flattened = ttmlir::utils::flatten(permutedAxesGroups);
    // // Check if the sequence is increasing (monotonically increasing), not
    // // necessarily that it equals the full range [0, 1, 2, ..., rank-1].
    // return !flattened.empty() && llvm::is_sorted(flattened);
    return true;
  }

  bool isCommuteDownwardsFavorable(ReshapeOp op,
                                   PermuteOp permuteOperand) const override {
    // Favorable if all reshape users are permutes (reshape-permute-reshape) or
    // input of permute is reshape (permute-reshape-permute). Reduces TMs.
    // auto reshapeUsers = op.getResult().getUsers();
    // bool allReshapeUsersArePermute =
    //     !reshapeUsers.empty() &&
    //     llvm::all_of(reshapeUsers,
    //                  [](Operation *user) { return isa<PermuteOp>(user); });
    // llvm::outs() << "allReshapeUsersArePermute: " <<
    // allReshapeUsersArePermute
    //              << "\n";
    // llvm::outs() << permuteOperand.getInput().getDefiningOp() << "\n";
    // bool permuteInputIsReshape =
    //     permuteOperand.getInput().getDefiningOp() &&
    //     dyn_cast<ReshapeOp>(permuteOperand.getInput().getDefiningOp()) !=
    //         nullptr;
    // llvm::outs() << "permuteInputIsReshape: " << permuteInputIsReshape <<
    // "\n"; return allReshapeUsersArePermute || permuteInputIsReshape;
    return true;
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
