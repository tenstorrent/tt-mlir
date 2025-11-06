// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

namespace mlir::tt::ttir {
// This function will return std::nullopt if the reshape places broadcasted
// data along the same axes as the original data. If this is the case for
// a particular broadcast -> reshape sequence, the reshape cannot be commuted
// above the broadcast.
//
// Otherwise, it will return the shape of the commuted reshape and the new
// broadcast dimensions.
static std::optional<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>>
getNewReshapeAndBroadcastDims(ArrayRef<int64_t> originalShape,
                              ArrayRef<int64_t> finalShape,
                              ArrayRef<int64_t> broadcastDims) {

  // This is meant to represent some volume of data in a tensor, and whether or
  // not it is original, or broadcasted data.
  struct VolumePartition {
    int64_t size;
    bool isOriginalData;
  };

  // A tensor after a broadcast can have its volume separated into chunks of
  // contiguous "original" and "broadcasted" data. For example, if we broadcast
  // a tensor with shape (2, 1, 1, 64) to (2, 32, 32, 64). We its volume
  // partitions will be:
  //  [{size: 2, isOriginalData: true}, {size: 1024, isOriginalData: false},
  //   {size: 64, isOriginalData: true}].
  SmallVector<VolumePartition> volumePartitions;

  // The current partition we are working on. We cannot correctly initialize
  // this before the loop because we do not know where the first partition
  // starts and whether or not is broadcasted or real data.
  VolumePartition currentPartition = {-1, false};
  for (uint64_t i = 0; i < originalShape.size(); i++) {
    assert(originalShape[i] > 1 && broadcastDims[i] == 1 ||
           originalShape[i] == 1 && broadcastDims[i] >= 1 &&
               "Broadcast dimensions should always be 1 when the input shape "
               "is > 1");

    // Create new "original" partiton OR fuse volume of this dimension with
    // current "original" partition.
    if (broadcastDims[i] == 1 && originalShape[i] > 1) {
      if (currentPartition.size == -1) {
        currentPartition = {1, true};
      }
      if (currentPartition.isOriginalData) {
        currentPartition.size *= originalShape[i];
      } else {
        volumePartitions.push_back(currentPartition);
        currentPartition = {originalShape[i], true};
      }
    }
    // Create new "broadcasted" partiton OR fuse volume of this dimension with
    // current "broadcasted" partition.
    else if (broadcastDims[i] > 1) {
      if (currentPartition.size == -1) {
        currentPartition = {1, false};
      }
      if (!currentPartition.isOriginalData) {
        currentPartition.size *= broadcastDims[i];
      } else {
        volumePartitions.push_back(currentPartition);
        currentPartition = {broadcastDims[i], false};
      }
    }
    // Otherwise, both broadcast dim and original shape are 1 and whether it is
    // broadcasted or not is both undefined and irrevlevant.
  }
  volumePartitions.push_back(currentPartition);

  // Consider the example from the comment above: Original tensor: (2, 1, 1,
  // 64). Broadcasted tensor: (2, 32, 32, 64). Consider the following reshape to
  // output a tensor of shape (2, 8, 4, 2, 16, 8, 8). Remember, our volume
  // partitions from above are:
  //    [{size: 2, isOriginalData: true}, {size: 1024, isOriginalData: false},
  //     {size: 64, isOriginalData: true}].
  //
  // We are going to iterate over this final shape from back-to-front. We will
  // start drawing volume from the partition at the back of the list, and move
  // to the next partition towards the front when we "run out" of volume in the
  // current partition.
  //
  // Starting with finalShape[6] = 8:
  //  We can draw 8 from the current partition. Now we must divide the volume
  //  left in the partition by 8.
  //  After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 1024, isOriginalData: false}, {size: 8, isOriginalData: true}]
  //
  // finalShape[5] = 8:
  //  We can draw 8 from the current partition. Now we must divide the volume
  //  left in the partition by 8.
  //  After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 1024, isOriginalData: false}, {size: 1, isOriginalData: true}]
  //    We now focus on the partition at index 1.
  //
  //
  // finalShape[4] = 16:
  //  We can draw 16 from the current partition. Now we must divide the volume
  //  left in the partition by 16.
  //  After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 64, isOriginalData: false}, ...]
  //    We are drawing from broadcasted data, the new broadcast dimension at
  //    dim 4 should be 16. And the new reshape size at dim 4 should be 1.
  //
  // finalShape[3] = 2:
  //  We can draw 2 from the current partition. Now we must divide the volume
  //  left in the partition by 2. After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 32, isOriginalData: false}, ...]
  //    We are drawing from broadcasted data, the new broadcast dimension at
  //    dim 3 should be 2. And the new reshape size at dim 3 should be 1.
  //
  // finalShape[2] = 4:
  //  We can draw 4 from the current partition. Now we must divide the volume
  //  left in the partition by 4. After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 8, isOriginalData: false}, ...]
  //    We are drawing from broadcasted data, the new broadcast dimension at
  //    dim 2 should be 8. And the new reshape size at dim 2 should be 1.
  //
  // finalShape[1] = 8:
  //  We can draw 8 from the current partition. Now we must divide the volume
  //  left in the partition by 8. After:
  //    volumePartitions = [{size: 2, isOriginalData: true},
  //      {size: 1, isOriginalData: false}, ...]
  //   We are drawing from broadcasted data, the new broadcast dimension at
  //   dim 1 should be 8. And the new reshape size at dim 1 should be 1.
  //   We can now focus on the partition at index 0.
  //
  // finalShape[0] = 2:
  //  We can draw 2 from the current partition. Now we must divide the volume
  //  left in the partition by 2. After:
  //    volumePartitions = [{size: 1, isOriginalData: true}, ...]
  //
  // And so we can commute the reshape above.
  //
  // IMPORTANT: If at any point the size of finalShape at some index i is
  // greater than the size of the current partition, this would imply that
  //            the reshape has placed broacasted and original data along the
  //            same axis. And so, the reshape cannot commute through the
  //            broadcast.

  int64_t partitionIndex = volumePartitions.size() - 1;

  // Initialize the broadcast dims to all 1s and populate them as we go.
  // The new reshape shape will be the same as the final shape. And we will
  // drop the size of a dimension to 1 if we know it is broadcasted data.
  SmallVector<int64_t> newBroadcastDims(finalShape.size(), 1);
  SmallVector<int64_t> newReshapeShape(finalShape);
  for (int64_t i = static_cast<int64_t>(finalShape.size()) - 1; i >= 0; i--) {
    if (partitionIndex < 0 && finalShape[i] != 1) {
      return std::nullopt;
    }
    if (finalShape[i] == 1) {
      continue;
    }
    if (volumePartitions[partitionIndex].isOriginalData) {
      if (finalShape[i] > volumePartitions[partitionIndex].size) {
        return std::nullopt;
      }
      volumePartitions[partitionIndex].size /= finalShape[i];
    } else {
      if (finalShape[i] > volumePartitions[partitionIndex].size) {
        return std::nullopt;
      }
      volumePartitions[partitionIndex].size /= finalShape[i];

      newBroadcastDims[i] = finalShape[i];
      newReshapeShape[i] = 1;
    }

    if (volumePartitions[partitionIndex].size == 1) {
      partitionIndex--;
    }
  }
  assert(partitionIndex == -1 && "All data should have been accounted for.");

  return std::make_tuple(newReshapeShape, newBroadcastDims);
}

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommuteReshapeThroughBroadcast
    : public TTIRCommuteOpRewritePattern<ttir::ReshapeOp, ttir::BroadcastOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ttir::ReshapeOp, ttir::BroadcastOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(ttir::BroadcastOp op,
                                    ttir::ReshapeOp reshapeUser,
                                    PatternRewriter &rewriter) const override {

    auto originalShape = op.getInput().getType().getShape();
    // auto broadcastShape = op.getResult().getType().getShape();
    auto tmResultType = reshapeUser.getResult().getType();
    auto finalShape = tmResultType.getShape();

    // This must return something, since we know that the commute is viable.
    // If this returns std::nullopt, then there is a bug in isCommuteViable or
    // within getNewReshapeAndBroadcastDims.
    auto [newReshapeShape, newBroadcastDimensions] =
        *getNewReshapeAndBroadcastDims(originalShape, finalShape,
                                       op.getBroadcastDimensions());

    // Now that we know which shape the reshape should have and which broadcast
    // dimensions the broadcast should have, we can generate the new ops.
    auto newTMResultType =
        RankedTensorType::get(newReshapeShape, tmResultType.getElementType(),
                              tmResultType.getEncoding());

    auto newReshape = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, reshapeUser.getLoc(), newTMResultType, op.getInput(),
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(newReshapeShape.begin(),
                                                      newReshapeShape.end())));

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newReshape,
        newBroadcastDimensions);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(reshapeUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
    }

    for (auto *user : users) {
      rewriter.replaceOp(user, newBroadcast);
    }
  }

  void
  performCommuteDownwardsRewrite(ttir::BroadcastOp op,
                                 ttir::ReshapeOp reshapeOperand,
                                 PatternRewriter &rewriter) const override {
    // TODO(@LPanosTT): implement this
    llvm_unreachable("Not implemented, this should not be called.");
  }

private:
  bool isCommuteUpwardsViable(ttir::BroadcastOp op,
                              ttir::ReshapeOp reshapeUser) const override {

    // If we have a broadcast -> reshape sequence, where the reshape places
    // broadcasted data along one or more of the same axes as real data. The
    // reshape cannot be commuted above the broadcast. This is because there is
    // no way to achieve the same result tensor with a reshape -> broadcast
    // sequence as the broadcast operation cannot result in real data along the
    // same axes as broadcasted data.

    auto originalShape = op.getInput().getType().getShape();
    auto finalShape = reshapeUser.getResult().getType().getShape();
    auto broadcastDims = op.getBroadcastDimensions();

    // There is a bug when broadcasting across more than 4 dimensions.
    // Will remove this check once TTNN provides a permanent fix.
    // https://github.com/tenstorrent/tt-metal/issues/21967
    return finalShape.size() <= 4 &&
           getNewReshapeAndBroadcastDims(originalShape, finalShape,
                                         broadcastDims)
               .has_value();
  }

  bool isCommuteUpwardsFavorable(ttir::BroadcastOp op,
                                 ttir::ReshapeOp) const override {
    // We should always commute a reshape above a broadcast if all users are an
    // identical reshape. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ttir::BroadcastOp op,
                                ttir::ReshapeOp) const override {
    // TODO(@LPanosTT, #3950): performCommuteDownwardsRewrite is not
    // implemented, thus it is not viable for now
    return false;
  }

  bool isCommuteDownwardsFavorable(ttir::BroadcastOp op,
                                   ttir::ReshapeOp) const override {
    // It is always favorable to commute a reshape below a broadcast
    // if it is not already on a consteval-able path. This condition
    // is already checked by the base class.
    return true;
  }
};
} // namespace

namespace {
template <CommuteDirection commuteDirection>
class TTIRCommutePermuteThroughBroadcast
    : public TTIRCommuteOpRewritePattern<ttir::PermuteOp, ttir::BroadcastOp,
                                         commuteDirection> {
public:
  using TTIRCommuteOpRewritePattern<
      ttir::PermuteOp, ttir::BroadcastOp,
      commuteDirection>::TTIRCommuteOpRewritePattern;

  void performCommuteUpwardsRewrite(ttir::BroadcastOp op,
                                    ttir::PermuteOp permuteUser,
                                    PatternRewriter &rewriter) const override {
    auto operand = op.getInput();
    auto operandShape = operand.getType().getShape();
    auto tmResultType = permuteUser.getResult().getType();

    // Commuting a permute above a broadcast requires us to permute which dims
    // are broadcasted.
    auto permutation = permuteUser.getPermutation();
    SmallVector<int64_t> newShape =
        ttmlir::utils::applyPermutation(operandShape, permutation);
    SmallVector<int64_t> newBroadcastDimensions =
        ttmlir::utils::applyPermutation(op.getBroadcastDimensions(),
                                        permutation);

    auto newPermute = ttir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, permuteUser->getLoc(), newShape,
        tmResultType.getElementType(), tmResultType.getEncoding(), operand,
        permutation);

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newPermute,
        newBroadcastDimensions);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "isCommuteUpwardsViable/Favorable should have ensured all users "
             "are identical TMs");
    }

    for (auto *user : users) {
      rewriter.replaceOp(user, newBroadcast);
    }
  }

  void
  performCommuteDownwardsRewrite(ttir::BroadcastOp op,
                                 ttir::PermuteOp permuteOperand,
                                 PatternRewriter &rewriter) const override {
    // TODO(@LPanosTT): implement this
    llvm_unreachable("Not implemented, this should not be called.");
  }

private:
  bool isCommuteUpwardsViable(ttir::BroadcastOp op,
                              ttir::PermuteOp) const override {
    // We can always commute a permute above a broadcast.
    return true;
  }

  bool isCommuteUpwardsFavorable(ttir::BroadcastOp op,
                                 ttir::PermuteOp) const override {
    // We should always commute a permute above a broadcast if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return !users.empty() && checkAllUsersAreIdenticalTms(users);
  }

  bool isCommuteDownwardsViable(ttir::BroadcastOp op,
                                ttir::PermuteOp) const override {
    // We can always commute a permute below a broadcast if it is not already on
    // a consteval-able path. This condition is already checked by the base
    // class.

    // TODO(@LPanosTT): performCommuteDownwardsRewrite is not implemented, thus
    // it is not viable for now
    return false;
  }

  bool isCommuteDownwardsFavorable(ttir::BroadcastOp op,
                                   ttir::PermuteOp) const override {
    // It is always favorable to commute a permute below a broadcast
    return true;
  }
};
} // namespace

template <CommuteDirection commuteDirection>
void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  patterns.add<TTIRCommuteReshapeThroughBroadcast<commuteDirection>,
               TTIRCommutePermuteThroughBroadcast<commuteDirection>>(ctx);
}

template void populateBroadcastCommutePatterns<CommuteDirection::UPWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
template void populateBroadcastCommutePatterns<CommuteDirection::DOWNWARDS>(
    MLIRContext *ctx, RewritePatternSet &patterns);
} // namespace mlir::tt::ttir
