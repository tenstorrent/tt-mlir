// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/EraseInverseOps/EraseInverseOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Utils.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <tuple>

namespace mlir::tt::ttir {

namespace {
class TTIRCommuteTransposesAboveBroadcast
    : public TTIRCommuteOpRewritePattern<ttir::TransposeOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteOpRewritePattern<
      ttir::TransposeOp, ttir::BroadcastOp>::TTIRCommuteOpRewritePattern;

  void performCommuteRewrite(ttir::BroadcastOp op,
                             ttir::TransposeOp transposeUser,
                             PatternRewriter &rewriter) const override {

    auto operand = op.getInput();
    auto tmResultType = transposeUser.getResult().getType();

    SmallVector<int64_t> newShape(operand.getType().getShape());
    std::swap(newShape[transposeUser.getDim0()],
              newShape[transposeUser.getDim1()]);

    // Commuting a transpose above a broadcast requires us to swap the broadcast
    // dimensions according to the transpose dimensions.
    SmallVector<int64_t> newBroadcastDimensions(op.getBroadcastDimensions());
    std::swap(newBroadcastDimensions[transposeUser.getDim0()],
              newBroadcastDimensions[transposeUser.getDim1()]);

    auto newTranspose = ttmlir::utils::createDPSOp<ttir::TransposeOp>(
        rewriter, op->getLoc(), newShape, tmResultType.getElementType(),
        tmResultType.getEncoding(), operand, transposeUser.getDim0(),
        transposeUser.getDim1());

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));

    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newTranspose,
        newBroadcastDimensions);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(transposeUser, user) &&
             "shouldCommute should have ensured this is true");
      rewriter.replaceOp(user, newBroadcast);
    }
  }

private:
  LogicalResult isCommuteViable(ttir::BroadcastOp op,
                                ttir::TransposeOp) const override {
    // We can always commute a transpose above a broadcast.
    return success();
  }

  LogicalResult isCommuteFavorable(ttir::BroadcastOp op,
                                   ttir::TransposeOp) const override {
    // We should always commute a transpose above a broadcast if all users are
    // an identical transpose. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return success(!users.empty() && checkAllUsersAreIdenticalTms(users));
  }
};
} // namespace

namespace {
struct DataPartition {
  int64_t size;
  bool real;
};

std::optional<std::tuple<SmallVector<int64_t>, SmallVector<int64_t>>>
getNewReshapeAndBroadcastDims(ArrayRef<int64_t> originalShape,
                              ArrayRef<int64_t> finalShape,
                              ArrayRef<int64_t> broadcastDims) {
  SmallVector<DataPartition> dataPartitions;
  DataPartition currentPartition = {-1, false};
  for (uint64_t i = 0; i < originalShape.size(); i++) {
    assert(originalShape[i] > 1 && broadcastDims[i] == 1 ||
           originalShape[i] == 1 && broadcastDims[i] >= 1 &&
               "Broadcast dimensions should always be 1 when the input shape "
               "is > 1");
    if (broadcastDims[i] == 1 && originalShape[i] > 1) {
      if (currentPartition.size == -1) {
        currentPartition = {1, true};
      }
      if (currentPartition.real) {
        currentPartition.size *= originalShape[i];
      } else {
        dataPartitions.push_back(currentPartition);
        currentPartition = {originalShape[i], true};
      }
    } else if (broadcastDims[i] > 1) {
      if (currentPartition.size == -1) {
        currentPartition = {1, false};
      }
      if (!currentPartition.real) {
        currentPartition.size *= broadcastDims[i];
      } else {
        dataPartitions.push_back(currentPartition);
        currentPartition = {broadcastDims[i], false};
      }
    }
  }
  dataPartitions.push_back(currentPartition);

  int64_t partitionIndex = dataPartitions.size() - 1;
  SmallVector<int64_t> newBroadcastDims(finalShape.size(), 1);
  SmallVector<int64_t> newReshapeShape(finalShape);
  for (int64_t i = finalShape.size() - 1; i >= 0; i--) {
    if (partitionIndex < 0 && finalShape[i] != 1) {
      return std::nullopt;
    }
    if (finalShape[i] == 1) {
      continue;
    }
    if (dataPartitions[partitionIndex].real) {
      if (finalShape[i] > dataPartitions[partitionIndex].size) {
        return std::nullopt;
      }
      dataPartitions[partitionIndex].size /= finalShape[i];
    } else {
      if (finalShape[i] > dataPartitions[partitionIndex].size) {
        return std::nullopt;
      }
      dataPartitions[partitionIndex].size /= finalShape[i];

      newBroadcastDims[i] = finalShape[i];
      newReshapeShape[i] = 1;
    }

    if (dataPartitions[partitionIndex].size == 1) {
      partitionIndex--;
    }
  }
  assert(partitionIndex == -1 && "All data should have been accounted for.");

  return std::make_tuple(newReshapeShape, newBroadcastDims);
}
class TTIRCommuteReshapeAboveBroadcast
    : public TTIRCommuteOpRewritePattern<ttir::ReshapeOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteOpRewritePattern<
      ttir::ReshapeOp, ttir::BroadcastOp>::TTIRCommuteOpRewritePattern;
  void performCommuteRewrite(ttir::BroadcastOp op, ttir::ReshapeOp reshapeUser,
                             PatternRewriter &rewriter) const override {

    auto originalShape = op.getInput().getType().getShape();
    // auto broadcastShape = op.getResult().getType().getShape();
    auto tmResultType = reshapeUser.getResult().getType();
    auto finalShape = tmResultType.getShape();

    // This must return something, since we know that the commute is viable.
    // If dataPartitions is nullopt, then there is a bug in isCommuteViable or
    // within getDataPartitions.
    auto [newReshapeShape, newBroadcastDimensions] =
        *getNewReshapeAndBroadcastDims(originalShape, finalShape,
                                       op.getBroadcastDimensions());

    // Now that we know which shape the reshape should have and which broadcast
    // dimensions the broadcast should have, we can generate the new ops.
    auto newTMResultType =
        RankedTensorType::get(newReshapeShape, tmResultType.getElementType(),
                              tmResultType.getEncoding());

    auto newReshape = ttmlir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, op->getLoc(), newTMResultType, op.getInput(),
        rewriter.getI32ArrayAttr(SmallVector<int32_t>(newReshapeShape.begin(),
                                                      newReshapeShape.end())));

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newReshape,
        newBroadcastDimensions);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(reshapeUser, user) &&
             "shouldCommute should have ensured this is true");
      rewriter.replaceOp(user, newBroadcast);
    }
  }

private:
  LogicalResult isCommuteViable(ttir::BroadcastOp op,
                                ttir::ReshapeOp reshapeUser) const override {
    // TODO Make new explanitory comment about when a reshape can and cannot
    // commute through a broadcast.

    auto originalShape = op.getInput().getType().getShape();
    auto finalShape = reshapeUser.getResult().getType().getShape();
    auto broadcastDims = op.getBroadcastDimensions();

    return success(getNewReshapeAndBroadcastDims(originalShape, finalShape,
                                                 broadcastDims));
  }

  LogicalResult isCommuteFavorable(ttir::BroadcastOp op,
                                   ttir::ReshapeOp) const override {
    // We should always commute a reshape above a broadcast if all users are an
    // identical reshape. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return success(users.size() > 0 && checkAllUsersAreIdenticalTms(users));
  }
};
} // namespace

namespace {
class TTIRCommutePermuteAboveBroadcast
    : public TTIRCommuteOpRewritePattern<ttir::PermuteOp, ttir::BroadcastOp> {
public:
  using TTIRCommuteOpRewritePattern<
      ttir::PermuteOp, ttir::BroadcastOp>::TTIRCommuteOpRewritePattern;

  void performCommuteRewrite(ttir::BroadcastOp op, ttir::PermuteOp permuteUser,
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

    auto newPermute = ttmlir::utils::createDPSOp<ttir::PermuteOp>(
        rewriter, op->getLoc(), newShape, tmResultType.getElementType(),
        tmResultType.getEncoding(), operand, permutation);

    assert(newBroadcastDimensions.size() ==
           static_cast<size_t>(tmResultType.getRank()));
    auto newBroadcast = ttmlir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, op->getLoc(), tmResultType, newPermute,
        newBroadcastDimensions);

    SmallVector<Operation *> users(op->getUsers());
    for (auto *user : users) {
      assert(checkIdenticalTms(permuteUser, user) &&
             "shouldCommute should have ensured this is true");
      rewriter.replaceOp(user, newBroadcast);
    }
  }

private:
  LogicalResult isCommuteViable(ttir::BroadcastOp op,
                                ttir::PermuteOp) const override {
    // We can always commute a permute above a broadcast.
    return success();
  }

  LogicalResult isCommuteFavorable(ttir::BroadcastOp op,
                                   ttir::PermuteOp) const override {
    // We should always commute a permute above a broadcast if all users are an
    // identical permutation. This includes the case where there is one user.
    SmallVector<Operation *> users(op->getUsers());
    return success(!users.empty() && checkAllUsersAreIdenticalTms(users));
  }
};
} // namespace

void populateBroadcastCommutePatterns(MLIRContext *ctx,
                                      RewritePatternSet &patterns) {
  patterns
      .add<TTIRCommuteTransposesAboveBroadcast,
           TTIRCommuteReshapeAboveBroadcast, TTIRCommutePermuteAboveBroadcast>(
          ctx);
}

} // namespace mlir::tt::ttir
