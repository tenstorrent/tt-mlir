// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOFUSINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// Fuses concatenate + reshape patterns into broadcast_in_dim operations.
//
// Example transformation:
//   Input:
//     %0 = stablehlo.concatenate %arg0, %arg0, %arg0, dim = 0 :
//            (tensor<2x4xbf16>, tensor<2x4xbf16>, tensor<2x4xbf16>) ->
//            tensor<6x4xbf16>
//     %1 = stablehlo.reshape %0 : (tensor<6x4xbf16>) -> tensor<3x2x4xbf16>
//
//   Output:
//     %0 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] :
//            (tensor<2x4xbf16>) -> tensor<3x2x4xbf16>
//
// The concatenate -> reshape fusion pattern is valid when the number of repeats
// (3 in this example) appears in the reshape output shape before the
// concatenate dimension. Here, the repeat count 3 appears at dimension 0 in the
// output shape <3x2x4xbf16>, which comes just before the original concatenate
// dimension 0 (now dimension 1 in the output: <3x2x4xbf16>). The broadcast
// dims [1, 2] correspond to the original tensor dimensions, while dimension 0
// handles the repeat.
//
// Fusion is skipped if any of the operations involved have sharded inputs or
// outputs, as sharding adds complexity that requires separate handling.
//
// TODO(@ddilbazTT): Add support for reshape -> concat fusion.
class ConcatenateToBroadcastInDimFusionPattern
    : public OpRewritePattern<::mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<::mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(::mlir::stablehlo::ConcatenateOp concatOp,
                                ::mlir::PatternRewriter &rewriter) const final {
    // Check if fusable.
    if (!isFusable(concatOp)) {
      return failure();
    }
    // Fuse concat -> reshape.
    // To find broadcast dims, remove the concat dim from reshape output shape.
    ::mlir::stablehlo::ReshapeOp reshapeOp =
        mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(
            *concatOp.getResult().getUsers().begin());
    if (!reshapeOp) {
      return failure();
    }
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();
    uint64_t dim = concatOp.getDimension();
    SmallVector<int64_t> broadcastDims;
    for (size_t i = 0; i < reshapeOutputShape.size(); i++) {
      if (i != dim) {
        broadcastDims.push_back(i);
      }
    }
    // Create broadcast_in_dim op with concatenate input and broadcast dims.
    // Replace reshape op with broadcast_in_dim op.
    ::mlir::stablehlo::BroadcastInDimOp broadcastInDimOp =
        rewriter.create<::mlir::stablehlo::BroadcastInDimOp>(
            concatOp.getLoc(), reshapeOp.getResult().getType(),
            concatOp.getInputs()[0], broadcastDims);
    rewriter.replaceOp(reshapeOp, broadcastInDimOp.getResult());
    return success();
  }

private:
  bool isFusable(::mlir::stablehlo::ConcatenateOp concatOp) const {
    // Check all arguments of concat are the same.
    if (!::llvm::all_equal(concatOp.getInputs())) {
      return false;
    }

    // Case 1: concat -> reshape.
    // Check concat has one user and that user is a reshape op.
    if (concatOp.getResult().hasOneUse()) {
      if (auto reshapeOp = mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(
              *concatOp.getResult().getUsers().begin())) {
        if (!checkConcatThenReshape(concatOp, reshapeOp)) {
          return false;
        }
        // Do not fuse if either concatenate or reshape op inputs or outputs are
        // sharded. Otherwise, would need to accommodate for sharding attributes
        // which is out of scope for this fusion pass.
        if (shardy_utils::opHasShardySharding(reshapeOp.getOperation()) ||
            shardy_utils::opHasShardySharding(concatOp.getOperation())) {
          return false;
        }
        return true;
      }
    }

    // Note: Not yet added support for reshape -> concat fusion.
    return false;
  }

  bool checkConcatThenReshape(::mlir::stablehlo::ConcatenateOp concatOp,
                              ::mlir::stablehlo::ReshapeOp reshapeOp) const {
    auto concatInput =
        mlir::cast<TypedValue<RankedTensorType>>(concatOp.getInputs()[0]);
    ArrayRef<int64_t> concatInputShape = concatInput.getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();
    // Number of dimensions of concat should be -1 number of dimensions of
    // reshape.
    if (concatInputShape.size() !=
        static_cast<size_t>(reshapeOutputShape.size() - 1)) {
      return false;
    }
    // If concat dim is i, reshape_output[i] = repeat_dim, reshape_output[i+1] =
    // concat_input[i]
    uint64_t dim = concatOp.getDimension();
    // num repeats is the number of concat inputs.
    int64_t numRepeats = static_cast<int64_t>(concatOp.getInputs().size());
    if (reshapeOutputShape[dim] != numRepeats) {
      return false;
    }
    if (reshapeOutputShape[dim + 1] != concatInputShape[dim]) {
      return false;
    }
    return true;
  }
};

struct StableHLOFusingPass
    : public impl::StableHLOFusingPassBase<StableHLOFusingPass> {
public:
  using impl::StableHLOFusingPassBase<
      StableHLOFusingPass>::StableHLOFusingPassBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConcatenateToBroadcastInDimFusionPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace mlir::tt::stablehlo
