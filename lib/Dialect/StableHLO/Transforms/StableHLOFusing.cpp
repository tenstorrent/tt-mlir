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

// Fuses concatenate / concatenate + reshape patterns into broadcast_in_dim
// operations.
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
class ConcatenateToBroadcastInDimFusionPattern
    : public OpRewritePattern<::mlir::stablehlo::ConcatenateOp> {
  using OpRewritePattern<::mlir::stablehlo::ConcatenateOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(::mlir::stablehlo::ConcatenateOp concatOp,
                                ::mlir::PatternRewriter &rewriter) const final {
    // Case 1: concat -> reshape.
    if (auto reshapeOp = getConcatThenReshapeOp(concatOp)) {
      return fuseConcatThenReshape(concatOp, *reshapeOp, rewriter);
    }
    // Case 2: reshape -> concat (reshape inserts size-1 dim at concat dim).
    if (auto reshapeOp = getReshapeThenConcatOp(concatOp)) {
      return fuseReshapeThenConcat(concatOp, *reshapeOp, rewriter);
    }
    // Case 3: concat of identical tensors along a size-1 dim ->
    // broadcast_in_dim.
    if (checkConcatToBroadcast(concatOp)) {
      return fuseConcatToBroadcast(concatOp, rewriter);
    }
    return failure();
  }

private:
  LogicalResult fuseConcatThenReshape(::mlir::stablehlo::ConcatenateOp concatOp,
                                      ::mlir::stablehlo::ReshapeOp reshapeOp,
                                      ::mlir::PatternRewriter &rewriter) const {
    // To find broadcast dims, remove the concat dim from reshape output shape.
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

  bool checkConcatToBroadcast(::mlir::stablehlo::ConcatenateOp concatOp) const {
    // All inputs must be identical.
    if (!::llvm::all_equal(concatOp.getInputs())) {
      return false;
    }
    // Do not fuse if concat op has sharding attributes.
    if (shardy_utils::opHasShardySharding(concatOp.getOperation())) {
      return false;
    }
    // Concat dim in the input must have size 1 so that broadcast_in_dim can
    // expand it. broadcast_in_dim requires the mapped input dim to be either
    // equal to the output dim or 1.
    auto concatInput =
        mlir::cast<TypedValue<RankedTensorType>>(concatOp.getInputs()[0]);
    uint64_t dim = concatOp.getDimension();
    return concatInput.getType().getShape()[dim] == 1;
  }

  LogicalResult fuseConcatToBroadcast(::mlir::stablehlo::ConcatenateOp concatOp,
                                      ::mlir::PatternRewriter &rewriter) const {
    auto concatInput =
        mlir::cast<TypedValue<RankedTensorType>>(concatOp.getInputs()[0]);
    int64_t rank = concatInput.getType().getRank();
    SmallVector<int64_t> broadcastDims;
    for (int64_t i = 0; i < rank; i++) {
      broadcastDims.push_back(i);
    }
    ::mlir::stablehlo::BroadcastInDimOp broadcastInDimOp =
        rewriter.create<::mlir::stablehlo::BroadcastInDimOp>(
            concatOp.getLoc(), concatOp.getResult().getType(), concatInput,
            broadcastDims);
    rewriter.replaceOp(concatOp, broadcastInDimOp.getResult());
    return success();
  }

  // Returns the reshape op if reshapeOp -> concatOp is a fusable pattern,
  // otherwise returns std::nullopt.
  std::optional<::mlir::stablehlo::ReshapeOp>
  getReshapeThenConcatOp(::mlir::stablehlo::ConcatenateOp concatOp) const {
    // All inputs must be identical.
    if (!::llvm::all_equal(concatOp.getInputs())) {
      return std::nullopt;
    }
    auto reshapeOp = mlir::dyn_cast_or_null<::mlir::stablehlo::ReshapeOp>(
        concatOp.getInputs()[0].getDefiningOp());
    if (!reshapeOp || !checkReshapeThenConcat(concatOp, reshapeOp)) {
      return std::nullopt;
    }
    return reshapeOp;
  }

  bool checkReshapeThenConcat(::mlir::stablehlo::ConcatenateOp concatOp,
                              ::mlir::stablehlo::ReshapeOp reshapeOp) const {
    // Do not fuse if either op has sharding attributes.
    if (shardy_utils::opHasShardySharding(reshapeOp.getOperation()) ||
        shardy_utils::opHasShardySharding(concatOp.getOperation())) {
      return false;
    }
    auto reshapeInput =
        mlir::cast<TypedValue<RankedTensorType>>(reshapeOp.getOperand());
    auto reshapeOutput =
        mlir::cast<TypedValue<RankedTensorType>>(reshapeOp.getResult());
    ArrayRef<int64_t> inputShape = reshapeInput.getType().getShape();
    ArrayRef<int64_t> outputShape = reshapeOutput.getType().getShape();

    // Reshape must add exactly one dimension.
    if (outputShape.size() != inputShape.size() + 1) {
      return false;
    }
    // The inserted dimension must be at the concat dim and have size 1.
    uint64_t dim = concatOp.getDimension();
    if (outputShape[dim] != 1) {
      return false;
    }
    // All other dims must be preserved in order.
    for (size_t i = 0; i < inputShape.size(); i++) {
      size_t outIdx = i < dim ? i : i + 1;
      if (outputShape[outIdx] != inputShape[i]) {
        return false;
      }
    }
    return true;
  }

  LogicalResult fuseReshapeThenConcat(::mlir::stablehlo::ConcatenateOp concatOp,
                                      ::mlir::stablehlo::ReshapeOp reshapeOp,
                                      ::mlir::PatternRewriter &rewriter) const {
    auto reshapeInput =
        mlir::cast<TypedValue<RankedTensorType>>(reshapeOp.getOperand());
    int64_t inputRank = reshapeInput.getType().getRank();
    uint64_t dim = concatOp.getDimension();
    // Broadcast dims map reshape input dims to concat output dims, skipping
    // the inserted dim D: input dim i -> output dim (i < D ? i : i+1).
    SmallVector<int64_t> broadcastDims;
    for (int64_t i = 0; i < inputRank; i++) {
      broadcastDims.push_back(
          static_cast<int64_t>(i) < static_cast<int64_t>(dim) ? i : i + 1);
    }
    ::mlir::stablehlo::BroadcastInDimOp broadcastInDimOp =
        rewriter.create<::mlir::stablehlo::BroadcastInDimOp>(
            concatOp.getLoc(), concatOp.getResult().getType(), reshapeInput,
            broadcastDims);
    rewriter.replaceOp(concatOp, broadcastInDimOp.getResult());
    return success();
  }

  // Returns the reshape op if concatOp -> reshapeOp is a fusable pattern,
  // otherwise returns std::nullopt.
  std::optional<::mlir::stablehlo::ReshapeOp>
  getConcatThenReshapeOp(::mlir::stablehlo::ConcatenateOp concatOp) const {
    // Check all arguments of concat are the same.
    if (!::llvm::all_equal(concatOp.getInputs())) {
      return std::nullopt;
    }
    if (!concatOp.getResult().hasOneUse()) {
      return std::nullopt;
    }
    auto reshapeOp = mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(
        *concatOp.getResult().getUsers().begin());
    if (!reshapeOp || !checkConcatThenReshape(concatOp, reshapeOp)) {
      return std::nullopt;
    }
    return reshapeOp;
  }

  bool checkConcatThenReshape(::mlir::stablehlo::ConcatenateOp concatOp,
                              ::mlir::stablehlo::ReshapeOp reshapeOp) const {
    // Do not fuse if either concatenate or reshape op inputs or outputs are
    // sharded. Otherwise, would need to accommodate for sharding attributes
    // which is out of scope for this fusion pass.
    if (shardy_utils::opHasShardySharding(reshapeOp.getOperation()) ||
        shardy_utils::opHasShardySharding(concatOp.getOperation())) {
      return false;
    }
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

// Fuses reshape + gather when the reshape is an unsqueeze at index_vector_dim
// and index_vector_dim is the last dim.
//
class ReshapeGatherFusionPattern
    : public OpRewritePattern<::mlir::stablehlo::GatherOp> {
  using OpRewritePattern<::mlir::stablehlo::GatherOp>::OpRewritePattern;

public:
  LogicalResult matchAndRewrite(::mlir::stablehlo::GatherOp gatherOp,
                                ::mlir::PatternRewriter &rewriter) const final {
    auto reshapeOp = mlir::dyn_cast_or_null<::mlir::stablehlo::ReshapeOp>(
        gatherOp.getStartIndices().getDefiningOp());
    if (!reshapeOp) {
      return failure();
    }

    if (shardy_utils::opHasShardySharding(reshapeOp.getOperation()) ||
        shardy_utils::opHasShardySharding(gatherOp.getOperation())) {
      return failure();
    }

    auto reshapeInput = reshapeOp.getOperand();
    auto reshapeOutput = reshapeOp.getResult();
    ArrayRef<int64_t> inputShape = reshapeInput.getType().getShape();
    ArrayRef<int64_t> outputShape = reshapeOutput.getType().getShape();

    // Reshape must add exactly one dimension.
    if (outputShape.size() != inputShape.size() + 1) {
      return failure();
    }

    auto dimNumbers = gatherOp.getDimensionNumbers();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    // The unsqueezed dimension must be at index_vector_dim and have size 1.
    if (indexVectorDim >= static_cast<int64_t>(outputShape.size()) ||
        outputShape[indexVectorDim] != 1) {
      return failure();
    }

    // All other dims must be preserved in order.
    for (size_t i = 0; i < inputShape.size(); i++) {
      size_t outIdx = static_cast<int64_t>(i) < indexVectorDim ? i : i + 1;
      if (outputShape[outIdx] != inputShape[i]) {
        return failure();
      }
    }

    // index_vector_dim must be the last dim.
    if (indexVectorDim != static_cast<int64_t>(outputShape.size()) - 1) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<::mlir::stablehlo::GatherOp>(
        gatherOp, gatherOp.getResult().getType(), gatherOp.getOperand(),
        reshapeInput, gatherOp.getDimensionNumbers(), gatherOp.getSliceSizes(),
        gatherOp.getIndicesAreSorted());

    return success();
  }
};

// Fuses a disguised row scatter into a single-axis one: index
// concat(real_row, column_iota) at dims=[0,1] flattens to a single-core 1D
// scatter (L1 clash). Re-emit with dims=[0], column as window.
class CollapseRowScatterPattern
    : public OpRewritePattern<::mlir::stablehlo::ScatterOp> {
  using OpRewritePattern<::mlir::stablehlo::ScatterOp>::OpRewritePattern;

  // True if `value` is an iota (through broadcast_in_dim). `axis` >= 0 requires
  // the iota to vary along that axis, rejecting a row-axis iota.
  static bool tracesToIota(::mlir::Value value, int64_t axis = -1) {
    while (value) {
      ::mlir::Operation *def = value.getDefiningOp();
      if (!def) {
        return false;
      }
      if (auto iota = ::mlir::dyn_cast<::mlir::stablehlo::IotaOp>(def)) {
        return axis < 0 ||
               static_cast<int64_t>(iota.getIotaDimension()) == axis;
      }
      auto bcast = ::mlir::dyn_cast<::mlir::stablehlo::BroadcastInDimOp>(def);
      if (!bcast) {
        return false;
      }
      if (axis < 0) {
        value = bcast.getOperand();
        continue;
      }
      // Map result `axis` to its operand axis; a broadcast dim can't carry it.
      ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
      int64_t mapped = -1;
      for (int64_t j = 0; j < static_cast<int64_t>(dims.size()); ++j) {
        if (dims[j] == axis) {
          mapped = j;
          break;
        }
      }
      if (mapped < 0) {
        return false;
      }
      value = bcast.getOperand();
      axis = mapped;
    }
    return false;
  }

public:
  LogicalResult matchAndRewrite(::mlir::stablehlo::ScatterOp op,
                                ::mlir::PatternRewriter &rewriter) const final {
    auto dimNumbers = op.getScatterDimensionNumbers();
    if (op.getInputs().size() != 1) {
      return failure();
    }
    auto operandType =
        mlir::cast<RankedTensorType>(op.getInputs()[0].getType());
    if (operandType.getRank() != 2) {
      return failure();
    }
    if (!dimNumbers.getUpdateWindowDims().empty()) {
      return failure(); // element-wise only
    }
    ArrayRef<int64_t> scatterDims = dimNumbers.getScatterDimsToOperandDims();
    if (scatterDims.size() != 2 || scatterDims[0] != 0 || scatterDims[1] != 1) {
      return failure();
    }

    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto concat = op.getScatterIndices()
                      .getDefiningOp<::mlir::stablehlo::ConcatenateOp>();
    if (!concat ||
        static_cast<int64_t>(concat.getDimension()) != indexVectorDim ||
        concat.getInputs().size() != 2) {
      return failure();
    }

    // Component 0 = row, component 1 = column iota.
    Value rowComp = concat.getInputs()[0];
    Value colComp = concat.getInputs()[1];
    auto rowType = mlir::cast<RankedTensorType>(rowComp.getType());
    if (rowType.getRank() != 3 || rowType.getShape()[indexVectorDim] != 1) {
      return failure();
    }
    if (tracesToIota(rowComp) || !tracesToIota(colComp, /*columnAxis=*/1)) {
      return failure();
    }
    int64_t numRows = rowType.getShape()[0];

    // Row is constant across columns -> take column 0.
    Location loc = op.getLoc();
    auto slicedType =
        RankedTensorType::get({numRows, 1, 1}, rowType.getElementType());
    Value sliced = rewriter.create<::mlir::stablehlo::SliceOp>(
        loc, slicedType, rowComp, rewriter.getDenseI64ArrayAttr({0, 0, 0}),
        rewriter.getDenseI64ArrayAttr({numRows, 1, 1}),
        rewriter.getDenseI64ArrayAttr({1, 1, 1}));
    auto rowIndexType =
        RankedTensorType::get({numRows, 1}, rowType.getElementType());
    Value rowIndex = rewriter.create<::mlir::stablehlo::ReshapeOp>(
        loc, rowIndexType, sliced);

    // Re-emit as a single-axis scatter; the dead concat/iota is left for DCE.
    auto newDimNumbers = ::mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(), /*update_window_dims=*/{1},
        /*inserted_window_dims=*/{0}, /*input_batching_dims=*/{},
        /*scatter_indices_batching_dims=*/{},
        /*scatter_dims_to_operand_dims=*/{0}, /*index_vector_dim=*/1);
    auto newScatter = rewriter.create<::mlir::stablehlo::ScatterOp>(
        loc, op.getResultTypes(), op.getInputs(), rowIndex, op.getUpdates(),
        newDimNumbers, op.getIndicesAreSorted(), op.getUniqueIndices());
    newScatter.getUpdateComputation().takeBody(op.getUpdateComputation());
    rewriter.replaceOp(op, newScatter.getResults());
    return success();
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
    patterns.add<ReshapeGatherFusionPattern>(&getContext());
    patterns.add<CollapseRowScatterPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace mlir::tt::stablehlo
