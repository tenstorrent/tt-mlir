// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_STABLEHLOSIMPLIFYPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

// torch scatter_reduce(dim=0) lowers to a multi-dim scatter whose column index
// is a positional iota -- really a single-axis row scatter. At dims=[0,1] it
// misses the embedding-backward lowering and hits the generic 1D flatten
// (single-core -> L1 clash under sharding). Rewrite concat(row, iota) to a
// single-axis scatter (dims=[0], column as window) so it stays multi-core.
// Conservative & in-place: rank-2, element-wise, concat(real, iota); never
// erases the iota/concat, so a CSE'd shared iota is fine.
class CollapseRowScatterPattern
    : public OpRewritePattern<::mlir::stablehlo::ScatterOp> {
  using OpRewritePattern<::mlir::stablehlo::ScatterOp>::OpRewritePattern;

  // True if `value` is a positional iota. The frontend builds the column index
  // as iota -> broadcast_in_dim -> reshape (shape-only ops that keep the
  // 0..C-1 values), so we look through them to the IotaOp. With `axis` >= 0 the
  // iota's varying dimension must land on `axis` of `value`, so a row/batch
  // iota is not mistaken for the column iota; `axis` < 0 matches any iota.
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
      // broadcast_in_dim: result `axis` is fed by operand axis j with
      // broadcast_dimensions[j] == axis; if none, `axis` is a broadcast
      // (constant) dim and cannot carry the iota.
      if (auto bcast =
              ::mlir::dyn_cast<::mlir::stablehlo::BroadcastInDimOp>(def)) {
        if (axis < 0) {
          value = bcast.getOperand();
          continue;
        }
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
        continue;
      }
      // reshape: track `axis` only through unit-dim insert/remove, mapping it
      // to the operand axis at the same position among the non-unit dims.
      if (auto reshape = ::mlir::dyn_cast<::mlir::stablehlo::ReshapeOp>(def)) {
        if (axis < 0) {
          value = reshape.getOperand();
          continue;
        }
        auto resType = mlir::cast<RankedTensorType>(reshape.getType());
        auto inType =
            mlir::cast<RankedTensorType>(reshape.getOperand().getType());
        if (resType.getDimSize(axis) == 1) {
          return false;
        }
        int64_t nonUnitBefore = 0;
        for (int64_t i = 0; i < axis; ++i) {
          if (resType.getDimSize(i) != 1) {
            ++nonUnitBefore;
          }
        }
        int64_t mapped = -1, seen = 0;
        for (int64_t i = 0; i < inType.getRank(); ++i) {
          if (inType.getDimSize(i) == 1) {
            continue;
          }
          if (seen == nonUnitBefore) {
            mapped = i;
            break;
          }
          ++seen;
        }
        if (mapped < 0) {
          return false;
        }
        value = reshape.getOperand();
        axis = mapped;
        continue;
      }
      return false;
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

    // Component 0 = real row coord, component 1 = positional column iota
    // (must vary along the column axis 1; a row-axis iota is not the identity
    // column map and would miscompile if collapsed).
    Value rowComp = concat.getInputs()[0];
    Value colComp = concat.getInputs()[1];
    auto rowType = mlir::cast<RankedTensorType>(rowComp.getType());
    if (rowType.getRank() != 3 || rowType.getShape()[indexVectorDim] != 1) {
      return failure();
    }
    // Graph traversal last — gated by the cheap structural checks above.
    if (tracesToIota(rowComp) || !tracesToIota(colComp, /*columnAxis=*/1)) {
      return failure();
    }
    int64_t numRows = rowType.getShape()[0];

    Location loc = op.getLoc();
    // Row coord is constant along the column axis -> take column 0: [N,1,1].
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

    // Scatter only dim 0; the column (operand dim 1) becomes the update window.
    auto newDimNumbers = ::mlir::stablehlo::ScatterDimensionNumbersAttr::get(
        rewriter.getContext(), /*update_window_dims=*/{1},
        /*inserted_window_dims=*/{0}, /*input_batching_dims=*/{},
        /*scatter_indices_batching_dims=*/{},
        /*scatter_dims_to_operand_dims=*/{0}, /*index_vector_dim=*/1);

    rewriter.modifyOpInPlace(op, [&]() {
      op.getScatterIndicesMutable().assign(rowIndex);
      op.setScatterDimensionNumbersAttr(newDimNumbers);
    });
    return success();
  }
};

struct StableHLOSimplifyPass
    : public impl::StableHLOSimplifyPassBase<StableHLOSimplifyPass> {
public:
  using impl::StableHLOSimplifyPassBase<
      StableHLOSimplifyPass>::StableHLOSimplifyPassBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapseRowScatterPattern>(&getContext());

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace mlir::tt::stablehlo
