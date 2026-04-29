// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "stablehlo/dialect/StablehloOps.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_SATURATINGSHIFTFOLDINGPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

class FoldI64SaturatingRightShift
    : public OpRewritePattern<mlir::stablehlo::ShiftRightLogicalOp> {
public:
  using OpRewritePattern<
      mlir::stablehlo::ShiftRightLogicalOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ShiftRightLogicalOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType) {
      return failure();
    }

    auto intType = dyn_cast<IntegerType>(resultType.getElementType());
    if (!intType || intType.getWidth() != 64) {
      return failure();
    }

    // Look through stablehlo.broadcast_in_dim to find a backing splat constant.
    Operation *def = op.getRhs().getDefiningOp();
    while (auto bcast =
               dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(def)) {
      def = bcast.getOperand().getDefiningOp();
    }
    auto constOp = dyn_cast_or_null<mlir::stablehlo::ConstantOp>(def)
    if (!constOp) {
      return failure();
    }
    auto attr = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
    if (!attr || !attr.isSplat() || attr.getSplatValue<APInt>().ult(32)) {
      return failure();
    }

    auto zeroAttr = DenseElementsAttr::get(resultType, APInt(64, 0));
    rewriter.replaceOpWithNewOp<mlir::stablehlo::ConstantOp>(op, resultType,
                                                             zeroAttr);
    return success();
  }
};

struct SaturatingShiftFoldingPass
    : public impl::SaturatingShiftFoldingPassBase<SaturatingShiftFoldingPass> {
  using impl::SaturatingShiftFoldingPassBase<
      SaturatingShiftFoldingPass>::SaturatingShiftFoldingPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldI64SaturatingRightShift>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
