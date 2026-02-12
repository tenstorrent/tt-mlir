// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFOLDCONSTANTRESHAPEBROADCAST
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Check if a reshape op's result is consumed only by elementwise binary ops
// (add, sub, mul, div, pow) or broadcast ops.
static bool isConsumedByElementwiseBinaryOrBroadcastOp(ReshapeOp reshapeOp) {
  for (Operation *user : reshapeOp->getUsers()) {
    if (!isa<ttir::AddOp, ttir::SubtractOp, ttir::MultiplyOp, ttir::DivOp,
             ttir::PowOp, ttir::BroadcastOp>(user)) {
      return false;
    }
  }
  return !reshapeOp->getUsers().empty();
}

// Trace through broadcast/reshape chains to find a splat constant.
// Returns the splat DenseElementsAttr if found, nullptr otherwise.
// This handles patterns like:
//   reshape(broadcast(reshape(constant))) where constant is a splat.
static DenseElementsAttr findSplatConstantThroughChain(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return nullptr;
  }

  // Direct constant case.
  if (auto constantOp = dyn_cast<ConstantOp>(defOp)) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (denseAttr && denseAttr.isSplat()) {
      return denseAttr;
    }
    return nullptr;
  }

  // Trace through reshape - a splat reshaped is still a splat.
  if (auto reshapeOp = dyn_cast<ReshapeOp>(defOp)) {
    return findSplatConstantThroughChain(reshapeOp.getInput());
  }

  // Trace through broadcast - a splat broadcasted is still a splat.
  if (auto broadcastOp = dyn_cast<BroadcastOp>(defOp)) {
    return findSplatConstantThroughChain(broadcastOp.getInput());
  }

  return nullptr;
}

// Helper to create a splat DenseElementsAttr with a new type.
// Returns nullptr if the element type is not supported (only Float/Integer).
static DenseElementsAttr createSplatWithNewType(DenseElementsAttr splatAttr,
                                                RankedTensorType newType) {
  Type elementType = splatAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    return DenseElementsAttr::get(newType, splatAttr.getSplatValue<APFloat>());
  }
  if (isa<IntegerType>(elementType)) {
    return DenseElementsAttr::get(newType, splatAttr.getSplatValue<APInt>());
  }
  return nullptr;
}

// Pattern to fold reshape of a splat constant (direct or through a chain of
// reshape/broadcast ops) into a new splat constant with the target shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<f32>}
//   %reshaped = ttir.reshape(%const) -> tensor<1x1xf32>
// Or chained:
//   %const = ttir.constant() {value = dense<2.0> : tensor<f32>}
//   %reshaped1 = ttir.reshape(%const) -> tensor<1x1x1xf32>
//   %broadcast = ttir.broadcast(%reshaped1) -> tensor<1x128x360xf32>
//   %reshaped2 = ttir.reshape(%broadcast) -> tensor<128x360xf32>
//
// Only handles splat constants consumed by elementwise binary ops or broadcast.
class FoldSplatConstantReshapePattern : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // Only fold if consumed by elementwise binary ops or broadcast.
    if (!isConsumedByElementwiseBinaryOrBroadcastOp(reshapeOp)) {
      return failure();
    }

    // Try to find a splat constant through the chain (handles direct constant
    // too).
    DenseElementsAttr splatAttr =
        findSplatConstantThroughChain(reshapeOp.getInput());
    if (!splatAttr) {
      return failure();
    }

    // Get the target type from the reshape op's result.
    auto newType = cast<RankedTensorType>(reshapeOp.getType());

    // Create a new splat constant with the target shape.
    auto newDenseAttr = createSplatWithNewType(splatAttr, newType);
    if (!newDenseAttr) {
      return failure();
    }

    // Replace the reshape op with the new constant.
    rewriter.replaceOpWithNewOp<ConstantOp>(reshapeOp, newType, newDenseAttr);
    return success();
  }
};

// Pattern to fold broadcast of a splat constant (direct or through a chain of
// reshape/broadcast ops) into a new splat constant with the broadcasted shape.
//
// Matches patterns like:
//   %const = ttir.constant() {value = dense<2.0> : tensor<1x1xf32>}
//   %broadcast = ttir.broadcast(%const) -> tensor<128x360xf32>
// Or chained:
//   %const = ttir.constant() {value = dense<2.0> : tensor<f32>}
//   %reshaped = ttir.reshape(%const) -> tensor<1x1xf32>
//   %broadcast = ttir.broadcast(%reshaped) -> tensor<128x360xf32>
class FoldSplatConstantBroadcastPattern : public OpRewritePattern<BroadcastOp> {
public:
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    // Try to find a splat constant through the chain (handles direct constant
    // too).
    DenseElementsAttr splatAttr =
        findSplatConstantThroughChain(broadcastOp.getInput());
    if (!splatAttr) {
      return failure();
    }

    // Get the target type from the broadcast op's result.
    auto newType = cast<RankedTensorType>(broadcastOp.getType());

    // Create a new splat constant with the target shape.
    auto newDenseAttr = createSplatWithNewType(splatAttr, newType);
    if (!newDenseAttr) {
      return failure();
    }

    // Replace the broadcast op with the new constant.
    rewriter.replaceOpWithNewOp<ConstantOp>(broadcastOp, newType, newDenseAttr);
    return success();
  }
};

class TTIRFoldConstantReshapeBroadcast
    : public impl::TTIRFoldConstantReshapeBroadcastBase<
          TTIRFoldConstantReshapeBroadcast> {
public:
  using impl::TTIRFoldConstantReshapeBroadcastBase<
      TTIRFoldConstantReshapeBroadcast>::TTIRFoldConstantReshapeBroadcastBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldSplatConstantReshapePattern,
                 FoldSplatConstantBroadcastPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));

    if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  }
};

} // namespace
} // namespace mlir::tt::ttir
