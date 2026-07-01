// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFOLDTOMEMORYCONFIG
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

// Returns the TTNNLayoutAttr encoding of a value's ranked tensor type, or
// nullptr if it has none. to_memory_config encodes its target memory config
// in the result type's encoding (it has no memory-config attribute), so the
// layout encoding is the source of truth for both patterns below.
static TTNNLayoutAttr getTTNNLayout(mlir::Value value) {
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!tensorType) {
    return nullptr;
  }
  return mlir::dyn_cast<TTNNLayoutAttr>(tensorType.getEncoding());
}

// to_memory_config(x) whose input already has the target memory config is a
// no-op and folds away to its input.
class FoldIdentityToMemoryConfig
    : public mlir::OpRewritePattern<ToMemoryConfigOp> {
public:
  using mlir::OpRewritePattern<ToMemoryConfigOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ToMemoryConfigOp op,
                  mlir::PatternRewriter &rewriter) const override {
    TTNNLayoutAttr inputLayout = getTTNNLayout(op.getInput());
    TTNNLayoutAttr outputLayout = getTTNNLayout(op.getResult());
    if (!inputLayout || !outputLayout || inputLayout != outputLayout) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, op.getInput());
    return mlir::success();
  }
};

// Two consecutive to_memory_config ops collapse to a single one targeting the
// outer config, because the op depends only on (input data, target config) and
// not the path taken:
//
//   %a = to_memory_config(%x)   // -> cfgA
//   %b = to_memory_config(%a)   // -> cfgB     ==>   %b = to_memory_config(%x) // -> cfgB
//
// Combined with the identity pattern above, this also erases
// shard->stage->shard round-trips (where cfgB == the layout of %x).
class FoldConsecutiveToMemoryConfig
    : public mlir::OpRewritePattern<ToMemoryConfigOp> {
public:
  using mlir::OpRewritePattern<ToMemoryConfigOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ToMemoryConfigOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto producerOp = op.getInput().getDefiningOp<ToMemoryConfigOp>();
    if (!producerOp) {
      return mlir::failure();
    }

    // Preserve intentional DRAM staging: if there are ops between producer and
    // consumer, and the producer parks the tensor in DRAM while the consumer
    // moves it to L1, folding would bypass the DRAM staging that downstream
    // scheduling may rely on. Only fold then if the net change (producer input
    // -> consumer output) is a genuine no-op, i.e. the tensor round-trips back
    // to the exact same layout it started in and the staging is truly
    // redundant.
    Operation *nextOp = producerOp->getNextNode();
    bool hasOpsBetween = (nextOp != op.getOperation());
    if (hasOpsBetween) {
      TTNNLayoutAttr producerOut = getTTNNLayout(producerOp.getResult());
      TTNNLayoutAttr consumerOut = getTTNNLayout(op.getResult());
      if (producerOut && consumerOut &&
          producerOut.getBufferType() == BufferType::DRAM &&
          consumerOut.getBufferType() == BufferType::L1) {
        TTNNLayoutAttr producerIn = getTTNNLayout(producerOp.getInput());
        bool isRoundTrip = producerIn && producerIn == consumerOut;
        if (!isRoundTrip) {
          return mlir::failure();
        }
      }
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op.getInputMutable().set(producerOp.getInput()); });
    return mlir::success();
  }
};

class TTNNFoldToMemoryConfigPass
    : public impl::TTNNFoldToMemoryConfigBase<TTNNFoldToMemoryConfigPass> {
public:
  using impl::TTNNFoldToMemoryConfigBase<
      TTNNFoldToMemoryConfigPass>::TTNNFoldToMemoryConfigBase;

  void runOnOperation() final {
    if (!foldingEnabled) {
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<FoldIdentityToMemoryConfig, FoldConsecutiveToMemoryConfig>(
        &getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
