// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/DitFusedDistributedRmsnormFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include <atomic>

namespace mlir::tt::ttir::fusing {

namespace {

std::string getUniqueDecompName() {
  static std::atomic<uint64_t> counter{0};
  return "dit_fused_distributed_rmsnorm_decomp_" +
         std::to_string(counter.fetch_add(1));
}

// Walk a unique-use reshape/permute chain starting from `root`.
SmallVector<Operation *> collectReshapePermuteChain(Value root) {
  SmallVector<Operation *> chain;
  Value cur = root;
  while (cur.hasOneUse()) {
    Operation *user = *cur.getUsers().begin();
    if (isa<ReshapeOp, PermuteOp>(user) && user->getNumResults() == 1) {
      chain.push_back(user);
      cur = user->getResult(0);
      continue;
    }
    break;
  }
  return chain;
}

bool matchMetalHeadsLayout(RankedTensorType inputType,
                           RankedTensorType outputType, int64_t &numHeads) {
  ArrayRef<int64_t> in = inputType.getShape();
  ArrayRef<int64_t> out = outputType.getShape();
  if (out.size() != 4 || out[0] != 1 || out[1] <= 1) {
    return false;
  }

  int64_t batch = 1;
  int64_t seq = 0;
  int64_t hidden = 0;
  if (inputType.getRank() == 3) {
    if (in[0] != 1) {
      return false;
    }
    seq = in[1];
    hidden = in[2];
  } else if (inputType.getRank() == 4 && in[0] == 1) {
    batch = in[1];
    seq = in[2];
    hidden = in[3];
  } else {
    return false;
  }

  numHeads = out[1];
  if (hidden % numHeads != 0) {
    return false;
  }
  int64_t headDim = hidden / numHeads;
  return out[2] == batch * seq && out[3] == headDim;
}

// Metal `dit_fused_distributed_rmsnorm` reads `weight.logical_shape()[-2]`,
// so γ must be rank >= 2 (`[1, H]` broadcast). Torch RMSNorm is `[H]`.
// Rank-up here, before TTNNLayout, so tilize sees `[1, H]` (BatchNorm /
// AdamW style). A TILE reshape of `[H]` at resolve/runtime is a host crash.
Value unsqueezeRank1WeightToBroadcast(OpBuilder &builder, Location loc,
                                      Value weight) {
  auto type = mlir::cast<RankedTensorType>(weight.getType());
  if (type.getRank() != 1) {
    return weight;
  }
  SmallVector<int64_t, 2> shape2D = {1, type.getShape()[0]};
  SmallVector<int32_t, 2> shapeI32 = {
      1, static_cast<int32_t>(type.getShape()[0])};
  auto resultType = RankedTensorType::get(shape2D, type.getElementType(),
                                          type.getEncoding());
  return builder
      .create<ReshapeOp>(loc, resultType, weight,
                         builder.getI32ArrayAttr(shapeI32))
      .getResult();
}

// Fallback decomp still uses `distributed_rms_norm`, which requires 1D γ.
Value squeezeBroadcastWeightTo1D(OpBuilder &builder, Location loc,
                                 Value weight) {
  auto type = mlir::cast<RankedTensorType>(weight.getType());
  if (type.getRank() != 2 || type.getShape()[0] != 1) {
    return weight;
  }
  SmallVector<int64_t, 1> shape1D = {type.getShape()[1]};
  SmallVector<int32_t, 1> shapeI32 = {
      static_cast<int32_t>(type.getShape()[1])};
  auto resultType = RankedTensorType::get(shape1D, type.getElementType(),
                                          type.getEncoding());
  return builder
      .create<ReshapeOp>(loc, resultType, weight,
                         builder.getI32ArrayAttr(shapeI32))
      .getResult();
}

func::FuncOp buildDecompFunc(OpBuilder &builder, Location loc,
                             DistributedRMSNormOp rmsOp, Value compositeWeight,
                             ArrayRef<Operation *> chain,
                             RankedTensorType resultType) {
  auto inputType = mlir::cast<RankedTensorType>(rmsOp.getInput().getType());
  auto weightType = mlir::cast<RankedTensorType>(compositeWeight.getType());

  auto funcType = builder.getFunctionType({inputType, weightType}, {resultType});
  auto funcOp = func::FuncOp::create(loc, getUniqueDecompName(), funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(builder.getContext());
  fb.setInsertionPointToStart(block);

  Value input = block->getArgument(0);
  Value weight =
      squeezeBroadcastWeightTo1D(fb, loc, block->getArgument(1));

  auto rms = fb.create<DistributedRMSNormOp>(
      loc, rmsOp.getType(), input, weight, /*residual=*/Value(),
      rmsOp.getClusterAxisAttr(), rmsOp.getEpsilonAttr());

  IRMapping mapping;
  mapping.map(rmsOp.getResult(), rms.getResult());
  Value last = rms.getResult();
  for (Operation *op : chain) {
    Operation *cloned = fb.clone(*op, mapping);
    last = cloned->getResult(0);
  }
  fb.create<func::ReturnOp>(loc, ValueRange{last});
  return funcOp;
}

} // namespace

mlir::LogicalResult DitFusedDistributedRmsnormFusingPattern::matchAndRewrite(
    DistributedRMSNormOp srcOp, mlir::PatternRewriter &rewriter) const {
  if (utils::isInsideCompositeDecomposition(srcOp)) {
    return failure();
  }
  if (!srcOp.getWeight() || srcOp.getResidual()) {
    return rewriter.notifyMatchFailure(
        srcOp, "requires weight and no residual for DiT Q/K RMS fusion");
  }

  auto inputType = mlir::cast<RankedTensorType>(srcOp.getInput().getType());
  if (inputType.getShape().back() %
          static_cast<int64_t>(mlir::tt::ttnn::TILE_HEIGHT) !=
      0) {
    return rewriter.notifyMatchFailure(
        srcOp, "hidden dim must be a multiple of tile height");
  }

  SmallVector<Operation *> chain =
      collectReshapePermuteChain(srcOp.getResult());
  if (chain.empty()) {
    return rewriter.notifyMatchFailure(
        srcOp, "expected unique-use reshape/permute chain to split heads");
  }

  Operation *lastOp = chain.back();
  auto resultType = mlir::cast<RankedTensorType>(lastOp->getResult(0).getType());
  int64_t numHeads = 0;
  if (!matchMetalHeadsLayout(inputType, resultType, numHeads)) {
    return rewriter.notifyMatchFailure(
        srcOp, "reshape/permute chain must produce [1, heads, seq, head_dim]");
  }

  Value metalWeight = unsqueezeRank1WeightToBroadcast(
      rewriter, srcOp.getLoc(), srcOp.getWeight());

  auto moduleOp = srcOp->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  auto decompFunc = buildDecompFunc(moduleBuilder, srcOp.getLoc(), srcOp,
                                    metalWeight, chain, resultType);
  moduleBuilder.insert(decompFunc);

  SmallVector<NamedAttribute> attrs = {
      rewriter.getNamedAttr("cluster_axis",
                            rewriter.getI32IntegerAttr(static_cast<int32_t>(
                                srcOp.getClusterAxis()))),
      rewriter.getNamedAttr("epsilon", srcOp.getEpsilonAttr()),
      rewriter.getNamedAttr("num_heads_per_device",
                            rewriter.getI32IntegerAttr(
                                static_cast<int32_t>(numHeads))),
      rewriter.getNamedAttr("per_head_norm", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("has_bias", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("has_rope", rewriter.getBoolAttr(false)),
  };

  SmallVector<Value> inputs = {srcOp.getInput(), metalWeight};
  auto composite = rewriter.create<ttcore::CompositeOp>(
      lastOp->getLoc(), TypeRange{resultType}, inputs,
      rewriter.getStringAttr("dit_fused_distributed_rmsnorm"),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      DictionaryAttr::get(rewriter.getContext(), attrs));

  rewriter.replaceOp(lastOp, composite.getResults());
  for (Operation *op : llvm::reverse(chain)) {
    if (op != lastOp && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }
  if (srcOp->use_empty()) {
    rewriter.eraseOp(srcOp);
  }
  return success();
}

} // namespace mlir::tt::ttir::fusing
