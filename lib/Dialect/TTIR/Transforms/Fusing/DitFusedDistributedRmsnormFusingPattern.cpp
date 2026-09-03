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

#include "llvm/ADT/SmallVector.h"

#include <atomic>

namespace mlir::tt::ttir::fusing {

namespace {

constexpr int64_t kLlamaTransMatTile = 32;

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

Value skipReshapeBroadcast(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<ReshapeOp, BroadcastOp>(defOp)) {
      v = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

// The half-rotation form widens a `head_dim/2` cache to `head_dim` as
// `cat(c, c)`. Metal wants the pair-interleaved `[c0, c0, c1, c1, ...]`
// instead, which resolve builds from the narrow cache. Hand the narrow one
// to the composite so that conversion has something to work with.
Value unwrapDuplicatedHalfCache(Value v) {
  v = skipReshapeBroadcast(v);
  auto concat = dyn_cast_or_null<ConcatOp>(v.getDefiningOp());
  if (!concat || concat.getInputs().size() != 2 ||
      concat.getInputs()[0] != concat.getInputs()[1]) {
    return v;
  }
  auto type = mlir::cast<RankedTensorType>(concat.getType());
  if (concat.getDim() != type.getRank() - 1) {
    return v;
  }
  return skipReshapeBroadcast(concat.getInputs()[0]);
}

std::optional<std::pair<int64_t, int64_t>>
getSliceOnDim(SliceStaticOp sliceOp, int64_t targetDim) {
  auto inputType = mlir::cast<RankedTensorType>(sliceOp.getInput().getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  ArrayAttr begins = sliceOp.getBegins();
  ArrayAttr ends = sliceOp.getEnds();
  ArrayAttr steps = sliceOp.getStep();
  if (static_cast<int64_t>(begins.size()) != rank) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < rank; ++i) {
    int64_t begin = mlir::cast<IntegerAttr>(begins[i]).getInt();
    int64_t end = mlir::cast<IntegerAttr>(ends[i]).getInt();
    int64_t step = mlir::cast<IntegerAttr>(steps[i]).getInt();
    if (step != 1) {
      return std::nullopt;
    }
    if (i == targetDim) {
      continue;
    }
    if (begin != 0 || end != inputShape[i]) {
      return std::nullopt;
    }
  }
  int64_t begin = mlir::cast<IntegerAttr>(begins[targetDim]).getInt();
  int64_t end = mlir::cast<IntegerAttr>(ends[targetDim]).getInt();
  return std::make_pair(begin, end);
}

// Llama 32x32 pair-rotation TILE used by metal fused RMS RoPE.
Value createLlamaTransformationMat(OpBuilder &builder, Location loc,
                                   Type elemType) {
  auto matType = RankedTensorType::get(
      {1, 1, kLlamaTransMatTile, kLlamaTransMatTile}, elemType);
  SmallVector<APFloat> values(
      kLlamaTransMatTile * kLlamaTransMatTile,
      APFloat::getZero(APFloat::BFloat()));
  APFloat one(APFloat::BFloat(), "1");
  APFloat negOne(APFloat::BFloat(), "-1");
  for (int64_t i = 0; i < kLlamaTransMatTile; i += 2) {
    values[i * kLlamaTransMatTile + (i + 1)] = one;
    values[(i + 1) * kLlamaTransMatTile + i] = negOne;
  }
  auto dense = DenseElementsAttr::get(matType, values);
  return builder.create<ConstantOp>(loc, matType, dense).getResult();
}

struct HalfRotationRoPE {
  SmallVector<Operation *> deinterleaveOps;
  SmallVector<Operation *> ropeOps;
  PermuteOp headsPermute;
  Value deinterleaved;
  Value cosCache;
  Value sinCache;
  AddOp addOp;
  int64_t numHeads = 0;
  int64_t seq = 0;
  int64_t headDim = 0;
};

// RMS → reshape [1,S,H,D/2,2] → permute {0,1,2,4,3} → reshape [1,S,H,D].
std::optional<HalfRotationRoPE>
matchDeinterleave(DistributedRMSNormOp rmsOp, int64_t hidden) {
  if (!rmsOp.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto reshape5 = dyn_cast<ReshapeOp>(*rmsOp.getResult().getUsers().begin());
  if (!reshape5) {
    return std::nullopt;
  }
  auto s5 = mlir::cast<RankedTensorType>(reshape5.getType()).getShape();
  if (s5.size() != 5 || s5[0] != 1 || s5[4] != 2) {
    return std::nullopt;
  }
  int64_t seq = s5[1];
  int64_t numHeads = s5[2];
  int64_t halfDim = s5[3];
  if (numHeads <= 1 || halfDim * 2 * numHeads != hidden) {
    return std::nullopt;
  }
  if (!reshape5.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto perm = dyn_cast<PermuteOp>(*reshape5.getResult().getUsers().begin());
  if (!perm || perm.getPermutation() != ArrayRef<int64_t>({0, 1, 2, 4, 3})) {
    return std::nullopt;
  }
  if (!perm.getResult().hasOneUse()) {
    return std::nullopt;
  }
  auto reshape4 = dyn_cast<ReshapeOp>(*perm.getResult().getUsers().begin());
  if (!reshape4) {
    return std::nullopt;
  }
  auto s4 = mlir::cast<RankedTensorType>(reshape4.getType()).getShape();
  if (s4.size() != 4 || s4[0] != 1 || s4[1] != seq || s4[2] != numHeads ||
      s4[3] != halfDim * 2) {
    return std::nullopt;
  }
  HalfRotationRoPE m;
  m.deinterleaveOps = {reshape5, perm, reshape4};
  m.deinterleaved = reshape4.getResult();
  m.numHeads = numHeads;
  m.seq = seq;
  m.headDim = halfDim * 2;
  return m;
}

// deinterleaved * cos + concat(-second, first) * sin
bool matchRotateHalf(HalfRotationRoPE &m) {
  MultiplyOp mulCos;
  SliceStaticOp sliceFirst;
  SliceStaticOp sliceSecond;
  for (Operation *user : m.deinterleaved.getUsers()) {
    if (auto mul = dyn_cast<MultiplyOp>(user)) {
      if (!mulCos) {
        mulCos = mul;
      }
      continue;
    }
    if (auto slice = dyn_cast<SliceStaticOp>(user)) {
      auto range = getSliceOnDim(slice, /*targetDim=*/3);
      if (!range) {
        return false;
      }
      if (range->first == 0 && range->second == m.headDim / 2) {
        sliceFirst = slice;
      } else if (range->first == m.headDim / 2 && range->second == m.headDim) {
        sliceSecond = slice;
      }
      continue;
    }
    return false;
  }
  if (!mulCos || !sliceFirst || !sliceSecond) {
    return false;
  }

  Value cosOperand = mulCos.getLhs() == m.deinterleaved ? mulCos.getRhs()
                                                        : mulCos.getLhs();
  if (skipReshapeBroadcast(cosOperand) == skipReshapeBroadcast(m.deinterleaved)) {
    return false;
  }

  if (!sliceSecond.getResult().hasOneUse()) {
    return false;
  }
  auto neg = dyn_cast<NegOp>(*sliceSecond.getResult().getUsers().begin());
  if (!neg || !neg.getResult().hasOneUse()) {
    return false;
  }
  auto concat = dyn_cast<ConcatOp>(*neg.getResult().getUsers().begin());
  if (!concat || concat.getDim() != 3 || concat.getInputs().size() != 2) {
    return false;
  }
  if (concat.getInputs()[0] != neg.getResult() ||
      concat.getInputs()[1] != sliceFirst.getResult()) {
    return false;
  }
  if (!concat.getResult().hasOneUse()) {
    return false;
  }
  auto mulSin = dyn_cast<MultiplyOp>(*concat.getResult().getUsers().begin());
  if (!mulSin) {
    return false;
  }
  Value sinOperand = mulSin.getLhs() == concat.getResult() ? mulSin.getRhs()
                                                           : mulSin.getLhs();

  AddOp addOp;
  for (Operation *user : mulCos.getResult().getUsers()) {
    if (auto add = dyn_cast<AddOp>(user)) {
      if (add.getLhs() == mulSin.getResult() ||
          add.getRhs() == mulSin.getResult()) {
        addOp = add;
        break;
      }
    }
  }
  if (!addOp) {
    for (Operation *user : mulSin.getResult().getUsers()) {
      if (auto add = dyn_cast<AddOp>(user)) {
        if (add.getLhs() == mulCos.getResult() ||
            add.getRhs() == mulCos.getResult()) {
          addOp = add;
          break;
        }
      }
    }
  }
  if (!addOp) {
    return false;
  }

  m.cosCache = unwrapDuplicatedHalfCache(cosOperand);
  m.sinCache = unwrapDuplicatedHalfCache(sinOperand);
  m.addOp = addOp;
  // `addOp` is deliberately absent: it is the tail of the subgraph, so one of
  // the two rewrite branches always hands it to `replaceOp`, which frees it.
  // Keeping it here would leave a dangling pointer for `eraseOpsIfDead`.
  m.ropeOps = {mulCos, sliceFirst, sliceSecond, neg, concat, mulSin};

  if (addOp.getResult().hasOneUse()) {
    if (auto perm = dyn_cast<PermuteOp>(*addOp.getResult().getUsers().begin())) {
      if (perm.getPermutation() == ArrayRef<int64_t>({0, 2, 1, 3})) {
        auto outTy = mlir::cast<RankedTensorType>(perm.getType());
        if (outTy.getShape() ==
            ArrayRef<int64_t>({1, m.numHeads, m.seq, m.headDim})) {
          m.headsPermute = perm;
        }
      }
    }
  }
  return true;
}

func::FuncOp buildHeadsDecompFunc(OpBuilder &builder, Location loc,
                                  DistributedRMSNormOp rmsOp, Value weight,
                                  ArrayRef<Type> inputTypes,
                                  RankedTensorType resultType,
                                  int64_t numHeads) {
  auto funcType = builder.getFunctionType(inputTypes, {resultType});
  auto funcOp = func::FuncOp::create(loc, getUniqueDecompName(), funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(builder.getContext());
  fb.setInsertionPointToStart(block);

  Value input = block->getArgument(0);
  Value w = squeezeBroadcastWeightTo1D(fb, loc, block->getArgument(1));
  auto rms = fb.create<DistributedRMSNormOp>(
      loc, rmsOp.getType(), input, w, /*residual=*/Value(),
      rmsOp.getClusterAxisAttr(), rmsOp.getEpsilonAttr());

  auto rmsType = mlir::cast<RankedTensorType>(rms.getType());
  ArrayRef<int64_t> in = rmsType.getShape();
  int64_t seq = in.size() == 3 ? in[1] : in[2];
  int64_t hidden = in.back();
  int64_t headDim = hidden / numHeads;
  SmallVector<int64_t, 4> seqMajor = {1, seq, numHeads, headDim};
  SmallVector<int32_t, 4> seqMajorI32 = {
      1, static_cast<int32_t>(seq), static_cast<int32_t>(numHeads),
      static_cast<int32_t>(headDim)};
  auto seqMajorType =
      RankedTensorType::get(seqMajor, rmsType.getElementType());
  Value reshaped =
      fb.create<ReshapeOp>(loc, seqMajorType, rms.getResult(),
                           fb.getI32ArrayAttr(seqMajorI32))
          .getResult();

  auto resultShape = resultType.getShape();
  if (resultShape == ArrayRef<int64_t>(seqMajor)) {
    fb.create<func::ReturnOp>(loc, ValueRange{reshaped});
    return funcOp;
  }
  Value last =
      fb.create<PermuteOp>(loc, resultType, reshaped,
                           ArrayRef<int64_t>({0, 2, 1, 3}))
          .getResult();
  fb.create<func::ReturnOp>(loc, ValueRange{last});
  return funcOp;
}

// The narrow cache feeding the composite must be expressible as
// [1, seq, mid, head_dim/2] with `mid` broadcastable against the heads dim,
// otherwise the decomposition cannot rebuild the widened cache. Returns `mid`.
std::optional<int64_t> getCacheHeadsDim(Value cache, int64_t seq,
                                        int64_t headDim, int64_t numHeads) {
  auto type = mlir::dyn_cast<RankedTensorType>(cache.getType());
  if (!type || type.getRank() < 1 || type.getShape().back() != headDim / 2) {
    return std::nullopt;
  }
  int64_t numElements = type.getNumElements();
  int64_t plane = seq * (headDim / 2);
  if (plane == 0 || numElements % plane != 0) {
    return std::nullopt;
  }
  int64_t mid = numElements / plane;
  if (mid != 1 && mid != numHeads) {
    return std::nullopt;
  }
  return mid;
}

ArrayAttr makeSliceOnLastDim(OpBuilder &builder, ArrayRef<int64_t> shape,
                             int64_t begin, int64_t end, bool isEnds) {
  SmallVector<int32_t> values;
  for (int64_t i = 0, e = shape.size(); i < e; ++i) {
    if (i + 1 == e) {
      values.push_back(static_cast<int32_t>(isEnds ? end : begin));
      continue;
    }
    values.push_back(isEnds ? static_cast<int32_t>(shape[i]) : 0);
  }
  return builder.getI32ArrayAttr(values);
}

// Rebuilds the fused op op-by-op: distributed RMS, the pair deinterleave, the
// half rotation against the widened caches, and the head-major permute. The
// composite is only correct to inline if this stays faithful to the subgraph
// the pattern consumed.
func::FuncOp buildRopeDecompFunc(OpBuilder &builder, Location loc,
                                 DistributedRMSNormOp rmsOp, Value weight,
                                 ArrayRef<Type> inputTypes,
                                 RankedTensorType resultType,
                                 const HalfRotationRoPE &rope, int64_t cosMid,
                                 int64_t sinMid) {
  auto funcType = builder.getFunctionType(inputTypes, {resultType});
  auto funcOp = func::FuncOp::create(loc, getUniqueDecompName(), funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(builder.getContext());
  fb.setInsertionPointToStart(block);

  Value w = squeezeBroadcastWeightTo1D(fb, loc, block->getArgument(1));
  auto rms = fb.create<DistributedRMSNormOp>(
      loc, rmsOp.getType(), block->getArgument(0), w, /*residual=*/Value(),
      rmsOp.getClusterAxisAttr(), rmsOp.getEpsilonAttr());

  Type elemType =
      mlir::cast<RankedTensorType>(rms.getType()).getElementType();
  const int64_t seq = rope.seq;
  const int64_t heads = rope.numHeads;
  const int64_t headDim = rope.headDim;
  const int64_t halfDim = headDim / 2;

  auto reshapeTo = [&](Value v, ArrayRef<int64_t> shape, Type elem) {
    SmallVector<int32_t> shapeI32(llvm::map_range(
        shape, [](int64_t d) { return static_cast<int32_t>(d); }));
    return fb.create<ReshapeOp>(loc, RankedTensorType::get(shape, elem), v,
                                fb.getI32ArrayAttr(shapeI32))
        .getResult();
  };

  // Deinterleave the [even, odd] pairs into [first half, second half].
  Value paired = reshapeTo(rms.getResult(), {1, seq, heads, halfDim, 2},
                           elemType);
  Value swapped =
      fb.create<PermuteOp>(
            loc,
            RankedTensorType::get({1, seq, heads, 2, halfDim}, elemType),
            paired, ArrayRef<int64_t>({0, 1, 2, 4, 3}))
          .getResult();
  auto headsType =
      RankedTensorType::get({1, seq, heads, headDim}, elemType);
  Value x = reshapeTo(swapped, headsType.getShape(), elemType);

  // The half-rotation form consumes cat(c, c) over the last dim.
  auto widenCache = [&](Value cache, int64_t mid) {
    Type cacheElem =
        mlir::cast<RankedTensorType>(cache.getType()).getElementType();
    Value narrow = reshapeTo(cache, {1, seq, mid, halfDim}, cacheElem);
    auto wideType =
        RankedTensorType::get({1, seq, mid, headDim}, cacheElem);
    return fb.create<ConcatOp>(loc, wideType, ValueRange{narrow, narrow},
                               fb.getSI32IntegerAttr(3))
        .getResult();
  };
  Value cos = widenCache(block->getArgument(2), cosMid);
  Value sin = widenCache(block->getArgument(3), sinMid);

  // rotate_half(x) = concat(-x[..., d/2:], x[..., :d/2])
  auto halfType =
      RankedTensorType::get({1, seq, heads, halfDim}, elemType);
  ArrayAttr steps = fb.getI32ArrayAttr({1, 1, 1, 1});
  auto first = fb.create<SliceStaticOp>(
      loc, halfType, x,
      makeSliceOnLastDim(fb, headsType.getShape(), 0, halfDim, false),
      makeSliceOnLastDim(fb, headsType.getShape(), 0, halfDim, true), steps);
  auto second = fb.create<SliceStaticOp>(
      loc, halfType, x,
      makeSliceOnLastDim(fb, headsType.getShape(), halfDim, headDim, false),
      makeSliceOnLastDim(fb, headsType.getShape(), halfDim, headDim, true),
      steps);
  auto negSecond = fb.create<NegOp>(loc, halfType, second.getResult());
  auto rotated = fb.create<ConcatOp>(
      loc, headsType, ValueRange{negSecond.getResult(), first.getResult()},
      fb.getSI32IntegerAttr(3));

  // MultiplyOp broadcasts the caches over the heads dim implicitly.
  auto xCos = fb.create<MultiplyOp>(loc, headsType, x, cos);
  auto rotSin =
      fb.create<MultiplyOp>(loc, headsType, rotated.getResult(), sin);
  Value result =
      fb.create<AddOp>(loc, headsType, xCos.getResult(), rotSin.getResult())
          .getResult();

  if (resultType.getShape() != headsType.getShape()) {
    result = fb.create<PermuteOp>(loc, resultType, result,
                                  ArrayRef<int64_t>({0, 2, 1, 3}))
                 .getResult();
  }
  fb.create<func::ReturnOp>(loc, ValueRange{result});
  return funcOp;
}

ttcore::CompositeOp
emitComposite(PatternRewriter &rewriter, Location loc,
              DistributedRMSNormOp rmsOp, Value weight,
              ArrayRef<Value> extraInputs, RankedTensorType resultType,
              int64_t numHeads, bool hasRope, func::FuncOp decompFunc) {
  SmallVector<NamedAttribute> attrs = {
      rewriter.getNamedAttr("cluster_axis",
                            rewriter.getI32IntegerAttr(static_cast<int32_t>(
                                rmsOp.getClusterAxis()))),
      rewriter.getNamedAttr("epsilon", rmsOp.getEpsilonAttr()),
      rewriter.getNamedAttr("num_heads_per_device",
                            rewriter.getI32IntegerAttr(
                                static_cast<int32_t>(numHeads))),
      rewriter.getNamedAttr("per_head_norm", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("has_bias", rewriter.getBoolAttr(false)),
      rewriter.getNamedAttr("has_rope", rewriter.getBoolAttr(hasRope)),
  };
  SmallVector<Value> inputs = {rmsOp.getInput(), weight};
  inputs.append(extraInputs.begin(), extraInputs.end());
  return rewriter.create<ttcore::CompositeOp>(
      loc, TypeRange{resultType}, inputs,
      rewriter.getStringAttr("dit_fused_distributed_rmsnorm"),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      DictionaryAttr::get(rewriter.getContext(), attrs));
}

void eraseOpsIfDead(PatternRewriter &rewriter, ArrayRef<Operation *> ops) {
  for (Operation *op : llvm::reverse(ops)) {
    if (op && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }
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

  Value metalWeight = unsqueezeRank1WeightToBroadcast(
      rewriter, srcOp.getLoc(), srcOp.getWeight());
  auto moduleOp = srcOp->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());

  int64_t hidden = inputType.getShape().back();
  if (auto rope = matchDeinterleave(srcOp, hidden);
      rope && matchRotateHalf(*rope)) {
    std::optional<int64_t> cosMid = getCacheHeadsDim(
        rope->cosCache, rope->seq, rope->headDim, rope->numHeads);
    std::optional<int64_t> sinMid = getCacheHeadsDim(
        rope->sinCache, rope->seq, rope->headDim, rope->numHeads);
    if (!cosMid || !sinMid) {
      return rewriter.notifyMatchFailure(
          srcOp, "RoPE cache shape cannot be expressed in the decomposition");
    }

    RankedTensorType metalOutType = RankedTensorType::get(
        {1, rope->numHeads, rope->seq, rope->headDim},
        inputType.getElementType());
    Value transMat = createLlamaTransformationMat(
        rewriter, srcOp.getLoc(), inputType.getElementType());
    SmallVector<Value> extra = {rope->cosCache, rope->sinCache, transMat};
    SmallVector<Type> decompInputs = {
        inputType, mlir::cast<RankedTensorType>(metalWeight.getType()),
        mlir::cast<RankedTensorType>(rope->cosCache.getType()),
        mlir::cast<RankedTensorType>(rope->sinCache.getType()),
        mlir::cast<RankedTensorType>(transMat.getType())};
    auto decompFunc = buildRopeDecompFunc(
        moduleBuilder, srcOp.getLoc(), srcOp, metalWeight, decompInputs,
        metalOutType, *rope, *cosMid, *sinMid);
    moduleBuilder.insert(decompFunc);

    // The rewriter is positioned at the anchor (the RMS norm), but the cos/sin
    // caches are materialized further down the block. Emit the composite where
    // the subgraph ends so its operands dominate it.
    rewriter.setInsertionPoint(rope->headsPermute
                                   ? rope->headsPermute.getOperation()
                                   : rope->addOp.getOperation());

    if (rope->headsPermute) {
      auto composite =
          emitComposite(rewriter, rope->headsPermute.getLoc(), srcOp,
                        metalWeight, extra, metalOutType, rope->numHeads,
                        /*hasRope=*/true, decompFunc);
      rewriter.replaceOp(rope->headsPermute, composite.getResults());
      // The permute was the only user of the add. Erase it ahead of `ropeOps`
      // so its operands are already unused when they are visited.
      if (rope->addOp->use_empty()) {
        rewriter.eraseOp(rope->addOp);
      }
    } else {
      auto composite = emitComposite(
          rewriter, rope->addOp.getLoc(), srcOp, metalWeight, extra,
          metalOutType, rope->numHeads, /*hasRope=*/true, decompFunc);
      SmallVector<int64_t, 4> seqMajor = {1, rope->seq, rope->numHeads,
                                          rope->headDim};
      auto seqMajorType = RankedTensorType::get(seqMajor,
                                                inputType.getElementType());
      Value seqMajorVal =
          rewriter
              .create<PermuteOp>(rope->addOp.getLoc(), seqMajorType,
                                 composite.getResult(0),
                                 ArrayRef<int64_t>({0, 2, 1, 3}))
              .getResult();
      rewriter.replaceOp(rope->addOp, seqMajorVal);
    }
    eraseOpsIfDead(rewriter, rope->ropeOps);
    eraseOpsIfDead(rewriter, rope->deinterleaveOps);
    if (srcOp->use_empty()) {
      rewriter.eraseOp(srcOp);
    }
    return success();
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

  SmallVector<Type> decompInputs = {
      inputType, mlir::cast<RankedTensorType>(metalWeight.getType())};
  auto decompFunc =
      buildHeadsDecompFunc(moduleBuilder, srcOp.getLoc(), srcOp, metalWeight,
                           decompInputs, resultType, numHeads);
  moduleBuilder.insert(decompFunc);

  auto composite =
      emitComposite(rewriter, lastOp->getLoc(), srcOp, metalWeight, {},
                    resultType, numHeads, /*hasRope=*/false, decompFunc);
  rewriter.replaceOp(lastOp, composite.getResults());
  eraseOpsIfDead(rewriter, chain);
  if (srcOp->use_empty()) {
    rewriter.eraseOp(srcOp);
  }
  return success();
}

} // namespace mlir::tt::ttir::fusing
