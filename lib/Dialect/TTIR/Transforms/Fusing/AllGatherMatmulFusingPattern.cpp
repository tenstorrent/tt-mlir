// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/AllGatherMatmulFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"

#include <atomic>
#include <numeric>
#include <optional>
#include <type_traits>

namespace mlir::tt::ttir::fusing {

namespace {

// Name of the composite this fusion emits. Must match the key registered in
// TTNNResolveComposites' composite registry.
constexpr llvm::StringLiteral kCompositeName =
    "all_gather_minimal_matmul_async";

std::string getUniqueDecompName() {
  static std::atomic<uint64_t> counter{0};
  return "all_gather_minimal_matmul_async_decomp_" +
         std::to_string(counter.fetch_add(1));
}

// Coefficient for the addcmul epilogue's scalar slot, which computes
//   result = addcmul_input1 + scalar * proj * addcmul_input2
// The gated residual we fuse carries no coefficient, so this is the identity
// 1.0 (see the addcmul pattern for the full derivation).
constexpr double kAddcmulScalar = 1.0;

// True if the all_gather gathers the last axis of its input. The fused
// minimal-matmul kernel only supports gathering the contraction (last) dim of
// the activation, so gathers on any other axis must not fuse.
bool gathersLastAxis(AllGatherOp allGatherOp) {
  auto inputType =
      mlir::cast<RankedTensorType>(allGatherOp.getInput().getType());
  int64_t rank = inputType.getRank();
  int64_t gatherDim = allGatherOp.getAllGatherDim();
  if (gatherDim < 0) {
    gatherDim += rank;
  }
  return gatherDim == rank - 1;
}

// True if `v` is a per-channel (row-broadcast) tensor: every dim except the
// last is 1, e.g. `[1, N]` or `[1, 1, N]`. The fused addcmul epilogue applies
// the gate per-channel, broadcasting it across the row (M) dim; a full
// `[M, N]` gate would be silently collapsed to its first row, so only a
// row-broadcast gate may fuse.
bool isRowBroadcast(mlir::Value v) {
  auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!type) {
    return false;
  }
  return llvm::all_of(type.getShape().drop_back(),
                      [](int64_t dim) { return dim == 1; });
}

// True if `reshape` only inserts or removes a leading unit dimension, e.g.
// `[M, N] <-> [1, M, N]`. DiT lowers the projection in 2D then unsqueezes
// a batch dim before the gated residual; that reshape must not block addcmul.
bool isLeadingUnitDimReshape(ReshapeOp reshapeOp) {
  auto inType = mlir::dyn_cast<RankedTensorType>(reshapeOp.getInput().getType());
  auto outType =
      mlir::dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
  if (!inType || !outType) {
    return false;
  }
  ArrayRef<int64_t> inShape = inType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  if (outType.getRank() == inType.getRank() + 1 && outShape.front() == 1 &&
      outShape.drop_front() == inShape) {
    return true;
  }
  if (inType.getRank() == outType.getRank() + 1 && inShape.front() == 1 &&
      inShape.drop_front() == outShape) {
    return true;
  }
  return false;
}

// Skip a one-use leading-unit reshape so matchers see the 2D projection
// underneath. Returns `v` unchanged when there is no such reshape.
Value skipLeadingUnitReshape(Value v) {
  auto reshapeOp = v.getDefiningOp<ReshapeOp>();
  if (!reshapeOp || !reshapeOp.getResult().hasOneUse() ||
      !isLeadingUnitDimReshape(reshapeOp)) {
    return v;
  }
  return reshapeOp.getInput();
}

// True if this matmul/linear result flows into a gated-residual epilogue
// (optionally through a leading-unit reshape, then a multiply then an add),
// which the addcmul pattern folds in whole.
template <typename MatmulLikeOp>
bool feedsGatedResidualEpilogue(MatmulLikeOp matmulOp) {
  if (!matmulOp.getResult().hasOneUse()) {
    return false;
  }
  Operation *user = *matmulOp.getResult().getUsers().begin();
  if (auto reshapeOp = mlir::dyn_cast<ReshapeOp>(user)) {
    if (!isLeadingUnitDimReshape(reshapeOp) ||
        !reshapeOp.getResult().hasOneUse()) {
      return false;
    }
    user = *reshapeOp.getResult().getUsers().begin();
  }
  auto mulOp = mlir::dyn_cast<MultiplyOp>(user);
  if (!mulOp || !mulOp.getResult().hasOneUse()) {
    return false;
  }
  return mlir::isa<AddOp>(*mulOp.getResult().getUsers().begin());
}

// Swap the last two dims of `input` so a `transpose_b` weight `[N, K]` becomes
// the `[K, N]` layout the fused kernel expects. Lives outside the composite
// (and is typically const-eval'd for parameter weights).
Value createTransposePermute(OpBuilder &builder, Location loc, Value input) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  SmallVector<int64_t> permutation(rank);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[rank - 2], permutation[rank - 1]);

  SmallVector<int64_t> permutedShape(inputType.getShape());
  std::swap(permutedShape[rank - 2], permutedShape[rank - 1]);
  auto permutedType = RankedTensorType::get(
      permutedShape, inputType.getElementType(), inputType.getEncoding());
  return builder.create<PermuteOp>(loc, permutedType, input, permutation);
}

Value reshapeToType(OpBuilder &builder, Location loc, Value input,
                     RankedTensorType resultType) {
  if (input.getType() == resultType) {
    return input;
  }
  SmallVector<int32_t> shapeI32(resultType.getShape().begin(),
                                resultType.getShape().end());
  return builder.create<ReshapeOp>(loc, resultType, input,
                                    builder.getI32ArrayAttr(shapeI32));
}

// Squeeze leading unit dims until `v` has `targetRank`. Used to capture a
// 3D residual/gate against a 2D projection without changing addcmul math.
Value squeezeLeadingUnitToRank(OpBuilder &builder, Location loc, Value v,
                                int64_t targetRank) {
  auto type = mlir::cast<RankedTensorType>(v.getType());
  if (type.getRank() == targetRank) {
    return v;
  }
  if (type.getRank() < targetRank) {
    return Value();
  }
  ArrayRef<int64_t> shape = type.getShape();
  int64_t leading = type.getRank() - targetRank;
  if (!llvm::all_of(shape.take_front(leading),
                     [](int64_t dim) { return dim == 1; })) {
    return Value();
  }
  auto squeezedType =
      RankedTensorType::get(shape.drop_front(leading), type.getElementType(),
                             type.getEncoding());
  return reshapeToType(builder, loc, v, squeezedType);
}

void clearTransposeAttrs(Operation *op) {
  if (auto matmulOp = mlir::dyn_cast<MatmulOp>(op)) {
    matmulOp.setTransposeA(false);
    matmulOp.setTransposeB(false);
  } else if (auto linearOp = mlir::dyn_cast<LinearOp>(op)) {
    linearOp.setTransposeA(false);
    linearOp.setTransposeB(false);
  }
}

int64_t getArrayI64(ArrayAttr attr, unsigned i) {
  return mlir::cast<IntegerAttr>(attr[i]).getInt();
}

// True if `slice` takes a full-extent contiguous range on every dim except
// the last, where it takes `[begin, end)` with step 1.
bool isContiguousLastDimSlice(SliceStaticOp sliceOp, RankedTensorType inputType,
                               int64_t &begin, int64_t &end) {
  int64_t rank = inputType.getRank();
  ArrayAttr begins = sliceOp.getBeginsAttr();
  ArrayAttr ends = sliceOp.getEndsAttr();
  ArrayAttr steps = sliceOp.getStepAttr();
  if (static_cast<int64_t>(begins.size()) != rank ||
      static_cast<int64_t>(ends.size()) != rank ||
      static_cast<int64_t>(steps.size()) != rank) {
    return false;
  }
  ArrayRef<int64_t> shape = inputType.getShape();
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (getArrayI64(begins, i) != 0 || getArrayI64(ends, i) != shape[i] ||
        getArrayI64(steps, i) != 1) {
      return false;
    }
  }
  if (getArrayI64(steps, rank - 1) != 1) {
    return false;
  }
  begin = getArrayI64(begins, rank - 1);
  end = getArrayI64(ends, rank - 1);
  return begin >= 0 && end > begin && end <= shape.back();
}

// If every user of `proj` (optionally through a leading-unit reshape) is a
// last-dim slice, and those slices partition N into C>=2 equal parts with no
// overlap/gap, return them in last-dim order. Otherwise empty.
//
// Wan self-attn QKV is the motivating case: one fused `to_qkv` linear to
// `3*head_dim`, then three equal last-dim slices. Unequal Q vs K/V widths
// stay unchunked (chunks=1).
struct EqualLastDimChunks {
  SmallVector<SliceStaticOp> slices;
  ReshapeOp throughReshape;
};

std::optional<EqualLastDimChunks> matchEqualLastDimChunks(Value proj) {
  EqualLastDimChunks result;
  Value sliced = proj;

  auto collectSlices = [&](Value src) -> bool {
    result.slices.clear();
    if (src.use_empty()) {
      return false;
    }
    for (Operation *user : src.getUsers()) {
      auto sliceOp = mlir::dyn_cast<SliceStaticOp>(user);
      if (!sliceOp || sliceOp.getInput() != src) {
        return false;
      }
      result.slices.push_back(sliceOp);
    }
    return !result.slices.empty();
  };

  if (!collectSlices(proj)) {
    if (!proj.hasOneUse()) {
      return std::nullopt;
    }
    auto reshapeOp = mlir::dyn_cast<ReshapeOp>(*proj.getUsers().begin());
    if (!reshapeOp || !isLeadingUnitDimReshape(reshapeOp) ||
        !collectSlices(reshapeOp.getResult())) {
      return std::nullopt;
    }
    result.throughReshape = reshapeOp;
    sliced = reshapeOp.getResult();
  }

  auto slicedType = mlir::dyn_cast<RankedTensorType>(sliced.getType());
  if (!slicedType || slicedType.getRank() < 1) {
    return std::nullopt;
  }

  struct Chunk {
    int64_t begin;
    int64_t end;
    SliceStaticOp op;
  };
  SmallVector<Chunk> chunks;
  chunks.reserve(result.slices.size());
  for (SliceStaticOp sliceOp : result.slices) {
    int64_t begin = 0;
    int64_t end = 0;
    if (!isContiguousLastDimSlice(sliceOp, slicedType, begin, end)) {
      return std::nullopt;
    }
    chunks.push_back({begin, end, sliceOp});
  }

  if (chunks.size() < 2) {
    return std::nullopt;
  }

  llvm::sort(chunks, [](const Chunk &a, const Chunk &b) {
    return a.begin < b.begin;
  });

  int64_t n = slicedType.getShape().back();
  int64_t width = chunks.front().end - chunks.front().begin;
  if (width <= 0 || n % static_cast<int64_t>(chunks.size()) != 0 ||
      n / static_cast<int64_t>(chunks.size()) != width ||
      chunks.front().begin != 0 || chunks.back().end != n) {
    return std::nullopt;
  }
  for (unsigned i = 1; i < chunks.size(); ++i) {
    if (chunks[i].end - chunks[i].begin != width ||
        chunks[i].begin != chunks[i - 1].end) {
      return std::nullopt;
    }
  }

  result.slices.clear();
  for (const Chunk &chunk : chunks) {
    result.slices.push_back(chunk.op);
  }
  return result;
}

// Last-dim slices of `proj` that the chunks=C fallback must reconstruct.
SmallVector<Value>
createLastDimChunkSlices(OpBuilder &builder, Location loc, Value proj,
                          ArrayRef<SliceStaticOp> slices) {
  auto projType = mlir::cast<RankedTensorType>(proj.getType());
  int64_t rank = projType.getRank();
  ArrayRef<int64_t> shape = projType.getShape();
  SmallVector<Value> results;
  results.reserve(slices.size());
  SmallVector<int32_t> steps(rank, 1);
  for (SliceStaticOp sliceOp : slices) {
    ArrayAttr origBegins = sliceOp.getBeginsAttr();
    ArrayAttr origEnds = sliceOp.getEndsAttr();
    int64_t begin = getArrayI64(origBegins, origBegins.size() - 1);
    int64_t end = getArrayI64(origEnds, origEnds.size() - 1);
    SmallVector<int32_t> begins(rank, 0);
    SmallVector<int32_t> ends(shape.begin(), shape.end());
    begins.back() = static_cast<int32_t>(begin);
    ends.back() = static_cast<int32_t>(end);
    SmallVector<int64_t> chunkShape(shape);
    chunkShape.back() = end - begin;
    auto chunkType =
        RankedTensorType::get(chunkShape, projType.getElementType(),
                              projType.getEncoding());
    results.push_back(builder.create<SliceStaticOp>(
        loc, chunkType, proj, builder.getI32ArrayAttr(begins),
        builder.getI32ArrayAttr(ends), builder.getI32ArrayAttr(steps)));
  }
  return results;
}

func::FuncOp buildDecompositionFunc(
    OpBuilder &builder, Location loc, ArrayRef<Value> captures,
    AllGatherOp allGatherOp, Operation *projOp, bool clearProjTranspose,
    bool hasAddcmul, TypeRange resultTypes,
    ArrayRef<SliceStaticOp> chunkSlices = {}) {
  auto argTypes =
      llvm::map_to_vector(captures, [](Value v) { return v.getType(); });
  auto funcOp = func::FuncOp::create(
      loc, getUniqueDecompName(),
      builder.getFunctionType(argTypes, resultTypes));
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(block, block->end());

  // Captures: input, weight, [bias], [residual, gate]. Map the original
  // projection's A/B (and bias) onto those args so clone() rewires them even
  // when the weight capture is a permute of the original B.
  IRMapping mapping;
  mapping.map(allGatherOp.getInput(), block->getArgument(0));
  if (auto matmulOp = mlir::dyn_cast<MatmulOp>(projOp)) {
    mapping.map(matmulOp.getB(), block->getArgument(1));
  } else if (auto linearOp = mlir::dyn_cast<LinearOp>(projOp)) {
    mapping.map(linearOp.getB(), block->getArgument(1));
    if (linearOp.getBias()) {
      mapping.map(linearOp.getBias(), block->getArgument(2));
    }
  }

  Operation *clonedGather = fb.clone(*allGatherOp, mapping);
  mapping.map(allGatherOp.getResult(), clonedGather->getResult(0));
  Operation *clonedProj = fb.clone(*projOp, mapping);
  if (clearProjTranspose) {
    clearTransposeAttrs(clonedProj);
  }

  Value result = clonedProj->getResult(0);
  if (hasAddcmul) {
    // residual/gate are the last two captures; they are already squeezed to
    // the projection rank so the fallback is a 2D mul+add.
    unsigned residualIdx = captures.size() - 2;
    Value residual = block->getArgument(residualIdx);
    Value gate = block->getArgument(residualIdx + 1);
    auto mulType = mlir::cast<RankedTensorType>(result.getType());
    auto mulOp = fb.create<MultiplyOp>(loc, mulType, result, gate);
    auto addOp = fb.create<AddOp>(loc, mulType, residual, mulOp.getResult());
    result = addOp.getResult();
  }

  if (chunkSlices.empty()) {
    fb.create<func::ReturnOp>(loc, result);
    return funcOp;
  }

  SmallVector<Value> chunkResults =
      createLastDimChunkSlices(fb, loc, result, chunkSlices);
  fb.create<func::ReturnOp>(loc, chunkResults);
  return funcOp;
}

SmallVector<NamedAttribute> buildCompositeAttrs(OpBuilder &rewriter,
                                               AllGatherOp allGatherOp,
                                               bool hasBias, bool hasAddcmul,
                                               int32_t chunks = 1) {
  mlir::MLIRContext *ctx = rewriter.getContext();
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(
      StringAttr::get(ctx, "all_gather_dim"),
      rewriter.getSI32IntegerAttr(allGatherOp.getAllGatherDim()));
  attrs.emplace_back(StringAttr::get(ctx, "cluster_axis"),
                     rewriter.getUI32IntegerAttr(allGatherOp.getClusterAxis()));
  attrs.emplace_back(StringAttr::get(ctx, "has_bias"),
                     rewriter.getBoolAttr(hasBias));
  attrs.emplace_back(StringAttr::get(ctx, "has_addcmul"),
                     rewriter.getBoolAttr(hasAddcmul));
  if (hasAddcmul) {
    attrs.emplace_back(StringAttr::get(ctx, "scalar"),
                       rewriter.getF32FloatAttr(kAddcmulScalar));
  }
  // Omit chunks=1 so existing dumps / CHECKs stay unchanged; C>1 is the
  // QKV split the fused kernel implements natively.
  if (chunks > 1) {
    attrs.emplace_back(StringAttr::get(ctx, "chunks"),
                       rewriter.getSI32IntegerAttr(chunks));
  }
  return attrs;
}

} // namespace

// Match, with no gated-residual epilogue, and fold into the composite:
//
//   proj = matmul(all_gather(input), weight) + bias
//
// where `+ bias` applies only to the linear variant (matmul has no bias).
// If a `residual + gate * proj` epilogue follows, defer to
// AllGatherMatmulAddcmulFusing so the whole thing folds at once.
// `transpose_b` is materialized as a permute of the weight to `[K, N]`;
// `transpose_a` is rejected because the kernel gathers A's last dim.
template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulFusing<MatmulLikeOp>::matchAndRewrite(
    MatmulLikeOp matmulOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(matmulOp)) {
    return mlir::failure();
  }

  AllGatherOp allGatherOp =
      matmulOp.getA().template getDefiningOp<AllGatherOp>();
  if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // The fused kernel only gathers the matmul's contraction (last) dim.
  if (!gathersLastAxis(allGatherOp)) {
    return mlir::failure();
  }

  if (matmulOp.getTransposeA()) {
    return mlir::failure();
  }

  // Let the addcmul pattern fold the whole gated-residual epilogue instead.
  if (feedsGatedResidualEpilogue(matmulOp)) {
    return mlir::failure();
  }

  Value bias;
  if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
    bias = matmulOp.getBias();
  }

  bool transposeB = matmulOp.getTransposeB();
  Value weight = matmulOp.getB();
  if (transposeB) {
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    if (weightType.getRank() < 2) {
      return mlir::failure();
    }
    rewriter.setInsertionPoint(matmulOp);
    weight = createTransposePermute(rewriter, matmulOp.getLoc(), weight);
  }

  auto projType = mlir::cast<RankedTensorType>(matmulOp.getResult().getType());

  std::optional<EqualLastDimChunks> qkvChunks =
      matchEqualLastDimChunks(matmulOp.getResult());
  int32_t chunks = qkvChunks ? static_cast<int32_t>(qkvChunks->slices.size())
                             : 1;

  SmallVector<Type> resultTypes;
  if (chunks > 1) {
    SmallVector<int64_t> chunkShape(projType.getShape());
    chunkShape.back() = projType.getShape().back() / chunks;
    auto chunkType = RankedTensorType::get(
        chunkShape, projType.getElementType(), projType.getEncoding());
    resultTypes.assign(chunks, chunkType);
  } else {
    resultTypes.push_back(projType);
  }

  // Captures feed the composite/decomposition in order: input, weight, [bias].
  SmallVector<Value> captures{allGatherOp.getInput(), weight};
  if (bias) {
    captures.push_back(bias);
  }

  Operation *anchor = matmulOp.getOperation();
  ModuleOp moduleOp = anchor->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, matmulOp.getLoc(), captures, allGatherOp, anchor,
      /*clearProjTranspose=*/transposeB, /*hasAddcmul=*/false, resultTypes,
      chunks > 1 ? ArrayRef<SliceStaticOp>(qkvChunks->slices)
                  : ArrayRef<SliceStaticOp>());
  moduleBuilder.insert(decompFunc);

  rewriter.setInsertionPoint(matmulOp);
  if (chunks == 1) {
    rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
        anchor, TypeRange{projType}, captures,
        rewriter.getStringAttr(kCompositeName),
        FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
        DictionaryAttr::get(rewriter.getContext(),
                             buildCompositeAttrs(rewriter, allGatherOp,
                                                 bias != nullptr,
                                                 /*hasAddcmul=*/false)));
    return mlir::success();
  }

  auto compositeOp = rewriter.create<ttcore::CompositeOp>(
      matmulOp.getLoc(), resultTypes, captures,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      DictionaryAttr::get(rewriter.getContext(),
                           buildCompositeAttrs(rewriter, allGatherOp,
                                               bias != nullptr,
                                               /*hasAddcmul=*/false, chunks)));

  for (auto [i, sliceOp] : llvm::enumerate(qkvChunks->slices)) {
    Value chunk = reshapeToType(
        rewriter, sliceOp.getLoc(), compositeOp.getResult(i),
        mlir::cast<RankedTensorType>(sliceOp.getResult().getType()));
    rewriter.replaceOp(sliceOp, chunk);
  }
  if (qkvChunks->throughReshape &&
      qkvChunks->throughReshape->use_empty()) {
    rewriter.eraseOp(qkvChunks->throughReshape);
  }
  rewriter.eraseOp(anchor);
  return mlir::success();
}

template <typename MatmulLikeOp>
mlir::LogicalResult AllGatherMatmulAddcmulFusing<MatmulLikeOp>::matchAndRewrite(
    AddOp addOp, mlir::PatternRewriter &rewriter) const {
  // Don't re-fuse the primitive ops we cloned into a decomposition body.
  if (utils::isInsideCompositeDecomposition(addOp)) {
    return mlir::failure();
  }

  // Match the DiT gated residual:
  //
  //   result = residual + gate * proj,
  //   where proj = matmul(all_gather(input), weight) + bias
  //   (`+ bias` applies only to the linear variant; matmul has no bias)
  //
  // walking backwards from the anchor `add`:  add -> multiply ->
  // [leading-unit reshape] -> matmul -> all_gather. Both the add and the
  // multiply are commutative, so we try each operand order.
  //
  // This maps to tt-metal's `addcmul` epilogue, whose fixed formula is
  //
  //   result = addcmul_input1 + scalar * proj * addcmul_input2
  //
  // The gated residual carries no coefficient in front of the product, so
  // `scalar` is the multiplicative identity (kAddcmulScalar == 1.0):
  //
  //   residual + 1.0 * proj * gate  ==  residual + gate * proj
  //
  // matching tt-metal, where every DiT call site passes 1.0; the slot stays
  // configurable only because the underlying addcmul kernel is general.

  // add: one operand is the `gate * proj` multiply, the other is the residual.
  MultiplyOp gateMulOp = addOp.getLhs().getDefiningOp<MultiplyOp>();
  mlir::Value residual = addOp.getRhs();
  if (!gateMulOp) {
    gateMulOp = addOp.getRhs().getDefiningOp<MultiplyOp>();
    residual = addOp.getLhs();
  }
  if (!gateMulOp || !gateMulOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // multiply: one operand is the (possibly reshaped) projection, the other
  // is the gate.
  auto matchProj = [](Value v) -> MatmulLikeOp {
    Value skipped = skipLeadingUnitReshape(v);
    return skipped.getDefiningOp<MatmulLikeOp>();
  };
  MatmulLikeOp projOp = matchProj(gateMulOp.getLhs());
  mlir::Value gate = gateMulOp.getRhs();
  if (!projOp) {
    projOp = matchProj(gateMulOp.getRhs());
    gate = gateMulOp.getLhs();
  }
  if (!projOp || !projOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // The fused addcmul epilogue applies the gate per-channel (broadcast across
  // the M/row dim). A full `[M, N]` gate would be silently collapsed to its
  // first row, so leave the full-gate case unfused (it stays as the primitive
  // matmul + multiply + add).
  if (!isRowBroadcast(gate)) {
    return mlir::failure();
  }

  AllGatherOp allGatherOp = projOp.getA().template getDefiningOp<AllGatherOp>();
  if (!allGatherOp || !allGatherOp.getResult().hasOneUse()) {
    return mlir::failure();
  }

  // The fused kernel only gathers the matmul's contraction (last) dim.
  if (!gathersLastAxis(allGatherOp)) {
    return mlir::failure();
  }

  if (projOp.getTransposeA()) {
    return mlir::failure();
  }

  Value bias;
  if constexpr (std::is_same_v<MatmulLikeOp, LinearOp>) {
    bias = projOp.getBias();
  }

  bool transposeB = projOp.getTransposeB();
  rewriter.setInsertionPoint(addOp);
  Value weight = projOp.getB();
  if (transposeB) {
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    if (weightType.getRank() < 2) {
      return mlir::failure();
    }
    weight = createTransposePermute(rewriter, addOp.getLoc(), weight);
  }

  auto projType = mlir::cast<RankedTensorType>(projOp.getResult().getType());
  Value squeezedResidual = squeezeLeadingUnitToRank(
      rewriter, addOp.getLoc(), residual, projType.getRank());
  Value squeezedGate =
      squeezeLeadingUnitToRank(rewriter, addOp.getLoc(), gate, projType.getRank());
  if (!squeezedResidual || !squeezedGate) {
    return mlir::failure();
  }
  if (!isRowBroadcast(squeezedGate)) {
    return mlir::failure();
  }

  // Captures feed the composite/decomposition in order:
  //   input, weight, [bias], residual, gate.
  SmallVector<Value> captures{allGatherOp.getInput(), weight};
  if (bias) {
    captures.push_back(bias);
  }
  captures.push_back(squeezedResidual);
  captures.push_back(squeezedGate);

  ModuleOp moduleOp = addOp->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  func::FuncOp decompFunc = buildDecompositionFunc(
      moduleBuilder, addOp.getLoc(), captures, allGatherOp,
      projOp.getOperation(), /*clearProjTranspose=*/transposeB,
      /*hasAddcmul=*/true, TypeRange{projType});
  moduleBuilder.insert(decompFunc);

  auto compositeOp = rewriter.create<ttcore::CompositeOp>(
      addOp.getLoc(), TypeRange{projType}, captures,
      rewriter.getStringAttr(kCompositeName),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      DictionaryAttr::get(rewriter.getContext(),
                           buildCompositeAttrs(rewriter, allGatherOp,
                                                bias != nullptr,
                                                /*hasAddcmul=*/true)));

  // The fused kernel writes the 2D projection layout. If the original add
  // was 3D (`[1, M, N]`), restore that for downstream users.
  auto addType = mlir::cast<RankedTensorType>(addOp.getResult().getType());
  Value result = reshapeToType(rewriter, addOp.getLoc(),
                                 compositeOp.getResult(0), addType);
  rewriter.replaceOp(addOp, result);
  return mlir::success();
}

// Explicit template instantiations.
template class AllGatherMatmulFusing<MatmulOp>;
template class AllGatherMatmulFusing<LinearOp>;
template class AllGatherMatmulAddcmulFusing<MatmulOp>;
template class AllGatherMatmulAddcmulFusing<LinearOp>;

} // namespace mlir::tt::ttir::fusing
