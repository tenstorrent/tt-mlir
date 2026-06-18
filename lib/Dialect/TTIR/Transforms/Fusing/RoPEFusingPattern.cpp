// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Fusing/RoPEFusingPattern.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include <atomic>

namespace mlir::tt::ttir::fusing {

namespace {

// Walk backward through transparent TM ops to find the original value.
// These ops don't change the semantic content, just type/shape/layout.
Value skipTMs(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp, ReshapeOp, BroadcastOp>(defOp)) {
      v = defOp->getOperand(0);
    } else {
      break;
    }
  }
  return v;
}

// Walk backward through BroadcastOp only, keeping typecast/reshape.
// This gives us a 4D value suitable as an op input.
Value skipBroadcastOnly(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (isa<BroadcastOp>(defOp)) {
      v = defOp->getOperand(0);
    } else {
      break;
    }
  }
  return v;
}

// Get slice parameters for a SliceStaticOp on a specific dimension.
// Returns (begin, end) on that dimension if the slice is identity on all
// other dimensions (begin=0, end=dimSize, step=1).
std::optional<std::pair<int64_t, int64_t>> getSliceOnDim(SliceStaticOp sliceOp,
                                                         int64_t targetDim) {
  auto inputType = mlir::cast<RankedTensorType>(sliceOp.getOperand().getType());
  auto inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();

  auto begins = sliceOp.getBegins();
  auto ends = sliceOp.getEnds();
  auto steps = sliceOp.getStep();

  for (int64_t i = 0; i < rank; ++i) {
    auto begin = mlir::cast<IntegerAttr>(begins[i]).getInt();
    auto end = mlir::cast<IntegerAttr>(ends[i]).getInt();
    auto step = mlir::cast<IntegerAttr>(steps[i]).getInt();

    if (step != 1) {
      return std::nullopt;
    }

    if (i == targetDim) {
      continue;
    }

    // Non-target dimensions must be identity slices.
    if (begin != 0 || end != inputShape[i]) {
      return std::nullopt;
    }
  }

  auto begin = mlir::cast<IntegerAttr>(begins[targetDim]).getInt();
  auto end = mlir::cast<IntegerAttr>(ends[targetDim]).getInt();
  return std::make_pair(begin, end);
}

// Check that a slice takes the first half [0, D/2) on the last dimension.
bool isFirstHalfSlice(SliceStaticOp sliceOp) {
  auto inputType = mlir::cast<RankedTensorType>(sliceOp.getOperand().getType());
  if (inputType.getRank() < 1) {
    return false;
  }
  int64_t lastDim = inputType.getRank() - 1;
  int64_t fullSize = inputType.getShape()[lastDim];
  if (fullSize % 2 != 0) {
    return false;
  }
  int64_t halfSize = fullSize / 2;
  auto params = getSliceOnDim(sliceOp, lastDim);
  return params && params->first == 0 && params->second == halfSize;
}

// Check that a slice takes the second half [D/2, D) on the last dimension.
bool isSecondHalfSlice(SliceStaticOp sliceOp) {
  auto inputType = mlir::cast<RankedTensorType>(sliceOp.getOperand().getType());
  if (inputType.getRank() < 1) {
    return false;
  }
  int64_t lastDim = inputType.getRank() - 1;
  int64_t fullSize = inputType.getShape()[lastDim];
  if (fullSize % 2 != 0) {
    return false;
  }
  int64_t halfSize = fullSize / 2;
  auto params = getSliceOnDim(sliceOp, lastDim);
  return params && params->first == halfSize && params->second == fullSize;
}

// Result of matching the rotate_half subgraph.
struct RotateHalfMatch {
  Value xSource;         // The original input tensor before slicing.
  ConcatOp concatOp;     // The concat that reassembles the rotated halves.
  SliceStaticOp hiSlice; // slice(x, D/2:)
  SliceStaticOp loSlice; // slice(x, :D/2)
  NegOp negOp;           // neg(slice(x, D/2:))
};

// Try to match the rotate_half subgraph:
//   concat(neg(slice(x, D/2:)), slice(x, :D/2))
std::optional<RotateHalfMatch> matchRotateHalf(Value rotatedValue) {
  auto concatOp = dyn_cast_or_null<ConcatOp>(rotatedValue.getDefiningOp());
  if (!concatOp) {
    return std::nullopt;
  }

  // Must concat on the last dimension with exactly 2 operands.
  auto resultType = mlir::cast<RankedTensorType>(concatOp.getType());
  if (concatOp.getDim() != resultType.getRank() - 1) {
    return std::nullopt;
  }
  if (concatOp.getNumOperands() != 2) {
    return std::nullopt;
  }

  Value firstArg = concatOp.getOperand(0);
  Value secondArg = concatOp.getOperand(1);

  // First operand must be neg(slice(x, D/2:)).
  auto negOp = dyn_cast_or_null<NegOp>(firstArg.getDefiningOp());
  if (!negOp) {
    return std::nullopt;
  }

  auto hiSlice =
      dyn_cast_or_null<SliceStaticOp>(negOp.getOperand().getDefiningOp());
  if (!hiSlice) {
    return std::nullopt;
  }

  // Second operand must be slice(x, :D/2).
  auto loSlice = dyn_cast_or_null<SliceStaticOp>(secondArg.getDefiningOp());
  if (!loSlice) {
    return std::nullopt;
  }

  // Both slices must come from the same source.
  if (hiSlice.getOperand() != loSlice.getOperand()) {
    return std::nullopt;
  }

  // hiSlice must be [D/2, D), loSlice must be [0, D/2).
  if (!isSecondHalfSlice(hiSlice) || !isFirstHalfSlice(loSlice)) {
    return std::nullopt;
  }

  RotateHalfMatch match;
  match.xSource = hiSlice.getOperand();
  match.concatOp = concatOp;
  match.hiSlice = hiSlice;
  match.loSlice = loSlice;
  match.negOp = negOp;
  return match;
}

// Given a MulOp, try to identify which operand is x (matching expectedXSource)
// and which is the cos/sin embedding.
// Checks direct equality first (x is often used directly), then falls back
// to walking through TMs. This mirrors TTNN's findCommonTMAncestor approach.
// Returns (x_operand, embedding_operand) or nullopt.
std::optional<std::pair<Value, Value>>
identifyXAndEmbedding(MultiplyOp mulOp, Value expectedXSource) {
  Value op0 = mulOp.getOperand(0);
  Value op1 = mulOp.getOperand(1);

  // Direct match first — handles the common case where x is used as-is
  // (even if it's the output of a reshape/typecast).
  if (op0 == expectedXSource) {
    return std::make_pair(op0, op1);
  }
  if (op1 == expectedXSource) {
    return std::make_pair(op1, op0);
  }

  // Fall back to tracing through TMs for cases where x is behind a
  // typecast in the cos branch (e.g. x_bf16 -> typecast -> x_f32).
  Value src0 = skipTMs(op0);
  Value src1 = skipTMs(op1);

  if (src0 == expectedXSource) {
    return std::make_pair(op0, op1);
  }
  if (src1 == expectedXSource) {
    return std::make_pair(op1, op0);
  }
  return std::nullopt;
}

// Get a 4D cos/sin input value suitable for the composite op.
// Walks back past broadcast but keeps typecast/reshape to ensure 4D.
Value get4DEmbeddingInput(Value embValue) {
  Value candidate = skipBroadcastOnly(embValue);
  auto candidateType = mlir::dyn_cast<RankedTensorType>(candidate.getType());
  if (candidateType && candidateType.getRank() == 4) {
    return candidate;
  }
  // If not 4D after skipping broadcast, use the broadcast output.
  return embValue;
}

// Generate a unique name for the RoPE decomposition function.
std::string getUniqueDecompName() {
  static std::atomic<uint64_t> counter{0};
  return "rotary_embedding_decomp_" + std::to_string(counter.fetch_add(1));
}

// Build a helper to create ArrayAttr of I32 for slice begins/ends/steps.
ArrayAttr makeSliceI32ArrayAttr(OpBuilder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t v : values) {
    attrs.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(v)));
  }
  return builder.getArrayAttr(attrs);
}

// Create a private func.func implementing the RoPE decomposition.
//
// Mirrors the TTNN RotaryEmbeddingDecompositionRewritePattern:
//
// When isSelfConcatCosSin is true (cos/sin are full-dim concat(half, half)):
//   Complex rotation form (fewer ops, no full-D multiplies):
//     result = concat(x_lo*cosH - x_hi*sinH, x_hi*cosH + x_lo*sinH)
//   where cosH/sinH are obtained by slicing cos/sin to [:D/2].
//   MultiplyOp implicit broadcasting handles shape mismatches.
//
// When isSelfConcatCosSin is false (cos/sin are full-dim):
//   Rotate_half form:
//     result = x * cos + rotate_half(x) * sin
//     where rotate_half(x) = concat(neg(x[D/2:]), x[:D/2])
func::FuncOp buildRoPEDecompositionFunc(OpBuilder &builder, Location loc,
                                        RankedTensorType inputType,
                                        RankedTensorType cosType,
                                        RankedTensorType sinType,
                                        RankedTensorType resultType,
                                        bool isSelfConcatCosSin) {
  auto funcName = getUniqueDecompName();
  auto funcType =
      builder.getFunctionType({inputType, cosType, sinType}, {resultType});

  auto funcOp = func::FuncOp::create(loc, funcName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  funcOp->setAttr(utils::kCompositeDecompositionAttr,
                  UnitAttr::get(builder.getContext()));

  Block *block = funcOp.addEntryBlock();
  OpBuilder fb(builder.getContext());
  fb.setInsertionPointToStart(block);

  Value input = block->getArgument(0);
  Value cos = block->getArgument(1);
  Value sin = block->getArgument(2);

  auto inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  int64_t lastDim = rank - 1;
  int64_t headDim = inputShape[lastDim];
  int64_t halfDim = headDim / 2;

  // Build slice shapes: same as input but with last dim = halfDim.
  SmallVector<int64_t> halfShape(inputShape);
  halfShape[lastDim] = halfDim;
  auto halfType = RankedTensorType::get(halfShape, inputType.getElementType());

  // Build slice attributes.
  SmallVector<int64_t> loBegins(rank, 0);
  SmallVector<int64_t> loEnds(inputShape);
  loEnds[lastDim] = halfDim;
  SmallVector<int64_t> steps(rank, 1);

  SmallVector<int64_t> hiBegins(rank, 0);
  hiBegins[lastDim] = halfDim;
  SmallVector<int64_t> hiEnds(inputShape);

  // x_lo = x[:D/2], x_hi = x[D/2:]
  auto xLo = fb.create<SliceStaticOp>(
      loc, halfType, input, makeSliceI32ArrayAttr(fb, loBegins),
      makeSliceI32ArrayAttr(fb, loEnds), makeSliceI32ArrayAttr(fb, steps));
  auto xHi = fb.create<SliceStaticOp>(
      loc, halfType, input, makeSliceI32ArrayAttr(fb, hiBegins),
      makeSliceI32ArrayAttr(fb, hiEnds), makeSliceI32ArrayAttr(fb, steps));

  if (isSelfConcatCosSin) {
    // Complex rotation form: cos/sin are full-dim concat(half, half).
    // Slice to get the half-D values, then use half-D ops only.
    // MultiplyOp implicit broadcasting handles any remaining shape mismatches.
    SmallVector<int64_t> cosShape(cosType.getShape());
    SmallVector<int64_t> cosHalfShape(cosShape);
    cosHalfShape[lastDim] = cosShape[lastDim] / 2;
    auto cosHalfType =
        RankedTensorType::get(cosHalfShape, cosType.getElementType());

    SmallVector<int64_t> cosLoBegins(rank, 0);
    SmallVector<int64_t> cosLoEnds(cosShape);
    cosLoEnds[lastDim] = cosShape[lastDim] / 2;
    SmallVector<int64_t> cosSteps(rank, 1);

    auto cosHalf = fb.create<SliceStaticOp>(
        loc, cosHalfType, cos, makeSliceI32ArrayAttr(fb, cosLoBegins),
        makeSliceI32ArrayAttr(fb, cosLoEnds),
        makeSliceI32ArrayAttr(fb, cosSteps));

    SmallVector<int64_t> sinShape(sinType.getShape());
    SmallVector<int64_t> sinHalfShape(sinShape);
    sinHalfShape[lastDim] = sinShape[lastDim] / 2;
    auto sinHalfType =
        RankedTensorType::get(sinHalfShape, sinType.getElementType());

    auto sinHalf = fb.create<SliceStaticOp>(
        loc, sinHalfType, sin, makeSliceI32ArrayAttr(fb, cosLoBegins),
        makeSliceI32ArrayAttr(fb, cosLoEnds),
        makeSliceI32ArrayAttr(fb, cosSteps));

    // first = x_lo*cos - x_hi*sin
    // MultiplyOp supports implicit broadcasting, no explicit broadcast needed.
    auto loCos = fb.create<MultiplyOp>(loc, halfType, xLo, cosHalf);
    auto hiSin = fb.create<MultiplyOp>(loc, halfType, xHi, sinHalf);
    auto first = fb.create<SubtractOp>(loc, halfType, loCos, hiSin);

    // second = x_hi*cos + x_lo*sin
    auto hiCos = fb.create<MultiplyOp>(loc, halfType, xHi, cosHalf);
    auto loSin = fb.create<MultiplyOp>(loc, halfType, xLo, sinHalf);
    auto second = fb.create<AddOp>(loc, halfType, hiCos, loSin);

    // result = concat(first, second)
    auto result =
        fb.create<ConcatOp>(loc, resultType, ValueRange{first, second},
                            fb.getSI32IntegerAttr(lastDim));
    fb.create<func::ReturnOp>(loc, ValueRange{result.getResult()});
  } else {
    // Rotate_half form.
    auto negHi = fb.create<NegOp>(loc, halfType, xHi);
    auto rotated = fb.create<ConcatOp>(loc, inputType, ValueRange{negHi, xLo},
                                       fb.getSI32IntegerAttr(lastDim));

    // MultiplyOp supports implicit broadcasting, no explicit broadcast needed.
    auto xCos = fb.create<MultiplyOp>(loc, resultType, input, cos);
    auto rotSin = fb.create<MultiplyOp>(loc, resultType, rotated, sin);
    auto result = fb.create<AddOp>(loc, resultType, xCos, rotSin);
    fb.create<func::ReturnOp>(loc, ValueRange{result.getResult()});
  }

  return funcOp;
}

// If `value` is concat(x, x, dim=last) — the half-D → full-D cos/sin
// self-duplication pattern — return the half-D input. Otherwise return nullopt.
// Mirrors TTNN's matchSelfConcatLastDim.
std::optional<Value> matchSelfConcatLastDim(Value value) {
  auto concatOp = dyn_cast_or_null<ConcatOp>(value.getDefiningOp());
  if (!concatOp || concatOp.getNumOperands() != 2 ||
      concatOp.getOperand(0) != concatOp.getOperand(1)) {
    return std::nullopt;
  }
  auto resultType = mlir::cast<RankedTensorType>(concatOp.getType());
  int64_t rank = resultType.getRank();
  int64_t lastDim = rank - 1;
  int64_t dim = concatOp.getDim();
  if (dim < 0) {
    dim += rank;
  }
  if (dim != lastDim) {
    return std::nullopt;
  }
  return concatOp.getOperand(0);
}

// Replace the anchor op with a ttcore.composite "rotary_embedding" that
// references a decomposition function.
// Detects self-concat cos/sin at the IR level (same as TTNN decomposition's
// matchSelfConcatLastDim) to choose the optimal decomposition form.
//
// The composite's cos/sin inputs are kept at full-dim (concat(half, half))
// so that ttnn::RotaryEmbeddingOp promotion (force-promote path) can still
// match the expected shapes. The decomposition body uses the half-dim
// operands directly via implicit broadcasting.
void replaceWithRoPEComposite(Operation *anchorOp,
                              mlir::PatternRewriter &rewriter, Value xSource,
                              Value cosInput, Value sinInput,
                              RankedTensorType resultType) {
  auto inputType = mlir::cast<RankedTensorType>(xSource.getType());
  auto cosType = mlir::cast<RankedTensorType>(cosInput.getType());
  auto sinType = mlir::cast<RankedTensorType>(sinInput.getType());

  // Detect self-concat cos/sin pattern regardless of which fusing pattern
  // matched (rotate_half or complex rotation can both produce self-concat).
  bool isSelfConcatCosSin = matchSelfConcatLastDim(cosInput).has_value() &&
                            matchSelfConcatLastDim(sinInput).has_value();

  // Find the parent module to insert the decomposition function.
  auto moduleOp = anchorOp->getParentOfType<ModuleOp>();

  // Build the decomposition function and insert it into the module.
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  auto decompFunc = buildRoPEDecompositionFunc(
      moduleBuilder, anchorOp->getLoc(), inputType, cosType, sinType,
      resultType, isSelfConcatCosSin);
  moduleBuilder.insert(decompFunc);

  // Create the ttcore.composite op.
  rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
      anchorOp, TypeRange{resultType}, ValueRange{xSource, cosInput, sinInput},
      rewriter.getStringAttr("rotary_embedding"),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      /*composite_attributes=*/nullptr);
}

} // namespace

//===----------------------------------------------------------------------===//
// RoPERotateHalfFusingPattern
//===----------------------------------------------------------------------===//

mlir::LogicalResult RoPERotateHalfFusingPattern::matchAndRewrite(
    AddOp srcOp, mlir::PatternRewriter &rewriter) const {

  // Don't fuse inside decomposition function bodies (infinite recursion).
  if (utils::isInsideCompositeDecomposition(srcOp)) {
    return failure();
  }

  // Both operands of the add must be multiplies.
  auto mul0 = dyn_cast_or_null<MultiplyOp>(srcOp.getOperand(0).getDefiningOp());
  auto mul1 = dyn_cast_or_null<MultiplyOp>(srcOp.getOperand(1).getDefiningOp());
  if (!mul0 || !mul1) {
    return failure();
  }

  // One multiply is x*cos, the other is rotate_half(x)*sin.
  // Try both orderings for which mul is the sin branch.
  MultiplyOp cosMul = nullptr;
  MultiplyOp sinMul = nullptr;
  std::optional<RotateHalfMatch> rotMatch;

  for (int attempt = 0; attempt < 2; ++attempt) {
    MultiplyOp candidateSinMul = (attempt == 0) ? mul1 : mul0;
    MultiplyOp candidateCosMul = (attempt == 0) ? mul0 : mul1;

    // Check both operands of candidateSinMul for rotate_half.
    for (int i = 0; i < 2; ++i) {
      rotMatch = matchRotateHalf(candidateSinMul->getOperand(i));
      if (rotMatch) {
        break;
      }
    }

    if (rotMatch) {
      sinMul = candidateSinMul;
      cosMul = candidateCosMul;
      break;
    }
  }

  if (!rotMatch) {
    return failure();
  }

  Value xSource = rotMatch->xSource;

  // Input must be 4D.
  auto inputType = mlir::cast<RankedTensorType>(xSource.getType());
  if (inputType.getRank() != 4) {
    return failure();
  }

  // In the cos branch, identify x and cos.
  auto cosXAndEmb = identifyXAndEmbedding(cosMul, xSource);
  if (!cosXAndEmb) {
    return failure();
  }
  auto [cosX, cosEmb] = *cosXAndEmb;

  // In the sin branch, the embedding is the operand that is NOT rotate_half.
  Value sinEmb;
  if (sinMul->getOperand(0) == rotMatch->concatOp.getResult()) {
    sinEmb = sinMul->getOperand(1);
  } else if (sinMul->getOperand(1) == rotMatch->concatOp.getResult()) {
    sinEmb = sinMul->getOperand(0);
  } else {
    return failure();
  }

  // Check single-use on critical intermediate ops to avoid incorrect fusion.
  if (!cosMul->hasOneUse() || !sinMul->hasOneUse() ||
      !rotMatch->concatOp->hasOneUse() || !rotMatch->negOp->hasOneUse()) {
    return failure();
  }

  // Get 4D cos/sin inputs for the op.
  Value cosInput = get4DEmbeddingInput(cosEmb);
  Value sinInput = get4DEmbeddingInput(sinEmb);

  auto resultType = mlir::cast<RankedTensorType>(srcOp.getType());
  replaceWithRoPEComposite(srcOp, rewriter, xSource, cosInput, sinInput,
                           resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// RoPEComplexRotationFusingPattern
//===----------------------------------------------------------------------===//

mlir::LogicalResult RoPEComplexRotationFusingPattern::matchAndRewrite(
    ConcatOp srcOp, mlir::PatternRewriter &rewriter) const {

  // Don't fuse inside decomposition function bodies (infinite recursion).
  if (utils::isInsideCompositeDecomposition(srcOp)) {
    return failure();
  }

  // Must concat on last dimension with exactly 2 operands.
  auto resultType = mlir::cast<RankedTensorType>(srcOp.getType());
  if (resultType.getRank() != 4) {
    return failure();
  }
  if (srcOp.getDim() != resultType.getRank() - 1) {
    return failure();
  }
  if (srcOp.getNumOperands() != 2) {
    return failure();
  }

  // Operand 0: subtract(x1*cos, x2*sin)
  auto subOp =
      dyn_cast_or_null<SubtractOp>(srcOp.getOperand(0).getDefiningOp());
  if (!subOp) {
    return failure();
  }

  // Operand 1: add(x2*cos, x1*sin)
  auto addOp = dyn_cast_or_null<AddOp>(srcOp.getOperand(1).getDefiningOp());
  if (!addOp) {
    return failure();
  }

  // All 4 operands of sub and add must be multiplies.
  auto subMul0 =
      dyn_cast_or_null<MultiplyOp>(subOp.getOperand(0).getDefiningOp());
  auto subMul1 =
      dyn_cast_or_null<MultiplyOp>(subOp.getOperand(1).getDefiningOp());
  auto addMul0 =
      dyn_cast_or_null<MultiplyOp>(addOp.getOperand(0).getDefiningOp());
  auto addMul1 =
      dyn_cast_or_null<MultiplyOp>(addOp.getOperand(1).getDefiningOp());
  if (!subMul0 || !subMul1 || !addMul0 || !addMul1) {
    return failure();
  }

  // Helper to find a SliceStaticOp operand of a MulOp.
  auto findSlice =
      [](MultiplyOp mulOp) -> std::optional<std::pair<SliceStaticOp, Value>> {
    for (int i = 0; i < 2; ++i) {
      Value op = mulOp->getOperand(i);
      Value other = mulOp->getOperand(1 - i);
      if (auto slice = dyn_cast_or_null<SliceStaticOp>(op.getDefiningOp())) {
        return std::make_pair(slice, other);
      }
    }
    return std::nullopt;
  };

  // Extract slice and embedding from each multiply.
  auto subMul0Info = findSlice(subMul0);
  auto subMul1Info = findSlice(subMul1);
  auto addMul0Info = findSlice(addMul0);
  auto addMul1Info = findSlice(addMul1);
  if (!subMul0Info || !subMul1Info || !addMul0Info || !addMul1Info) {
    return failure();
  }

  auto [subSlice0, subEmb0] = *subMul0Info;
  auto [subSlice1, subEmb1] = *subMul1Info;
  auto [addSlice0, addEmb0] = *addMul0Info;
  auto [addSlice1, addEmb1] = *addMul1Info;

  // All slices must come from the same source tensor.
  Value xSource = subSlice0.getOperand();
  if (subSlice1.getOperand() != xSource || addSlice0.getOperand() != xSource ||
      addSlice1.getOperand() != xSource) {
    return failure();
  }

  // Input must be 4D.
  auto inputType = mlir::cast<RankedTensorType>(xSource.getType());
  if (inputType.getRank() != 4) {
    return failure();
  }

  // sub: x1*cos - x2*sin => subSlice0 = x1 (first half), subSlice1 = x2
  // add: x2*cos + x1*sin => addSlice0 = x2, addSlice1 = x1
  if (!isFirstHalfSlice(subSlice0) || !isSecondHalfSlice(subSlice1)) {
    return failure();
  }
  if (!isSecondHalfSlice(addSlice0) || !isFirstHalfSlice(addSlice1)) {
    return failure();
  }

  // Verify cos is shared: subEmb0 and addEmb0 should trace to same source.
  Value cosSource = skipTMs(subEmb0);
  Value cosSourceFromAdd = skipTMs(addEmb0);
  if (cosSource != cosSourceFromAdd) {
    return failure();
  }

  // Verify sin is shared: subEmb1 and addEmb1 should trace to same source.
  Value sinSource = skipTMs(subEmb1);
  Value sinSourceFromAdd = skipTMs(addEmb1);
  if (sinSource != sinSourceFromAdd) {
    return failure();
  }

  // Check single-use on critical intermediates.
  if (!subOp->hasOneUse() || !addOp->hasOneUse() || !subMul0->hasOneUse() ||
      !subMul1->hasOneUse() || !addMul0->hasOneUse() || !addMul1->hasOneUse()) {
    return failure();
  }

  // cos/sin are half-dim. Create concat(half, half) as the composite's
  // cos/sin inputs so that ttnn::RotaryEmbeddingOp promotion can match
  // the expected full-dim shapes. The decomposition body will detect the
  // self-concat and use the half-dim operand directly.
  Value cosHalf = get4DEmbeddingInput(subEmb0);
  Value sinHalf = get4DEmbeddingInput(subEmb1);

  auto cosHalfType = mlir::cast<RankedTensorType>(cosHalf.getType());
  auto sinHalfType = mlir::cast<RankedTensorType>(sinHalf.getType());

  int64_t lastDim = resultType.getRank() - 1;

  SmallVector<int64_t> cosFullShape(cosHalfType.getShape());
  cosFullShape[lastDim] *= 2;
  auto cosFullType =
      RankedTensorType::get(cosFullShape, cosHalfType.getElementType());

  SmallVector<int64_t> sinFullShape(sinHalfType.getShape());
  sinFullShape[lastDim] *= 2;
  auto sinFullType =
      RankedTensorType::get(sinFullShape, sinHalfType.getElementType());

  auto cosFull = rewriter.create<ConcatOp>(
      srcOp.getLoc(), cosFullType, ValueRange{cosHalf, cosHalf},
      rewriter.getSI32IntegerAttr(lastDim));
  auto sinFull = rewriter.create<ConcatOp>(
      srcOp.getLoc(), sinFullType, ValueRange{sinHalf, sinHalf},
      rewriter.getSI32IntegerAttr(lastDim));

  replaceWithRoPEComposite(srcOp, rewriter, xSource, cosFull, sinFull,
                           resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// RoPEInterleavedPairFusingPattern
//===----------------------------------------------------------------------===//

namespace {

// Build a slice on one dim (begin, end, step), identity on all other dims.
static SliceStaticOp buildSliceOnDim(PatternRewriter &rewriter, Location loc,
                                     Value input, int64_t targetDim,
                                     int32_t begin, int32_t end,
                                     int32_t step = 1) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  int64_t rank = inputType.getRank();
  ArrayRef<int64_t> shape = inputType.getShape();

  SmallVector<int32_t> begins(rank, 0);
  SmallVector<int32_t> ends(rank);
  SmallVector<int32_t> steps(rank, 1);
  for (int64_t i = 0; i < rank; ++i) {
    ends[i] = static_cast<int32_t>(shape[i]);
  }
  begins[targetDim] = begin;
  ends[targetDim] = end;
  steps[targetDim] = step;

  SmallVector<int64_t> resultShape(shape);
  resultShape[targetDim] = llvm::divideCeil(end - begin, step);

  auto resultType =
      RankedTensorType::get(resultShape, inputType.getElementType());

  return rewriter.create<SliceStaticOp>(
      loc, resultType, input, rewriter.getI32ArrayAttr(begins),
      rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
}

// Helper: walk back through reshape/broadcast chain to find a SliceStaticOp.
// Returns the slice and the value it slices from.
static SliceStaticOp findSliceThroughTMs(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (auto slice = dyn_cast<SliceStaticOp>(defOp)) {
      return slice;
    }
    if (isa<ReshapeOp, BroadcastOp, TypecastOp>(defOp)) {
      v = defOp->getOperand(0);
      continue;
    }
    break;
  }
  return nullptr;
}

// For a slice on a 6D reshape of x at pair dim, returns the pair index
// (0 or 1) if the slice has the form [..., 0:1, 0:1] or [..., 0:1, 1:2]
// on the last two dims of a (..., 1, 2) shape. Returns nullopt otherwise.
static std::optional<int64_t> getPairIndex(SliceStaticOp slice) {
  auto inputType = mlir::cast<RankedTensorType>(slice.getOperand().getType());
  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return std::nullopt;
  }
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (inputShape[rank - 2] != 1 || inputShape[rank - 1] != 2) {
    return std::nullopt;
  }

  auto begins = slice.getBegins();
  auto ends = slice.getEnds();
  auto steps = slice.getStep();
  for (int64_t i = 0; i < rank; ++i) {
    auto step = mlir::cast<IntegerAttr>(steps[i]).getInt();
    if (step != 1) {
      return std::nullopt;
    }
    auto begin = mlir::cast<IntegerAttr>(begins[i]).getInt();
    auto end = mlir::cast<IntegerAttr>(ends[i]).getInt();
    if (i < rank - 1) {
      // All dims except the last must be identity (or [0:1] on size-1 dim).
      if (begin != 0 || end != inputShape[i]) {
        return std::nullopt;
      }
    }
  }
  auto lastBegin = mlir::cast<IntegerAttr>(begins[rank - 1]).getInt();
  auto lastEnd = mlir::cast<IntegerAttr>(ends[rank - 1]).getInt();
  if (lastBegin == 0 && lastEnd == 1) {
    return 0;
  }
  if (lastBegin == 1 && lastEnd == 2) {
    return 1;
  }
  return std::nullopt;
}

// For a slice on freqs_cis (..., D/2, 2, 2), returns the column index (0 or 1)
// if the slice picks one column of the trailing 2x2 (i.e. slices [..., :,
// c:c+1] where c is 0 or 1) and is identity elsewhere.
static std::optional<int64_t> getColumnIndex(SliceStaticOp slice) {
  auto inputType = mlir::cast<RankedTensorType>(slice.getOperand().getType());
  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return std::nullopt;
  }
  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (inputShape[rank - 1] != 2) {
    return std::nullopt;
  }

  auto begins = slice.getBegins();
  auto ends = slice.getEnds();
  auto steps = slice.getStep();
  for (int64_t i = 0; i < rank - 1; ++i) {
    auto step = mlir::cast<IntegerAttr>(steps[i]).getInt();
    auto begin = mlir::cast<IntegerAttr>(begins[i]).getInt();
    auto end = mlir::cast<IntegerAttr>(ends[i]).getInt();
    if (step != 1 || begin != 0 || end != inputShape[i]) {
      return std::nullopt;
    }
  }
  auto lastStep = mlir::cast<IntegerAttr>(steps[rank - 1]).getInt();
  auto lastBegin = mlir::cast<IntegerAttr>(begins[rank - 1]).getInt();
  auto lastEnd = mlir::cast<IntegerAttr>(ends[rank - 1]).getInt();
  if (lastStep != 1) {
    return std::nullopt;
  }
  if (lastBegin == 0 && lastEnd == 1) {
    return 0;
  }
  if (lastBegin == 1 && lastEnd == 2) {
    return 1;
  }
  return std::nullopt;
}

// Tracks one half of the matched pattern (cos branch or sin branch).
struct InterleavedBranch {
  MultiplyOp mulOp;
  SliceStaticOp xSlice;     // slice on the 6D reshape of x; pair index encoded
  SliceStaticOp freqsSlice; // slice on freqs_cis; column index encoded
  int64_t pairIdx;          // 0 or 1
  int64_t colIdx;           // 0 or 1
  Value xReshape6D;         // the 6D reshape of x (slice's input)
  Value freqsSrc;           // freqs_cis source tensor
};

// Try to interpret a MultiplyOp as one branch of the interleaved RoPE pattern.
// Returns the populated branch if both operands trace back to a pair-slice on
// x's 6D reshape and a column-slice on freqs_cis; nullopt otherwise.
static std::optional<InterleavedBranch> matchBranch(MultiplyOp mulOp) {
  for (int swap = 0; swap < 2; ++swap) {
    Value xSide = mulOp.getOperand(swap);
    Value freqsSide = mulOp.getOperand(1 - swap);

    SliceStaticOp xSlice = findSliceThroughTMs(xSide);
    SliceStaticOp freqsSlice = findSliceThroughTMs(freqsSide);
    if (!xSlice || !freqsSlice) {
      continue;
    }

    auto pairIdx = getPairIndex(xSlice);
    if (!pairIdx) {
      continue;
    }
    auto colIdx = getColumnIndex(freqsSlice);
    if (!colIdx) {
      continue;
    }

    InterleavedBranch br;
    br.mulOp = mulOp;
    br.xSlice = xSlice;
    br.freqsSlice = freqsSlice;
    br.pairIdx = *pairIdx;
    br.colIdx = *colIdx;
    br.xReshape6D = xSlice.getOperand();
    br.freqsSrc = freqsSlice.getOperand();
    return br;
  }
  return std::nullopt;
}

} // namespace

mlir::LogicalResult RoPEInterleavedPairFusingPattern::matchAndRewrite(
    AddOp addOp, mlir::PatternRewriter &rewriter) const {
  // 1. Both add operands must be multiplies.
  auto mul0 = dyn_cast_or_null<MultiplyOp>(addOp.getOperand(0).getDefiningOp());
  auto mul1 = dyn_cast_or_null<MultiplyOp>(addOp.getOperand(1).getDefiningOp());
  if (!mul0 || !mul1) {
    return failure();
  }

  // 2. Identify both branches.
  auto br0 = matchBranch(mul0);
  auto br1 = matchBranch(mul1);
  if (!br0 || !br1) {
    return failure();
  }

  // 3. Both branches must share the same x 6D reshape and same freqs_cis.
  if (br0->xReshape6D != br1->xReshape6D || br0->freqsSrc != br1->freqsSrc) {
    return failure();
  }

  // 4. The two branches must select pair indices {0, 1} and columns {0, 1}.
  //    The "real" branch pairs pair[0] with col[0]; "imag" branch pairs
  //    pair[1] with col[1]. We accept either order across mul0/mul1.
  if (br0->pairIdx == br1->pairIdx || br0->colIdx == br1->colIdx) {
    return failure();
  }
  if (br0->pairIdx != br0->colIdx || br1->pairIdx != br1->colIdx) {
    return failure();
  }

  // 5. The x 6D reshape must come from a ReshapeOp that produced shape
  //    (..., D/2, 1, 2) from (..., D).
  auto xReshapeOp =
      dyn_cast_or_null<ReshapeOp>(br0->xReshape6D.getDefiningOp());
  if (!xReshapeOp) {
    return failure();
  }
  Value x4d = xReshapeOp.getOperand();
  auto x4dType = mlir::cast<RankedTensorType>(x4d.getType());
  if (x4dType.getRank() != 4) {
    return failure();
  }
  int64_t D = x4dType.getShape().back();
  if (D % 2 != 0 || D < 2) {
    return failure();
  }
  int64_t halfD = D / 2;

  auto x6dType = mlir::cast<RankedTensorType>(br0->xReshape6D.getType());
  ArrayRef<int64_t> x6dShape = x6dType.getShape();
  int64_t x6dRank = x6dShape.size();
  if (x6dRank != 6 || x6dShape[x6dRank - 3] != halfD ||
      x6dShape[x6dRank - 2] != 1 || x6dShape[x6dRank - 1] != 2) {
    return failure();
  }

  // 6. freqs_cis must be 6D shape (..., D/2, 2, 2).
  auto freqsType = mlir::cast<RankedTensorType>(br0->freqsSrc.getType());
  ArrayRef<int64_t> freqsShape = freqsType.getShape();
  int64_t fRank = freqsShape.size();
  if (fRank != 6 || freqsShape[fRank - 3] != halfD ||
      freqsShape[fRank - 2] != 2 || freqsShape[fRank - 1] != 2) {
    return failure();
  }

  // 7. AddOp must have a single ReshapeOp user that flattens
  //    (..., D/2, 2) -> (..., D).
  if (!addOp->hasOneUse()) {
    return failure();
  }
  auto finalReshape = dyn_cast<ReshapeOp>(*addOp->getUsers().begin());
  if (!finalReshape) {
    return failure();
  }
  auto finalType = mlir::cast<RankedTensorType>(finalReshape.getType());
  if (finalType.getShape() != x4dType.getShape()) {
    return failure();
  }

  // 8. Single-use guards on critical intermediates.
  if (!mul0->hasOneUse() || !mul1->hasOneUse()) {
    return failure();
  }

  // -------- Rewrite --------
  Location loc = addOp.getLoc();
  Type elemType = x4dType.getElementType();

  // 9. Extract cos_half and sin_half from freqs_cis.
  //    cos = freqs_cis[..., 0, 0]  (row 0, col 0)
  //    sin = freqs_cis[..., 1, 0]  (row 1, col 0)
  // First slice on the row dim (rank-2), then on the col dim (rank-1).
  SliceStaticOp cosRowSlice =
      buildSliceOnDim(rewriter, loc, br0->freqsSrc, fRank - 2, 0, 1);
  SliceStaticOp cosColSlice =
      buildSliceOnDim(rewriter, loc, cosRowSlice.getResult(), fRank - 1, 0, 1);

  SliceStaticOp sinRowSlice =
      buildSliceOnDim(rewriter, loc, br0->freqsSrc, fRank - 2, 1, 2);
  SliceStaticOp sinColSlice =
      buildSliceOnDim(rewriter, loc, sinRowSlice.getResult(), fRank - 1, 0, 1);

  // Squeeze the trailing two singleton dims so the cache is 4D and
  // broadcastable to the 4D x. freqs_cis (B, S, ?, D/2, 1, 1) -> (B, S, ?, D/2)
  SmallVector<int64_t> cacheShape4D(freqsShape.begin(),
                                    freqsShape.begin() + fRank - 2);
  auto cacheType4D = RankedTensorType::get(cacheShape4D, elemType);
  SmallVector<int32_t> cacheShape4DAttr(cacheShape4D.size());
  for (size_t i = 0; i < cacheShape4D.size(); ++i) {
    cacheShape4DAttr[i] = static_cast<int32_t>(cacheShape4D[i]);
  }
  auto cosHalf =
      rewriter.create<ReshapeOp>(loc, cacheType4D, cosColSlice.getResult(),
                                 rewriter.getI32ArrayAttr(cacheShape4DAttr));
  auto sinHalf =
      rewriter.create<ReshapeOp>(loc, cacheType4D, sinColSlice.getResult(),
                                 rewriter.getI32ArrayAttr(cacheShape4DAttr));

  // 10. Duplicate caches to full D: cos_full = concat([cos_half, cos_half]).
  SmallVector<int64_t> cacheShapeFull(cacheShape4D);
  cacheShapeFull.back() = D;
  auto cacheTypeFull = RankedTensorType::get(cacheShapeFull, elemType);
  auto cosFull = rewriter.create<ConcatOp>(
      loc, cacheTypeFull, ValueRange{cosHalf.getResult(), cosHalf.getResult()},
      rewriter.getSI32IntegerAttr(cacheShapeFull.size() - 1));
  auto sinFull = rewriter.create<ConcatOp>(
      loc, cacheTypeFull, ValueRange{sinHalf.getResult(), sinHalf.getResult()},
      rewriter.getSI32IntegerAttr(cacheShapeFull.size() - 1));

  // 11. Permute x: interleaved [x0,x1,x2,x3,...] -> rotate-half
  // [x0,x2,...|x1,x3,...]
  //     via slice (step 2) on the last dim, then concat.
  auto xEvens =
      buildSliceOnDim(rewriter, loc, x4d, /*dim=*/3, 0, D, /*step=*/2);
  auto xOdds = buildSliceOnDim(rewriter, loc, x4d, /*dim=*/3, 1, D, /*step=*/2);
  auto xPerm = rewriter.create<ConcatOp>(
      loc, x4dType, ValueRange{xEvens.getResult(), xOdds.getResult()},
      rewriter.getSI32IntegerAttr(3));

  // 12. Apply ttcore.composite "rotary_embedding".
  // cosFull = concat(cosHalf, cosHalf) — always self-concat here.
  auto moduleOp = addOp->getParentOfType<ModuleOp>();
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointToEnd(moduleOp.getBody());
  auto decompFunc = buildRoPEDecompositionFunc(
      moduleBuilder, loc, x4dType,
      mlir::cast<RankedTensorType>(cosFull.getType()),
      mlir::cast<RankedTensorType>(sinFull.getType()), x4dType,
      /*isSelfConcatCosSin=*/true);
  moduleBuilder.insert(decompFunc);
  auto rotEmb = rewriter.create<ttcore::CompositeOp>(
      loc, TypeRange{x4dType},
      ValueRange{xPerm.getResult(), cosFull.getResult(), sinFull.getResult()},
      rewriter.getStringAttr("rotary_embedding"),
      FlatSymbolRefAttr::get(rewriter.getContext(), decompFunc.getName()),
      /*composite_attributes=*/nullptr);

  // 13. Permute back: rotate-half [r0,r1,...|i0,i1,...] -> interleaved
  // [r0,i0,r1,i1,...]
  //     via reshape -> permute (swap last two dims) -> reshape.
  // Reshape (B,S,H,D) -> (B,S,H,2,D/2): the leading "2" is "which half".
  SmallVector<int64_t> splitShape;
  for (int64_t d : x4dType.getShape().drop_back()) {
    splitShape.push_back(d);
  }
  splitShape.push_back(2);
  splitShape.push_back(halfD);
  auto splitType = RankedTensorType::get(splitShape, elemType);
  SmallVector<int32_t> splitShapeAttr(splitShape.size());
  for (size_t i = 0; i < splitShape.size(); ++i) {
    splitShapeAttr[i] = static_cast<int32_t>(splitShape[i]);
  }
  auto split =
      rewriter.create<ReshapeOp>(loc, splitType, rotEmb->getResult(0),
                                 rewriter.getI32ArrayAttr(splitShapeAttr));

  // Permute last two dims: (B,S,H,2,D/2) -> (B,S,H,D/2,2).
  SmallVector<int64_t> permShape(splitShape);
  std::swap(permShape[permShape.size() - 2], permShape[permShape.size() - 1]);
  auto permType = RankedTensorType::get(permShape, elemType);
  SmallVector<int64_t> perm = {0, 1, 2, 4, 3};
  auto permuted =
      rewriter.create<PermuteOp>(loc, permType, split.getResult(), perm);

  // Flatten back: (B,S,H,D/2,2) -> (B,S,H,D).
  SmallVector<int32_t> finalShapeAttr;
  for (int64_t d : x4dType.getShape()) {
    finalShapeAttr.push_back(static_cast<int32_t>(d));
  }
  auto flat =
      rewriter.create<ReshapeOp>(loc, x4dType, permuted.getResult(),
                                 rewriter.getI32ArrayAttr(finalShapeAttr));

  // 14. Replace the final reshape's result with our flattened output.
  rewriter.replaceOp(finalReshape, flat.getResult());
  return success();
}

} // namespace mlir::tt::ttir::fusing
