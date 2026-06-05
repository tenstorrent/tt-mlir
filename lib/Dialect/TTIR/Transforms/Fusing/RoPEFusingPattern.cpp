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
  int64_t lastDim = resultType.getRank() - 1;
  if (concatOp.getDim() != lastDim) {
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

} // namespace mlir::tt::ttir::fusing
