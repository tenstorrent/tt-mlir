// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>

namespace mlir::tt::ttnn::fusing {

namespace {

struct RoPEComponents {
  SmallVector<Value> x;   // Input tensor candidates (along TM chain)
  SmallVector<Value> cos; // Cosine embedding candidates (along TM chain)
  SmallVector<Value> sin; // Sine embedding candidates (along TM chain)
  AddOp addOp;            // Final add operation
  MultiplyOp cosMul;      // x * cos multiplication
  MultiplyOp sinMul;      // rotate_half(x) * sin multiplication
};

// Result of RoPE semantic analysis - the validated input values.
struct RoPEInputs {
  Value x;
  Value cos;
  Value sin;
  // Permutation to convert BHSD -> original output axis order.
  // Uses TTNN PermuteOp semantics: outputDim[i] = inputDim[permutation[i]].
  SmallVector<int64_t, 4> outPermutation;
};

// Semantic axis analyzer for RoPE fusion.
//
// RoPE (Rotary Position Embedding) computes:
//   output = (x * cos) + (rotate_half(x) * sin)
//
// The structural pattern matcher (RoPEFusing) identifies the ops and
// produces candidate values for x, cos, and sin — one candidate per point
// along each TM chain (permute/typecast). This analyzer determines which
// combination is valid by propagating symbolic axis labels {B, H, S, D}
// from the inputs through every op in the subgraph up to the root (the
// final add).
//
// Each op visitor preserves or transforms axis labels:
//   - Elementwise (add, mul, neg, typecast): labels must be compatible;
//     broadcast dims adopt the non-broadcast side's label.
//   - Concat: concatenation dimension must be D (the halved dimension).
//   - Slice: the sliced dimension must be D.
//   - Permute: reorders axis labels according to the permutation.
//
// A candidate combination is valid when the root resolves to a permutation
// of {B, H, S, D}. If that permutation is not the identity, the returned
// RoPEInputs includes an output permutation so the caller can insert a
// permute after rotary_embedding to restore the original axis order.
class RoPEAnalyzer {
public:
  std::optional<RoPEInputs> findValidInputs(Value root,
                                            const RoPEComponents &components) {
    SmallVector<Axis> expected = {Axis::B, Axis::H, Axis::S, Axis::D};

    for (Value xCandidate : components.x) {
      for (Value cosCandidate : components.cos) {
        for (Value sinCandidate : components.sin) {
          cache.clear();
          currentX = xCandidate;
          currentCos = cosCandidate;
          currentSin = sinCandidate;

          SemanticShape res = solve(root);
          if (!res.known || res.axes.size() != expected.size()) {
            continue;
          }

          // Result must be a permutation of {B,H,S,D} with no duplicates.
          if (!std::is_permutation(res.axes.begin(), res.axes.end(),
                                   expected.begin(), expected.end())) {
            continue;
          }

          SmallVector<int64_t, 4> outPerm;
          for (Axis a : res.axes) {
            outPerm.push_back(static_cast<int64_t>(a));
          }

          return RoPEInputs{xCandidate, cosCandidate, sinCandidate, outPerm};
        }
      }
    }
    return std::nullopt;
  }

private:
  enum class Axis : uint8_t { B, H, S, D, Other };

  // Tracks symbolic axis labels for a tensor value. Each dimension carries
  // an Axis label and a broadcast flag — broadcast dims (size 1) are flexible
  // and adopt the label from the non-broadcast side during merge.
  struct SemanticShape {
    bool known = false;
    SmallVector<Axis, 4> axes;
    SmallVector<bool, 4> isBroadcast;

    static SemanticShape unknown() { return {false, {}, {}}; }

    // Create a seed shape with optional broadcast awareness.
    // When a concrete tensor shape is provided, size-1 dims are marked as
    // broadcast (flexible axis label). Without a shape, no dims are broadcast.
    static SemanticShape seed(ArrayRef<Axis> a, ArrayRef<int64_t> shape = {}) {
      SemanticShape s;
      s.known = true;
      s.axes.assign(a.begin(), a.end());
      s.isBroadcast.resize(a.size());
      for (size_t i = 0; i < a.size(); ++i) {
        s.isBroadcast[i] = (!shape.empty() && shape[i] == 1);
      }
      return s;
    }

    bool isCompatible(const SemanticShape &other) const {
      if (!known || !other.known) {
        return true;
      }
      if (axes.size() != other.axes.size()) {
        return false;
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        if (isBroadcast[i] || other.isBroadcast[i]) {
          continue;
        }
        if (axes[i] != other.axes[i]) {
          return false;
        }
      }
      return true;
    }

    SemanticShape merge(const SemanticShape &other) const {
      if (!known) {
        return other;
      }
      if (!other.known) {
        return *this;
      }
      SemanticShape res = *this;
      for (size_t i = 0; i < axes.size(); ++i) {
        if (isBroadcast[i] && !other.isBroadcast[i]) {
          res.axes[i] = other.axes[i];
        }
        res.isBroadcast[i] = isBroadcast[i] && other.isBroadcast[i];
      }
      return res;
    }
  };

  Value currentX;
  Value currentCos;
  Value currentSin;
  mutable DenseMap<Value, SemanticShape> cache;

  SemanticShape solve(Value v) {
    if (cache.count(v)) {
      return cache[v];
    }

    if (v == currentX) {
      return cache[v] =
                 SemanticShape::seed({Axis::B, Axis::H, Axis::S, Axis::D});
    }
    if (v == currentCos || v == currentSin) {
      auto type = mlir::dyn_cast<RankedTensorType>(v.getType());
      if (type && type.getRank() == 4) {
        return cache[v] = SemanticShape::seed(
                   {Axis::B, Axis::H, Axis::S, Axis::D}, type.getShape());
      }
      return cache[v] =
                 SemanticShape::seed({Axis::B, Axis::H, Axis::S, Axis::D});
    }

    Operation *op = v.getDefiningOp();
    if (!op) {
      return cache[v] = SemanticShape::unknown();
    }

    SemanticShape result =
        llvm::TypeSwitch<Operation *, SemanticShape>(op)
            .Case<ttnn::AddOp>([&](auto add) { return visitAdd(add); })
            .Case<MultiplyOp>([&](auto mul) { return visitMul(mul); })
            .Case<ConcatOp>([&](auto cat) { return visitConcat(cat); })
            .Case<SliceStaticOp>([&](auto s) { return visitSlice(s); })
            .Case<PermuteOp>([&](auto p) { return visitPermute(p); })
            .Case<NegOp>([&](auto neg) { return visitNeg(neg); })
            .Case<TypecastOp>([&](auto tc) { return visitTypecast(tc); })
            .Case<RepeatOp>([&](auto rep) { return visitRepeat(rep); })
            .Default([&](Operation *) { return SemanticShape::unknown(); });

    return cache[v] = result;
  }

  SemanticShape visitElementwise(Value lhsVal, Value rhsVal) {
    auto lhs = solve(lhsVal);
    auto rhs = solve(rhsVal);

    if (lhs.known && rhs.known && lhs.isCompatible(rhs)) {
      return lhs.merge(rhs);
    }

    return SemanticShape::unknown();
  }

  SemanticShape visitAdd(ttnn::AddOp op) {
    return visitElementwise(op.getLhs(), op.getRhs());
  }

  SemanticShape visitMul(MultiplyOp op) {
    return visitElementwise(op.getLhs(), op.getRhs());
  }

  SemanticShape visitConcat(ConcatOp op) {
    auto lhs = solve(op.getOperand(0));
    auto rhs = solve(op.getOperand(1));

    if (lhs.known && rhs.known && lhs.isCompatible(rhs)) {
      int64_t dim = op.getDim();
      if (dim >= 0 && static_cast<size_t>(dim) < lhs.axes.size() &&
          lhs.axes[dim] == Axis::D) {
        return lhs.merge(rhs);
      }
    }

    return SemanticShape::unknown();
  }

  // Slice validation is lightweight here — the structural matcher
  // (isValidHalfRotationSlices) already validates complementary halves,
  // full-range on non-sliced dims, and step=1. We only need to confirm
  // the sliced dimension is semantic D.
  SemanticShape visitSlice(SliceStaticOp op) {
    auto input = solve(op.getInput());
    if (!input.known) {
      return SemanticShape::unknown();
    }

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    // Any dimension that changed size must be the semantic D axis.
    for (size_t i = 0; i < input.axes.size(); ++i) {
      if (inputShape[i] != outputShape[i] && input.axes[i] != Axis::D) {
        return SemanticShape::unknown();
      }
    }

    return input;
  }

  SemanticShape visitNeg(NegOp op) { return solve(op.getInput()); }
  SemanticShape visitTypecast(TypecastOp op) { return solve(op.getInput()); }
  SemanticShape visitRepeat(RepeatOp op) { return solve(op.getInput()); }

  SemanticShape visitPermute(PermuteOp op) {
    auto input = solve(op.getInput());
    if (!input.known) {
      return SemanticShape::unknown();
    }

    auto permutation = op.getPermutation();
    if (permutation.size() != input.axes.size()) {
      return SemanticShape::unknown();
    }

    SemanticShape result;
    result.known = true;
    result.axes.resize(permutation.size());
    result.isBroadcast.resize(permutation.size());

    for (size_t i = 0; i < permutation.size(); ++i) {
      int64_t srcIdx = permutation[i];
      if (srcIdx < 0 || static_cast<size_t>(srcIdx) >= input.axes.size()) {
        return SemanticShape::unknown();
      }
      result.axes[i] = input.axes[srcIdx];
      result.isBroadcast[i] = input.isBroadcast[srcIdx];
    }

    return result;
  }
};

// ---------------------------------------------------------------------------
// TM chain utilities
// ---------------------------------------------------------------------------

// Skip transparent TM ops (typecast, permute) to reach the semantic source.
Value skipTMs(Value v) {
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp, PermuteOp>(defOp)) {
      v = defOp->getOperand(0);
    } else {
      break;
    }
  }

  return v;
}

// Collect a value and all predecessors reachable through TM ops.
// Includes RepeatOp so that pre-broadcast values (with size-1 dims) are
// available as candidates — this preserves broadcast information that the
// semantic analyzer uses to handle non-standard axis orderings.
SmallVector<Value> collectCandidates(Value v) {
  SmallVector<Value> candidates{v};
  while (Operation *defOp = v.getDefiningOp()) {
    if (isa<TypecastOp, PermuteOp, RepeatOp>(defOp)) {
      v = defOp->getOperand(0);
      candidates.push_back(v);
      continue;
    }
    break;
  }

  return candidates;
}

// Walk two TM chains (permute/typecast) and return the first shared value.
Value findCommonTMAncestor(Value a, Value b) {
  DenseSet<Value> aChain;
  for (Value v = a; v;) {
    aChain.insert(v);
    Operation *op = v.getDefiningOp();
    if (!op || !isa<TypecastOp, PermuteOp>(op)) {
      break;
    }
    v = op->getOperand(0);
  }
  for (Value v = b; v;) {
    if (aChain.contains(v)) {
      return v;
    }
    Operation *op = v.getDefiningOp();
    if (!op || !isa<TypecastOp, PermuteOp>(op)) {
      break;
    }
    v = op->getOperand(0);
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Slice validation helpers
// ---------------------------------------------------------------------------

struct SliceParams {
  ArrayRef<int64_t> inputShape;
  SmallVector<int64_t> begins;
  SmallVector<int64_t> ends;
  SmallVector<int64_t> steps;
};

std::optional<SliceParams> getSliceParams(SliceStaticOp slice) {
  auto inputType = mlir::dyn_cast<RankedTensorType>(slice.getInput().getType());
  if (!inputType) {
    return std::nullopt;
  }

  auto begins = ttmlir::utils::getIntegerVector<int64_t>(slice.getBegins());
  auto ends = ttmlir::utils::getIntegerVector<int64_t>(slice.getEnds());
  auto steps = ttmlir::utils::getIntegerVector<int64_t>(slice.getStep());
  if (!begins || !ends || !steps) {
    return std::nullopt;
  }

  return SliceParams{inputType.getShape(), std::move(*begins), std::move(*ends),
                     std::move(*steps)};
}

// Find the single dimension that is sliced (not full range).
// Returns nullopt if zero or more than one dimension is sliced.
std::optional<size_t> findSlicedDim(const SliceParams &p) {
  std::optional<size_t> slicedDim;
  for (size_t i = 0; i < p.inputShape.size(); ++i) {
    bool isFull =
        (p.begins[i] == 0 && p.ends[i] == p.inputShape[i] && p.steps[i] == 1);
    if (isFull) {
      continue;
    }
    if (slicedDim.has_value()) {
      return std::nullopt; // More than one sliced dimension
    }
    slicedDim = i;
  }
  return slicedDim;
}

// Validates that two slices form complementary halves for rotate_half.
// rotate_half: concat(neg(slice[half:]), slice[:half]) -> [-second, first]
bool isValidHalfRotationSlices(SliceStaticOp lhsSlice, SliceStaticOp rhsSlice,
                               bool lhsNeg, bool rhsNeg) {
  auto lhsOpt = getSliceParams(lhsSlice);
  auto rhsOpt = getSliceParams(rhsSlice);
  if (!lhsOpt || !rhsOpt) {
    return false;
  }
  const SliceParams &lhs = *lhsOpt;
  const SliceParams &rhs = *rhsOpt;

  if (lhs.inputShape != rhs.inputShape || lhs.inputShape.empty()) {
    return false;
  }

  // Exactly one dimension must be sliced, and it must be the same for both.
  auto lhsDimOpt = findSlicedDim(lhs);
  auto rhsDimOpt = findSlicedDim(rhs);
  if (!lhsDimOpt || !rhsDimOpt || *lhsDimOpt != *rhsDimOpt) {
    return false;
  }
  size_t dim = *lhsDimOpt;

  // The sliced dimension must be even (to split into two halves) with step 1.
  int64_t dimSize = lhs.inputShape[dim];
  if ((dimSize % 2) != 0 || lhs.steps[dim] != 1 || rhs.steps[dim] != 1) {
    return false;
  }

  // The two slices must cover [0, half) and [half, dimSize) exactly.
  int64_t half = dimSize / 2;
  bool lhsIsSecondHalf = (lhs.begins[dim] == half && lhs.ends[dim] == dimSize);
  bool rhsIsFirstHalf = (rhs.begins[dim] == 0 && rhs.ends[dim] == half);

  // rotate_half = concat(neg(second_half), first_half), so concat
  // operand 0 must be the negated second-half and operand 1 must be the
  // non-negated first-half. Reject the swapped order [first, neg(second)]
  // which is not a valid RoPE rotation.
  if (!(lhsNeg && lhsIsSecondHalf && rhsIsFirstHalf)) {
    return false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// Structural pattern matching
// ---------------------------------------------------------------------------

// Match concat(neg(slice(x)), slice(x)) and return the common source x.
Value matchRotateHalfSource(Value v) {
  v = skipTMs(v);

  auto concatOp = v.getDefiningOp<ConcatOp>();
  if (!concatOp || concatOp.getNumOperands() != 2) {
    return nullptr;
  }

  auto peelToSlice = [&](Value operand, bool &wasNeg) -> SliceStaticOp {
    operand = skipTMs(operand);
    if (auto negOp = operand.getDefiningOp<NegOp>()) {
      wasNeg = true;
      operand = skipTMs(negOp.getInput());
    } else {
      wasNeg = false;
    }
    return operand.getDefiningOp<SliceStaticOp>();
  };

  bool lhsNeg = false;
  bool rhsNeg = false;
  SliceStaticOp lhsSlice = peelToSlice(concatOp.getOperand(0), lhsNeg);
  SliceStaticOp rhsSlice = peelToSlice(concatOp.getOperand(1), rhsNeg);

  if (!lhsSlice || !rhsSlice) {
    return nullptr;
  }

  // Exactly one side should be negated for rotate_half.
  if (!(lhsNeg ^ rhsNeg)) {
    return nullptr;
  }

  Value source = findCommonTMAncestor(lhsSlice.getInput(), rhsSlice.getInput());
  if (!source) {
    return nullptr;
  }

  if (!isValidHalfRotationSlices(lhsSlice, rhsSlice, lhsNeg, rhsNeg)) {
    return nullptr;
  }

  return source;
}

// Match: (x * cos) + (rotate_half(x) * sin)
bool matchRope(RoPEComponents &c) {
  Value addLhs = skipTMs(c.addOp.getLhs());
  Value addRhs = skipTMs(c.addOp.getRhs());

  auto mul1 = addLhs.getDefiningOp<MultiplyOp>();
  auto mul2 = addRhs.getDefiningOp<MultiplyOp>();
  if (!mul1 || !mul2) {
    return false;
  }

  // Identify the sin branch: the multiply that has a rotate_half operand,
  // i.e. concat(neg(slice(x, half:)), slice(x, :half)). We check both
  // operand positions since multiply is commutative.
  auto tryMatchSinBranch = [&](MultiplyOp mul, Value &outX,
                               Value &outSin) -> bool {
    Value lhs = mul.getLhs();
    Value rhs = mul.getRhs();

    if (Value x = matchRotateHalfSource(lhs)) {
      outX = x;
      outSin = rhs;
      return true;
    }
    if (Value x = matchRotateHalfSource(rhs)) {
      outX = x;
      outSin = lhs;
      return true;
    }
    return false;
  };

  Value xFromSinBranch;
  Value sinValue;
  MultiplyOp sinMul = nullptr;
  MultiplyOp cosMul = nullptr;

  if (tryMatchSinBranch(mul1, xFromSinBranch, sinValue)) {
    sinMul = mul1;
    cosMul = mul2;
  } else if (tryMatchSinBranch(mul2, xFromSinBranch, sinValue)) {
    sinMul = mul2;
    cosMul = mul1;
  } else {
    return false;
  }

  // Disambiguate x vs cos in the cos branch (x * cos or cos * x).
  // One operand must share a common ancestor with xFromSinBranch through
  // TM ops (permute/typecast), identifying it as x. The other is cos.
  // Exactly one side must match; if both or neither do, we bail.
  Value cosLhs = cosMul.getLhs();
  Value cosRhs = cosMul.getRhs();

  Value lhsAncestor = findCommonTMAncestor(cosLhs, xFromSinBranch);
  Value rhsAncestor = findCommonTMAncestor(cosRhs, xFromSinBranch);

  if (static_cast<bool>(lhsAncestor) == static_cast<bool>(rhsAncestor)) {
    return false;
  }

  Value xValue = lhsAncestor ? lhsAncestor : rhsAncestor;
  Value cosValue = lhsAncestor ? cosRhs : cosLhs;

  // Collect candidate values for x, cos, sin. Each list includes the
  // matched value plus all values reachable by walking back through TM ops
  // (permute/typecast). The semantic analyzer tries all combinations to
  // find one where axis labels resolve to a valid BHSD permutation.
  c.x = collectCandidates(xValue);
  c.cos = collectCandidates(cosValue);
  c.sin = collectCandidates(sinValue);
  c.cosMul = cosMul;
  c.sinMul = sinMul;

  return true;
}

// ---------------------------------------------------------------------------
// Fused op creation
// ---------------------------------------------------------------------------

// Check whether any of the RoPE multiply/add ops compute in f32 and, if so,
// return a DeviceComputeKernelConfig with fp32_dest_acc_en set.
DeviceComputeKernelConfigAttr buildComputeConfig(mlir::MLIRContext *ctx,
                                                 const RoPEComponents &c) {
  auto usesF32 = [](Operation *op) {
    auto resultType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
    return resultType.getElementType().isF32();
  };
  if (usesF32(c.cosMul) || usesF32(c.sinMul) || usesF32(c.addOp)) {
    return DeviceComputeKernelConfigAttr::get(ctx).withFp32DestAccEn(true);
  }
  return nullptr;
}

mlir::LogicalResult createFusedRoPEOp(mlir::PatternRewriter &rewriter,
                                      AddOp srcOp, const RoPEInputs &inputs,
                                      const RoPEComponents &components) {
  op_model::ScopedSingletonDeviceGuard deviceGuard;

  auto computeConfig = buildComputeConfig(rewriter.getContext(), components);

  auto ropeOp = rewriter.create<RotaryEmbeddingOp>(
      srcOp.getLoc(), inputs.x.getType(), inputs.x, inputs.cos, inputs.sin,
      /*token_index=*/nullptr,
      /*memory_config=*/nullptr,
      /*compute_config=*/computeConfig);

  // Validate the fused op. If validation fails, try the workaround-padded
  // version since the workaround pass (seq_len tile alignment) hasn't run yet.
  std::vector<TTNNLayoutAttr> inputLayouts =
      utils::extractInputLayouts(ropeOp.getOperation());
  auto resultType = mlir::cast<RankedTensorType>(ropeOp.getType());
  OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
  auto validationResult = op_constraint_validation::validateOperation(
      ropeOp.getOperation(), inputLayouts, config);

  if (!validationResult.isSuccess()) {
    auto workaround =
        workarounds::decomposition::getWorkaroundedOp(ropeOp, rewriter);
    if (workaround) {
      auto paddedOp = workaround->first;
      auto sliceOp = workaround->second;
      inputLayouts = utils::extractInputLayouts(paddedOp.getOperation());
      resultType = mlir::cast<RankedTensorType>(paddedOp.getType());
      config = OpConfig(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      validationResult = op_constraint_validation::validateOperation(
          paddedOp.getOperation(), inputLayouts, config);
      rewriter.eraseOp(sliceOp);
      rewriter.eraseOp(paddedOp);
    }

    if (!validationResult.isSuccess()) {
      rewriter.eraseOp(ropeOp);
      return failure();
    }
  }

  Value result = ropeOp.getResult();
  if (!llvm::equal(inputs.outPermutation,
                   llvm::seq<int64_t>(0, inputs.outPermutation.size()))) {
    DenseI64ArrayAttr permutationAttr =
        rewriter.getDenseI64ArrayAttr(inputs.outPermutation);
    auto permuted = rewriter.create<ttnn::PermuteOp>(
        srcOp.getLoc(), srcOp.getType(), result, permutationAttr,
        ttnn::MemoryConfigAttr(), mlir::FloatAttr());
    result = permuted.getResult();
  }

  rewriter.replaceOp(srcOp, result);
  return success();
}

} // namespace

// =============================================================================
// RoPEFusing
// =============================================================================

mlir::LogicalResult
RoPEFusing::matchAndRewrite(AddOp srcOp,
                            mlir::PatternRewriter &rewriter) const {
  RoPEComponents c;
  c.addOp = srcOp;

  if (!matchRope(c)) {
    return failure();
  }

  RoPEAnalyzer analyzer;
  auto validInputs = analyzer.findValidInputs(srcOp.getResult(), c);
  if (!validInputs) {
    return failure();
  }

  return createFusedRoPEOp(rewriter, srcOp, *validInputs, c);
}

// =============================================================================
// RoPEDecodeFusing
// =============================================================================

mlir::LogicalResult
RoPEDecodeFusing::matchAndRewrite(PermuteOp permuteOp,
                                  mlir::PatternRewriter &rewriter) const {
  auto ropeOp = permuteOp.getInput().getDefiningOp<RotaryEmbeddingOp>();
  if (!ropeOp) {
    return failure();
  }

  // Already in decode mode.
  if (ropeOp.getTokenIndex()) {
    return failure();
  }

  // RoPE result must feed only into this permute.
  if (!ropeOp.getResult().hasOneUse()) {
    return failure();
  }

  // Permutation must be 4D and put S axis (dim 2 in BHSD) at position 0.
  auto perm = permuteOp.getPermutation();
  if (perm.size() != 4 || perm[0] != 2) {
    return failure();
  }

  // Input must be rank 4 with S dim (dim 2) == 1.
  auto inputType = mlir::cast<RankedTensorType>(ropeOp.getInput().getType());
  if (inputType.getRank() != 4 || inputType.getShape()[2] != 1) {
    return failure();
  }

  // cos/sin must be single-position (dim -2 == 1).
  auto cosType = mlir::cast<RankedTensorType>(ropeOp.getCosCache().getType());
  if (cosType.getShape()[cosType.getRank() - 2] != 1) {
    return failure();
  }

  op_model::ScopedSingletonDeviceGuard deviceGuard;

  // Create pre-permute on the original RoPE input: BHSD -> permuted order.
  auto prePermute = ttir_to_ttnn::utils::generatePermute(
      mlir::cast<TypedValue<RankedTensorType>>(ropeOp.getInput()),
      llvm::ArrayRef(perm), rewriter, ropeOp.getLoc());

  auto tokenIndex = rewriter.getIntegerAttr(
      rewriter.getIntegerType(32, /*isSigned=*/false), 0);

  auto newRope = rewriter.create<RotaryEmbeddingOp>(
      ropeOp.getLoc(), prePermute.getType(), prePermute.getResult(),
      ropeOp.getCosCache(), ropeOp.getSinCache(), tokenIndex,
      ropeOp.getMemoryConfigAttr(), ropeOp.getComputeConfigAttr());

  // Validate the fused op. If validation fails, try the workaround-padded
  // version since the workaround pass (seq_len tile alignment) hasn't run yet.
  std::vector<TTNNLayoutAttr> inputLayouts =
      utils::extractInputLayouts(newRope.getOperation());
  auto resultType = mlir::cast<RankedTensorType>(newRope.getType());
  OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
  auto validationResult = op_constraint_validation::validateOperation(
      newRope.getOperation(), inputLayouts, config);

  if (!validationResult.isSuccess()) {
    auto workaround =
        workarounds::decomposition::getWorkaroundedOp(newRope, rewriter);
    if (workaround) {
      auto paddedOp = workaround->first;
      auto sliceOp = workaround->second;
      inputLayouts = utils::extractInputLayouts(paddedOp.getOperation());
      resultType = mlir::cast<RankedTensorType>(paddedOp.getType());
      config = OpConfig(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      validationResult = op_constraint_validation::validateOperation(
          paddedOp.getOperation(), inputLayouts, config);
      rewriter.eraseOp(sliceOp);
      rewriter.eraseOp(paddedOp);
    }

    if (!validationResult.isSuccess()) {
      rewriter.eraseOp(newRope);
      rewriter.eraseOp(prePermute);
      return failure();
    }
  }

  // Replace the permute's uses with the new RoPE result.
  // The old RoPE op becomes dead and is cleaned up by the rewriter.
  rewriter.replaceOp(permuteOp, newRope.getResult());
  return success();
}

} // namespace mlir::tt::ttnn::fusing
