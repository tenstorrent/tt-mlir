// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

#ifdef TTMLIR_ENABLE_OPMODEL
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidation.h"
#include "ttmlir/OpModel/TTNN/SingletonDeviceContext.h"
#endif

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

template <typename ActivationOp>
class TTNNConv2dWithActivation : public mlir::OpRewritePattern<Conv2dOp> {
  using TTNNConv2dWithActivation::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp = getActivationOp(srcOp);
    Value activationInput = activationOp.getInput();

    auto activation = getActivationOpType(rewriter);

    ttcore::DataType weightDtype = ttcore::elementTypeToDataType(
        srcOp.getWeight().getType().getElementType());
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation)
                           .withWeightsDtype(weightDtype);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the activation op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(activationOp, activationInput);

    return mlir::success();
  }

private:
  ActivationOp getActivationOp(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsersOfType<ReshapeOp, ActivationOp>(srcOp)) &&
           "Conv2d should have either activation or Reshape as user.");

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    assert(reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp) &&
           "Reshape should have only one user and that user should be "
           "activation.");
    return mlir::cast<ActivationOp>(*reshapeOp.getResult().getUsers().begin());
  }

  ttnn::UnaryOpType getActivationOpType(mlir::PatternRewriter &rewriter) const {
    // Extract op name from full operation name (e.g., "ttnn.relu" -> "relu")
    // and convert to enum
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    auto activation = ttnn::symbolizeUnaryOpType(opName);
    assert(activation.has_value() && "Unsupported activation op");
    return activation.value();
  }

  bool isFusable(Conv2dOp srcOp) const {
    if (srcOp.getConv2dConfig() && srcOp.getConv2dConfig()->hasActivation()) {
      return false;
    }

    // Conv2d has multiple uses so we cannot fuse.
    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    // Conv2d only user is activation so we can fuse.
    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    // Since window flattening will add rehape after conv we need to check
    // if there is reshape right after conv2d.
    if (!ttmlir::utils::allUsersOfType<ReshapeOp>(srcOp)) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If we want to fuse activation to conv we need to make sure that reshape
    // has only one user and that user is activation.
    return reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp);
  }
};

template <typename SrcOp, typename ActivationOp>
class TTNNMatmulAndLinearWithActivation : public mlir::OpRewritePattern<SrcOp> {
  using TTNNMatmulAndLinearWithActivation::template OpRewritePattern<
      SrcOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(SrcOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp =
        mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    Value activationInput = activationOp.getInput();
    auto activationStr = getActivationString();

    rewriter.modifyOpInPlace(srcOp, [&]() {
      srcOp.setActivationAttr(rewriter.getStringAttr(activationStr));
    });

    rewriter.replaceAllUsesWith(activationOp, activationInput);
    return mlir::success();
  }

private:
  // After tt-metal resolves this issue:
  // https://github.com/tenstorrent/tt-metal/issues/31393, we can use the
  // UnaryWithParam enum directly instead of string.
  std::string getActivationString() const {
    llvm::StringLiteral fullOpName = ActivationOp::getOperationName();
    llvm::StringRef opName = fullOpName.rsplit('.').second;
    return opName.str();
  }

  bool isFusable(SrcOp srcOp) const {
    if (srcOp.getActivation()) {
      return false;
    }

    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    return false;
  }
};

#ifdef TTMLIR_ENABLE_OPMODEL

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

class RoPEAnalyzer {
public:
  // Analyzes the RoPE computation graph and returns valid input candidates.
  // Iterates over all candidate combinations of x, cos, sin to find one that
  // satisfies the semantic constraints (axes are some permutation of
  // [B, H, S, D]). If the output axes are not [B, H, S, D], the returned
  // RoPEInputs includes an output permutation that can be applied after
  // rotary_embedding to match the original output axis order.
  // Returns nullopt if no valid combination is found.
  std::optional<RoPEInputs> findValidInputs(Value root,
                                            const RoPEComponents &components) {
    SmallVector<Axis> expected = {Axis::B, Axis::H, Axis::S, Axis::D};

    // Try all combinations of x, cos, sin candidates
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

          // Accept any permutation of {B,H,S,D}.
          llvm::SmallDenseSet<Axis, 4> seen;
          bool ok = true;
          for (Axis a : res.axes) {
            if (a != Axis::B && a != Axis::H && a != Axis::S && a != Axis::D) {
              ok = false;
              break;
            }
            if (!seen.insert(a).second) {
              ok = false;
              break;
            }
          }
          if (!ok || seen.size() != expected.size()) {
            continue;
          }

          // Compute permutation to convert BHSD -> res.axes (original output
          // axis order). For each output axis label, pick the index in BHSD.
          SmallVector<int64_t, 4> outPerm;
          outPerm.reserve(expected.size());
          for (Axis outAxis : res.axes) {
            auto it = llvm::find(expected, outAxis);
            if (it == expected.end()) {
              ok = false;
              break;
            }
            outPerm.push_back(static_cast<int64_t>(it - expected.begin()));
          }
          if (!ok) {
            continue;
          }

          return RoPEInputs{xCandidate, cosCandidate, sinCandidate, outPerm};
        }
      }
    }
    return std::nullopt;
  }

private:
  // --- Structures (Same as before) ---
  enum class Axis : uint8_t { B, H, S, D, Other };

  struct SemanticShape {
    bool known = false;
    SmallVector<Axis, 4> axes;
    SmallVector<bool, 4> isBroadcast;

    static SemanticShape unknown() { return {false, {}, {}}; }

    // Create a seed shape (e.g., for X)
    static SemanticShape seed(ArrayRef<Axis> a) {
      SemanticShape s;
      s.known = true;
      s.axes.assign(a.begin(), a.end());
      s.isBroadcast.assign(a.size(), false); // Default: no broadcast
      return s;
    }

    // Check compatibility (Are these shapes fuseable?)
    bool isCompatible(const SemanticShape &other) const {
      if (!known || !other.known) {
        return true; // Optimistic
      }
      if (axes.size() != other.axes.size()) {
        return false;
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] != other.axes[i]) {
          return false;
        }
      }
      return true;
    }

    // Merge shapes (Propagate broadcast info)
    SemanticShape merge(const SemanticShape &other) const {
      if (!known) {
        return other;
      }
      if (!other.known) {
        return *this;
      }
      SemanticShape res = *this;
      for (size_t i = 0; i < axes.size(); ++i) {
        // Result is broadcast only if BOTH inputs are broadcast
        res.isBroadcast[i] = isBroadcast[i] && other.isBroadcast[i];
      }
      return res;
    }
  };

  // Current candidate values being tested
  Value currentX;
  Value currentCos;
  Value currentSin;
  mutable DenseMap<Value, SemanticShape> cache;

  SemanticShape solve(Value v) {
    if (cache.count(v)) {
      return cache[v];
    }

    // Base Cases: check against current candidates
    if (v == currentX) {
      return cache[v] =
                 SemanticShape::seed({Axis::B, Axis::H, Axis::S, Axis::D});
    }
    if (v == currentCos || v == currentSin) {
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
            .Default([&](Operation *) { return SemanticShape::unknown(); });

    return cache[v] = result;
  }

  // --- Individual Op Rules (The "Business Logic") ---
  // Each function focuses on ONE op. Easy to read, easy to debug.

  SemanticShape visitAdd(ttnn::AddOp op) {
    auto lhs = solve(op.getLhs());
    auto rhs = solve(op.getRhs());

    if (lhs.known && rhs.known && lhs.isCompatible(rhs)) {
      return lhs.merge(rhs);
    }

    return SemanticShape::unknown();
  }

  SemanticShape visitMul(MultiplyOp op) {
    auto lhs = solve(op.getLhs());
    auto rhs = solve(op.getRhs());

    if (lhs.known && rhs.known && lhs.isCompatible(rhs)) {
      return lhs.merge(rhs);
    }

    return SemanticShape::unknown();
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

  SemanticShape visitSlice(SliceStaticOp op) {
    auto input = solve(op.getInput());
    if (!input.known) {
      return SemanticShape::unknown();
    }

    // For RoPE, slices should only happen on the D (head_dim) dimension to
    // split the tensor in half for rotate_half pattern.
    //
    // Enforce:
    // - Full range slice on all dims except semantic D
    //   (begin=0, end=inputShape[i]).
    // - Step must be 1 for all dims.
    // - The semantic D dim must be even, and the slice range on that dim must
    //   be either [0, D/2) or [D/2, D).
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();
    if (inputShape.empty()) {
      return SemanticShape::unknown();
    }

    // Find which physical dimension currently corresponds to semantic D.
    size_t dDim = input.axes.size();
    for (size_t i = 0; i < input.axes.size(); ++i) {
      if (input.axes[i] == Axis::D) {
        if (dDim != input.axes.size()) {
          // Multiple D labels is ambiguous.
          return SemanticShape::unknown();
        }
        dDim = i;
      }
    }
    if (dDim == input.axes.size()) {
      return SemanticShape::unknown();
    }

    int64_t dSize = inputShape[dDim];
    if (dSize <= 0 || (dSize % 2) != 0) {
      return SemanticShape::unknown();
    }
    int64_t halfDim = dSize / 2;

    auto beginsOrErr = ttmlir::utils::getIntegerVector<int64_t>(op.getBegins());
    auto endsOrErr = ttmlir::utils::getIntegerVector<int64_t>(op.getEnds());
    auto stepsOrErr = ttmlir::utils::getIntegerVector<int64_t>(op.getStep());
    if (!beginsOrErr || !endsOrErr || !stepsOrErr) {
      return SemanticShape::unknown();
    }

    ArrayRef<int64_t> begins = *beginsOrErr;
    ArrayRef<int64_t> ends = *endsOrErr;
    ArrayRef<int64_t> steps = *stepsOrErr;
    if (begins.size() != inputShape.size() ||
        ends.size() != inputShape.size() || steps.size() != inputShape.size()) {
      return SemanticShape::unknown();
    }

    // Check that all dimensions except semantic D remain unchanged and that the
    // slice uses full-range + unit step on those dims.
    for (size_t i = 0; i < input.axes.size(); ++i) {
      if (steps[i] != 1) {
        return SemanticShape::unknown();
      }

      if (i == dDim) {
        continue;
      }

      if (inputShape[i] != outputShape[i]) {
        return SemanticShape::unknown();
      }
      if (begins[i] != 0 || ends[i] != inputShape[i]) {
        return SemanticShape::unknown();
      }
    }

    // Validate half-slice on semantic D dimension.
    int64_t b = begins[dDim];
    int64_t e = ends[dDim];
    bool isFirstHalf = (b == 0 && e == halfDim);
    bool isSecondHalf = (b == halfDim && e == dSize);
    if (!isFirstHalf && !isSecondHalf) {
      return SemanticShape::unknown();
    }

    // Optional consistency: output D dim should be half.
    if (outputShape[dDim] != halfDim) {
      return SemanticShape::unknown();
    }

    return input;
  }

  // Negation is element-wise and preserves semantic shape.
  // Used in rotate_half: concat(neg(slice(...)), slice(...))
  SemanticShape visitNeg(NegOp op) { return solve(op.getInput()); }

  // Typecast is element-wise dtype conversion and preserves semantic shape.
  SemanticShape visitTypecast(TypecastOp op) { return solve(op.getInput()); }

  SemanticShape visitPermute(PermuteOp op) {
    auto input = solve(op.getInput());
    if (!input.known) {
      return SemanticShape::unknown();
    }

    // Apply the permutation to reorder the semantic axes
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

class RoPENewFusing : public mlir::OpRewritePattern<AddOp> {
  using RoPENewFusing::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const override {
    // 1. Match Anchors (Structural Discovery)
    RoPEComponents c;
    c.addOp = srcOp;

    if (!matchRope(c)) {
      return failure();
    }

    // 2. Validate Semantics (Abstract Interpretation)
    RoPEAnalyzer analyzer;
    auto validInputs = analyzer.findValidInputs(srcOp.getResult(), c);
    if (!validInputs) {
      return failure();
    }

    // 3. Rewrite - create fused RotaryEmbeddingOp
    Value ropeResult = createFusedRoPEOp(rewriter, srcOp, *validInputs);
    rewriter.replaceOp(srcOp, ropeResult);
    return success();
  }

private:
  // Creates the fused RotaryEmbeddingOp from validated inputs.
  Value createFusedRoPEOp(mlir::PatternRewriter &rewriter, AddOp srcOp,
                          const RoPEInputs &inputs) const {
    // Create rotary_embedding in the axis order of the chosen x input. If the
    // original output axis order differs from BHSD, we permute the result back.
    auto ropeOp = rewriter.create<RotaryEmbeddingOp>(
        srcOp.getLoc(), inputs.x.getType(), inputs.x, inputs.cos, inputs.sin,
        /*token_index=*/nullptr,
        /*memory_config=*/nullptr,
        /*compute_config=*/nullptr);

    // If outPermutation is identity, no permute is needed.
    bool isIdentity = true;
    for (int64_t i = 0; i < static_cast<int64_t>(inputs.outPermutation.size());
         ++i) {
      if (inputs.outPermutation[i] != i) {
        isIdentity = false;
        break;
      }
    }
    if (isIdentity) {
      return ropeOp.getResult();
    }

    DenseI64ArrayAttr permutationAttr =
        rewriter.getDenseI64ArrayAttr(inputs.outPermutation);
    auto permuted = rewriter.create<ttnn::PermuteOp>(
        srcOp.getLoc(), srcOp.getType(), ropeOp.getResult(), permutationAttr,
        ttnn::MemoryConfigAttr(), mlir::FloatAttr());
    return permuted.getResult();
  }

  Value skipTMs(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp, PermuteOp>(defOp)) {
        v = defOp->getOperand(0);
      } else {
        break;
      }
    }

    return v;
  }

  SmallVector<Value> collectCandidates(Value v) const {
    SmallVector<Value> candidates{v};
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp, PermuteOp>(defOp)) {
        v = defOp->getOperand(0);
        candidates.push_back(v);
        continue;
      }
      break;
    }

    return candidates;
  }

  Value findCommonTMAncestor(Value a, Value b) const {
    DenseSet<Value> seen;
    auto walk = [&](Value v, auto &&onVisit) {
      while (v) {
        onVisit(v);
        Operation *op = v.getDefiningOp();
        if (!op) {
          break;
        }
        if (!isa<TypecastOp, PermuteOp>(op)) {
          break;
        }
        v = op->getOperand(0);
      }
    };

    walk(a, [&](Value v) { seen.insert(v); });

    Value lca = nullptr;
    walk(b, [&](Value v) {
      if (!lca && seen.contains(v)) {
        lca = v;
      }
    });

    return lca;
  }

  Value matchRotateHalfSource(Value v) const {
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

    Value source =
        findCommonTMAncestor(lhsSlice.getInput(), rhsSlice.getInput());
    if (!source) {
      return nullptr;
    }

    if (!isValidHalfRotationSlices(lhsSlice, rhsSlice, lhsNeg, rhsNeg)) {
      return nullptr;
    }

    return source;
  }

  // Helper to extract slice parameters from a SliceStaticOp.
  struct SliceParams {
    ArrayRef<int64_t> inputShape;
    SmallVector<int64_t> begins;
    SmallVector<int64_t> ends;
    SmallVector<int64_t> steps;
    bool valid = false;
  };

  SliceParams getSliceParams(SliceStaticOp slice) const {
    SliceParams p;
    auto inputType =
        mlir::dyn_cast<RankedTensorType>(slice.getInput().getType());
    if (!inputType) {
      return p;
    }

    auto begins = ttmlir::utils::getIntegerVector<int64_t>(slice.getBegins());
    auto ends = ttmlir::utils::getIntegerVector<int64_t>(slice.getEnds());
    auto steps = ttmlir::utils::getIntegerVector<int64_t>(slice.getStep());
    if (!begins || !ends || !steps) {
      return p;
    }

    p.inputShape = inputType.getShape();
    p.begins = std::move(*begins);
    p.ends = std::move(*ends);
    p.steps = std::move(*steps);
    p.valid = true;
    return p;
  }

  // Find the single dimension that is sliced (not full range).
  // Returns nullopt if zero or more than one dimension is sliced.
  std::optional<size_t> findSlicedDim(const SliceParams &p) const {
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
                                 bool lhsNeg, bool rhsNeg) const {
    SliceParams lhs = getSliceParams(lhsSlice);
    SliceParams rhs = getSliceParams(rhsSlice);
    if (!lhs.valid || !rhs.valid) {
      return false;
    }

    // Both must have same input shape
    if (lhs.inputShape != rhs.inputShape || lhs.inputShape.empty()) {
      return false;
    }

    // Find sliced dimension - must be exactly one and same for both
    auto lhsDimOpt = findSlicedDim(lhs);
    auto rhsDimOpt = findSlicedDim(rhs);
    if (!lhsDimOpt || !rhsDimOpt || *lhsDimOpt != *rhsDimOpt) {
      return false;
    }
    size_t dim = *lhsDimOpt;

    // Sliced dimension must be even and have step 1
    int64_t dimSize = lhs.inputShape[dim];
    if (dimSize <= 0 || (dimSize % 2) != 0) {
      return false;
    }
    if (lhs.steps[dim] != 1 || rhs.steps[dim] != 1) {
      return false;
    }

    // Check complementary halves: [0, half) and [half, dimSize)
    int64_t half = dimSize / 2;
    bool lhsIsFirst = (lhs.begins[dim] == 0 && lhs.ends[dim] == half);
    bool lhsIsSecond = (lhs.begins[dim] == half && lhs.ends[dim] == dimSize);
    bool rhsIsFirst = (rhs.begins[dim] == 0 && rhs.ends[dim] == half);
    bool rhsIsSecond = (rhs.begins[dim] == half && rhs.ends[dim] == dimSize);

    if (!((lhsIsFirst && rhsIsSecond) || (lhsIsSecond && rhsIsFirst))) {
      return false;
    }

    // rotate_half: [-second_half, first_half]. Negated slice must be second
    // half.
    if ((lhsNeg && !lhsIsSecond) || (rhsNeg && !rhsIsSecond)) {
      return false;
    }

    return true;
  }

  // Match the RoPE pattern: (x * cos) + (rotate_half(x) * sin)
  // Returns true if pattern matches and populates RoPEComponents.
  bool matchRope(RoPEComponents &c) const {
    // Step 1: Find the two multiply operations from the add.
    //         Pattern: add(mul(...), mul(...))
    Value addLhs = skipTMs(c.addOp.getLhs());
    Value addRhs = skipTMs(c.addOp.getRhs());

    auto mul1 = addLhs.getDefiningOp<MultiplyOp>();
    auto mul2 = addRhs.getDefiningOp<MultiplyOp>();
    if (!mul1 || !mul2) {
      return false;
    }

    // Step 2: Identify which multiply is the sin branch (contains rotate_half).
    //         Sin branch pattern: rotate_half(x) * sin  OR  sin *
    //         rotate_half(x)
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

    Value cosLhs = cosMul.getLhs();
    Value cosRhs = cosMul.getRhs();

    Value commonWithLhs = findCommonTMAncestor(cosLhs, xFromSinBranch);
    Value commonWithRhs = findCommonTMAncestor(cosRhs, xFromSinBranch);

    Value xValue;
    Value cosValue;
    if (commonWithLhs && !commonWithRhs) {
      xValue = commonWithLhs;
      cosValue = cosRhs;
    } else if (commonWithRhs && !commonWithLhs) {
      xValue = commonWithRhs;
      cosValue = cosLhs;
    } else {
      return false;
    }

    c.x = collectCandidates(xValue);
    c.cos = collectCandidates(cosValue);
    c.sin = collectCandidates(sinValue);
    c.cosMul = cosMul;
    c.sinMul = sinMul;

    return true;
  }
};

} // namespace

// This is rope fusing pattern. Given this formula:
// (x * cos) + (rotate_half(x) * sin)
// into ttnn rotary_embedding op from ttnn.
//
// rotate_half is defined as half rotation of the last dimension:
// rotate_half([x1, x2, x3, x4]) = [-x3, -x4, x1, x2]
//
// This pattern is broken down into two sub-patterns:
// 1. unrotatedProjection: matches x * cos
// 2. rotatedProjection: matches rotate_half(x) * sin
//
// unrotatedProjection comes as following sequence of operations:
// cos_unsqueezed = unsqueeze(cos) - where unsqueeze will be reshape which
// prepends dimensions to cos to match x's rank.
// unrotated_projection = multiply(x, cos_unsqueezed)
//
// rotatedProjections comes as following sequence of operations:
// sin_unsqueezed = unsqueeze(sin) - where unsqueeze will be reshape which
// prepends dimensions to sin to match x's rank.
// rotated_second_half = slice(x, ..., start = last_dim/2, end = last_dim)
// neg_rotated_second_half = negate(rotated_second_half)
// rotated_first_half = slice(x, ..., start = 0, end = last_dim/2)
// rotated_x = concatenate(neg_rotated_second_half, rotated_first_half, axis =
// last_dim) rotated_projection = multiply(rotated_x, sin_unsqueezed)
//
// Finally there is an add operation which adds unrotated_projection and
// rotated_projection. x_embedding = add(unrotated_projection,
// rotated_projection)
//
// This whole pattern is replaced with ttnn.rotary_embedding op.
// This op accepts x, cos, sin and trans_mat. First three are self explanatory.
// Trans matrix is used to perform the half rotation. If we take that we have
// x of shape [batch_size, num_heads, seq_len, head_dim] then trans mat
// will be of shape[head_dim, head_dim] and will look like this (lets take
// head_dim = 4):
// [[0, 0, 1, 0],
//  [0, 0, 0, 1],
//  [-1,0, 0, 0],
//  [0,-1, 0, 0]]
//
// So when we multiply x with this matrix we get the desired half rotation.
// class RoPEFusing : public mlir::OpRewritePattern<AddOp> {
//   using RoPEFusing::OpRewritePattern<AddOp>::OpRewritePattern;

// public:
//   mlir::LogicalResult
//   matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const
//   override {
//     RoPEComponents c;
//     c.addOp = srcOp;

//     // Match: add(multiply, multiply) where one is rotated projection
//     if (!matchProjections(srcOp, c)) {
//       llvm::errs() << "Failed to match projections for RoPE\n";
//       return failure();
//     }

//     // Validate semantic constraints before modifying IR
//     if (!validateSemantics(c)) {
//       llvm::errs() << "Failed to validate semantics for RoPE\n";
//       return failure();
//     }

//     return createRotaryEmbeddingOp(rewriter, c);
//   }

// private:
//   template <typename Op, typename LhsOp, typename RhsOp>
//   struct OpWithTracingInfo {
//     Op operation;
//     std::pair<LhsOp, llvm::SmallVector<PermuteOp>>* lhsInfo;
//     std::pair<RhsOp, llvm::SmallVector<PermuteOp>>* rhsInfo;
//   };

//   struct RoPEComponents {
//     Value x;            // Input tensor
//     Value cos;          // Cosine embedding (unsqueezed)
//     Value sin;          // Sine embedding (unsqueezed)
//     OpWithTracingInfo<AddOp, MultiplyOp, MultiplyOp> addOp;        // Final
//     add operation OpWithTracingInfo<MultiplyOp, CosOp, Value> cosMul;  // x *
//     cos multiplication OpWithTracingInfo<MultiplyOp, SinOp, Value> sinMul; //
//     rotate_half(x) * sin multiplication OpWithTracingInfo<ConcatOp, NegOp,
//     SliceStaticOp> concatOp;  // concat operation
//   };

//   //
//   ============================================================================
//   // Op Utilities
//   //
//   ============================================================================
//   //

//   std::pair<Value, llvm::SmallVector<PermuteOp>>
//   skipTransparentAndTrackPermutes(Value v) const {
//     llvm::SmallVector<PermuteOp> permutes;
//     while (Operation *defOp = v.getDefiningOp()) {
//       if (auto typecastOp = dyn_cast<TypecastOp>(defOp)) {
//         v = typecastOp.getOperand();
//       }

//       if (auto permuteOp = dyn_cast<PermuteOp>(defOp)) {
//         v = permuteOp.getOperand();
//         permutes.push_back(permuteOp);
//       }

//       break;
//     }

//     return {v, permutes};
//   }

//   /// Overload that takes a vector of PermuteOps directly.
//   /// Assumes permutes are in backwards-trace order (closest to output
//   first).
//   /// Reverses internally to get correct composition order.
//   static llvm::SmallVector<int64_t>
//   composePermutations(llvm::ArrayRef<PermuteOp> permuteOps) {
//     if (permuteOps.empty()) {
//       return {};
//     }

//     return llvm::SmallVector<int64_t>(permuteOps.size());
//   }

//   //
//   ============================================================================
//   // Pattern Matching
//   //
//   ============================================================================

//   bool matchProjections(RoPEComponents &c) {
//     auto [lhs, lhsPermutes] =
//     skipTransparentAndTrackPermutes(c.addOp.operation); auto [rhs,
//     rhsPermutes] = skipTransparentAndTrackPermutes(c.addOp.operation);

//     auto lhsMul = lhs.getDefiningOp<MultiplyOp>();
//     auto rhsMul = rhs.getDefiningOp<MultiplyOp>();
//     if (!lhsMul || !rhsMul) {
//       return false;
//     }

//     auto lhsStripped =
//     skipTransparentAndTrackPermutes(lhsMul.getLhs()).first; auto rhsStripped
//     = skipTransparentAndTrackPermutes(lhsMul.getRhs()).first;

//     if (isa<SinOp>(lhsStripped.getDefiningOp())) {
//       OpWithTracingInfo<MultiplyOp, SinOp, Value> sinMulInfo;
//       std::pair<SinOp, llvm::SmallVector<PermuteOp>>*

//       sinMulInfo.lhsInfo = new OpWithTracingInfo<MultiplyOp, SinOp, Value>();

//       c.sinMul.mulOp = lhsMul;
//       c.sinMul.sin = lhsStripped;
//       c.sinMul.rotatedValue =
//       skipTransparentAndTrackPermutes(lhsMul.getRhs()).first;
//     }

//     return true;
//   }

//   // Match: add(unrotated_projection, rotated_projection)
//   // where unrotated = x * cos, rotated = rotate_half(x) * sin
//   bool matchProjections(AddOp addOp, RoPEComponents &c) const {
//     Value lhs = skipNeutral(addOp.getLhs());
//     Value rhs = skipNeutral(addOp.getRhs());

//     auto lhsMul = lhs.getDefiningOp<MultiplyOp>();
//     auto rhsMul = rhs.getDefiningOp<MultiplyOp>();

//     if (!lhsMul || !rhsMul) {
//       llvm::errs() << "LHS or RHS is not a multiply op\n";
//       return false;
//     }

//     // Try to match rotation pattern on each multiply
//     auto [lhsRotationInput, lhsRotatedValue] = matchRotationPattern(lhsMul);
//     auto [rhsRotationInput, rhsRotatedValue] = matchRotationPattern(rhsMul);

//     // Exactly one must have the rotation pattern
//     if ((lhsRotationInput != nullptr) == (rhsRotationInput != nullptr)) {
//       llvm::errs() << "Exactly one must have the rotation pattern\n";
//       return false;
//     }

//     // Identify which is rotated vs unrotated
//     MultiplyOp rotatedMul = lhsRotationInput ? lhsMul : rhsMul;
//     MultiplyOp unrotatedMul = lhsRotationInput ? rhsMul : lhsMul;
//     Value x = lhsRotationInput ? lhsRotationInput : rhsRotationInput;
//     Value rotatedValue = lhsRotationInput ? lhsRotatedValue :
//     rhsRotatedValue;

//     // Match unrotated projection: x * cos (cos is the operand that isn't x)
//     Value cosEmbed;
//     if (!matchUnrotatedProjection(unrotatedMul, x, cosEmbed)) {
//       llvm::errs() << "Failed to match unrotated projection for RoPE\n";
//       return false;
//     }

//     // Match rotated projection: rotate_half(x) * sin (sin is the operand
//     that
//     // isn't the rotated value)
//     Value sinEmbed;
//     if (!matchRotatedProjection(rotatedMul, rotatedValue, sinEmbed)) {
//       llvm::errs() << "Failed to match rotated projection for RoPE\n";
//       return false;
//     }

//     // Populate components
//     c.x = x;
//     c.cos = cosEmbed;
//     c.sin = sinEmbed;
//     c.cosMul = unrotatedMul;
//     c.sinMul = rotatedMul;

//     return true;
//   }

//   // Check if multiply contains rotate_half pattern.
//   // Returns {input_x, rotated_value} if found, {nullptr, nullptr} otherwise.
//   // rotate_half(x) = concat(neg(slice(x, half:end)), slice(x, 0:half))
//   std::pair<Value, Value> matchRotationPattern(MultiplyOp mulOp) const {
//     Value lhs = skipNeutral(mulOp.getLhs());
//     Value rhs = skipNeutral(mulOp.getRhs());

//     // Check both operands for rotation pattern
//     if (Value input = getRotationInput(lhs)) {
//       return {input, lhs};
//     }
//     if (Value input = getRotationInput(rhs)) {
//       return {input, rhs};
//     }
//     return {nullptr, nullptr};
//   }

//   // Extract input x from rotate_half pattern
//   Value getRotationInput(Value v) const {
//     auto concatOp = v.getDefiningOp<ConcatOp>();
//     if (!concatOp || concatOp.getNumOperands() != 2) {
//       return nullptr;
//     }

//     Value negHalf = skipNeutral(concatOp.getOperand(0));
//     Value firstHalf = skipNeutral(concatOp.getOperand(1));

//     auto negOp = negHalf.getDefiningOp<NegOp>();
//     if (!negOp) {
//       return nullptr;
//     }

//     Value secondHalf = negOp.getInput();

//     auto firstSlice = skipNeutral(firstHalf).getDefiningOp<SliceStaticOp>();
//     auto secondSlice =
//     skipNeutral(secondHalf).getDefiningOp<SliceStaticOp>();

//     if (!firstSlice || !secondSlice) {
//       return nullptr;
//     }

//     Value input = skipNeutral(firstSlice.getInput());
//     if (input != skipNeutral(secondSlice.getInput())) {
//       return nullptr;
//     }

//     if (!validateRotationSlices(firstSlice, secondSlice, input)) {
//       return nullptr;
//     }

//     return input;
//   }

//   // Validate slice ranges: first_half [0, half), second_half [half, end)
//   bool validateRotationSlices(SliceStaticOp firstSlice,
//                               SliceStaticOp secondSlice, Value input) const {
//     auto inputType = mlir::cast<RankedTensorType>(input.getType());
//     ArrayRef<int64_t> inputShape = inputType.getShape();

//     // rotary_embedding requires 4D: [batch, num_heads, seq_len, head_dim]
//     if (inputShape.size() != 4) {
//       return false;
//     }

//     int64_t lastDim = inputShape.back();
//     if (lastDim <= 0 || lastDim % 2 != 0) {
//       return false;
//     }

//     int64_t halfDim = lastDim / 2;

//     auto firstBegins =
//         ttmlir::utils::getIntegerVector<int64_t>(firstSlice.getBegins());
//     auto firstEnds =
//         ttmlir::utils::getIntegerVector<int64_t>(firstSlice.getEnds());
//     if (!firstBegins || !firstEnds || firstBegins->back() != 0 ||
//         firstEnds->back() != halfDim) {
//       return false;
//     }

//     auto secondBegins =
//         ttmlir::utils::getIntegerVector<int64_t>(secondSlice.getBegins());
//     auto secondEnds =
//         ttmlir::utils::getIntegerVector<int64_t>(secondSlice.getEnds());
//     if (!secondBegins || !secondEnds || secondBegins->back() != halfDim ||
//         secondEnds->back() != lastDim) {
//       return false;
//     }

//     return true;
//   }

//   // Match unrotated projection: x * cos
//   // We know x from rotation pattern, cos is the other operand.
//   // Under trust boundary, we use skipNeutral consistently for tracing.
//   bool matchUnrotatedProjection(MultiplyOp mulOp, Value expectedX,
//                                 Value &cosEmbed) const {
//     Value lhs = skipNeutral(mulOp.getLhs());
//     Value rhs = skipNeutral(mulOp.getRhs());

//     if (lhs == expectedX) {
//       cosEmbed = skipNeutral(mulOp.getRhs());
//       return true;
//     }
//     if (rhs == expectedX) {
//       cosEmbed = skipNeutral(mulOp.getLhs());
//       return true;
//     }
//     return false;
//   }

//   // Match rotated projection: rotate_half(x) * sin
//   // We know the rotated value, sin is the other operand
//   // Note: rotatedValue has already been skipNeutral'd in
//   matchRotationPattern. bool matchRotatedProjection(MultiplyOp mulOp, Value
//   rotatedValue,
//                               Value &sinEmbed) const {
//     Value lhs = skipNeutral(mulOp.getLhs());
//     Value rhs = skipNeutral(mulOp.getRhs());

//     if (lhs == rotatedValue) {
//       sinEmbed = rhs;
//       return true;
//     }
//     if (rhs == rotatedValue) {
//       sinEmbed = lhs;
//       return true;
//     }
//     return false;
//   }

//   //
//   ============================================================================
//   // Compute Config
//   //
//   ============================================================================

//   // Create compute config with FP32 dest accumulation if the multiply or add
//   // operations use F32 element type. Returns nullptr if F32 is not used.
//   static DeviceComputeKernelConfigAttr
//   createComputeConfigIfNeeded(MLIRContext *context, RoPEComponents &c) {
//     auto cosMulType =
//         mlir::cast<RankedTensorType>(c.cosMul.getResult().getType());
//     auto sinMulType =
//         mlir::cast<RankedTensorType>(c.sinMul.getResult().getType());
//     auto addType =
//     mlir::cast<RankedTensorType>(c.addOp.getResult().getType());

//     bool usesF32 = cosMulType.getElementType().isF32() ||
//                    sinMulType.getElementType().isF32() ||
//                    addType.getElementType().isF32();

//     if (!usesF32) {
//       return nullptr;
//     }

//     return DeviceComputeKernelConfigAttr::get(
//         context,
//         /*mathFidelity=*/MathFidelity::HiFi4,
//         /*mathApproxMode=*/BoolAttr::get(context, false),
//         /*fp32DestAccEn=*/BoolAttr::get(context, true),
//         /*packerL1Acc=*/BoolAttr::get(context, true),
//         /*dstFullSyncEn=*/nullptr);
//   }

//   //
//   ============================================================================
//   // Validation
//   //
//   ============================================================================

//   bool validateSemantics(RoPEComponents &c) const {
//     if (!c.x || !c.cos || !c.sin) {
//       return false;
//     }

//     auto inputType = mlir::cast<RankedTensorType>(c.x.getType());
//     if (inputType.getRank() != 4) {
//       return false;
//     }

//     ArrayRef<int64_t> inputShape = inputType.getShape();
//     int64_t headDim = inputShape.back();

//     if (headDim <= 0 || headDim % 2 != 0) {
//       return false;
//     }

//     // Validate input and output shapes are compatible
//     auto outputType = mlir::cast<RankedTensorType>(c.addOp.getType());
//     ArrayRef<int64_t> outputShape = outputType.getShape();

//     if (!validateInputOutputShapes(inputShape, outputShape)) {
//       return false;
//     }

//     // Validate cos/sin shapes are broadcastable to input
//     if (!isBroadcastableTo(c.cos, inputShape) ||
//         !isBroadcastableTo(c.sin, inputShape)) {
//       return false;
//     }

//     return true;
//   }

//   // Validate that input and output shapes are compatible for RoPE.
//   // Either shapes match exactly, or input can be permuted to match output.
//   bool validateInputOutputShapes(ArrayRef<int64_t> inputShape,
//                                  ArrayRef<int64_t> outputShape) const {
//     if (inputShape.size() != 4 || outputShape.size() != 4) {
//       return false;
//     }

//     // Check if shapes match exactly
//     if (inputShape == outputShape) {
//       return true;
//     }

//     // Check if input can be permuted to match output
//     // First verify they have the same elements (necessary for valid
//     permutation) SmallVector<int64_t> sortedInput(inputShape);
//     SmallVector<int64_t> sortedOutput(outputShape);
//     llvm::sort(sortedInput);
//     llvm::sort(sortedOutput);

//     if (sortedInput != sortedOutput) {
//       return false;
//     }

//     // Generate the permutation and verify last two dims (seq_len, head_dim)
//     // match after permutation
//     auto permutation =
//         ttmlir::utils::generatePermutation(inputShape, outputShape);
//     auto permutedInput =
//         ttmlir::utils::applyPermutation(inputShape, permutation);

//     // After permutation, shapes should match
//     return permutedInput == SmallVector<int64_t>(outputShape);
//   }

//   // Check if embedding tensor shape is broadcastable to target shape
//   bool isBroadcastableTo(Value embed, ArrayRef<int64_t> targetShape) const {
//     auto embedType = mlir::dyn_cast<RankedTensorType>(embed.getType());
//     if (!embedType) {
//       return false;
//     }

//     ArrayRef<int64_t> embedShape = embedType.getShape();

//     // Must have same rank for rotary embedding
//     if (embedShape.size() != targetShape.size()) {
//       return false;
//     }

//     // Each dimension must match or be 1 (broadcastable)
//     for (size_t i = 0; i < embedShape.size(); ++i) {
//       if (embedShape[i] != targetShape[i] && embedShape[i] != 1) {
//         return false;
//       }
//     }

//     return true;
//   }

//   //
//   ============================================================================
//   // Input Preparation
//   //
//   ============================================================================

//   // Prepare inputs for RoPE operation.
//   // If input x shape differs from output shape, permute x to match.
//   void prepareInputs(RoPEComponents &c, mlir::PatternRewriter &rewriter)
//   const {
//     auto inputType = mlir::cast<RankedTensorType>(c.x.getType());
//     auto outputType = mlir::cast<RankedTensorType>(c.addOp.getType());

//     ArrayRef<int64_t> inputShape = inputType.getShape();
//     ArrayRef<int64_t> outputShape = outputType.getShape();

//     // If shapes already match, no permutation needed
//     if (inputShape == outputShape) {
//       return;
//     }

//     // Generate permutation to transform input shape to output shape
//     auto permutation =
//         ttmlir::utils::generatePermutation(inputShape, outputShape);

//     // Apply permutation to x
//     c.x = ttir_to_ttnn::utils::generatePermute(
//         mlir::cast<TypedValue<RankedTensorType>>(c.x),
//         llvm::to_vector(permutation), rewriter, c.addOp.getLoc());
//   }

//   //
//   ============================================================================
//   // Op Creation
//   //
//   ============================================================================

//   mlir::LogicalResult createRotaryEmbeddingOp(mlir::PatternRewriter
//   &rewriter,
//                                               RoPEComponents &c) const {
//     op_model::ScopedSingletonDeviceGuard deviceGuard;

//     // Prepare inputs (permute x if needed to match output shape)
//     prepareInputs(c, rewriter);

//     // Create compute config with FP32 dest accumulation if original ops used
//     F32 DeviceComputeKernelConfigAttr computeConfig =
//       createComputeConfigIfNeeded(rewriter.getContext(), c);

//     auto resultType = c.addOp.getType();
//     auto ropeOp = rewriter.create<RotaryEmbeddingOp>(
//         c.addOp.getLoc(), resultType, c.x, c.cos, c.sin,
//         /*token_index=*/nullptr,
//         /*memory_config=*/nullptr, computeConfig);

//     std::vector<TTNNLayoutAttr> inputLayouts =
//         utils::extractInputLayouts(ropeOp.getOperation());

//     OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
//     auto result = op_constraint_validation::validateOperation(
//         ropeOp.getOperation(), inputLayouts, config);

//     if (!result.isSuccess()) {
//       llvm::errs() << "Failed to validate operation for RoPE: \n";
//       llvm::errs() << result.errorMessage << "\n";
//       rewriter.eraseOp(ropeOp);
//       return failure();
//     }

//     rewriter.replaceOp(c.addOp, ropeOp.getResult());
//     return mlir::success();
//   }
// };

// ============================================================================
// SDPA Fusing
// ============================================================================
//
// Matches Scaled Dot Product Attention:
//   Attention(Q, K, V) = softmax((Q @ K^T) * scale + mask) @ V
//
// Anchors on the final matmul (attention @ V) and walks backward:
//
//   matmul (attention @ V)
//      |
//   [where]          <- optional causal masking
//      |
//   softmax
//      |
//   [add(mask)]      <- optional attention mask
//      |
//   [multiply(scale) | divide(scale)] <- optional scaling factor
//      |
//   matmul (Q @ K^T)
//
// Uses skipTransparent() to handle type conversions and layout ops that don't
// change semantics, making the pattern robust to variations in the IR.
//
class SDPAFusing : public mlir::OpRewritePattern<MatmulOp> {
  using SDPAFusing::OpRewritePattern<MatmulOp>::OpRewritePattern;

  // SDPA Query, Key, Value tensors have shape [B, H, S, D]
  // (Batch, NumHeads, SeqLen, HeadDim).
  static constexpr int64_t kNumHeadsDim = 1;
  static constexpr int64_t kSeqLenDim = 2;

  // Permutation to convert query from [B, H, S, D] -> [S, B, H, D] for SDPA
  // decode op.
  static constexpr std::array<int64_t, 4> kToDecodePermutation = {2, 0, 1, 3};

  // Permutation to un-transpose key from [B, H, D, S] -> [B, H, S, D].
  // Used when key comes from SplitQueryKeyValueAndSplitHeadsOp with
  // transpose_key=true.
  static constexpr std::array<int64_t, 4> kUnTransposeKeyPermutation = {0, 1, 3,
                                                                        2};

public:
  mlir::LogicalResult
  matchAndRewrite(MatmulOp srcOp,
                  mlir::PatternRewriter &rewriter) const override {
    SDPAComponents c;
    c.attentionMatmul = srcOp;
    c.value = srcOp.getB();

    // Match: matmul -> [where] -> softmax -> score
    if (!matchSoftmaxPath(srcOp.getA(), c)) {
      return failure();
    }

    if (!matchScoreComputation(c.softmax.getInput(), c)) {
      return failure();
    }

    // Validate semantic constraints (single-use of intermediate ops, valid
    // scale range) before modifying the IR.
    if (!validateSemantics(c)) {
      return failure();
    }

    // Prepare inputs for SDPA: normalize Q/K/V/mask by skipping transparent ops
    // and dropping matmul-only transforms (e.g. K^T permute, GQA head
    // expansion). Key un-transpose for SDPA op legality is handled during input
    // canonicalization (see unTransposeKeyIfNeeded()).
    prepareInputsForSDPA(c, rewriter);

    return createSDPAOp(rewriter, c);
  }

private:
  struct SDPAComponents {
    Value query, key, value, mask;
    std::optional<float> scale;
    MatmulOp attentionMatmul;
    SoftmaxOp softmax;
    Operation *scoreOp = nullptr;
  };

  // ============================================================================
  // Transparent Op Utilities
  // ============================================================================

  // Operations that don't change semantic meaning - can be traced through.
  static bool isTransparentOp(Operation *op) {
    return isa<ToLayoutOp, ToMemoryConfigOp, TypecastOp>(op);
  }

  // Skip through transparent ops to find the semantic operation.
  Value skipTransparent(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (!isTransparentOp(defOp)) {
        break;
      }
      v = defOp->getOperand(0);
    }
    return v;
  }

  // ============================================================================
  // Layout / Transpose Utilities
  // ============================================================================

  // Check if a permutation is a transpose on the last two dimensions.
  // For a 4D tensor [B, H, S, D], a transpose permutation would be [0, 1, 3,
  // 2]. This is the typical transpose used before matrix multiplication.
  static bool isTransposeOnLastTwoDims(ArrayRef<int64_t> perm) {
    if (perm.size() < 2) {
      return false;
    }

    size_t n = perm.size();
    // Check that all dimensions except the last two are identity.
    for (size_t i = 0; i < n - 2; ++i) {
      if (perm[i] != static_cast<int64_t>(i)) {
        return false;
      }
    }

    // Check that the last two dimensions are swapped.
    return perm[n - 2] == static_cast<int64_t>(n - 1) &&
           perm[n - 1] == static_cast<int64_t>(n - 2);
  }

  // Check if key is transposed by looking at its source operation or shape.
  // Returns true if:
  // 1. Key came from SplitQueryKeyValueAndSplitHeadsOp with transpose_key=true
  // 2. Key shape suggests transposition: K[B, H, D, S] where D matches Q's
  //    head_dim and S matches V's seq_len
  bool isKeyTransposed(Value key, Value query, Value value) const {
    // Check explicit source first
    Operation *defOp = key.getDefiningOp();
    if (auto splitOp =
            dyn_cast_or_null<SplitQueryKeyValueAndSplitHeadsOp>(defOp)) {
      return splitOp.getTransposeKey();
    }

    // Shape-based detection for keys transposed via permute operations
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

    if (!kType || !qType || !vType || kType.getRank() != 4 ||
        qType.getRank() != 4 || vType.getRank() != 4) {
      return false;
    }

    auto kShape = kType.getShape();
    auto qShape = qType.getShape();
    auto vShape = vType.getShape();

    // Q: [B, H, S_q, head_dim], K_normal: [B, H, S_k, head_dim]
    // K_transposed: [B, H, head_dim, S_k]
    int64_t qHeadDim = qShape[3];
    int64_t vSeqLen = vShape[kSeqLenDim];

    // If K's dim[2] matches Q's head_dim and K's dim[3] matches V's seq_len,
    // then K is transposed: [B, H, head_dim, seq_k]
    bool kDim2MatchesHeadDim = kShape[2] == qHeadDim;
    bool kDim3MatchesSeqLen = kShape[3] == vSeqLen;

    return kDim2MatchesHeadDim && kDim3MatchesSeqLen;
  }

  // ============================================================================
  // Constant Extraction
  // ============================================================================

  std::optional<float> extractConstant(Value v) const {
    // Skip transparent ops to find the actual constant.
    v = skipTransparent(v);

    // Direct FullOp.
    if (auto fullOp = v.getDefiningOp<FullOp>()) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        return attr.getValue().convertToFloat();
      }
    }

    // Try load_cached - look up the const_eval function and find FullOp inside.
    if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
      auto callee = loadCached.getCallee();
      auto moduleOp = loadCached->getParentOfType<ModuleOp>();
      if (!moduleOp) {
        return std::nullopt;
      }

      auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
      if (!funcOp) {
        return std::nullopt;
      }

      // Walk the function body to find a FullOp.
      std::optional<float> result;
      funcOp.walk([&](FullOp fullOp) {
        if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
          result = attr.getValue().convertToFloat();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      return result;
    }

    return std::nullopt;
  }

  // ============================================================================
  // Q/K Extraction with Scale Handling
  // ============================================================================

  // Extract tensor and its scale. Checks if skipping transparent ops leads to a
  // multiply with a constant scale. If so, extracts the scale and returns the
  // tensor input. Otherwise returns the original value unchanged.
  std::pair<Value, std::optional<float>> extractTensorWithScale(Value v) const {
    std::optional<float> scale;

    // Check if transparent ops lead to a multiply (scale applied to tensor).
    Value skipped = skipTransparent(v);
    if (auto mulOp = skipped.getDefiningOp<MultiplyOp>()) {
      if (auto s = extractConstant(mulOp.getRhs())) {
        scale = s;
        return {mulOp.getLhs(), scale};
      }
      if (auto s = extractConstant(mulOp.getLhs())) {
        scale = s;
        return {mulOp.getRhs(), scale};
      }
    }

    // No multiply found - return original value unchanged.
    return {v, scale};
  }

  // Returns false if we find both post-matmul scaling AND pre-scaling on Q/K,
  // which would indicate this is likely not a standard SDPA pattern.
  // Also rejects if Q or K comes from a LoadCachedOp (const-eval function).
  bool extractQKWithScales(Value a, Value b, SDPAComponents &c) const {
    auto [query, qScale] = extractTensorWithScale(a);
    auto [key, kScale] = extractTensorWithScale(b);

    // Reject if we found both post-matmul scale and pre-scaling on Q/K.
    // Standard SDPA uses one or the other, not both.
    bool hasPostMatmulScale = c.scale.has_value();
    bool hasPreScale = qScale.has_value() || kScale.has_value();
    if (hasPostMatmulScale && hasPreScale) {
      return false;
    }

    c.query = query;
    c.key = key;

    // Combine pre-scales if present: Q*s and K*s → combined scale = s*s.
    if (hasPreScale) {
      float qs = qScale.value_or(1.0f);
      float ks = kScale.value_or(1.0f);
      c.scale = qs * ks;
    }
    return true;
  }

  // ============================================================================
  // Pattern Matching with Backtracking
  // ============================================================================

  // Match: [Typecast] -> [where(cond, zeros, softmax)] -> softmax
  bool matchSoftmaxPath(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try where(cond, zeros, softmax) pattern first
    if (auto whereOp = v.getDefiningOp<WhereOp>()) {
      Value softmaxCandidate = skipTransparent(whereOp.getThird());
      if (auto softmax = softmaxCandidate.getDefiningOp<SoftmaxOp>()) {
        c.softmax = softmax;
        return true;
      }
    }

    // Direct softmax
    if (auto softmax = v.getDefiningOp<SoftmaxOp>()) {
      c.softmax = softmax;
      return true;
    }

    return false;
  }

  // Match score computation with backtracking for different orderings.
  // Patterns (in order of priority):
  //   1. [transparent] -> linear(Q_scaled, K_scaled, mask)
  //   2. [transparent] -> add(score_chain, mask)
  //   3. [transparent] -> score_chain (no mask)
  bool matchScoreComputation(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Try linear(Q_scaled, K_scaled, mask) first
    if (auto linearOp = v.getDefiningOp<LinearOp>()) {
      c.scoreOp = linearOp;
      if (!extractQKWithScales(linearOp.getA(), linearOp.getB(), c)) {
        return false;
      }
      if (linearOp.getBias()) {
        c.mask = linearOp.getBias();
      }
      return true;
    }

    // Try add(score, mask) with both operand orderings
    if (auto addOp = v.getDefiningOp<AddOp>()) {
      // Try lhs as score, rhs as mask
      if (matchScoreChain(addOp.getLhs(), c)) {
        c.mask = addOp.getRhs();
        return true;
      }
      // Try rhs as score, lhs as mask
      if (matchScoreChain(addOp.getRhs(), c)) {
        c.mask = addOp.getLhs();
        return true;
      }
      return false;
    }

    // No add - try direct score chain (no mask)
    return matchScoreChain(v, c);
  }

  // Match: [transparent] -> [multiply(*, scale) | divide(*, scale)] ->
  //        [transparent] -> matmul
  // Extracts scale if present, then matches the Q@K matmul.
  bool matchScoreChain(Value v, SDPAComponents &c) const {
    v = skipTransparent(v);

    // Optional multiply for scale (post-matmul scaling)
    if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
      if (auto scale = extractConstant(mulOp.getRhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getLhs());
      } else if (auto scale = extractConstant(mulOp.getLhs())) {
        c.scale = scale;
        v = skipTransparent(mulOp.getRhs());
      }
    }

    // Optional divide for scale (post-matmul scaling, e.g. SegFormer style)
    // Division by X is equivalent to multiply by 1/X.
    else if (auto divOp = v.getDefiningOp<DivideOp>()) {
      if (auto divisor = extractConstant(divOp.getRhs())) {
        if (*divisor != 0.0f) {
          c.scale = 1.0f / *divisor;
          v = skipTransparent(divOp.getLhs());
        }
      }
    }

    // Must end with matmul (different from attention matmul)
    if (auto matmul = v.getDefiningOp<MatmulOp>()) {
      if (matmul != c.attentionMatmul) {
        c.scoreOp = matmul;
        if (!extractQKWithScales(matmul.getA(), matmul.getB(), c)) {
          return false;
        }
        return true;
      }
    }

    return false;
  }

  // ============================================================================
  // Input Canonicalization (dtype/TM/mask)
  // ============================================================================

  static Value restoreElementTypeIfNeeded(Value v, Type elementType,
                                          PatternRewriter &rewriter) {
    auto vType = cast<RankedTensorType>(v.getType());
    if (vType.getElementType() == elementType) {
      return v;
    }

    // Convert MLIR element type to ttcore::DataType.
    auto dataType = ttcore::elementTypeToDataType(elementType);

    // Create new tensor type with correctly updated encoding.
    auto castType = utils::RankedTensorTypeFactory::create(vType, dataType);
    return rewriter.create<TypecastOp>(
        v.getLoc(), castType, v,
        ttcore::DataTypeAttr::get(rewriter.getContext(), dataType));
  }

  // Find the element type at the end of a "TM chain" (typecast/reshape/permute/
  // repeat_interleave), without allocating a temporary vector. This keeps dtype
  // expectations stable when we later drop some of these ops for SDPA inputs.
  static Type getTargetElementType(Value v) {
    Type lastSeen = cast<RankedTensorType>(v.getType()).getElementType();
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp, ReshapeOp, PermuteOp, RepeatInterleaveOp>(defOp)) {
        v = defOp->getOperand(0);
        lastSeen = cast<RankedTensorType>(v.getType()).getElementType();
        continue;
      }

      break;
    }
    return lastSeen;
  }

  std::pair<Value, Type> analyzeQ(Value v) const {
    if (auto typecastOp = v.getDefiningOp<TypecastOp>()) {
      v = typecastOp.getInput();
    }

    // If Q comes from load_cached, trace through const-eval function to find
    // the original dtype before any f32 conversions.
    if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
      auto funcOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          loadCached, loadCached.getCalleeAttr());
      if (funcOp) {
        // Find the return op and get the value corresponding to this result.
        unsigned resultIdx = cast<OpResult>(v).getResultNumber();
        for (auto &block : funcOp.getBody()) {
          if (auto returnOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
            Value innerV = returnOp.getOperand(resultIdx);
            // Extract original tensor before scaling to get the true dtype.
            auto [originalTensor, scale] = extractTensorWithScale(innerV);
            return {v, getTargetElementType(originalTensor)};
          }
        }
      }
    }

    return {v, getTargetElementType(v)};
  }

  // Analyze K tensor: trace through TMs we can drop,
  // and track whether we skipped a K^T permute.
  std::tuple<Value, Type, bool> analyzeK(Value v) const {
    Type targetDtype = getTargetElementType(v);
    bool skippedTranspose = false;

    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
        // Only skip if it's GQA head expansion (on dim 1 in [B,H,S,D])
        if (repeatOp.getDim() == kNumHeadsDim) {
          v = repeatOp.getInput();
          continue;
        }
        break;
      }

      if (auto permuteOp = dyn_cast<PermuteOp>(defOp)) {
        // Only skip if it's a transpose on last two dims (K^T for matmul)
        if (isTransposeOnLastTwoDims(permuteOp.getPermutation())) {
          v = permuteOp.getInput();
          skippedTranspose = true;
          continue;
        }
      }

      break;
    }

    return {v, targetDtype, skippedTranspose};
  }

  // Analyze V tensor: trace through TMs we can drop.
  std::pair<Value, Type> analyzeV(Value v) const {
    Type targetDtype = getTargetElementType(v);

    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatInterleaveOp>(defOp)) {
        // Only skip if it's GQA head expansion (on dim 1 in [B,H,S,D])
        if (repeatOp.getDim() == kNumHeadsDim) {
          v = repeatOp.getInput();
          continue;
        }
        break;
      }

      break;
    }

    return {v, targetDtype};
  }

  // Trace mask back through broadcast materialization ops (e.g. RepeatOp).
  //
  // Many frontends materialize attention mask broadcasts early (often via
  // `ttnn.repeat`) to match the score tensor shape. For SDPA we prefer to keep
  // the original mask and let broadcastMaskForSDPA() re-broadcast to the exact
  // shape required by the fused op.
  Value prepareMask(Value v) const {
    while (Operation *defOp = v.getDefiningOp()) {
      if (isa<TypecastOp>(defOp)) {
        v = defOp->getOperand(0);
        continue;
      }

      if (auto repeatOp = dyn_cast<RepeatOp>(defOp)) {
        v = repeatOp.getInput();
        continue;
      }

      break;
    }
    return v;
  }

  // Slice mask on head dimension if it was broadcasted.
  //
  // TTNN SDPA expects mask with shape [B, 1, S_q, S_kv], but some frontends
  // may broadcast the mask to [B, H, S_q, S_kv] matching Q's num_heads.
  // We slice to [B, 1, S_q, S_kv] which SDPA can then broadcast internally.
  Value sliceMaskOnHeadDimIfNeeded(Value mask, PatternRewriter &rewriter,
                                   Location loc) const {
    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto maskShape = maskType.getShape();

    // Only handle 4D masks.
    if (maskShape.size() != 4) {
      return mask;
    }

    // If head dim (dim 1) is already 1, no slicing needed.
    if (maskShape[1] == 1) {
      return mask;
    }

    // Slice to get [B, 1, S_q, S_kv].
    SmallVector<int32_t> begins = {0, 0, 0, 0};
    SmallVector<int32_t> ends = {static_cast<int32_t>(maskShape[0]), 1,
                                 static_cast<int32_t>(maskShape[2]),
                                 static_cast<int32_t>(maskShape[3])};
    SmallVector<int32_t> steps = {1, 1, 1, 1};

    SmallVector<int64_t> resultShape = {maskShape[0], 1, maskShape[2],
                                        maskShape[3]};
    auto resultType =
        utils::RankedTensorTypeFactory::create(maskType, resultShape);

    return rewriter.create<SliceStaticOp>(
        loc, resultType, mask, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(steps));
  }

  // Prepare matched inputs for SDPA operation.
  //
  // This normalizes inputs while keeping the pattern robust to frontend
  // variations:
  // - Skip transparent ops (ToLayout, ToMemoryConfig, Typecast)
  // - Drop matmul-only transforms on K/V (K^T permute, typecast wrappers, GQA
  //   head expansion via repeat_interleave)
  // - Trace mask through broadcast materialization (RepeatOp) to recover the
  //   original mask and let broadcastMaskForSDPA() re-broadcast precisely
  //
  // Each preparation step is only committed if shapes remain SDPA-legal.
  void prepareInputsForSDPA(SDPAComponents &c,
                            PatternRewriter &rewriter) const {
    // Analyze all inputs upfront before committing any changes.
    // This ensures K and V are validated together (important for GQA where
    // both may need repeat_interleave traced through).
    auto [preparedQ, preparedQElementType] = analyzeQ(c.query);
    auto [preparedK, preparedKElementType, skippedKTranspose] = analyzeK(c.key);
    auto [preparedV, preparedVElementType] = analyzeV(c.value);

    // Validate and commit Q.
    if (validateShapes(preparedQ, c.key, c.value)) {
      c.query =
          restoreElementTypeIfNeeded(preparedQ, preparedQElementType, rewriter);
    } else {
      c.query =
          restoreElementTypeIfNeeded(c.query, preparedQElementType, rewriter);
    }

    // Validate K and V together - both must be prepared or neither.
    // This handles GQA where K and V are both traced through repeat_interleave.
    if (validateShapes(c.query, preparedK, preparedV)) {
      c.key = preparedK;
      c.value = preparedV;
    }

    // If key is still in a transposed form, materialize an un-transpose so the
    // fused SDPA op sees the expected [B, H, S, D] shape. Do this before
    // restoring element type so the permute operates on the traced-back value.
    //  Only do this if we didn't already skip a K^T permute during
    // tracing, to avoid adding a unneeded transpose when shapes are ambiguous
    // (e.g., when seq_k == head_dim).
    if (!skippedKTranspose) {
      c.key = unTransposeKeyIfNeeded(c.query, c.key, c.value, rewriter,
                                     c.attentionMatmul.getLoc());
    }

    // Restore element types for K and V after any shape transformations.
    c.key = restoreElementTypeIfNeeded(c.key, preparedKElementType, rewriter);
    c.value =
        restoreElementTypeIfNeeded(c.value, preparedVElementType, rewriter);

    if (c.mask) {
      c.mask = prepareMask(c.mask);

      // If mask is broadcasted on head dimension (dim 1), slice it to
      // [B, 1, S_q, S_kv] since TTNN SDPA doesn't support this broadcast.
      c.mask = sliceMaskOnHeadDimIfNeeded(c.mask, rewriter,
                                          c.attentionMatmul.getLoc());

      // The mask should have the same element type as the qkv tensors.
      c.mask =
          restoreElementTypeIfNeeded(c.mask, preparedQElementType, rewriter);
    }
  }

  // ============================================================================
  // Key Un-transpose
  // ============================================================================

  // If key appears transposed (via source op or shape heuristic), generate a
  // permute to restore the expected shape [B, H, S, D] for SDPA.
  Value unTransposeKeyIfNeeded(Value query, Value key, Value value,
                               mlir::PatternRewriter &rewriter,
                               Location loc) const {
    if (!isKeyTransposed(key, query, value)) {
      return key;
    }

    // Generate permute to un-transpose: [B, H, D, S] -> [B, H, S, D]
    return ttir_to_ttnn::utils::generatePermute(
        mlir::cast<TypedValue<RankedTensorType>>(key),
        llvm::to_vector(kUnTransposeKeyPermutation), rewriter, loc);
  }

  // ============================================================================
  // Validation
  // ============================================================================

  // Check if an SDPA validation error can be recovered by TTNNWorkarounds pass.
  // These errors are handled by
  // ScaledDotProductAttentionPadTileDimsRewritePattern which pads:
  // - sequence dimensions to be divisible by chunk size (32) when mask is
  //   present
  // - head dimensions to be divisible by tile width (32) always
  static bool isRecoverableSDPAError(const std::string &errorMessage) {
    // Q sequence length not divisible by q_chunk_size (default 32)
    if (errorMessage.find(
            "Q sequence length must be divisible by q_chunk_size") !=
        std::string::npos) {
      return true;
    }
    // K sequence length not divisible by k_chunk_size (default 32)
    if (errorMessage.find(
            "K sequence length must be divisible by k_chunk_size") !=
        std::string::npos) {
      return true;
    }
    // Head dimension not tile-aligned (requires padding)
    if (errorMessage.find("Padding is not supported on the head_dim") !=
        std::string::npos) {
      return true;
    }
    return false;
  }

  bool validateShapes(Value query, Value key, Value value) const {
    if (!query || !key || !value) {
      return false;
    }

    auto qType = mlir::dyn_cast<RankedTensorType>(query.getType());
    auto kType = mlir::dyn_cast<RankedTensorType>(key.getType());
    auto vType = mlir::dyn_cast<RankedTensorType>(value.getType());

    if (!qType || !kType || !vType) {
      return false;
    }

    // Validate shapes: Q, K, V should be 4D tensors.
    // Q shape: [batch, num_heads, seq_q, head_dim]
    // K shape: [batch, num_kv_heads, seq_k, head_dim] or
    //          [batch, num_kv_heads, head_dim, seq_k] if transposed
    // V shape: [batch, num_kv_heads, seq_v, head_dim]
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();
    auto vShape = vType.getShape();

    if (qShape.size() != 4 || kShape.size() != 4 || vShape.size() != 4) {
      return false;
    }

    int64_t qHeadDim = qShape[3];
    int64_t vSeqLen = vShape[kSeqLenDim];
    int64_t vHeadDim = vShape[3];

    bool keyTransposed = isKeyTransposed(key, query, value);
    int64_t kSeqLen = keyTransposed ? kShape[3] : kShape[kSeqLenDim];
    int64_t kHeadDim = keyTransposed ? kShape[kSeqLenDim] : kShape[3];

    // Key and Value must have the same sequence length.
    if (kSeqLen != vSeqLen) {
      return false;
    }

    // Head dimensions must match across Q, K, V.
    if (qHeadDim != kHeadDim || kHeadDim != vHeadDim) {
      return false;
    }

    // Batch dimensions must match.
    if (qShape[0] != kShape[0] || kShape[0] != vShape[0]) {
      return false;
    }

    // Validate num_heads:
    // - K and V must have the same num_heads (num_kv_heads)
    // - Q's num_heads must be divisible by num_kv_heads (for GQA/MQA support)
    int64_t qNumHeads = qShape[kNumHeadsDim];
    int64_t kNumHeads = kShape[kNumHeadsDim];
    int64_t vNumHeads = vShape[kNumHeadsDim];

    if (kNumHeads != vNumHeads) {
      return false;
    }

    if (qNumHeads % kNumHeads != 0) {
      return false;
    }

    return true;
  }

  bool validateSemantics(const SDPAComponents &c) const {
    if (!c.query || !c.key || !c.value || !c.softmax || !c.scoreOp) {
      return false;
    }

    if (!validateShapes(c.query, c.key, c.value)) {
      return false;
    }

    if (!c.softmax->hasOneUse()) {
      return false;
    }

    // If softmax feeds into a typecast, verify the typecast also has one use.
    if (auto *softmaxUser = *c.softmax->getUsers().begin()) {
      if (isa<TypecastOp>(softmaxUser) && !softmaxUser->hasOneUse()) {
        return false;
      }
    }

    if (c.scale.has_value() && (*c.scale <= 0.0f || *c.scale > 1.0f)) {
      return false;
    }

    return true;
  }

  // Broadcast attention mask to the required shape for SDPA operations.
  //
  // For regular SDPA:
  //   Target mask shape: [batch, 1, query_seq, key_seq]
  //   - Dimension 1 (heads) stays as 1
  //
  // For decode SDPA:
  //   Target mask shape: [batch, 1, num_heads, key_seq]
  //   - Dimension 1 is query seq (always 1 for decode)
  //   - Dimension 2 must explicitly match num_heads
  Value broadcastMaskForSDPA(Value mask, RankedTensorType qType,
                             RankedTensorType kType, bool isDecode,
                             mlir::PatternRewriter &rewriter,
                             Location loc) const {
    if (!mask) {
      return mask;
    }

    auto maskType = mlir::cast<RankedTensorType>(mask.getType());
    auto qShape = qType.getShape();
    auto kShape = kType.getShape();

    // Compute target mask shape based on SDPA variant.
    // Q shape: [batch, num_heads, seq_len, head_dim]
    // K shape: [batch, num_heads, key_seq, head_dim]
    SmallVector<int64_t> targetShape;
    if (isDecode) {
      // Decode: [batch, 1, num_heads, key_seq]
      targetShape = {qShape[0], 1, qShape[kNumHeadsDim], kShape[kSeqLenDim]};
    } else {
      // Regular: [batch, 1, query_seq, key_seq]
      targetShape = {qShape[0], 1, qShape[kSeqLenDim], kShape[kSeqLenDim]};
    }

    // Check if broadcast is needed.
    if (llvm::equal(maskType.getShape(), targetShape)) {
      return mask;
    }

    auto broadcastType =
        utils::RankedTensorTypeFactory::create(maskType, targetShape);
    auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
        maskType.getShape(), targetShape);
    auto shapeAttr = ShapeAttr::get(rewriter.getContext(), broadcastDims);

    return rewriter.create<RepeatOp>(loc, broadcastType, mask, shapeAttr);
  }

  mlir::LogicalResult createSDPAOp(mlir::PatternRewriter &rewriter,
                                   SDPAComponents &c) const {
    op_model::ScopedSingletonDeviceGuard deviceGuard;

    // When no scale is found in the pattern, explicitly set scale=1.0 to
    // prevent tt-metal from applying the default 1/sqrt(head_dim) scaling.
    float scale = c.scale.value_or(1.0f);
    FloatAttr scaleAttr = rewriter.getF32FloatAttr(scale);

    // Capture original output element type to restore after SDPA if needed.
    auto originalOutputType =
        mlir::cast<RankedTensorType>(c.attentionMatmul.getResult().getType());
    Type originalElementType = originalOutputType.getElementType();

    auto qType = mlir::cast<RankedTensorType>(c.query.getType());
    auto qShape = qType.getShape();
    auto kType = mlir::cast<RankedTensorType>(c.key.getType());

    // Check if this is decode mode (query seq_len == 1)
    // Query shape: [batch x num_heads x seq_len x head_size]
    bool isDecode = qShape.size() == 4 && qShape[kSeqLenDim] == 1;
    // Broadcast mask to the required shape for the SDPA variant.
    Value attentionMask = broadcastMaskForSDPA(
        c.mask, qType, kType, isDecode, rewriter, c.attentionMatmul.getLoc());
    if (isDecode) {
      // Permute query: [B, H, 1, D] -> [1, B, H, D]
      Value permutedQuery = ttir_to_ttnn::utils::generatePermute(
          mlir::cast<TypedValue<RankedTensorType>>(c.query),
          llvm::to_vector(kToDecodePermutation), rewriter,
          c.attentionMatmul.getLoc());

      auto decodeOp = rewriter.create<ScaledDotProductAttentionDecodeOp>(
          c.attentionMatmul.getLoc(), permutedQuery.getType(), permutedQuery,
          c.key, c.value,
          /*is_causal=*/rewriter.getBoolAttr(false), attentionMask,
          /*cur_pos_tensor=*/Value(),
          /*attention_sink=*/Value(), scaleAttr,
          /*memory_config=*/MemoryConfigAttr(),
          /*program_config=*/SDPAProgramConfigAttr());

      // Validate the operation using op constraint validation
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(decodeOp.getOperation());

      auto resultType =
          mlir::cast<RankedTensorType>(decodeOp.getResult().getType());
      OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      auto result = op_constraint_validation::validateOperation(
          decodeOp.getOperation(), inputLayouts, config);

      if (!result.isSuccess() && !isRecoverableSDPAError(result.errorMessage)) {
        rewriter.eraseOp(decodeOp);
        return failure();
      }

      // Permute result back: [1, B, H, D] -> [B, H, 1, D].
      Value finalResult = ttir_to_ttnn::utils::generatePermute(
          decodeOp.getResult(),
          ttmlir::utils::inversePermutation(kToDecodePermutation), rewriter,
          c.attentionMatmul.getLoc());

      // Restore original element type if SDPA produced a different dtype.
      finalResult = restoreElementTypeIfNeeded(finalResult, originalElementType,
                                               rewriter);

      rewriter.replaceOp(c.attentionMatmul, finalResult);
    } else {
      auto sdpaOp = rewriter.create<ScaledDotProductAttentionOp>(
          c.attentionMatmul.getLoc(), c.query.getType(), c.query, c.key,
          c.value, attentionMask,
          /*is_causal=*/rewriter.getBoolAttr(false), scaleAttr,
          /*sliding_window_size=*/IntegerAttr(),
          /*memory_config=*/MemoryConfigAttr());

      // Validate the operation using op constraint validation
      std::vector<TTNNLayoutAttr> inputLayouts =
          utils::extractInputLayouts(sdpaOp.getOperation());

      auto resultType =
          mlir::cast<RankedTensorType>(sdpaOp.getResult().getType());
      OpConfig config(mlir::cast<TTNNLayoutAttr>(resultType.getEncoding()));
      auto result = op_constraint_validation::validateOperation(
          sdpaOp.getOperation(), inputLayouts, config);

      if (!result.isSuccess() && !isRecoverableSDPAError(result.errorMessage)) {
        rewriter.eraseOp(sdpaOp);
        return failure();
      }

      // Restore original element type if SDPA produced a different dtype.
      Value finalResult = restoreElementTypeIfNeeded(
          sdpaOp.getResult(), originalElementType, rewriter);

      rewriter.replaceOp(c.attentionMatmul, finalResult);
    }

    return mlir::success();
  }
};

#endif // TTMLIR_ENABLE_OPMODEL

class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    // TODO(mvasiljevic): Add HardsigmoidOp once tt-metal issue is resolved
    // https://github.com/tenstorrent/tt-metal/issues/30973
    patterns.add<
        TTNNConv2dWithActivation<ReluOp>, TTNNConv2dWithActivation<Relu6Op>,
        TTNNConv2dWithActivation<SiluOp>, TTNNConv2dWithActivation<SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SigmoidOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, SiluOp>,
        TTNNMatmulAndLinearWithActivation<MatmulOp, GeluOp>,
        TTNNMatmulAndLinearWithActivation<LinearOp, GeluOp>>(&getContext());

#ifdef TTMLIR_ENABLE_OPMODEL
    if (enableOpConstraints) {
      patterns.add<RoPENewFusing>(&getContext());
      patterns.add<SDPAFusing>(&getContext());
    }
#endif // TTMLIR_ENABLE_OPMODEL

    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
