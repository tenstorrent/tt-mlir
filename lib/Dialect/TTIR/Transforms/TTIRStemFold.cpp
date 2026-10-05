// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// clang-format off
// TTIRStemFold
// ============
//
// Folds a linear image stem - a 1x1 channel-mix conv, a channel concat of two
// space-to-depth branches and a 1x1 conv - into pixel_unshuffle(channels_last)
// + one 1x1 conv. The pass matches the input TTIR as the frontend emits it;
// nothing has to run in front of it.
//
// BEFORE (symbolic shapes; every conv2d carries batch_dim=0, height_dim=1,
// width_dim=2, channel_dim=3 and {channel_last}):
//
//   %x    : N x Cin x H x W (NCHW)      %Wm : Cm x Cin x 1 x 1    %bm : 1 x 1 x 1 x Cm   (mix conv)
//   %Wa   : Cu x 1 x kh x kw (average taps, groups = Cu, stride s, kh, kw <= s)
//   %W1   : Cout x Cfeat x 1 x 1         %b1 : 1 x 1 x 1 x Cout   (Cfeat = Cy*R^2 + Cu*(R/s)^2)
//
//   %m1 = transpose x2 / permute [0,2,3,1] (%x)                 -> N x H x W x Cin
//   %m2 = conv2d(%m1, %Wm, %bm) k=1x1 s=1 p=0 g=1               -> N x H x W x Cm          (a) mix
//   %m  = transpose x2 / permute [0,3,1,2] (%m2)                -> N x Cm x H x W          mix, NCHW
//   -- direct branch --
//   %d0 = index(%m) dim=1 [a,b)  /  slice_static                -> N x Cy x H x W
//   %d1 = reshape(%d0) [N,Cy,H/R,R,W/R,R] ; transposes/permutes -> composed {0,3,5,1,2,4} or {0,1,3,5,2,4}
//   %d  = reshape [N,Cy*R^2,H/R,W/R]                            -> N x Cy*R^2 x H/R x W/R  (b) s2d(R)
//   -- averaged branch --
//   %u0 = index(%m) dim=1 [c,d)                                 -> N x Cu x H x W
//   %u1 = permute [0,2,3,1] (%u0)                               -> N x H x W x Cu
//   %u2 = conv2d(%u1, %Wa) k=kh x kw, s=s, p=0, g=Cu, no bias    -> N x H/s x W/s x Cu      (c) average
//   %u3 = permute [0,3,1,2] (%u2)                               -> N x Cu x H/s x W/s
//   %u4 = reshape(%u3) [N,Cu,H/R,R/s,W/R,R/s] ; transposes      -> composed as above
//   %u  = reshape [N,Cu*(R/s)^2,H/R,W/R]                        -> N x Cu*(R/s)^2 x H/R x W/R  (c) s2d(R/s)
//   -- head --
//   %f  = concat(%d, %u) dim=1 (or -3)                          -> N x Cfeat x H/R x W/R   (d)
//   %f1 = permute [0,2,3,1] (%f)                                -> N x H/R x W/R x Cfeat
//   %y  = conv2d(%f1, %W1, %b1) k=1x1 s=1 p=0 g=1               -> N x H/R x W/R x Cout    (e)  <- anchor
//   %z  = clamp_scalar / relu6 (%y)                              -> N x H/R x W/R x Cout         (kept)
//
// AFTER:
//
//   %W, %b = fold of (%Wm, %bm, %Wa, %W1, %b1)                   Cout x Cp x 1 x 1, 1 x 1 x 1 x Cout
//                                                                 (parameters only -> const-eval hoists it)
//   %pu = pixel_unshuffle(%x) {downscale_factor = R, channel_order = channel_major,
//                              channels_last = true, padded_channels = Cp}      -> N x H/R x W/R x Cp (NHWC)
//   %y  = conv2d(%pu, %W, %b)  k=1x1 s=1 p=0 g=1 {channel_last}                 -> N x H/R x W/R x Cout
//   %z  = clamp_scalar / relu6 (%y)                                              (unchanged)
//
// with Cp = Cin*R^2 rounded up to a multiple of 32 (the kernel zero-pads the
// channels, W gets zero rows for them).
//
// Why it is exact: every op in (a)..(e) is linear and output pixel (u, v) only
// reads the R x R input block rows R*u..R*u+R-1, cols R*v..R*v+R-1 (the average
// reads rows R*u + s*ey + i, cols R*v + s*ex + k of that block, i < kh <= s,
// k < kw <= s). A linear shift-invariant map from an R x R x Cin patch to Cout
// values is an R x R / stride-R conv, i.e. a 1x1 conv on pixel_unshuffle(x, R).
//
// Weights. Fused input channel k = c*R^2 + q, q = dy*R + dx (channel-major
// order of the new pixel_unshuffle). Feature index j = position in the concat;
// inside a branch of C channels and factor r it is dy*(r*C) + dx*C + c
// (spatial-major) or c*r^2 + dy*r + dx (channel-major), as the branch's
// transpose chain dictates.
//
//   Wm   = reshape(%Wm) -> Cm x Cin ;  bm = reshape(%bm) -> 1 x Cm ;  W1 = reshape(%W1) -> Cout x Cfeat
//   -- per branch b (direct: C_b = Cy, r = R, T = 1 tap, V = Wm rows; averaged: C_b = Cu, r = R/s, T = kh*kw):
//   V_b  = Wa[c_b, t] * Wm[c0 + c_b, c]  (broadcast multiply C_b x T x 1 * C_b x 1 x Cin) -> reshape [C_b*T, Cin]
//   Q_b  = 0/1 constant [F_b*R^2, C_b*T] : row (j, q), col (c_b, t) = 1 iff feature j is channel c_b at
//          sub-position (ey, ex) and q == (s*ey + i)*R + (s*ex + k), tap t = i*kw + k
//   G_b  = reshape(permute(reshape(Q_b @ V_b -> [F_b*R^2, Cin], [F_b, R^2, Cin]), [0,2,1]), [F_b, Cin*R^2])
//   -- fold:
//   G    = concat(G_b over the branches, in concat order) dim 0  -> Cfeat x Cin*R^2
//   W_eq = W1 @ G                                                -> Cout x Cin*R^2
//   W    = reshape(pad(W_eq, [0,0, 0,Cp - Cin*R^2], 0.0), [Cout, Cp, 1, 1])
//   -- bias (the mix bias pushed through the head):
//   coef_b = bm[c0 + c_b] * Wa[c_b, t]  -> [1, C_b*T]       (bm slice itself for the direct branch)
//   Qq_b   = 0/1 constant [C_b*T, F_b] : row (c_b, t), col j = 1 iff feature j belongs to channel c_b
//   f0     = reshape(concat(coef_b @ Qq_b over the branches) dim 1, [Cfeat, 1])
//   b      = reshape(W1 @ f0, [1,1,1,Cout]) + %b1              (b_eq = W1 @ f0 + b1)
//
// Match restrictions (this stem and nothing else): a 1x1 conv with bias whose
// input is the NHWC view of a two-operand channel concat of space-to-depth
// blocks, one fed directly by a channel slice of the mix output and one fed
// through a grouped [kh,kw]/stride-s average conv (kh, kw <= s, no bias) of a
// channel slice of the same mix output; the mix output is the NCHW view of a
// 1x1 conv on the NCHW->NHWC view of the input; r_direct == r_avg * s == R;
// every intermediate value has a single use and the mix output is used by the
// two slices only; all weights are parameters (block arguments) or constants.
// Space-to-depth blocks are matched in the raw reshape -> transpose chain ->
// reshape form (channel order read from the composed 6D permutation) or as an
// already fused ttir.pixel_unshuffle.

// clang-format on

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSTEMFOLD
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Small matching helpers
//===----------------------------------------------------------------------===//

// Walk backwards through transpose / permute ops, composing them into one
// permutation (composed[i] = input dim that lands on output dim i). The chain
// output itself may have several users; every earlier op must have one.
static Value peelPermutationChain(Value v, SmallVector<int64_t> &composed) {
  auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
  if (!ty) {
    return {};
  }
  const int64_t rank = ty.getRank();
  composed.resize(rank);
  std::iota(composed.begin(), composed.end(), 0);
  Value cur = v;
  bool first = true;
  while (Operation *def = cur.getDefiningOp()) {
    SmallVector<int64_t> opPerm(rank);
    if (auto t = mlir::dyn_cast<TransposeOp>(def)) {
      if (!first && !t.getResult().hasOneUse()) {
        return {};
      }
      std::iota(opPerm.begin(), opPerm.end(), 0);
      int64_t d0 = t.getDim0() < 0 ? t.getDim0() + rank : t.getDim0();
      int64_t d1 = t.getDim1() < 0 ? t.getDim1() + rank : t.getDim1();
      if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank) {
        return {};
      }
      std::swap(opPerm[d0], opPerm[d1]);
      cur = t.getInput();
    } else if (auto p = mlir::dyn_cast<PermuteOp>(def)) {
      if (!first && !p.getResult().hasOneUse()) {
        return {};
      }
      if (static_cast<int64_t>(p.getPermutation().size()) != rank) {
        return {};
      }
      llvm::copy(p.getPermutation(), opPerm.begin());
      cur = p.getInput();
    } else {
      break;
    }
    first = false;
    SmallVector<int64_t> next(rank);
    for (int64_t i = 0; i < rank; ++i) {
      next[i] = opPerm[composed[i]];
    }
    composed = next;
  }
  return cur;
}

static bool isPerm(ArrayRef<int64_t> p, ArrayRef<int64_t> want) {
  return p.size() == want.size() &&
         std::equal(p.begin(), p.end(), want.begin());
}

// `v` is the NHWC view of an NCHW value (transposes or permute [0,2,3,1]).
static Value peelNchwToNhwc(Value v) {
  SmallVector<int64_t> composed;
  Value in = peelPermutationChain(v, composed);
  return (in && isPerm(composed, {0, 2, 3, 1})) ? in : Value{};
}

// `v` is the NCHW view of an NHWC value (transposes or permute [0,3,1,2]).
static Value peelNhwcToNchw(Value v) {
  SmallVector<int64_t> composed;
  Value in = peelPermutationChain(v, composed);
  return (in && isPerm(composed, {0, 3, 1, 2})) ? in : Value{};
}

// Weights must be function parameters or constants: the folded weights are
// built from them and the const-eval hoist moves that computation out.
static bool isParamLike(Value v) {
  if (mlir::isa<BlockArgument>(v)) {
    return true;
  }
  Operation *def = v.getDefiningOp();
  return def && def->hasTrait<OpTrait::ConstantLike>();
}

// conv2d stride / padding / dilation come as a scalar or an array attribute.
static SmallVector<int32_t> getI32Array(Attribute attr, int64_t n) {
  SmallVector<int32_t> out;
  if (auto d = mlir::dyn_cast_or_null<DenseI32ArrayAttr>(attr)) {
    out.assign(d.asArrayRef().begin(), d.asArrayRef().end());
  } else if (auto i = mlir::dyn_cast_or_null<IntegerAttr>(attr)) {
    out.assign(n, static_cast<int32_t>(i.getInt()));
  }
  return out;
}

static bool allEqual(ArrayRef<int32_t> v, int32_t x) {
  return !v.empty() && llvm::all_of(v, [x](int32_t e) { return e == x; });
}

static bool hasNhwcDims(Conv2dOp conv) {
  return conv.getBatchDim() == 0 && conv.getHeightDim() == 1 &&
         conv.getWidthDim() == 2 && conv.getChannelDim() == 3;
}

// 1x1, stride 1, unpadded, ungrouped NHWC conv with parameter weight/bias.
static bool isPointwiseConv(Conv2dOp conv) {
  auto wTy = mlir::dyn_cast<RankedTensorType>(conv.getWeight().getType());
  if (!wTy || wTy.getRank() != 4 || wTy.getDimSize(2) != 1 ||
      wTy.getDimSize(3) != 1 || conv.getGroups() != 1 || !hasNhwcDims(conv)) {
    return false;
  }
  return allEqual(getI32Array(conv.getStrideAttr(), 2), 1) &&
         allEqual(getI32Array(conv.getPaddingAttr(), 4), 0) &&
         allEqual(getI32Array(conv.getDilationAttr(), 2), 1) &&
         isParamLike(conv.getWeight()) &&
         (!conv.getBias() || isParamLike(conv.getBias()));
}

//===----------------------------------------------------------------------===//
// The matched stem
//===----------------------------------------------------------------------===//

// One operand of the channel concat.
struct Branch {
  enum Kind { Direct, Averaged } kind = Direct;
  int64_t c0 = 0, c1 = 0; // mix channels [c0, c1) feeding this branch
  int64_t r = 0;          // space-to-depth factor of this branch
  PixelUnshuffleChannelOrder order = PixelUnshuffleChannelOrder::ChannelMajor;
  // Averaged only: grouped [kh, kw] / stride-s conv in front of the block.
  int64_t s = 1, kh = 1, kw = 1;
  Value avgWeight; // [C, 1, kh, kw]

  int64_t channels() const { return c1 - c0; }
  int64_t features() const { return channels() * r * r; }
  int64_t taps() const { return kh * kw; }
};

struct StemMatch {
  Conv2dOp outConv; // (e) the 1x1 conv Cfeat -> Cout, anchor of the match
  Conv2dOp mixConv; // (a) the 1x1 conv Cin -> Cm
  Value x;          // NCHW network input [N, Cin, H, W]
  SmallVector<Branch, 2> branches; // in concat order
  int64_t N = 0, Cin = 0, H = 0, W = 0, R = 0, Cm = 0, Cfeat = 0, Cout = 0;
};

// Channel-range slice of an NCHW value: ttir.index(dim 1, step 1) or a
// ttir.slice_static that keeps every other dim whole.
static Value matchChannelSlice(Value v, int64_t &c0, int64_t &c1) {
  if (auto idx = v.getDefiningOp<IndexOp>()) {
    if (!idx.getResult().hasOneUse() || idx.getDim() != 1 ||
        idx.getStep() != 1) {
      return {};
    }
    c0 = idx.getBegin();
    c1 = idx.getEnd();
    return idx.getInput();
  }
  auto sl = v.getDefiningOp<SliceStaticOp>();
  if (!sl || !sl.getResult().hasOneUse()) {
    return {};
  }
  auto inTy = mlir::dyn_cast<RankedTensorType>(sl.getInput().getType());
  auto begins = sl.getBegins(), ends = sl.getEnds(), steps = sl.getStep();
  if (!inTy || inTy.getRank() != 4 || begins.size() != 4 || ends.size() != 4 ||
      steps.size() != 4) {
    return {};
  }
  auto asInt = [](Attribute a) { return mlir::cast<IntegerAttr>(a).getInt(); };
  for (int64_t d = 0; d < 4; ++d) {
    if (asInt(steps[d]) != 1) {
      return {};
    }
    if (d != 1 &&
        (asInt(begins[d]) != 0 || asInt(ends[d]) != inTy.getDimSize(d))) {
      return {};
    }
  }
  c0 = asInt(begins[1]);
  c1 = asInt(ends[1]);
  return sl.getInput();
}

// Space-to-depth block ending at `v`:
//   reshape [N,C,H,W] -> [N,C,H/r,r,W/r,r] -> transpose/permute chain ->
//   reshape [N,C*r^2,H/r,W/r]
// with the chain composing to {0,3,5,1,2,4} (spatial-major, c_out = dy*(r*C) +
// dx*C + c) or {0,1,3,5,2,4} (channel-major, c_out = c*r^2 + dy*r + dx); or an
// already fused ttir.pixel_unshuffle. Returns the 4D NCHW input of the block.
static Value matchSpaceToDepth(Value v, int64_t &r,
                               PixelUnshuffleChannelOrder &order) {
  if (auto pu = v.getDefiningOp<PixelUnshuffleOp>()) {
    if (!pu.getResult().hasOneUse() || pu.getChannelsLast()) {
      return {};
    }
    r = pu.getDownscaleFactor();
    order = pu.getChannelOrder();
    return pu.getInput();
  }
  auto reshape4D = v.getDefiningOp<ReshapeOp>();
  if (!reshape4D || !reshape4D.getResult().hasOneUse()) {
    return {};
  }
  auto outTy = mlir::dyn_cast<RankedTensorType>(reshape4D.getType());
  auto midTy = mlir::dyn_cast<RankedTensorType>(reshape4D.getInput().getType());
  if (!outTy || !midTy || outTy.getRank() != 4 || midTy.getRank() != 6) {
    return {};
  }
  SmallVector<int64_t> composed;
  Value chainIn = peelPermutationChain(reshape4D.getInput(), composed);
  if (!chainIn || composed.size() != 6) {
    return {};
  }
  if (isPerm(composed, {0, 3, 5, 1, 2, 4})) {
    order = PixelUnshuffleChannelOrder::SpatialMajor;
  } else if (isPerm(composed, {0, 1, 3, 5, 2, 4})) {
    order = PixelUnshuffleChannelOrder::ChannelMajor;
  } else {
    return {};
  }
  auto reshape6D = chainIn.getDefiningOp<ReshapeOp>();
  if (!reshape6D || !reshape6D.getResult().hasOneUse()) {
    return {};
  }
  auto inTy = mlir::dyn_cast<RankedTensorType>(reshape6D.getInput().getType());
  auto sixTy = mlir::dyn_cast<RankedTensorType>(reshape6D.getType());
  if (!inTy || !sixTy || inTy.getRank() != 4 || sixTy.getRank() != 6 ||
      !inTy.hasStaticShape()) {
    return {};
  }
  auto in = inTy.getShape(), six = sixTy.getShape(), out = outTy.getShape();
  const int64_t rr = six[3];
  if (rr <= 1 || six[5] != rr || six[0] != in[0] || six[1] != in[1] ||
      six[2] * rr != in[2] || six[4] * rr != in[3] || out[0] != in[0] ||
      out[1] != in[1] * rr * rr || out[2] != in[2] / rr ||
      out[3] != in[3] / rr) {
    return {};
  }
  r = rr;
  return reshape6D.getInput();
}

// Grouped average conv of the averaged branch: NHWC, weight [C, 1, kh, kw],
// groups == C, stride [s, s], no padding/dilation/bias, kh, kw <= s.
static bool matchAverageConv(Conv2dOp conv, Branch &b) {
  auto wTy = mlir::dyn_cast<RankedTensorType>(conv.getWeight().getType());
  auto stride = getI32Array(conv.getStrideAttr(), 2);
  if (!wTy || wTy.getRank() != 4 || wTy.getDimSize(1) != 1 || conv.getBias() ||
      !hasNhwcDims(conv) || !conv.getResult().hasOneUse() ||
      !isParamLike(conv.getWeight()) || stride.size() != 2 ||
      stride[0] != stride[1] || stride[0] < 1 ||
      !allEqual(getI32Array(conv.getPaddingAttr(), 4), 0) ||
      !allEqual(getI32Array(conv.getDilationAttr(), 2), 1) ||
      conv.getGroups() != wTy.getDimSize(0)) {
    return false;
  }
  b.s = stride[0];
  b.kh = wTy.getDimSize(2);
  b.kw = wTy.getDimSize(3);
  b.avgWeight = conv.getWeight();
  return b.kh <= b.s && b.kw <= b.s;
}

// One concat operand -> Branch; returns the mix (NCHW) value it slices.
static Value matchBranch(Value operand, Branch &b) {
  Value blockIn = matchSpaceToDepth(operand, b.r, b.order);
  if (!blockIn) {
    return {};
  }
  // Direct: slice(mix) -> s2d.
  if (Value mix = matchChannelSlice(blockIn, b.c0, b.c1)) {
    b.kind = Branch::Direct;
    return mix;
  }
  // Averaged: nhwc->nchw(conv2d(nchw->nhwc(slice(mix)))) -> s2d.
  Value convOut = peelNhwcToNchw(blockIn);
  auto avg = convOut ? convOut.getDefiningOp<Conv2dOp>() : Conv2dOp();
  if (!avg || !matchAverageConv(avg, b)) {
    return {};
  }
  Value sliceOut = peelNchwToNhwc(avg.getInput());
  Value mix = sliceOut ? matchChannelSlice(sliceOut, b.c0, b.c1) : Value{};
  if (!mix ||
      mlir::cast<RankedTensorType>(b.avgWeight.getType()).getDimSize(0) !=
          b.channels()) {
    return {};
  }
  b.kind = Branch::Averaged;
  return mix;
}

// The whole stem, anchored on its last conv. Fails on anything that is not
// exactly the structure described in the file header.
static std::optional<StemMatch> matchStem(Conv2dOp outConv) {
  StemMatch m;
  m.outConv = outConv;
  if (!isPointwiseConv(outConv) || !outConv.getBias()) {
    return std::nullopt;
  }

  // (d) concat on the channel dim, seen through its NHWC view.
  Value featNchw = peelNchwToNhwc(outConv.getInput());
  auto concat = featNchw ? featNchw.getDefiningOp<ConcatOp>() : ConcatOp();
  if (!concat || !concat.getResult().hasOneUse() ||
      concat.getInputs().size() != 2) {
    return std::nullopt;
  }
  if ((concat.getDim() < 0 ? concat.getDim() + 4 : concat.getDim()) != 1) {
    return std::nullopt;
  }

  // (b), (c) the two branches, both slicing the same mix output.
  Value mixNchw;
  for (Value operand : concat.getInputs()) {
    Branch b;
    Value mix = matchBranch(operand, b);
    if (!mix || (mixNchw && mix != mixNchw)) {
      return std::nullopt;
    }
    mixNchw = mix;
    m.branches.push_back(b);
  }
  if (m.branches[0].kind == m.branches[1].kind) {
    return std::nullopt; // exactly one direct and one averaged branch
  }
  const Branch &direct =
      m.branches[0].kind == Branch::Direct ? m.branches[0] : m.branches[1];
  const Branch &avg =
      m.branches[0].kind == Branch::Direct ? m.branches[1] : m.branches[0];
  m.R = direct.r;
  if (m.R <= 1 || avg.r * avg.s != m.R) {
    return std::nullopt;
  }
  for (Operation *user : mixNchw.getUsers()) {
    if (!mlir::isa<IndexOp, SliceStaticOp>(user)) {
      return std::nullopt; // the mix output must die with the stem
    }
  }

  // (a) the mix conv: nhwc->nchw(conv2d 1x1(nchw->nhwc(x))).
  Value mixOut = peelNhwcToNchw(mixNchw);
  m.mixConv = mixOut ? mixOut.getDefiningOp<Conv2dOp>() : Conv2dOp();
  if (!m.mixConv || !isPointwiseConv(m.mixConv) ||
      !m.mixConv.getResult().hasOneUse()) {
    return std::nullopt;
  }
  m.x = peelNchwToNhwc(m.mixConv.getInput());
  auto xTy = m.x ? mlir::dyn_cast<RankedTensorType>(m.x.getType())
                 : RankedTensorType();
  if (!xTy || xTy.getRank() != 4 || !xTy.hasStaticShape()) {
    return std::nullopt;
  }
  m.N = xTy.getDimSize(0);
  m.Cin = xTy.getDimSize(1);
  m.H = xTy.getDimSize(2);
  m.W = xTy.getDimSize(3);
  m.Cm = mlir::cast<RankedTensorType>(m.mixConv.getWeight().getType())
             .getDimSize(0);
  auto w1Ty = mlir::cast<RankedTensorType>(outConv.getWeight().getType());
  m.Cout = w1Ty.getDimSize(0);
  m.Cfeat = w1Ty.getDimSize(1);

  // Shape consistency.
  if (m.H % m.R != 0 || m.W % m.R != 0) {
    return std::nullopt;
  }
  int64_t features = 0;
  for (const Branch &b : m.branches) {
    if (b.c0 < 0 || b.c0 >= b.c1 || b.c1 > m.Cm) {
      return std::nullopt;
    }
    features += b.features();
  }
  auto outTy = mlir::cast<RankedTensorType>(outConv.getType());
  if (features != m.Cfeat || outTy.getRank() != 4 ||
      outTy.getDimSize(1) != m.H / m.R || outTy.getDimSize(2) != m.W / m.R ||
      outTy.getDimSize(3) != m.Cout ||
      !mlir::isa<FloatType>(w1Ty.getElementType())) {
    return std::nullopt;
  }
  return m;
}

//===----------------------------------------------------------------------===//
// Folded weight and bias
//===----------------------------------------------------------------------===//

// Builds W [Cout, Cp, 1, 1] and b [1, 1, 1, Cout] from the stem's parameters
// with TTIR ops only (see the chain in the file header). Everything depends on
// parameters and constants, so the const-eval hoist runs it once.
class FoldedWeightBuilder {
public:
  FoldedWeightBuilder(PatternRewriter &rewriter, const StemMatch &match)
      : rw(rewriter), m(match), loc(m.outConv.getLoc()),
        elemTy(mlir::cast<RankedTensorType>(m.outConv.getWeight().getType())
                   .getElementType()),
        RR(m.R * m.R), K(m.Cin * RR), Cp(llvm::alignTo(K, 32)) {}

  int64_t paddedChannels() const { return Cp; }

  // Returns {W, b}.
  std::pair<Value, Value> build() {
    // Mix weight as [Cm, Cin] and mix bias as [1, Cm].
    Value wm = reshape(m.mixConv.getWeight(), {m.Cm, m.Cin}, "_stem_wmix");
    Value bm = m.mixConv.getBias()
                   ? reshape(m.mixConv.getBias(), {1, m.Cm}, "_stem_bmix")
                   : Value{};

    SmallVector<Value, 2> gBlocks, f0Blocks;
    for (size_t i = 0; i < m.branches.size(); ++i) {
      const Branch &b = m.branches[i];
      std::string sfx = "_stem_b" + std::to_string(i);
      gBlocks.push_back(branchG(b, wm, sfx));
      if (bm) {
        f0Blocks.push_back(branchF0(b, bm, sfx));
      }
    }

    // W_eq = W1 @ G -> pad the K = Cin*R^2 columns to Cp -> conv weight.
    Value G = concat(gBlocks, 0, {m.Cfeat, K}, "_stem_G");
    Value w1 = reshape(m.outConv.getWeight(), {m.Cout, m.Cfeat}, "_stem_w1");
    Value weq = matmul(w1, G, {m.Cout, K}, "_stem_weq");
    if (Cp != K) {
      weq = rw.create<PadOp>(suffix("_stem_wpad"), tensorTy({m.Cout, Cp}), weq,
                             rw.getDenseI32ArrayAttr(
                                 {0, 0, 0, static_cast<int32_t>(Cp - K)}),
                             rw.getF32FloatAttr(0.0f))
                .getResult();
    }
    Value wconv = reshape(weq, {m.Cout, Cp, 1, 1}, "_stem_wconv");

    // b_eq = W1 @ f0 + b1.
    Value bias;
    if (bm) {
      Value f0 = concat(f0Blocks, 1, {1, m.Cfeat}, "_stem_f0");
      Value f0c = reshape(f0, {m.Cfeat, 1}, "_stem_f0c");
      Value beq = matmul(w1, f0c, {m.Cout, 1}, "_stem_beq");
      bias = reshape(beq, {1, 1, 1, m.Cout}, "_stem_beq4");
    }
    Value b1 = reshape(m.outConv.getBias(), {1, 1, 1, m.Cout}, "_stem_b1");
    bias = bias ? rw.create<AddOp>(suffix("_stem_bias"),
                                   tensorTy({1, 1, 1, m.Cout}), bias, b1)
                      .getResult()
                : b1;
    return {wconv, bias};
  }

private:
  // G_b [F_b, Cin*R^2]: contribution of fused input channel k = c*R^2 + q to
  // each feature j of the branch.
  //   V [C_b*T, Cin]     = Wa[c_b, t] * Wm[c0 + c_b, c]
  //                        (Wa = 1, T = 1 for the direct branch)
  //   Q [F_b*R^2, C_b*T] = 0/1: row (j, q), col (c_b, t) is 1 iff feature j
  //                        is channel c_b at output sub-position (ey, ex) and
  //                        q == (s*ey + i)*R + (s*ex + k), tap t = i*kw + k
  //   G_b                = reshape(permute(reshape(Q @ V, [F_b, R^2, Cin]),
  //                        [0, 2, 1]), [F_b, Cin*R^2])
  Value branchG(const Branch &b, Value wm, const std::string &sfx) {
    const int64_t Cb = b.channels(), Fb = b.features(), T = b.taps();
    Value wmRows = slice(wm, 0, b.c0, b.c1, sfx + "_wm"); // [Cb, Cin]
    Value Q =
        selection({Fb * RR, Cb * T}, sfx + "_Q", [&](ArrayRef<int64_t> id) {
          const int64_t j = id[0] / RR, q = id[0] % RR, cb = id[1] / T,
                        t = id[1] % T;
          const int64_t dy = q / m.R, dx = q % m.R, i = t / b.kw, k = t % b.kw;
          if (dy < i || dx < k || (dy - i) % b.s != 0 || (dx - k) % b.s != 0) {
            return false;
          }
          const int64_t ey = (dy - i) / b.s, ex = (dx - k) / b.s;
          return ey < b.r && ex < b.r && j == featureIndex(b, cb, ey, ex);
        });
    Value V = wmRows;
    if (b.kind == Branch::Averaged) {
      Value wa = reshape(b.avgWeight, {Cb, T, 1}, sfx + "_wa");
      Value wm3 = reshape(wmRows, {Cb, 1, m.Cin}, sfx + "_wm3");
      Value prod =
          rw.create<MultiplyOp>(suffix(sfx + "_v"), tensorTy({Cb, T, m.Cin}),
                                broadcast(wa, {Cb, T, m.Cin}, sfx + "_wa_bc"),
                                broadcast(wm3, {Cb, T, m.Cin}, sfx + "_wm_bc"))
              .getResult();
      V = reshape(prod, {Cb * T, m.Cin}, sfx + "_V");
    }
    Value g = matmul(Q, V, {Fb * RR, m.Cin}, sfx + "_G"); // [(j,q), c]
    g = reshape(g, {Fb, RR, m.Cin}, sfx + "_G3");         // [j, q, c]
    g = rw.create<PermuteOp>(suffix(sfx + "_Gp"), tensorTy({Fb, m.Cin, RR}), g,
                             rw.getDenseI64ArrayAttr({0, 2, 1}))
            .getResult();                       // [j, c, q]
    return reshape(g, {Fb, K}, sfx + "_Gflat"); // [j, c*R^2 + q]
  }

  // f0_b [1, F_b]: the branch's features when the input is zero (mix bias
  // only).
  //   coef [1, C_b*T]  = bm[c0 + c_b] * Wa[c_b, t]
  //                      (= the bm slice for the direct branch)
  //   Qq [C_b*T, F_b]  = 0/1: row (c_b, t), col j is 1 iff feature j belongs
  //                      to channel c_b
  //   f0_b             = coef @ Qq
  Value branchF0(const Branch &b, Value bm, const std::string &sfx) {
    const int64_t Cb = b.channels(), Fb = b.features(), T = b.taps();
    Value Qq = selection({Cb * T, Fb}, sfx + "_Qq", [&](ArrayRef<int64_t> id) {
      const int64_t cb = id[0] / T, j = id[1];
      for (int64_t ey = 0; ey < b.r; ++ey) {
        for (int64_t ex = 0; ex < b.r; ++ex) {
          if (j == featureIndex(b, cb, ey, ex)) {
            return true;
          }
        }
      }
      return false;
    });
    Value coef = slice(bm, 1, b.c0, b.c1, sfx + "_bm"); // [1, Cb]
    if (b.kind == Branch::Averaged) {
      Value bm2 = reshape(coef, {Cb, 1}, sfx + "_bm2");
      Value wa2 = reshape(b.avgWeight, {Cb, T}, sfx + "_wa2");
      Value prod =
          rw.create<MultiplyOp>(suffix(sfx + "_coef"), tensorTy({Cb, T}),
                                broadcast(bm2, {Cb, T}, sfx + "_bm_bc"), wa2)
              .getResult();
      coef = reshape(prod, {1, Cb * T}, sfx + "_coef2");
    }
    return matmul(coef, Qq, {1, Fb}, sfx + "_f0");
  }

  // Position of (channel c, sub-position dy/dx) inside the branch's features.
  static int64_t featureIndex(const Branch &b, int64_t c, int64_t dy,
                              int64_t dx) {
    if (b.order == PixelUnshuffleChannelOrder::SpatialMajor) {
      return dy * (b.r * b.channels()) + dx * b.channels() + c;
    }
    return c * b.r * b.r + dy * b.r + dx;
  }

  // --- op factories (all typed with the stem weight's element type) ---
  Location suffix(StringRef tag) const {
    return ttmlir::utils::appendLocationSuffix(loc, tag);
  }
  RankedTensorType tensorTy(ArrayRef<int64_t> shape) const {
    return RankedTensorType::get(shape, elemTy);
  }
  Value reshape(Value v, ArrayRef<int64_t> shape, StringRef tag) {
    SmallVector<int32_t> s(shape.begin(), shape.end());
    return rw
        .create<ReshapeOp>(suffix(tag), tensorTy(shape), v,
                           rw.getI32ArrayAttr(s))
        .getResult();
  }
  Value matmul(Value a, Value b, ArrayRef<int64_t> shape, StringRef tag) {
    return rw.create<MatmulOp>(suffix(tag), tensorTy(shape), a, b).getResult();
  }
  Value concat(ValueRange vals, int64_t dim, ArrayRef<int64_t> shape,
               StringRef tag) {
    if (vals.size() == 1) {
      return vals.front();
    }
    return rw
        .create<ConcatOp>(suffix(tag), tensorTy(shape), vals,
                          rw.getSI32IntegerAttr(dim))
        .getResult();
  }
  // Range slice along one dim, as ttir.slice_static (ttir.index has no TTNN
  // lowering at this point of the pipeline).
  Value slice(Value v, int64_t dim, int64_t begin, int64_t end, StringRef tag) {
    SmallVector<int64_t> shape(
        mlir::cast<RankedTensorType>(v.getType()).getShape());
    SmallVector<int32_t> begins(shape.size(), 0), ends, steps(shape.size(), 1);
    for (int64_t d : shape) {
      ends.push_back(static_cast<int32_t>(d));
    }
    begins[dim] = static_cast<int32_t>(begin);
    ends[dim] = static_cast<int32_t>(end);
    shape[dim] = end - begin;
    return rw
        .create<SliceStaticOp>(
            suffix(tag), tensorTy(shape), v, rw.getI32ArrayAttr(begins),
            rw.getI32ArrayAttr(ends), rw.getI32ArrayAttr(steps))
        .getResult();
  }
  Value broadcast(Value v, ArrayRef<int64_t> shape, StringRef tag) {
    auto vShape = mlir::cast<RankedTensorType>(v.getType()).getShape();
    SmallVector<int64_t> factors;
    for (size_t i = 0; i < shape.size(); ++i) {
      factors.push_back(shape[i] / vShape[i]);
    }
    return rw.create<BroadcastOp>(suffix(tag), tensorTy(shape), v, factors)
        .getResult();
  }
  // Dense 0/1 constant of `shape`; pred(index) decides each element.
  Value selection(ArrayRef<int64_t> shape, StringRef tag,
                  llvm::function_ref<bool(ArrayRef<int64_t>)> pred) {
    const llvm::fltSemantics &sem =
        mlir::cast<FloatType>(elemTy).getFloatSemantics();
    const llvm::APFloat one(sem, static_cast<uint64_t>(1)),
        zero = llvm::APFloat::getZero(sem);
    int64_t total = 1;
    for (int64_t d : shape) {
      total *= d;
    }
    SmallVector<llvm::APFloat> vals;
    vals.reserve(total);
    SmallVector<int64_t> idx(shape.size());
    for (int64_t flat = 0; flat < total; ++flat) {
      int64_t rem = flat;
      for (int64_t d = shape.size() - 1; d >= 0; --d) {
        idx[d] = rem % shape[d];
        rem /= shape[d];
      }
      vals.push_back(pred(idx) ? one : zero);
    }
    auto ty = tensorTy(shape);
    return rw
        .create<ConstantOp>(suffix(tag), ty, DenseElementsAttr::get(ty, vals))
        .getResult();
  }

  PatternRewriter &rw;
  StemMatch m; // by value: op accessors are non-const
  Location loc;
  Type elemTy;
  const int64_t RR, K, Cp;
};

//===----------------------------------------------------------------------===//
// The rewrite
//===----------------------------------------------------------------------===//

class StemFoldPattern : public OpRewritePattern<Conv2dOp> {
public:
  using OpRewritePattern<Conv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2dOp outConv,
                                PatternRewriter &rewriter) const override {
    std::optional<StemMatch> m = matchStem(outConv);
    if (!m) {
      return failure();
    }

    FoldedWeightBuilder weights(rewriter, *m);
    auto [wconv, bias] = weights.build();
    const int64_t Cp = weights.paddedChannels();

    // pixel_unshuffle(x, R, channel_major, channels_last, Cp)
    //   -> [N, H/R, W/R, Cp] NHWC.
    auto xTy = mlir::cast<RankedTensorType>(m->x.getType());
    auto puTy = RankedTensorType::get({m->N, m->H / m->R, m->W / m->R, Cp},
                                      xTy.getElementType());
    Value pu = rewriter
                   .create<PixelUnshuffleOp>(
                       ttmlir::utils::appendLocationSuffix(outConv.getLoc(),
                                                           "_stem_pu"),
                       puTy, m->x,
                       rewriter.getUI32IntegerAttr(static_cast<uint32_t>(m->R)),
                       PixelUnshuffleChannelOrderAttr::get(
                           rewriter.getContext(),
                           PixelUnshuffleChannelOrder::ChannelMajor),
                       rewriter.getBoolAttr(true),
                       rewriter.getUI32IntegerAttr(static_cast<uint32_t>(Cp)))
                   .getResult();

    // 1x1 conv Cp -> Cout with the folded weight and bias; same result type and
    // discardable attributes (channel_last) as the anchor conv.
    auto newConv = rewriter.create<Conv2dOp>(
        outConv.getLoc(), outConv.getType(), pu, wconv, bias,
        rewriter.getDenseI32ArrayAttr({1, 1}),
        rewriter.getDenseI32ArrayAttr({0, 0, 0, 0}),
        rewriter.getDenseI32ArrayAttr({1, 1}), /*groups=*/1u,
        /*flattened_compat_info=*/nullptr);
    for (NamedAttribute attr : outConv->getDiscardableAttrs()) {
      newConv->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(outConv, newConv.getResult());
    return success(); // the old stem is now dead and removed by the driver
  }
};

class TTIRStemFoldPass : public impl::TTIRStemFoldBase<TTIRStemFoldPass> {
public:
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<StemFoldPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
