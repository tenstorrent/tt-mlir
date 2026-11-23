// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTIRStridedConvSpaceToDepthOpt
//
// Space-to-depth packing for narrow-channel, stride-2 stem convolutions.
//
// A model whose first layer is e.g. Conv 7x7 / stride 2 on a 3-channel NCHW
// image lowers to
//
//   permute({0,2,3,1})  [N,C,H,W] -> [N,H,W,C]          NHWC
//   conv2d              {channel_last, kHxkW, stride 2x2, groups 1}
//
// and the layout passes tilize before the permute, so the permute writes the
// narrow channel dimension padded to a 32-wide tile (10.7x write amplification
// for C=3), followed by a DRAM copy and, because the padded tensor often no
// longer fits L1, a sliced conv2d.
//
// A stride-2 conv is exactly a stride-1 conv on the space-to-depth image:
//
//   x_s2d = pixel_unshuffle(x, r=2)            [N, 4C, H/2, W/2]
//   y     = conv2d(x_s2d, W', b) {stride 1, kernel ceil(kH/2) x ceil(kW/2)}
//
// with W'[oc, c*4 + dy*2 + dx, i, j] = W[oc, c, 2i+dy, 2j+dx], zero where the
// original tap does not exist. That channel index is exactly
// ttir.pixel_unshuffle's channel_major ordering, so no extra reorder is needed.
//
// The activation is produced in channels_last form: pixel_unshuffle writes
// NHWC [N, H/2, W/2, Cp] directly, each pixel's 4C channels zero-padded to an
// L1-aligned width Cp. That is the row-major height-sharded activation conv2d
// reads, so neither the NCHW->NHWC permute nor a tilize is left in front of
// the conv. The packed weights get Cp - 4C zero input channels to match.
// TTNNPixelUnshuffleChannelsLast places the result on the conv's shard grid
// once the optimizer has chosen it.
//
// The original padding (pT, pL, pB, pR), with pT and pL even, becomes
// (pT/2, pL/2, pB', pR') where pB' = outH + ceil(kH/2) - 1 - H/2 - pT/2 and
// likewise pR'. The packed padding may cover rows the original conv never read,
// but those rows only ever meet zero kernel taps, so the computation is
// unchanged up to accumulation order.
//
// The weight rearrangement is built from rank<=4 pad/reshape/permute ops on the
// parameter, so ConstEvalHoistTransform lifts it out of the forward graph and
// it costs nothing per inference.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSTRIDEDCONVSPACETODEPTHOPT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Space-to-depth factor. Equal to the stride this pass matches on: a stride-r
// conv becomes a stride-1 conv on the r-fold space-to-depth image.
static constexpr int64_t kFactor = 2;
// The packed channel count must still fit one tile column, otherwise the
// rewrite reintroduces the padding it exists to remove.
static constexpr int64_t kTileWidth = 32;

// Channel width of the channels_last pixel_unshuffle output: the packed
// channel count rounded up to a 32 B row-major stick for 2-byte elements
// (16), or a full tile width. conv2d's row-major height-sharded input path
// aligns channels to the shard width (32, 16 or 8), and pixel_unshuffle needs
// an L1-aligned stick; 16 keeps the output as small as possible without a
// re-shard. packedC never exceeds kTileWidth (see the match guard).
static int64_t alignedChannels(int64_t packedC) {
  return packedC <= 16 ? 16 : kTileWidth;
}

static bool isPermute(Operation *op, ArrayRef<int64_t> expected) {
  auto p = mlir::dyn_cast_or_null<ttir::PermuteOp>(op);
  return p && p.getPermutation() == expected;
}

// conv2d stride/padding/dilation are `i32 | array<2xi32> | array<4xi32>`.
// Normalise to n elements; [pH, pW] expands to [pT, pL, pB, pR].
static bool expandAttr(Attribute attr, unsigned n,
                       SmallVectorImpl<int32_t> &out) {
  out.clear();
  if (auto arr = mlir::dyn_cast_or_null<DenseI32ArrayAttr>(attr)) {
    ArrayRef<int32_t> v = arr.asArrayRef();
    if (v.size() == n) {
      out.assign(v.begin(), v.end());
      return true;
    }
    if (n == 4 && v.size() == 2) {
      out.assign({v[0], v[1], v[0], v[1]});
      return true;
    }
    return false;
  }
  if (auto i = mlir::dyn_cast_or_null<IntegerAttr>(attr)) {
    out.assign(n, static_cast<int32_t>(i.getInt()));
    return true;
  }
  return false;
}

class TTIRStridedConvSpaceToDepthOptPass
    : public impl::TTIRStridedConvSpaceToDepthOptBase<
          TTIRStridedConvSpaceToDepthOptPass> {
public:
  using impl::TTIRStridedConvSpaceToDepthOptBase<
      TTIRStridedConvSpaceToDepthOptPass>::TTIRStridedConvSpaceToDepthOptBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    // Collect first: process() erases ops, which would invalidate a live walk.
    SmallVector<ttir::Conv2dOp, 4> candidates;
    mod.walk([&](ttir::Conv2dOp op) { candidates.push_back(op); });
    for (ttir::Conv2dOp convOp : candidates) {
      process(convOp);
    }
  }

private:
  void process(ttir::Conv2dOp convOp) {
    static const int64_t kNCHWtoNHWC[] = {0, 2, 3, 1};

    // ---- Match -------------------------------------------------------------
    if (!convOp->hasAttr("channel_last")) {
      return;
    }
    Operation *preOp = convOp.getInput().getDefiningOp();
    if (!isPermute(preOp, kNCHWtoNHWC)) {
      return;
    }
    auto prePerm = mlir::cast<ttir::PermuteOp>(preOp);
    Value nchwInput = prePerm.getInput(); // [N,C,H,W]

    auto inputTy = mlir::dyn_cast<RankedTensorType>(nchwInput.getType());
    auto weightTy =
        mlir::dyn_cast<RankedTensorType>(convOp.getWeight().getType());
    auto resultTy =
        mlir::dyn_cast<RankedTensorType>(convOp.getResult().getType());
    if (!inputTy || !weightTy || !resultTy || inputTy.getRank() != 4 ||
        weightTy.getRank() != 4 || resultTy.getRank() != 4 ||
        !inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !resultTy.hasStaticShape()) {
      return;
    }

    SmallVector<int32_t> stride, padding, dilation;
    if (!expandAttr(convOp.getStride(), 2, stride) ||
        !expandAttr(convOp.getPadding(), 4, padding) ||
        !expandAttr(convOp.getDilation(), 2, dilation)) {
      return;
    }
    if (stride[0] != kFactor || stride[1] != kFactor || dilation[0] != 1 ||
        dilation[1] != 1 || convOp.getGroups() != 1) {
      return;
    }
    const int32_t pT = padding[0], pL = padding[1], pB = padding[2],
                  pR = padding[3];
    // pT and pL set the phase of the space-to-depth grid; an odd leading pad
    // has no integer phase, so the rewrite does not apply.
    if (pT < 0 || pL < 0 || pB < 0 || pR < 0 || pT % kFactor != 0 ||
        pL % kFactor != 0) {
      return;
    }

    const int64_t N = inputTy.getDimSize(0);
    const int64_t C = inputTy.getDimSize(1);
    const int64_t H = inputTy.getDimSize(2);
    const int64_t W = inputTy.getDimSize(3);
    const int64_t OC = weightTy.getDimSize(0);
    const int64_t IC = weightTy.getDimSize(1);
    const int64_t kH = weightTy.getDimSize(2);
    const int64_t kW = weightTy.getDimSize(3);
    if (IC != C || H % kFactor != 0 || W % kFactor != 0) {
      return;
    }
    // Only worthwhile while the packed channels still fit one tile column
    // (C=3 -> 12, C=8 -> 32); wider inputs never suffered the amplification.
    const int64_t packedC = C * kFactor * kFactor;
    if (packedC > kTileWidth) {
      return;
    }
    // channel_last result is [N, outH, outW, OC].
    const int64_t outH = resultTy.getDimSize(1);
    const int64_t outW = resultTy.getDimSize(2);
    if (resultTy.getDimSize(0) != N || resultTy.getDimSize(3) != OC) {
      return;
    }

    const int64_t kHp = (kH + kFactor - 1) / kFactor; // packed kernel extent
    const int64_t kWp = (kW + kFactor - 1) / kFactor;
    const int64_t padTop = pT / kFactor;
    const int64_t padLeft = pL / kFactor;
    const int64_t padBottom = outH + kHp - 1 - H / kFactor - padTop;
    const int64_t padRight = outW + kWp - 1 - W / kFactor - padLeft;
    if (padBottom < 0 || padRight < 0) {
      return;
    }
    // The packed conv must cover every row/col the original one read. Taps
    // 2i+dy < kH are covered by construction; the packed padding may only
    // exceed the original into rows that meet zero taps.
    if (padBottom * kFactor < pB - (kHp * kFactor - kH) ||
        padRight * kFactor < pR - (kWp * kFactor - kW)) {
      return;
    }

    // ---- Rewrite -----------------------------------------------------------
    OpBuilder b(convOp);
    Location loc = convOp.getLoc();
    Type actElemTy = inputTy.getElementType();
    Type wElemTy = weightTy.getElementType();
    MLIRContext *ctx = b.getContext();

    auto channelOrderAttr = ttir::PixelUnshuffleChannelOrderAttr::get(
        ctx, ttir::PixelUnshuffleChannelOrder::ChannelMajor);
    auto factorAttr = b.getUI32IntegerAttr(static_cast<uint32_t>(kFactor));
    // Activation: pixel_unshuffle writes NHWC directly, each pixel's packedC
    // channels zero-padded to an aligned stick (e.g. 12 -> 16 for bf16). An
    // unaligned stick (12 x 2 B = 24 B) would make conv2d re-shard its input.
    const int64_t convC = alignedChannels(packedC);
    auto nhwcTy =
        RankedTensorType::get({N, H / kFactor, W / kFactor, convC}, actElemTy);
    Value nhwc = b.create<ttir::PixelUnshuffleOp>(
                      loc, nhwcTy, nchwInput, factorAttr, channelOrderAttr,
                      /*channels_last=*/b.getBoolAttr(true),
                      b.getUI32IntegerAttr(static_cast<uint32_t>(convC)))
                     .getResult();

    // Weight: W[OC,IC,kH,kW] -> W'[OC, IC*4, kHp, kWp] with
    // W'[oc, c*4+dy*2+dx, i, j] = Wpad[oc, c, 2i+dy, 2j+dx]. Every step is
    // rank <= 4 so const-eval hoisting can lift the whole chain.
    const int64_t Hk = kHp * kFactor, Wk = kWp * kFactor;
    Value w = convOp.getWeight();
    if (Hk != kH || Wk != kW) {
      // Zero-pad the kernel up to the packed footprint (7x7 -> 8x8); the extra
      // taps are zero, which is what makes the padding rewrite exact.
      auto wPadTy = RankedTensorType::get({OC, IC, Hk, Wk}, wElemTy);
      SmallVector<int32_t> pads = {
          0, 0, 0, 0, 0, static_cast<int32_t>(Hk - kH), 0,
          static_cast<int32_t>(Wk - kW)};
      w = b.create<ttir::PadOp>(loc, wPadTy, w, b.getDenseI32ArrayAttr(pads),
                                b.getF32FloatAttr(0.0f))
              .getResult();
    }
    auto i32shape = [&](ArrayRef<int64_t> s) {
      SmallVector<int32_t> v(s.begin(), s.end());
      return b.getI32ArrayAttr(v);
    };
    auto reshape = [&](Value v, ArrayRef<int64_t> shape) {
      return b.create<ttir::ReshapeOp>(loc,
                                       RankedTensorType::get(shape, wElemTy), v,
                                       i32shape(shape))
          .getResult();
    };
    auto permute = [&](Value v, ArrayRef<int64_t> perm) {
      auto ty = mlir::cast<RankedTensorType>(v.getType());
      SmallVector<int64_t> shape;
      for (int64_t p : perm) {
        shape.push_back(ty.getDimSize(p));
      }
      return b.create<ttir::PermuteOp>(
                  loc, RankedTensorType::get(shape, wElemTy), v,
                  b.getDenseI64ArrayAttr(perm))
          .getResult();
    };
    // Split the row axis r = 2i+dy, move dy outward, then do the same for
    // cols, leaving the channel axis ordered as c*4 + dy*2 + dx.
    Value r1 = reshape(w, {OC * IC, kHp, kFactor, Wk});     // (oc,c), i, dy, (j,dx)
    Value p1 = permute(r1, {0, 2, 1, 3});                   // (oc,c), dy, i, (j,dx)
    Value r2 = reshape(p1, {OC * IC * kFactor * kHp, kWp, kFactor});
    Value p2 = permute(r2, {0, 2, 1});                      // (oc,c,dy,i), dx, j
    Value r3 = reshape(p2, {OC * IC * kFactor, kHp, kFactor, kWp});
    Value p3 = permute(r3, {0, 2, 1, 3});                   // (oc,c,dy), dx, i, j
    Value wPacked = reshape(p3, {OC, packedC, kHp, kWp});
    if (convC != packedC) {
      // The activation's pad channels are zero; zero weights for them keep
      // the conv exact (and a zero weight never meets a NaN, since
      // pixel_unshuffle zero-fills the pad).
      auto wPadCTy = RankedTensorType::get({OC, convC, kHp, kWp}, wElemTy);
      SmallVector<int32_t> pads = {
          0, 0, 0, static_cast<int32_t>(convC - packedC), 0, 0, 0, 0};
      wPacked = b.create<ttir::PadOp>(loc, wPadCTy, wPacked,
                                      b.getDenseI32ArrayAttr(pads),
                                      b.getF32FloatAttr(0.0f))
                    .getResult();
    }

    auto newConv = b.create<ttir::Conv2dOp>(
        loc, resultTy, nhwc, wPacked, convOp.getBias(),
        /*stride=*/b.getDenseI32ArrayAttr({1, 1}),
        /*padding=*/
        b.getDenseI32ArrayAttr({static_cast<int32_t>(padTop),
                                static_cast<int32_t>(padLeft),
                                static_cast<int32_t>(padBottom),
                                static_cast<int32_t>(padRight)}),
        /*dilation=*/b.getDenseI32ArrayAttr({1, 1}),
        /*groups=*/static_cast<uint32_t>(1),
        /*flattened_compat_info=*/nullptr);
    // Carry the original channel_last attribute across verbatim: the frontend
    // emits it as `channel_last = true` (BoolAttr), not a UnitAttr, and
    // downstream passes match on the value.
    newConv->setAttr("channel_last", convOp->getAttr("channel_last"));
    for (StringRef name :
         {"batch_dim", "height_dim", "width_dim", "channel_dim"}) {
      if (Attribute a = convOp->getAttr(name)) {
        newConv->setAttr(name, a);
      }
    }

    convOp.getResult().replaceAllUsesWith(newConv.getResult());
    convOp.erase();
    if (prePerm.getResult().use_empty()) {
      prePerm.erase();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
