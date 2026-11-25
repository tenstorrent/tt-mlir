// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTIRPadUpsampleChannels
//
// A bilinear upsample needs its channel count to be a multiple of 32, so an
// NHWC `upsample2d(X[...,C]) -> conv2d` with C % 32 != 0 lowers to
//
//   pad X to Cp (TILE pad) -> untilize -> upsample -> tilize ->
//   slice [0:C] -> reshape -> conv
//
// where the slice only drops the zero channels again and forces a re-tilize.
// Zero channels are harmless to a conv whose weights are zero on them, so
// this pass pads X to Cp once, upsamples all Cp channels and gives the conv
// Cp input channels with zero weights on [C, Cp) (same result; the weight pad
// is folded by const-eval).

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRPADUPSAMPLECHANNELS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kChannelAlign = 32;

static void process(Upsample2dOp up) {
  if (up.getMode() != "bilinear" || !up.getResult().hasOneUse()) {
    return;
  }
  auto conv = mlir::dyn_cast<Conv2dOp>(*up.getResult().getUsers().begin());
  if (!conv || conv.getInput() != up.getResult() || conv.getGroups() != 1) {
    return;
  }
  // NHWC on both: channels last.
  auto cd = conv->getAttrOfType<IntegerAttr>("channel_dim");
  if (cd && cd.getInt() != 3) {
    return;
  }
  auto inTy = mlir::cast<RankedTensorType>(up.getInput().getType());
  auto upTy = mlir::cast<RankedTensorType>(up.getResult().getType());
  if (inTy.getRank() != 4 || !inTy.hasStaticShape()) {
    return;
  }
  const int64_t C = inTy.getDimSize(3);
  if (C % kChannelAlign == 0) {
    return;
  }
  const int64_t Cp = llvm::alignTo(C, kChannelAlign);

  // OIHW weights that are a parameter or constant (the pad is const-evaled).
  Value weight = conv.getWeight();
  auto wTy = mlir::cast<RankedTensorType>(weight.getType());
  auto wArg = mlir::dyn_cast<BlockArgument>(weight);
  auto func = conv->getParentOfType<func::FuncOp>();
  if (!wArg || !ttcore::isConstOrParamArg(wArg, func) || wTy.getRank() != 4 ||
      wTy.getDimSize(1) != C) {
    return;
  }

  OpBuilder b(up);
  Location loc = up.getLoc();
  auto padded = [&](RankedTensorType ty, int64_t dim) {
    SmallVector<int64_t> s(ty.getShape());
    s[dim] = Cp;
    return RankedTensorType::get(s, ty.getElementType(), ty.getEncoding());
  };
  // padding = [lo0, hi0, lo1, hi1, ...]
  auto padOn = [&](int64_t dim, int64_t rank) {
    SmallVector<int32_t> p(2 * rank, 0);
    p[2 * dim + 1] = static_cast<int32_t>(Cp - C);
    return b.getDenseI32ArrayAttr(p);
  };

  Value x = b.create<PadOp>(loc, padded(inTy, 3), up.getInput(), padOn(3, 4),
                            b.getF32FloatAttr(0.0f))
                .getResult();
  auto newUp = b.create<Upsample2dOp>(
      loc, padded(upTy, 3), x, up.getScaleFactorAttr(), up.getModeAttr());
  // Keep discardable attributes (e.g. channel_last).
  for (NamedAttribute attr : up->getDiscardableAttrs()) {
    newUp->setAttr(attr.getName(), attr.getValue());
  }

  b.setInsertionPoint(conv);
  Value w = b.create<PadOp>(conv.getLoc(), padded(wTy, 1), weight, padOn(1, 4),
                            b.getF32FloatAttr(0.0f))
                .getResult();
  conv.getInputMutable().assign(newUp.getResult());
  conv.getWeightMutable().assign(w);
  up.erase();
}

class TTIRPadUpsampleChannels
    : public impl::TTIRPadUpsampleChannelsBase<TTIRPadUpsampleChannels> {
public:
  using impl::TTIRPadUpsampleChannelsBase<
      TTIRPadUpsampleChannels>::TTIRPadUpsampleChannelsBase;

  void runOnOperation() final {
    SmallVector<Upsample2dOp> ups;
    getOperation()->walk([&](Upsample2dOp op) { ups.push_back(op); });
    for (Upsample2dOp up : ups) {
      process(up);
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
