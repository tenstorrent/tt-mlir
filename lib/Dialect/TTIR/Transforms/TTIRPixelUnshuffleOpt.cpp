// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TTIRPixelUnshuffleOpt: fuse reshape-6D -> transpose-chain -> reshape-4D
// into ttir.pixel_unshuffle(downscale_factor=r, channel_order=N).
//
// Detects the channel ordering from the composed permutation:
//   SpatialMajor {0,3,5,1,2,4}: ONNX SpaceToDepth  (c_out = rh*(r*C)+rw*C+c_in)
//   ChannelMajor {0,1,3,5,2,4}: PyTorch pixel_unshuffle (c_out = c_in*r²+rh*r+rw)
//
// The transpose chain may be any mix of ttir.TransposeOp / ttir.PermuteOp
// whose composed permutation equals one of the above on a 6D tensor
// [N, C, H/r, r, W/r, r].
//
// In practice the ONNX→TTIR lowering produces:
//   %6d   = ttir.reshape(%input) {shape=[N, C, H/r, r, W/r, r]}
//   %t1   = ttir.transpose(%6d)    {dim0=-5, dim1=-3}
//   %t2   = ttir.transpose(%t1)    {dim0=-4, dim1=-1}
//   %t3   = ttir.transpose(%t2)    {dim0=-2, dim1=-1}
//   %4d   = ttir.reshape(%t3)  {shape=[N, C*r^2, H/r, W/r]}
// =>
//   ttir.pixel_unshuffle(%input) {downscale_factor = r, channel_order = spatial_major (=1)}
//
// The channel_order attribute is passed through to TTNN and into the kernel
// compile-time args — no downstream reorder chain is needed for any C.
//
// For C=1 both orderings produce identical results and either may be detected.
// For C>1 the composed permutation uniquely determines the ordering.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRPIXELUNSHUFFLEOPT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

/// Returns r > 0 if `op` reshapes [N,C,H,W] -> [N,C, H/r,r, W/r,r], else 0.
static uint32_t matchPixelUnshuffleReshape6D(ttir::ReshapeOp op) {
  auto inType  = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  if (!inType || !outType) return 0;
  if (inType.getRank() != 4 || outType.getRank() != 6) return 0;

  auto inShape  = inType.getShape();
  auto outShape = outType.getShape();

  if (outShape[0] != inShape[0] || outShape[1] != inShape[1]) return 0;

  int64_t r = outShape[3];
  if (r <= 1 || outShape[5] != r) return 0;

  if (inShape[2] == mlir::ShapedType::kDynamic ||
      inShape[3] == mlir::ShapedType::kDynamic) return 0;
  if (outShape[2] * r != inShape[2] || outShape[4] * r != inShape[3]) return 0;

  return static_cast<uint32_t>(r);
}

/// Returns true if `op` reshapes a 6D tensor -> [N, C*r^2, H/r, W/r].
/// Works for both channel orderings: after composeTransposeChain the 6D shape
/// is either [N, r, r, C, H/r, W/r] (SpatialMajor) or [N, C, r, r, H/r, W/r]
/// (ChannelMajor).  In both cases dims 1-3 hold {r,r,C} or {C,r,r} and their
/// product equals C*r^2, so the reshape validation is ordering-independent.
static bool matchPixelUnshuffleReshape4D(ttir::ReshapeOp op) {
  auto inType  = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
  auto outType = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
  if (!inType || !outType) return false;
  if (inType.getRank() != 6 || outType.getRank() != 4) return false;

  auto inShape  = inType.getShape();
  auto outShape = outType.getShape();

  // dims 1-3 contain C, r, r in some order; product = C*r^2 regardless.
  int64_t C_out = inShape[1] * inShape[2] * inShape[3];
  return (outShape[0] == inShape[0] &&
          outShape[1] == C_out       &&
          outShape[2] == inShape[4]  &&
          outShape[3] == inShape[5]);
}

/// Walk backwards through a chain of ttir.TransposeOp / ttir.PermuteOp,
/// composing their permutations into a single permutation.
///
/// composed[i] answers: "which dim of the chain input maps to chain output dim i?"
/// All intermediate ops must have exactly one use (they are exclusively
/// consumed by the next op in the chain).
///
/// Returns the chain input (first value not defined by a transpose/permute),
/// or mlir::Value{} on error.
static mlir::Value composeTransposeChain(mlir::Value chainOutput, int64_t rank,
                                          llvm::SmallVector<int64_t> &composed) {
  composed.resize(rank);
  std::iota(composed.begin(), composed.end(), 0);

  mlir::Value current = chainOutput;

  while (true) {
    mlir::Operation *defOp = current.getDefiningOp();
    if (!defOp) break;

    llvm::SmallVector<int64_t> opPerm(rank);

    if (auto transposeOp = mlir::dyn_cast<ttir::TransposeOp>(defOp)) {
      if (!transposeOp.getResult().hasOneUse()) return {};
      std::iota(opPerm.begin(), opPerm.end(), 0);
      int64_t d0 = transposeOp.getDim0();
      int64_t d1 = transposeOp.getDim1();
      if (d0 < 0) d0 += rank;
      if (d1 < 0) d1 += rank;
      if (d0 < 0 || d0 >= rank || d1 < 0 || d1 >= rank) return {};
      std::swap(opPerm[d0], opPerm[d1]);
      current = transposeOp.getInput();
    } else if (auto permuteOp = mlir::dyn_cast<ttir::PermuteOp>(defOp)) {
      if (!permuteOp.getResult().hasOneUse()) return {};
      auto permAttr = permuteOp.getPermutation();
      if (static_cast<int64_t>(permAttr.size()) != rank) return {};
      for (int64_t i = 0; i < rank; ++i) opPerm[i] = permAttr[i];
      current = permuteOp.getInput();
    } else {
      break;
    }

    // Compose: new_composed[i] = opPerm[composed[i]]
    llvm::SmallVector<int64_t> newComposed(rank);
    for (int64_t i = 0; i < rank; ++i)
      newComposed[i] = opPerm[composed[i]];
    composed = newComposed;
  }

  return current;
}

/// Returns true if composed == {0, 3, 5, 1, 2, 4} (SpatialMajor / ONNX SpaceToDepth permutation).
static bool isSpatialMajorPermutation(const llvm::SmallVector<int64_t> &p) {
  static const int64_t kExpected[] = {0, 3, 5, 1, 2, 4};
  if (p.size() != 6) return false;
  for (int i = 0; i < 6; ++i)
    if (p[i] != kExpected[i]) return false;
  return true;
}

/// Returns true if composed == {0, 1, 3, 5, 2, 4} (ChannelMajor / PyTorch permutation).
static bool isChannelMajorPermutation(const llvm::SmallVector<int64_t> &p) {
  static const int64_t kExpected[] = {0, 1, 3, 5, 2, 4};
  if (p.size() != 6) return false;
  for (int i = 0; i < 6; ++i)
    if (p[i] != kExpected[i]) return false;
  return true;
}

/// Pattern: reshape-6D -> [transpose/permute chain composing to SpatialMajor
///          {0,3,5,1,2,4} or ChannelMajor {0,1,3,5,2,4}]
///          -> reshape-4D  =>  ttir.pixel_unshuffle(input, downscale_factor=r, channel_order=N)
struct PixelUnshufflePattern : public mlir::OpRewritePattern<ttir::ReshapeOp> {
  using mlir::OpRewritePattern<ttir::ReshapeOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ttir::ReshapeOp reshape4D,
                  mlir::PatternRewriter &rewriter) const override {
    // Must be the final 6D->4D reshape
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(reshape4D.getType());
    auto inType  = mlir::dyn_cast<mlir::RankedTensorType>(
        reshape4D.getInput().getType());
    if (!outType || !inType) return mlir::failure();
    if (outType.getRank() != 4 || inType.getRank() != 6)
      return mlir::failure();

    // Validate the 6D->4D output shape matches pixel_unshuffle form
    if (!matchPixelUnshuffleReshape4D(reshape4D))
      return mlir::failure();

    // Walk backwards through transpose/permute chain and compose permutation
    llvm::SmallVector<int64_t> composed;
    mlir::Value chainInput =
        composeTransposeChain(reshape4D.getInput(), 6, composed);
    if (!chainInput) return mlir::failure();

    // Determine channel ordering from the composed permutation.
    // SpatialMajor {0,3,5,1,2,4}: ONNX SpaceToDepth ordering (c_out = rh*(r*C)+rw*C+c_in)
    // ChannelMajor {0,1,3,5,2,4}: PyTorch ordering (c_out = c_in*r²+rh*r+rw)
    ttir::PixelUnshuffleChannelOrder channelOrder;
    if (isSpatialMajorPermutation(composed)) {
      channelOrder = ttir::PixelUnshuffleChannelOrder::SpatialMajor;
    } else if (isChannelMajorPermutation(composed)) {
      channelOrder = ttir::PixelUnshuffleChannelOrder::ChannelMajor;
    } else {
      return mlir::failure();
    }

    // Chain input must come from the 4D->6D reshape
    auto reshape6D = chainInput.getDefiningOp<ttir::ReshapeOp>();
    if (!reshape6D) return mlir::failure();

    uint32_t r = matchPixelUnshuffleReshape6D(reshape6D);
    if (r == 0) return mlir::failure();

    // Create ttir.pixel_unshuffle with the detected channel_order.
    // No downstream reorder chain needed — the kernel handles ordering directly.
    auto psOp = rewriter.create<ttir::PixelUnshuffleOp>(
        reshape4D.getLoc(),
        reshape4D.getType(),
        reshape6D.getInput(),
        rewriter.getUI32IntegerAttr(r),
        ttir::PixelUnshuffleChannelOrderAttr::get(rewriter.getContext(),
                                                  channelOrder));

    rewriter.replaceOp(reshape4D, psOp.getResult());
    return mlir::success();
  }
};

class TTIRPixelUnshuffleOptPass
    : public impl::TTIRPixelUnshuffleOptBase<TTIRPixelUnshuffleOptPass> {
public:
  void runOnOperation() final {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PixelUnshufflePattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
