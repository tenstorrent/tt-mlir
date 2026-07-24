// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TTIRSpatialRowGroupPackingOpt
//
// By the time this pass runs (inside createTTNNPipelineTTIRPasses, after TTIRFusing
// and canonicalization), the two-transpose pattern from ONNX 1×1 conv2d has been
// fused into single permute ops:
//
//   Before (ONNX-generated TTIR):
//     transpose(-3,-2) → transpose(-2,-1) → conv2d → transpose(-2,-1) → transpose(-3,-2)
//
//   After TTIRFusing / EraseInverseOps (what this pass actually sees):
//     permute({0,2,3,1})  [N,C,H,W]→[N,H,W,C] NHWC
//     conv2d              {channel_last, kernel=1×1, stride=1, pad=0}
//     permute({0,3,1,2})  [N,H,W,C]→[N,C,H,W] NCHW
//
// This pass replaces the pattern with NCHW-native spatial row-group packing
// (K=TILE_WIDTH for IC coprime to TILE_WIDTH, e.g. IC=3 → K=32):
//
//   ── Weight packing (ops on constant parameter → auto const-eval'd) ────────
//   Maps directly to _make_packed_weight() using 9 TTIR ops — NO embedded constant:
//
//   broadcast  %weight [OC,IC,1,1] → [OC,IC,K,K]   expand kH=kW=1 to K×K
//   arange     [1,1,K,K] arange_dim=2               val[0,0,k,*]=k  (row indices)
//   arange     [1,1,K,K] arange_dim=3               val[0,0,*,k]=k  (col indices)
//   eq         row_grid == col_grid → [1,1,K,K] bool  True on K×K diagonal only
//   typecast   bool → bf16           [1,1,K,K]       1.0/0.0 (identity I_K)
//   multiply   w_bc * i_k            [OC,IC,K,K]     zero off-diagonal elements
//   permute    [0,2,1,3]             [OC,IC,K,K]→[OC,K,IC,K]  produces W^T
//   reshape    [OC,K,IC,K]→[OC*K, IC*K]
//   reshape    → [1, 1, OC*K, IC*K]  transposed packed weight (A-operand)
//
//   No embedded tensor constant — I_K is computed from arange+eq (zero overhead).
//
//   ── Bias packing (ops on constant parameter → auto const-eval'd) ─────────
//   reshape(%bias, [OC,1])                   rank-2 to avoid tt-metal rank-1 bug
//   repeat_interleave([OC,1], K, dim=0)  → [OC*K, 1]
//   reshape([1, 1, OC*K, 1])                 column bias (broadcasts over spatial P)
//
//   ── Activation packing (runtime) ─────────────────────────────────────────
//   reshape(%input, [N, 1, C*K, H/K*W])  free NCHW view — no permute needed
//
//   ── NCHW-native linear → output unpack ───────────────────────────────────
//   linear(W_packedT[1,1,OC*K,IC*K], act[N,1,IC*K,P]) → [N,1,OC*K,P]
//   reshape([N, OC, H, W])               free NCHW view
//   (TTNNSpatialPackActivationRowMajorOpt inserts untilize before this reshape)
//
// Because %weight and %bias carry ttcore.argument_type=parameter,
// ConstEvalHoistTransform (which runs AFTER all TTIR passes) automatically lifts
// the weight/bias packing ops into const_eval functions.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSPATIALROWGROUPPACKINGOPT
#define GEN_PASS_DEF_TTIRDEPTHWISECONVSPATIALPACKINGOPT
#define GEN_PASS_DEF_TTIRPOINTWISECONV2DPARTIALPACKINGOPT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static constexpr int64_t TILE_WIDTH = 32;

static int64_t packingFactor(int64_t C) {
  return TILE_WIDTH /
         (int64_t)std::gcd((unsigned long long)C, (unsigned long long)TILE_WIDTH);
}

// No buildMask() — the block-diagonal identity is built from ttir.arange + ttir.eq
// so there is no embedded constant tensor in the IR.

// Check if op is ttir.permute with exactly the given permutation.
static bool isPermute(Operation *op, ArrayRef<int64_t> expected) {
  auto p = mlir::dyn_cast_or_null<ttir::PermuteOp>(op);
  if (!p)
    return false;
  return p.getPermutation() == expected;
}

// ── Pass ─────────────────────────────────────────────────────────────────────

class TTIRSpatialRowGroupPackingOptPass
    : public impl::TTIRSpatialRowGroupPackingOptBase<
          TTIRSpatialRowGroupPackingOptPass> {
public:
  using impl::TTIRSpatialRowGroupPackingOptBase<
      TTIRSpatialRowGroupPackingOptPass>::TTIRSpatialRowGroupPackingOptBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<ttir::Conv2dOp, 4> candidates;
    mod.walk([&](ttir::Conv2dOp op) { candidates.push_back(op); });

    for (ttir::Conv2dOp convOp : candidates)
      process(convOp, ctx);
  }

private:
  // ── NCHW-native path ───────────────────────────────────────────────────────
  void process(ttir::Conv2dOp convOp, MLIRContext *ctx) {
    // ── 1. Match permute → conv2d → permute pattern ───────────────────────
    // By the time this pass runs, TTIRFusing has fused the two transposes into
    // single permute ops:
    //   permute({0,2,3,1}) NCHW→NHWC → conv2d → permute({0,3,1,2}) NHWC→NCHW

    static const int64_t kNCHWtoNHWC[] = {0, 2, 3, 1};
    static const int64_t kNHWCtoNCHW[] = {0, 3, 1, 2};

    auto *prePermuteOp = convOp.getInput().getDefiningOp();
    if (!isPermute(prePermuteOp, kNCHWtoNHWC))
      return;
    auto prePerm = mlir::cast<ttir::PermuteOp>(prePermuteOp);
    Value origInput = prePerm.getInput(); // [N,C,H,W] NCHW

    // Post-conv: conv2d result must feed exactly one permute({0,3,1,2}).
    ttir::PermuteOp postPerm;
    for (Operation *u : convOp.getResult().getUsers()) {
      if (isPermute(u, kNHWCtoNCHW)) {
        postPerm = mlir::cast<ttir::PermuteOp>(u);
        break;
      }
    }
    if (!postPerm)
      return;

    // ── 2. Check applicability ─────────────────────────────────────────────
    // Weight is OIHW: [OC, IC, kH, kW]
    auto weightType = mlir::cast<RankedTensorType>(convOp.getWeight().getType());
    if (weightType.getRank() != 4)
      return;
    int64_t OC = weightType.getDimSize(0);
    int64_t IC = weightType.getDimSize(1);
    int64_t kH = weightType.getDimSize(2);
    int64_t kW = weightType.getDimSize(3);

    // Narrow-channel 1×1 pointwise only.
    if (IC >= TILE_WIDTH || kH != 1 || kW != 1)
      return;

    // Only apply when gcd(IC, TILE_WIDTH) == 1, i.e. K = TILE_WIDTH (full packing
    // factor).  This restricts packing to the YUV adapter (IC=3, K=32) and any
    // other narrow-channel conv where all tile rows fill completely with zero waste.
    //
    // Partial packing factors (e.g. IC=24 → K=4) accidentally match the narrow-
    // channel guard but produce wrong results: the packed reshape operates on L1
    // interleaved data whose stride mapping is inconsistent with the flattened view
    // fed to the linear, causing ~2.8% PCC degradation (0.99 → 0.972) for the
    // backbone's first 24-channel 1×1 conv at 320×576 resolution.
    //
    // Values coprime to 32 (K=32): IC = 3, 5, 7, 9, 11, 13, 15, ...
    // Values NOT coprime to 32 (K<32, SKIP): IC = 2,4,6,8,10,12,14,16,18,20,24,...
    if (packingFactor(IC) != TILE_WIDTH)
      return;

    // Stride = [1,1].
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getStride()))
      for (int32_t s : arr.asArrayRef())
        if (s != 1)
          return;

    // Padding all zeros.
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getPadding()))
      for (int32_t p : arr.asArrayRef())
        if (p != 0)
          return;

    if (convOp.getGroups() != 1)
      return;

    // origInput is [N,C,H,W] NCHW.
    auto inputType = mlir::cast<RankedTensorType>(origInput.getType());
    if (inputType.getRank() != 4)
      return;
    int64_t N = inputType.getDimSize(0);
    int64_t H = inputType.getDimSize(2);
    int64_t W = inputType.getDimSize(3);

    int64_t K = packingFactor(IC);
    if (H % K != 0)
      return;

    int64_t packedH  = H / K;
    int64_t packedIC = IC * K;
    int64_t packedOC = OC * K;
    auto bf16 = BFloat16Type::get(ctx);
    // Use the original activation element type (e.g. f32) for all non-weight
    // types so that downstream ops (slice_static etc.) see a consistent type.
    // ElementTypeNormalization keeps f32 as f32, so postPerm may be f32 even
    // though the hardcoded bf16 below would cause a type mismatch.
    auto actElemTy = mlir::cast<RankedTensorType>(origInput.getType()).getElementType();

    Value origWeight = convOp.getWeight(); // [OC, IC, 1, 1]
    Value origBias   = convOp.getBias();   // [1,  1,  1, OC] optional

    // stride/padding/dilation not needed — we use ttir.linear, not ttir.conv2d

    // ── 3. Build new ops ───────────────────────────────────────────────────
    OpBuilder b(convOp);
    Location loc = convOp.getLoc();

    // ── 3a. Weight packing (auto const-eval'd — origWeight is a parameter) ─
    //
    // Implements _make_packed_weight() using 9 TTIR ops — NO embedded constant:
    //
    //   broadcast  [OC,IC,1,1]→[OC,IC,K,K]   expand kH=kW=1 to K×K grid
    //   arange     [1,1,K,K] arange_dim=2    val[0,0,k,*]=k  (row indices)
    //   arange     [1,1,K,K] arange_dim=3    val[0,0,*,k]=k  (col indices)
    //   eq         row==col → [1,1,K,K] bool  True only on K×K diagonal
    //   typecast   bool→bf16  → I_K [1,1,K,K]  1.0/0.0, no dense constant
    //   multiply   w_bc * i_k → [OC,IC,K,K]  zero off-diagonal
    //   permute    [0,2,1,3] → [OC,K,IC,K]   OIHW-compatible (swap IC and K dims)
    //   reshape    → [IC*K, OC*K]
    //   reshape    → [IC*K, OC*K, 1, 1]  OIHW

    // broadcast [OC,IC,1,1] → [OC,IC,K,K]
    auto wBcTy = RankedTensorType::get({OC, IC, K, K}, bf16);
    Value wBc  =
        b.create<ttir::BroadcastOp>(loc, wBcTy, origWeight,
                                     b.getDenseI64ArrayAttr({1, 1, K, K}))
            .getResult();

    // arange [1,1,K,K] along dim=2: val[0,0,k,*]=k  (row index grid)
    auto kGridTy = RankedTensorType::get({1, 1, K, K},
                                          IntegerType::get(ctx, 64));
    Value kRow   =
        b.create<ttir::ArangeOp>(loc, kGridTy,
                                  /*start=*/(int64_t)0, /*end=*/K,
                                  /*step=*/(int64_t)1,
                                  /*arange_dimension=*/(uint64_t)2)
            .getResult();

    // arange [1,1,K,K] along dim=3: val[0,0,*,k]=k  (col index grid)
    Value kCol   =
        b.create<ttir::ArangeOp>(loc, kGridTy,
                                  /*start=*/(int64_t)0, /*end=*/K,
                                  /*step=*/(int64_t)1,
                                  /*arange_dimension=*/(uint64_t)3)
            .getResult();

    // eq: diagonal boolean mask [1,1,K,K] — True where row_index == col_index
    auto diagBoolTy = RankedTensorType::get({1, 1, K, K},
                                             IntegerType::get(ctx, 1));
    Value diagBool  =
        b.create<ttir::EqualOp>(loc, diagBoolTy, kRow, kCol).getResult();

    // typecast bool→bf16: produces I_K [1,1,K,K] (1.0 on diagonal, 0.0 off)
    auto iKTy = RankedTensorType::get({1, 1, K, K}, bf16);
    Value iK  = b.create<ttir::TypecastOp>(loc, iKTy, diagBool).getResult();

    // multiply: w_bc * i_k → [OC,IC,K,K]  (i_k broadcasts [1,1,K,K]→[OC,IC,K,K])
    auto wDiagTy = wBcTy;
    Value wDiag  =
        b.create<ttir::MultiplyOp>(loc, wDiagTy, wBc, iK).getResult();

    // permute [0,2,1,3]: [OC,IC,K,K]→[OC,K,IC,K]  produces transposed weight W^T
    //   result[oc,k,ic,k'] = wDiag[oc,ic,k,k'] = weight[oc,ic,0,0] if k==k'
    //   After reshape: W^T[oc*K+k, ic*K+k'] — NCHW linear computes W^T @ act
    auto wPermTy = RankedTensorType::get({OC, K, IC, K}, bf16);
    Value wPerm  =
        b.create<ttir::PermuteOp>(loc, wPermTy, wDiag,
                                   b.getDenseI64ArrayAttr({0, 2, 1, 3}))
            .getResult();

    // reshape [OC,K,IC,K] → [OC*K, IC*K]
    auto w2dPackedTy = RankedTensorType::get({packedOC, packedIC}, bf16);
    Value w2dPacked  =
        b.create<ttir::ReshapeOp>(
             loc, w2dPackedTy, wPerm,
             b.getI32ArrayAttr({(int32_t)packedOC, (int32_t)packedIC}))
            .getResult();

    // reshape [OC*K, IC*K] → [1, 1, OC*K, IC*K]  transposed weight (A-operand of ttir.linear)
    auto wPackedTy = RankedTensorType::get({1, 1, packedOC, packedIC}, bf16);
    Value wPacked  =
        b.create<ttir::ReshapeOp>(
             loc, wPackedTy, w2dPacked,
             b.getI32ArrayAttr({1, 1, (int32_t)packedOC, (int32_t)packedIC}))
            .getResult();

    // ── 3b. Bias packing (auto const-eval'd — origBias is a parameter) ────
    Value bPacked;
    if (origBias) {
      // reshape [1,1,1,OC] → [OC,1]  (rank-2, NOT rank-1)
      // tt-metal repeat_interleave bug: for a rank-1 tensor [OC] with dim=0,
      // normalized_dim=0 == input_rank-1=0 triggers an internal
      // transpose(-1,-2) on rank-1, calling get_normalized_index(-2) for
      // rank=1 → 1+(-2)=-1 → UINT64_MAX → TT_FATAL.
      // Using [OC,1] (rank-2): dim=0 != input_rank-1=1 → main path, no issue.
      auto bFlatTy = RankedTensorType::get({OC, 1}, actElemTy);
      Value bFlat  =
          b.create<ttir::ReshapeOp>(loc, bFlatTy, origBias,
                                     b.getI32ArrayAttr({(int32_t)OC, 1}))
              .getResult();

      // repeat_interleave([OC,1], K, dim=0) → [OC*K, 1]
      auto bRepTy = RankedTensorType::get({packedOC, 1}, actElemTy);
      Value bRep  =
          b.create<ttir::RepeatInterleaveOp>(
               loc, bRepTy, bFlat,
               b.getUI32IntegerAttr((uint32_t)K),
               b.getSI32IntegerAttr(0))
              .getResult();

      // reshape [OC*K, 1] → [1,1,OC*K,1]  column bias: broadcasts over spatial dim P
      auto bPackedTy = RankedTensorType::get({1, 1, packedOC, 1}, actElemTy);
      bPacked        =
          b.create<ttir::ReshapeOp>(loc, bPackedTy, bRep,
                                     b.getI32ArrayAttr(
                                         {1, 1, (int32_t)packedOC, 1}))
              .getResult();
    }

    // ── 3c. Activation packing (runtime) ──────────────────────────────────
    // Single NCHW-flat reshape: [N,C,H,W] → [N,1,C*K,H/K*W]  free ROW_MAJOR view
    // No permute — channels stay in dim[2] for NCHW-native linear (W^T @ act).
    // TTNNSpatialPackActivationRowMajorOpt moves the tilize to AFTER this reshape.
    int64_t packedSpatial = packedH * W;
    auto aFlatTy = RankedTensorType::get({N, 1, packedIC, packedSpatial}, actElemTy);
    Value aFlat  =
        b.create<ttir::ReshapeOp>(
             loc, aFlatTy, origInput,
             b.getI32ArrayAttr({(int32_t)N, 1, (int32_t)packedIC,
                                (int32_t)packedSpatial}))
            .getResult();

    // ── Single-linear path (NCHW-native) ────────────────────────────────────
    // result = wPacked[1,1,OC*K,IC*K] @ aFlat[N,1,IC*K,P] → [N,1,OC*K,P]
    // No activation permutes — eliminates the two bottleneck permutes (~1,271 µs).
    //
    // wPacked is operand A (not B): TTNNWeightDtypeConversion converts operand B
    // that traces to constant args. aFlat (operand B) is a runtime value so it is
    // not converted. wPacked (operand A) is skipped by the pass entirely.
    // The block-diagonal structure is preserved without a ttcore.weight_dtype attr.
    //
    // One linear per camera keeps L1 pressure halved (vs. per-slice approach that
    // created 2 linears, pushing Y pix-rearrange into UV kernel's CB region).
    auto linearOutTy = RankedTensorType::get({N, 1, packedOC, packedSpatial}, actElemTy);
    Value linearOut  =
        b.create<ttir::LinearOp>(loc, linearOutTy, wPacked, aFlat,
                                  bPacked ? bPacked : Value(),
                                  /*transpose_a=*/false,
                                  /*transpose_b=*/false)
            .getResult();

    // ── 3e. Output unpack ─────────────────────────────────────────────────
    // Single NCHW free-view reshape: [N,1,OC*K,H/K*W] → [N,OC,H,W]
    // No permute. TTNNSpatialPackActivationRowMajorOpt (TTNN level) inserts an
    // untilize (to_layout ROW_MAJOR) before this reshape so it remains a free view.
    auto finalTy = RankedTensorType::get({N, OC, H, W}, actElemTy);
    Value finalOut =
        b.create<ttir::ReshapeOp>(
             loc, finalTy, linearOut,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)OC, (int32_t)H, (int32_t)W}))
            .getResult();

    // ── 4. Replace old chain ──────────────────────────────────────────────
    postPerm.getResult().replaceAllUsesWith(finalOut);
    postPerm.erase();
    convOp.erase();
    prePerm.erase();
  }
};

// ── TTIRDepthwiseConvSpatialPackingOpt ──────────────────────────────────────
//
// Handles the UV AveragePool pattern:
//   permute({0,2,3,1}) → conv2d{groups=IC, channel_last=true} → permute({0,3,1,2})
// where IC < TILE_WIDTH and TILE_WIDTH % IC == 0.
//
// K = TILE_WIDTH / IC.  packed_IC = IC * K = TILE_WIDTH (tile-aligned, zero waste).
//
// Weight packing: repeat_interleave(K, dim=0) on [IC,1,kH,kW] → [IC*K,1,kH,kW]
//   Each original filter is replicated K times so each of the K spatial row
//   groups applies the same kernel.
//
// The packed permute reads packed_IC=TILE_WIDTH columns per tile → zero padding
// waste (vs IC=2 → padded to 32 = 93.8% zeros at baseline).
//
// Math correctness (for the reshape-based unpack):
//   Original output: [N, out_H, out_W, IC]
//   Packed output:   [N, packed_out_H, out_W, IC*K]
//   where packed_out_H = (H/K - kH) / sH + 1
//   Guard: K * packed_out_H == out_H  (checked at match time).
//
//   Packed NCHW [N, IC*K, packed_out_H, out_W] reshapes to [N, IC, out_H, out_W]
//   because flat index for packed channel (orig_ic*K+k) at row hp maps to
//   original (orig_ic, k*packed_out_H + hp).  In ROW_MAJOR this is a pure
//   contiguous view.

class TTIRDepthwiseConvSpatialPackingOptPass
    : public impl::TTIRDepthwiseConvSpatialPackingOptBase<
          TTIRDepthwiseConvSpatialPackingOptPass> {
public:
  using impl::TTIRDepthwiseConvSpatialPackingOptBase<
      TTIRDepthwiseConvSpatialPackingOptPass>::
      TTIRDepthwiseConvSpatialPackingOptBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<ttir::Conv2dOp, 4> candidates;
    mod.walk([&](ttir::Conv2dOp op) { candidates.push_back(op); });

    llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] " << candidates.size()
                 << " Conv2dOp candidates\n";
    for (ttir::Conv2dOp convOp : candidates)
      process(convOp, ctx);
  }

private:
  void process(ttir::Conv2dOp convOp, MLIRContext *ctx) {
    static const int64_t kNCHWtoNHWC[] = {0, 2, 3, 1};
    static const int64_t kNHWCtoNCHW[] = {0, 3, 1, 2};

    // ── 1. Match permute → depthwise conv2d → permute ────────────────────
    // By the time this pass runs (after CanonicalizerPass), two consecutive
    // TransposeOps have been converted to PermuteOps and folded into one:
    //   transpose(-3,-2) → transpose(-2,-1) → conv2d  becomes:
    //   permute({0,2,3,1})                  → conv2d

    auto *prePermuteOp = convOp.getInput().getDefiningOp();
    if (!isPermute(prePermuteOp, kNCHWtoNHWC))
      return;
    auto prePerm = mlir::cast<ttir::PermuteOp>(prePermuteOp);
    Value origInput = prePerm.getInput(); // [N,IC,H,W] NCHW

    ttir::PermuteOp postPerm;
    for (Operation *u : convOp.getResult().getUsers()) {
      if (isPermute(u, kNHWCtoNCHW)) {
        postPerm = mlir::cast<ttir::PermuteOp>(u);
        break;
      }
    }
    if (!postPerm)
      return;

    // ── 2. Guard checks ──────────────────────────────────────────────────
    // Weight OIHW: [OC, IC_per_group, kH, kW].  For depthwise OC=IC, IC_per_group=1.
    auto weightType = mlir::cast<RankedTensorType>(convOp.getWeight().getType());
    if (weightType.getRank() != 4)
      return;
    int64_t IC      = weightType.getDimSize(0); // = OC for depthwise
    int64_t icGroup = weightType.getDimSize(1); // must be 1 for depthwise
    int64_t kH      = weightType.getDimSize(2);
    int64_t kW      = weightType.getDimSize(3);
    if (icGroup != 1) {
      llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] SKIP IC=" << IC
                   << " icGroup=" << icGroup << " (not depthwise)\n";
      return;
    }

    // Must be depthwise: groups == IC.
    if (static_cast<int64_t>(convOp.getGroups()) != IC) {
      llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] SKIP groups="
                   << convOp.getGroups() << " != IC=" << IC << "\n";
      return;
    }

    // IC must be narrow and TILE_WIDTH must divide by IC exactly.
    if (IC >= TILE_WIDTH || (TILE_WIDTH % IC) != 0) {
      llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] SKIP IC=" << IC
                   << " fails narrow/tile-divisible guard (TILE_WIDTH=" << TILE_WIDTH << ")\n";
      return;
    }

    // Extract stride and padding.
    SmallVector<int32_t> stride, padding, dilation;
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getStride()))
      stride.assign(arr.asArrayRef().begin(), arr.asArrayRef().end());
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getPadding()))
      padding.assign(arr.asArrayRef().begin(), arr.asArrayRef().end());
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getDilation()))
      dilation.assign(arr.asArrayRef().begin(), arr.asArrayRef().end());
    if (stride.size() < 2 || padding.size() < 4 || dilation.size() < 2)
      return;
    int32_t sH = stride[0], sW = stride[1];

    // Verify input and output shapes.
    auto inputType = mlir::cast<RankedTensorType>(origInput.getType());
    if (inputType.getRank() != 4)
      return;
    int64_t N = inputType.getDimSize(0);
    int64_t H = inputType.getDimSize(2);
    int64_t W = inputType.getDimSize(3);

    // Get original output dims from the conv2d result (channel_last: [N,H',W',IC]).
    auto resultType = mlir::cast<RankedTensorType>(convOp.getResult().getType());
    if (resultType.getRank() != 4)
      return;
    int64_t out_H = resultType.getDimSize(1);
    int64_t out_W = resultType.getDimSize(2);

    int64_t K        = TILE_WIDTH / IC;
    int64_t packedIC = IC * K; // == TILE_WIDTH

    if (H % K != 0)
      return;
    int64_t packedH     = H / K;
    int64_t packedOut_H = (packedH - kH) / sH + 1;

    // Verify the unpack reshape is mathematically correct.
    if (K * packedOut_H != out_H)
      return;

    llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] MATCHED depthwise"
                 << " IC=" << IC << " K=" << K << " kH=" << kH << " kW=" << kW
                 << " H=" << H << " W=" << W << " packedIC=" << packedIC << "\n";

    auto bf16 = BFloat16Type::get(ctx);
    Value origWeight = convOp.getWeight(); // [IC, 1, kH, kW]

    // ── 3. Build new ops ─────────────────────────────────────────────────
    OpBuilder b(convOp);
    Location loc = convOp.getLoc();

    // ── 3a. Weight packing (auto const-eval'd) ───────────────────────────
    // repeat_interleave(K, dim=0): [IC,1,kH,kW] → [IC*K,1,kH,kW]
    // Each original filter replicated K times so every spatial row group
    // applies the same depthwise kernel.
    auto wPackedTy = RankedTensorType::get({packedIC, 1, kH, kW}, bf16);
    Value wPacked  =
        b.create<ttir::RepeatInterleaveOp>(
             loc, wPackedTy, origWeight,
             b.getUI32IntegerAttr((uint32_t)K),
             b.getSI32IntegerAttr(0)) // dim=0
            .getResult();

    // ── 3b. Activation packing (runtime) ─────────────────────────────────
    // reshape [N,IC,H,W] → [N,IC*K,H/K,W]  free view — row groups → channels
    auto aPkTy = RankedTensorType::get({N, packedIC, packedH, W}, bf16);
    Value aPk  =
        b.create<ttir::ReshapeOp>(
             loc, aPkTy, origInput,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)packedIC,
                                (int32_t)packedH, (int32_t)W}))
            .getResult();

    // permute({0,2,3,1}): [N,IC*K,H/K,W] → [N,H/K,W,IC*K]  NHWC packed
    // IC*K = TILE_WIDTH → zero tile-column padding, no DRAM inflation.
    auto aPkNhwcTy = RankedTensorType::get({N, packedH, W, packedIC}, bf16);
    Value aPkNhwc  =
        b.create<ttir::PermuteOp>(
             loc, aPkNhwcTy, aPk,
             b.getDenseI64ArrayAttr({0, 2, 3, 1}))
            .getResult();

    // ── 3c. Packed depthwise conv2d ──────────────────────────────────────
    // Input:  [N, H/K, W, IC*K]  NHWC  (channel_last=true)
    // Weight: [IC*K, 1, kH, kW]  OIHW  groups=IC*K  (still fully depthwise)
    // Output: [N, packed_out_H, out_W, IC*K]  NHWC
    // Use the custom builder: (resultType, input, weight, bias, stride,
    //   padding, dilation, uint32_t groups, FlattenedCompatInfoAttr=nullptr)
    // Default batch_dim=0, height_dim=1, width_dim=2, channel_dim=3 (NHWC).
    auto convPackedTy =
        RankedTensorType::get({N, packedOut_H, out_W, packedIC}, bf16);
    auto convPackedOp =
        b.create<ttir::Conv2dOp>(
             loc, convPackedTy,
             /*input=*/aPkNhwc,
             /*weight=*/wPacked,
             /*bias=*/Value(),
             /*stride=*/
             b.getDenseI32ArrayAttr({sH, sW}),
             /*padding=*/
             b.getDenseI32ArrayAttr({padding[0], padding[1],
                                     padding[2], padding[3]}),
             /*dilation=*/
             b.getDenseI32ArrayAttr({dilation[0], dilation[1]}),
             /*groups=*/(uint32_t)packedIC,
             /*flattened_compat_info=*/nullptr);
    convPackedOp->setAttr("channel_last", b.getUnitAttr());
    Value convPacked = convPackedOp.getResult();

    // ── 3d. Output unpack ────────────────────────────────────────────────
    // permute({0,3,1,2}): [N,packed_out_H,out_W,IC*K] → [N,IC*K,packed_out_H,out_W]
    auto oNchwTy = RankedTensorType::get({N, packedIC, packedOut_H, out_W}, bf16);
    Value oNchw  =
        b.create<ttir::PermuteOp>(
             loc, oNchwTy, convPacked,
             b.getDenseI64ArrayAttr({0, 3, 1, 2}))
            .getResult();

    // reshape [N,IC*K,packed_out_H,out_W] → [N,IC,out_H,out_W]  free view unpack
    // In ROW_MAJOR: element [cp, hp_out] where cp=orig_ic*K+k maps to
    // output [orig_ic, k*packed_out_H+hp_out] — mathematically correct.
    auto finalTy = RankedTensorType::get({N, IC, out_H, out_W}, bf16);
    Value finalOut =
        b.create<ttir::ReshapeOp>(
             loc, finalTy, oNchw,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)IC,
                                (int32_t)out_H, (int32_t)out_W}))
            .getResult();

    // ── 4. Replace old chain ─────────────────────────────────────────────
    postPerm.getResult().replaceAllUsesWith(finalOut);
    postPerm.erase();
    convOp.erase();
    prePerm.erase();
    llvm::errs() << "[TTIRDepthwiseConvSpatialPackingOpt] REWRITTEN depthwise"
                 << " IC=" << IC << " → packedIC=" << packedIC << "\n";
  }
};

// ── TTIRPointwiseConv2dPartialPackingOpt ────────────────────────────────────
//
// Same permute→conv2d→permute pattern as TTIRSpatialRowGroupPackingOpt but
// fires when packingFactor(IC) < TILE_WIDTH (partial K, e.g. IC=24 → K=4).
//
// Uses ttir.conv2d (not ttir.linear) to avoid TTNNSpatialPackActivationRowMajorOpt
// which specifically matches LinearOp and interferes with the linear path for
// partial K, degrading PCC from 0.99 to ~0.972.
//
// Weight format for conv2d OIHW [OC*K, IC*K, 1, 1]:
//   W_conv[oc*K+k, ic*K+k', 0, 0] = W[oc, ic]  if k==k'  else 0
//
// Building from W[OC,IC,1,1]:
//   broadcast [OC,IC,1,1] → [OC,IC,K,K]
//   arange(dim=2)+arange(dim=3)+eq → I_K [1,1,K,K] diagonal mask
//   multiply → [OC,IC,K,K] (zeroed off-diagonal)
//   permute [0,2,1,3] → [OC,K,IC,K]     ← OIHW reordering (vs [1,2,0,3] for linear)
//   reshape [OC*K, IC*K] → [OC*K, IC*K, 1, 1]
//
// Correctness:
//   out_pack[sp, oc*K+k] = Σ_ic x_pack[sp, ic*K+k] * W[oc,ic]
//                        = conv_out[oc, k*(H/K)+h', w]   where sp=h'*W+w  ✓

class TTIRPointwiseConv2dPartialPackingOptPass
    : public impl::TTIRPointwiseConv2dPartialPackingOptBase<
          TTIRPointwiseConv2dPartialPackingOptPass> {
public:
  using impl::TTIRPointwiseConv2dPartialPackingOptBase<
      TTIRPointwiseConv2dPartialPackingOptPass>::
      TTIRPointwiseConv2dPartialPackingOptBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<ttir::Conv2dOp, 4> candidates;
    mod.walk([&](ttir::Conv2dOp op) { candidates.push_back(op); });

    llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] " << candidates.size()
                 << " Conv2dOp candidates\n";
    for (ttir::Conv2dOp convOp : candidates)
      process(convOp, ctx);
  }

private:
  // Check if op is ttir.transpose with the given normalized dim pair (accepts
  // both negative and positive indices for a 4-D tensor).
  static bool isTranspose4D(Operation *op, int32_t d0, int32_t d1) {
    auto tr = mlir::dyn_cast_or_null<ttir::TransposeOp>(op);
    if (!tr)
      return false;
    auto norm = [](int32_t d) -> int32_t { return d < 0 ? d + 4 : d; };
    return norm(tr.getDim0()) == norm(d0) && norm(tr.getDim1()) == norm(d1);
  }

  void process(ttir::Conv2dOp convOp, MLIRContext *ctx) {
    static const int64_t kNCHWtoNHWC[] = {0, 2, 3, 1};
    static const int64_t kNHWCtoNCHW[] = {0, 3, 1, 2};

    // ── 0. Dump candidate pattern for diagnostic ──────────────────────────────
    {
      auto wt = mlir::cast<RankedTensorType>(convOp.getWeight().getType());
      auto it = mlir::cast<RankedTensorType>(convOp.getInput().getType());
      auto *inDefOp = convOp.getInput().getDefiningOp();
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt][CANDIDATE]"
                   << " weight=" << wt
                   << " input=" << it
                   << " channel_last=" << convOp->hasAttr("channel_last")
                   << " groups=" << convOp.getGroups()
                   << " inputDefOp=" << (inDefOp ? inDefOp->getName().getStringRef() : "<block-arg>")
                   << "\n";
      if (auto tr = mlir::dyn_cast_or_null<ttir::TransposeOp>(inDefOp))
        llvm::errs() << "  transpose dim0=" << tr.getDim0()
                     << " dim1=" << tr.getDim1() << "\n";
      if (auto perm = mlir::dyn_cast_or_null<ttir::PermuteOp>(inDefOp)) {
        llvm::errs() << "  permute dims=[";
        for (int64_t d : perm.getPermutation()) llvm::errs() << d << ",";
        llvm::errs() << "]\n";
      }
    }

    // ── 1. Pattern detection — two supported paths ────────────────────────────
    //
    // Path A (NCHW→NHWC permute wrapper, existing):
    //   permute({0,2,3,1}) [NCHW→NHWC] → conv2d → permute({0,3,1,2}) [NHWC→NCHW]
    //
    // Path B (backbone channel_last, new):
    //   transpose(-3,-2) → transpose(-2,-1) → conv2d{channel_last}  (no post-permute)
    //   The two transposes are equivalent to permute({0,2,3,1}) but TTIRFusing
    //   doesn't fold them when the downstream backbone stays fully channel_last.

    Value origInput; // NCHW [N, IC, H, W] source for packing (both paths)
    ttir::PermuteOp prePerm, postPerm;
    ttir::TransposeOp preTr1, preTr2;
    bool pathB = false; // true → channel_last output, no postPerm to erase

    auto *inputDefOp = convOp.getInput().getDefiningOp();

    if (isPermute(inputDefOp, kNCHWtoNHWC)) {
      // Path A: single permute NCHW→NHWC before conv2d.
      prePerm   = mlir::cast<ttir::PermuteOp>(inputDefOp);
      origInput = prePerm.getInput();
      for (Operation *u : convOp.getResult().getUsers()) {
        if (isPermute(u, kNHWCtoNCHW)) {
          postPerm = mlir::cast<ttir::PermuteOp>(u);
          break;
        }
      }
      // Path A1: permute→conv2d→permute  (NCHW output)
      // Path A2: permute→conv2d channel_last, no post-permute (NHWC output)
      //   TTIRFusing may fold two transposes into a single permute and the
      //   backbone can stay channel_last throughout.
      if (!postPerm) {
        if (!convOp->hasAttr("channel_last")) {
          llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP pathA"
                       << " no postPerm and not channel_last\n";
          return;
        }
        pathB = true; // reuse Path B output logic (channel_last result type)
      }
    } else if (isTranspose4D(inputDefOp, -2, -1)) {
      // Path B: transpose(-3,-2) → transpose(-2,-1) → conv2d{channel_last}
      // The second transpose (dim0=-2, dim1=-1) is the direct input to conv2d.
      preTr2    = mlir::cast<ttir::TransposeOp>(inputDefOp);
      auto *tr1Op = preTr2.getInput().getDefiningOp();
      if (!isTranspose4D(tr1Op, -3, -2))
        return;
      preTr1    = mlir::cast<ttir::TransposeOp>(tr1Op);
      origInput = preTr1.getInput(); // NCHW source [N, IC, H, W]
      pathB     = true;
      if (!convOp->hasAttr("channel_last"))
        return;
      for (Operation *u : convOp.getResult().getUsers())
        if (isPermute(u, kNHWCtoNCHW))
          return;
    } else {
      return; // no recognised pre-op pattern
    }

    // ── 2. Guard: partial-K 1×1 pointwise conv, non-depthwise ───────────────
    auto weightType = mlir::cast<RankedTensorType>(convOp.getWeight().getType());
    if (weightType.getRank() != 4)
      return;
    int64_t OC = weightType.getDimSize(0);
    int64_t IC = weightType.getDimSize(1);
    int64_t kH = weightType.getDimSize(2);
    int64_t kW = weightType.getDimSize(3);

    if (kH != 1 || kW != 1) {
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP IC=" << IC
                   << " kH=" << kH << " kW=" << kW << " (not 1x1)\n";
      return;
    }
    if (IC >= TILE_WIDTH) {
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP IC=" << IC
                   << " >= TILE_WIDTH=" << TILE_WIDTH << "\n";
      return;
    }

    int64_t K = packingFactor(IC);

    // Only handle partial packing (K < TILE_WIDTH).
    // K == TILE_WIDTH is handled by TTIRSpatialRowGroupPackingOpt (linear path).
    if (K == TILE_WIDTH) {
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP IC=" << IC
                   << " K=TILE_WIDTH=" << TILE_WIDTH << " (handled by RowGroupPacking)\n";
      return;
    }

    // Non-depthwise only.
    if (convOp.getGroups() != 1) {
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP IC=" << IC
                   << " groups=" << convOp.getGroups() << " (depthwise, skip)\n";
      return;
    }
    llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] Checking 1x1 conv"
                 << " IC=" << IC << " OC=" << OC << " K=" << K
                 << " path=" << (pathB ? "B(channel_last)" : "A(permute)") << "\n";

    // Stride [1,1], padding all zeros.
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getStride()))
      for (int32_t s : arr.asArrayRef())
        if (s != 1)
          return;
    if (auto arr = mlir::dyn_cast<DenseI32ArrayAttr>(convOp.getPadding()))
      for (int32_t p : arr.asArrayRef())
        if (p != 0)
          return;

    // origInput is always NCHW [N, IC, H, W]: dim0=N, dim1=IC, dim2=H, dim3=W.
    auto inputType = mlir::cast<RankedTensorType>(origInput.getType());
    if (inputType.getRank() != 4)
      return;
    int64_t N = inputType.getDimSize(0);
    int64_t H = inputType.getDimSize(2);
    int64_t W = inputType.getDimSize(3);

    if (H % K != 0) {
      llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] SKIP IC=" << IC
                   << " K=" << K << " H=" << H << " not divisible by K\n";
      return;
    }

    int64_t packedH  = H / K;
    int64_t packedIC = IC * K;
    int64_t packedOC = OC * K;

    llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] MATCHED 1x1 pointwise"
                 << " IC=" << IC << " OC=" << OC << " K=" << K
                 << " H=" << H << " W=" << W
                 << " packedIC=" << packedIC << " packedOC=" << packedOC
                 << " path=" << (pathB ? "B(channel_last)" : "A(permute)") << "\n";

    auto bf16     = BFloat16Type::get(ctx);
    auto actElemTy = mlir::cast<RankedTensorType>(origInput.getType()).getElementType();

    Value origWeight = convOp.getWeight(); // [OC, IC, 1, 1]
    Value origBias   = convOp.getBias();   // [1,  1,  1, OC] optional

    // ── 3. Build new ops ─────────────────────────────────────────────────────
    OpBuilder b(convOp);
    Location loc = convOp.getLoc();

    // ── 3a. Weight packing for conv2d OIHW [OC*K, IC*K, 1, 1] ──────────────
    //
    // broadcast [OC,IC,1,1] → [OC,IC,K,K]
    auto wBcTy = RankedTensorType::get({OC, IC, K, K}, bf16);
    Value wBc  =
        b.create<ttir::BroadcastOp>(loc, wBcTy, origWeight,
                                     b.getDenseI64ArrayAttr({1, 1, K, K}))
            .getResult();

    // arange(dim=2) row-index grid [1,1,K,K]
    auto kGridTy = RankedTensorType::get({1, 1, K, K},
                                          IntegerType::get(ctx, 64));
    Value kRow   =
        b.create<ttir::ArangeOp>(loc, kGridTy,
                                  (int64_t)0, K, (int64_t)1, (uint64_t)2)
            .getResult();

    // arange(dim=3) col-index grid [1,1,K,K]
    Value kCol   =
        b.create<ttir::ArangeOp>(loc, kGridTy,
                                  (int64_t)0, K, (int64_t)1, (uint64_t)3)
            .getResult();

    // diagonal bool mask [1,1,K,K]
    auto diagBoolTy = RankedTensorType::get({1, 1, K, K},
                                             IntegerType::get(ctx, 1));
    Value diagBool  =
        b.create<ttir::EqualOp>(loc, diagBoolTy, kRow, kCol).getResult();

    // typecast bool→bf16  I_K [1,1,K,K]
    auto iKTy = RankedTensorType::get({1, 1, K, K}, bf16);
    Value iK  = b.create<ttir::TypecastOp>(loc, iKTy, diagBool).getResult();

    // multiply: w_bc * i_k → [OC,IC,K,K]  zero off-diagonal
    Value wDiag = b.create<ttir::MultiplyOp>(loc, wBcTy, wBc, iK).getResult();

    // permute [0,2,1,3]: [OC,IC,K,K] → [OC,K,IC,K]  (OIHW reordering)
    //   result[oc,k,ic,k'] = wDiag[oc,ic,k,k']
    //   After reshape: w_conv[oc*K+k, ic*K+k'] = W[oc,ic] if k==k' else 0  ✓
    auto wPermTy = RankedTensorType::get({OC, K, IC, K}, bf16);
    Value wPerm  =
        b.create<ttir::PermuteOp>(loc, wPermTy, wDiag,
                                   b.getDenseI64ArrayAttr({0, 2, 1, 3}))
            .getResult();

    // reshape [OC,K,IC,K] → [OC*K, IC*K]
    auto w2dTy = RankedTensorType::get({packedOC, packedIC}, bf16);
    Value w2d  =
        b.create<ttir::ReshapeOp>(
             loc, w2dTy, wPerm,
             b.getI32ArrayAttr({(int32_t)packedOC, (int32_t)packedIC}))
            .getResult();

    // reshape [OC*K, IC*K] → [OC*K, IC*K, 1, 1]  OIHW conv2d weight
    auto wPackedTy = RankedTensorType::get({packedOC, packedIC, 1, 1}, bf16);
    Value wPacked  =
        b.create<ttir::ReshapeOp>(
             loc, wPackedTy, w2d,
             b.getI32ArrayAttr({(int32_t)packedOC, (int32_t)packedIC, 1, 1}))
            .getResult();

    // ── 3b. Bias packing (auto const-eval'd) ─────────────────────────────────
    // [1,1,1,OC] → [OC,1] → repeat_interleave(K,dim=0) → [OC*K,1] → [1,1,1,OC*K]
    Value bPacked;
    if (origBias) {
      auto bFlatTy = RankedTensorType::get({OC, 1}, actElemTy);
      Value bFlat  =
          b.create<ttir::ReshapeOp>(loc, bFlatTy, origBias,
                                     b.getI32ArrayAttr({(int32_t)OC, 1}))
              .getResult();

      auto bRepTy = RankedTensorType::get({packedOC, 1}, actElemTy);
      Value bRep  =
          b.create<ttir::RepeatInterleaveOp>(
               loc, bRepTy, bFlat,
               b.getUI32IntegerAttr((uint32_t)K),
               b.getSI32IntegerAttr(0))
              .getResult();

      auto bPackedTy = RankedTensorType::get({1, 1, 1, packedOC}, actElemTy);
      bPacked        =
          b.create<ttir::ReshapeOp>(loc, bPackedTy, bRep,
                                     b.getI32ArrayAttr(
                                         {1, 1, 1, (int32_t)packedOC}))
              .getResult();
    }

    // ── 3c. Activation packing (runtime) ─────────────────────────────────────
    // reshape [N,IC,H,W] → [N,IC*K,H/K,W]  free view
    auto aNTy = RankedTensorType::get({N, packedIC, packedH, W}, actElemTy);
    Value aN  =
        b.create<ttir::ReshapeOp>(
             loc, aNTy, origInput,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)packedIC,
                                (int32_t)packedH, (int32_t)W}))
            .getResult();

    // permute {0,2,3,1}: [N,IC*K,H/K,W] → [N,H/K,W,IC*K]  NHWC — 0% tile waste
    auto aNHWCTy = RankedTensorType::get({N, packedH, W, packedIC}, actElemTy);
    Value aNHWC  =
        b.create<ttir::PermuteOp>(
             loc, aNHWCTy, aN,
             b.getDenseI64ArrayAttr({0, 2, 3, 1}))
            .getResult();

    // ── 3d. Packed 1×1 conv2d (channel_last, groups=1) ───────────────────────
    // Input  [N,H/K,W,IC*K]  NHWC
    // Weight [OC*K, IC*K, 1, 1]  OIHW  (block-diagonal)
    // Output [N,H/K,W,OC*K]  NHWC
    // TTIRFlattenSlidingWindow will add the [N,H/K,W,C] → [1,1,N*H/K*W,C] reshape.
    auto convPackedTy =
        RankedTensorType::get({N, packedH, W, packedOC}, actElemTy);
    auto convPackedOp =
        b.create<ttir::Conv2dOp>(
             loc, convPackedTy,
             /*input=*/aNHWC,
             /*weight=*/wPacked,
             /*bias=*/bPacked ? bPacked : Value(),
             /*stride=*/b.getDenseI32ArrayAttr({1, 1}),
             /*padding=*/b.getDenseI32ArrayAttr({0, 0, 0, 0}),
             /*dilation=*/b.getDenseI32ArrayAttr({1, 1}),
             /*groups=*/(uint32_t)1,
             /*flattened_compat_info=*/nullptr);
    convPackedOp->setAttr("channel_last", b.getUnitAttr());
    Value convPacked = convPackedOp.getResult();

    // ── 3e. Output unpack ─────────────────────────────────────────────────────
    // permute {0,3,1,2}: [N,H/K,W,OC*K] → [N,OC*K,H/K,W]  NCHW
    auto oPerTy = RankedTensorType::get({N, packedOC, packedH, W}, actElemTy);
    Value oPer  =
        b.create<ttir::PermuteOp>(
             loc, oPerTy, convPacked,
             b.getDenseI64ArrayAttr({0, 3, 1, 2}))
            .getResult();

    // reshape [N,OC*K,H/K,W] → [N,OC,H,W]  free view unpack
    auto ncHWTy = RankedTensorType::get({N, OC, H, W}, actElemTy);
    Value ncHW  =
        b.create<ttir::ReshapeOp>(
             loc, ncHWTy, oPer,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)OC, (int32_t)H, (int32_t)W}))
            .getResult();

    // Path B: backbone stays channel_last → add NCHW→NHWC permute so the
    // result type matches the original conv2d output [N,H,W,OC] NHWC.
    Value finalOut;
    if (pathB) {
      auto nhwcTy = RankedTensorType::get({N, H, W, OC}, actElemTy);
      finalOut    =
          b.create<ttir::PermuteOp>(
               loc, nhwcTy, ncHW,
               b.getDenseI64ArrayAttr({0, 2, 3, 1}))
              .getResult();
    } else {
      finalOut = ncHW; // Path A: NCHW replaces postPerm's NCHW result
    }

    // ── 4. Replace old chain ──────────────────────────────────────────────────
    if (pathB) {
      // Path A2 / Path B: replace the channel_last conv2d output directly.
      convOp.getResult().replaceAllUsesWith(finalOut);
      convOp.erase();
      if (prePerm) {
        // Path A2: single pre-permute op, erase it.
        if (prePerm.getResult().use_empty())
          prePerm.erase();
      } else {
        // Path B: two-transpose chain; erase if dead.
        if (preTr2 && preTr2.getResult().use_empty())
          preTr2.erase();
        if (preTr1 && preTr1.getResult().use_empty())
          preTr1.erase();
      }
    } else {
      // Path A1: replace postPerm (NCHW output) and erase pre-permute chain.
      postPerm.getResult().replaceAllUsesWith(finalOut);
      postPerm.erase();
      convOp.erase();
      prePerm.erase();
    }
    llvm::errs() << "[TTIRPointwiseConv2dPartialPackingOpt] REWRITTEN 1x1 pointwise"
                 << " IC=" << IC << " OC=" << OC
                 << " → packedIC=" << packedIC << " packedOC=" << packedOC
                 << " path=" << (pathB ? "B(channel_last)" : "A(permute)") << "\n";
  }
};

} // namespace

} // namespace mlir::tt::ttir
