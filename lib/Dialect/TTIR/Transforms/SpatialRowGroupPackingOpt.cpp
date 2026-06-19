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
// This pass replaces the pattern with spatial row-group packing so C*K=96 fills
// tile rows completely (zero padding waste for C=3, K=32):
//
//   ── Weight packing (ops on constant parameter → auto const-eval'd) ────────
//   Maps directly to _make_packed_weight() in Python (9 ops, no embedded constant):
//
//   broadcast  %weight [OC,IC,1,1] → [OC,IC,K,K]   expand kH=kW=1 to K×K
//   arange     [1,1,K,K] arange_dim=2               val[0,0,k,*]=k  (row indices)
//   arange     [1,1,K,K] arange_dim=3               val[0,0,*,k]=k  (col indices)
//   eq         row_grid == col_grid → [1,1,K,K] bool  True on K×K diagonal only
//   typecast   bool → bf16           [1,1,K,K]       1.0/0.0 (identity I_K)
//   multiply   w_bc * i_k            [OC,IC,K,K]     zero off-diagonal elements
//              (i_k broadcasts [1,1,K,K] → [OC,IC,K,K] implicitly)
//   permute    [1,2,0,3]             [OC,IC,K,K]→[IC,K,OC,K]  absorbs W.T
//   reshape    [IC,K,OC,K]→[IC*K, OC*K]
//   reshape    → [IC*K, OC*K, 1, 1]  OIHW packed weight
//
//   No embedded tensor constant — I_K is computed from arange+eq (zero overhead).
//
//   ── Bias packing (ops on constant parameter → auto const-eval'd) ─────────
//   reshape(%bias, [OC,1])                   rank-2 to avoid tt-metal rank-1 bug
//   repeat_interleave([OC,1], K, dim=0)  → [OC*K, 1]
//   reshape([1, 1, 1, OC*K])
//
//   ── Activation packing (runtime) ─────────────────────────────────────────
//   reshape(%input, [N, C*K, H/K, W])    free view
//   permute({0, 2, 3, 1})                → [N, H/K, W, C*K]  NHWC  17.7 MB ✓
//
//   ── Packed conv2d → output unpack ────────────────────────────────────────
//   conv2d(packed_act, W_packed, B_packed, stride=1, pad=0)
//   → [N, H/K, W, OC*K]
//   permute({0, 3, 1, 2}) → [N, OC*K, H/K, W]
//   reshape([N, OC, H, W])               free view
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

#include <cmath>
#include <numeric>

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRSPATIALROWGROUPPACKINGOPT
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

    // permute [1,2,0,3]: [OC,IC,K,K]→[IC,K,OC,K]  absorbs W.T (matches Python _make_packed_weight)
    //   result[ic,k,oc,k'] = wDiag[oc,ic,k,k'] = weight[oc,ic,0,0] if k==k'
    //   After reshape: W[ic*K+k, oc*K+k'] is the linear weight matrix (row=input ch, col=output ch)
    auto wPermTy = RankedTensorType::get({IC, K, OC, K}, bf16);
    Value wPerm  =
        b.create<ttir::PermuteOp>(loc, wPermTy, wDiag,
                                   b.getDenseI64ArrayAttr({1, 2, 0, 3}))
            .getResult();

    // reshape [IC,K,OC,K] → [IC*K, OC*K]
    auto w2dPackedTy = RankedTensorType::get({packedIC, packedOC}, bf16);
    Value w2dPacked  =
        b.create<ttir::ReshapeOp>(
             loc, w2dPackedTy, wPerm,
             b.getI32ArrayAttr({(int32_t)packedIC, (int32_t)packedOC}))
            .getResult();

    // reshape [IC*K, OC*K] → [1, 1, IC*K, OC*K]  for ttir.linear
    // No prepare_conv2d_weights needed — linear takes weight directly
    auto wPackedTy = RankedTensorType::get({1, 1, packedIC, packedOC}, bf16);
    Value wPacked  =
        b.create<ttir::ReshapeOp>(
             loc, wPackedTy, w2dPacked,
             b.getI32ArrayAttr({1, 1, (int32_t)packedIC, (int32_t)packedOC}))
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
      auto bFlatTy = RankedTensorType::get({OC, 1}, bf16);
      Value bFlat  =
          b.create<ttir::ReshapeOp>(loc, bFlatTy, origBias,
                                     b.getI32ArrayAttr({(int32_t)OC, 1}))
              .getResult();

      // repeat_interleave([OC,1], K, dim=0) → [OC*K, 1]
      auto bRepTy = RankedTensorType::get({packedOC, 1}, bf16);
      Value bRep  =
          b.create<ttir::RepeatInterleaveOp>(
               loc, bRepTy, bFlat,
               b.getUI32IntegerAttr((uint32_t)K),
               b.getSI32IntegerAttr(0))
              .getResult();

      // reshape [OC*K, 1] → [1,1,1,OC*K]
      auto bPackedTy = RankedTensorType::get({1, 1, 1, packedOC}, bf16);
      bPacked        =
          b.create<ttir::ReshapeOp>(loc, bPackedTy, bRep,
                                     b.getI32ArrayAttr(
                                         {1, 1, 1, (int32_t)packedOC}))
              .getResult();
    }

    // ── 3c. Activation packing (runtime) ──────────────────────────────────
    // reshape [N,C,H,W] → [N,C*K,H/K,W]  free view
    auto aNTy = RankedTensorType::get({N, packedIC, packedH, W}, bf16);
    Value aN  =
        b.create<ttir::ReshapeOp>(
             loc, aNTy, origInput,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)packedIC,
                                (int32_t)packedH, (int32_t)W}))
            .getResult();

    // permute({0,2,3,1}): [N,C*K,H/K,W] → [N,H/K,W,C*K]  NHWC packed
    //   writes only 17.7 MB vs 151 MB baseline — C*K=96=3×TILE_WIDTH fills tiles 100%
    auto aNHWCTy = RankedTensorType::get({N, packedH, W, packedIC}, bf16);
    Value aNHWC  =
        b.create<ttir::PermuteOp>(
             loc, aNHWCTy, aN,
             b.getDenseI64ArrayAttr({0, 2, 3, 1}))
            .getResult();

    // reshape [N,H/K,W,C*K] → [N,1,H/K*W,C*K]  flatten spatial for ttir.linear
    int64_t packedSpatial = packedH * W;
    auto aFlatTy = RankedTensorType::get({N, 1, packedSpatial, packedIC}, bf16);
    Value aFlat  =
        b.create<ttir::ReshapeOp>(
             loc, aFlatTy, aNHWC,
             b.getI32ArrayAttr({(int32_t)N, 1, (int32_t)packedSpatial,
                                (int32_t)packedIC}))
            .getResult();

    // ── Single-linear path ───────────────────────────────────────────────────
    // ONE packed linear [packedSpatial × packedOC] for the full OC (e.g. OC=3).
    // Any downstream slice_static users on postPerm (Y and UV channel splits)
    // are automatically redirected to finalOut via replaceAllUsesWith below —
    // they survive unchanged and now slice the single packed output instead.
    //
    // This produces 1 ttnn.linear per camera (5 total for the 5-camera BEV
    // model). The previous per-slice approach created 2 linears per camera (10
    // total), doubling the L1 pressure from spatial packing. That extra L1
    // usage pushed the Y backbone pixel-rearrangement tensor down to address
    // ~622976 (inside the UV DramWidth kernel's static CB region [622592-695712])
    // causing the "CBs clash with L1 buffers" runtime crash. With a single linear
    // the L1 pressure is halved and the Y pix-rearrange lands at a safe address.
    //
    // Force the packed weight to stay in bf16.
    // TTNNWeightDtypeConversion applies the model's global weights_dtype (e.g.
    // bfp_bf4 in trace_enabled configs) to every LinearOp whose weight traces to
    // a constant parameter. For the Kronecker-packed weight this destroys the
    // block-diagonal structure and drops PCC to ~0.96.
    auto linearOutTy = RankedTensorType::get({N, 1, packedSpatial, packedOC}, bf16);
    auto *linearOp   =
        b.create<ttir::LinearOp>(loc, linearOutTy, aFlat, wPacked,
                                  bPacked ? bPacked : Value(),
                                  /*transpose_a=*/false,
                                  /*transpose_b=*/false)
            .getOperation();
    linearOp->setAttr("ttcore.weight_dtype", b.getStringAttr("bf16"));
    Value linearOut = linearOp->getResult(0);

    // ── 3e. Output unpack ─────────────────────────────────────────────────
    // reshape [N,1,H/K*W,OC*K] → [N,H/K,W,OC*K]  unflatten spatial
    auto oUnflatTy = RankedTensorType::get({N, packedH, W, packedOC}, bf16);
    Value oUnflat  =
        b.create<ttir::ReshapeOp>(
             loc, oUnflatTy, linearOut,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)packedH,
                                (int32_t)W, (int32_t)packedOC}))
            .getResult();

    // permute({0,3,1,2}): [N,H/K,W,OC*K] → [N,OC*K,H/K,W]
    auto oPerTy = RankedTensorType::get({N, packedOC, packedH, W}, bf16);
    Value oPer  =
        b.create<ttir::PermuteOp>(
             loc, oPerTy, oUnflat,
             b.getDenseI64ArrayAttr({0, 3, 1, 2}))
            .getResult();

    // reshape [N,OC*K,H/K,W] → [N,OC,H,W]  free view
    auto finalTy = RankedTensorType::get({N, OC, H, W}, bf16);
    Value finalOut =
        b.create<ttir::ReshapeOp>(
             loc, finalTy, oPer,
             b.getI32ArrayAttr({(int32_t)N, (int32_t)OC, (int32_t)H, (int32_t)W}))
            .getResult();

    // ── 4. Replace old chain ──────────────────────────────────────────────
    postPerm.getResult().replaceAllUsesWith(finalOut);
    postPerm.erase();
    convOp.erase();
    prePerm.erase();
  }
};

} // namespace

} // namespace mlir::tt::ttir
