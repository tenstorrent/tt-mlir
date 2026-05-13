// RUN: ttmlir-opt --split-input-file --d2m-fe-pipeline="enable-elementwise-fusion=true enable-eltwise-reduction-fusion=true override-device-shape=1,1" %s | FileCheck %s

// Eltwise (exp) followed by a single-dim reduction (sum over dim 1) should be
// fused into one compute d2m.generic. We expect tile_exp and tile_reduce_sum
// to live in the same d2m.generic region (no d2m.generic boundary between
// them).
module {
  // CHECK-LABEL: func.func @exp_then_sum(
  // CHECK: d2m.tile_exp
  // CHECK-NOT: d2m.generic
  // CHECK: d2m.tile_reduce_sum
  func.func @exp_then_sum(%arg0: tensor<64x64xbf16>) -> tensor<64x1xbf16> {
    %0 = "ttir.exp"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = "ttir.sum"(%0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<64x64xbf16>) -> tensor<64x1xbf16>
    return %1 : tensor<64x1xbf16>
  }
}

// -----

// Multi-dim reductions are not eligible for fusion (only single reduction dim
// is supported in V1). We expect tile_exp and tile_reduce_sum to land in
// separate d2m.generic ops.
module {
  // CHECK-LABEL: func.func @exp_then_sum_multidim(
  // CHECK: d2m.tile_exp
  // CHECK: d2m.generic
  // CHECK: d2m.tile_reduce_sum
  func.func @exp_then_sum_multidim(%arg0: tensor<64x64xbf16>) -> tensor<1x1xbf16> {
    %0 = "ttir.exp"(%arg0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %1 = "ttir.sum"(%0) <{dim_arg = [0 : i32, 1 : i32], keep_dim = true}> : (tensor<64x64xbf16>) -> tensor<1x1xbf16>
    return %1 : tensor<1x1xbf16>
  }
}
