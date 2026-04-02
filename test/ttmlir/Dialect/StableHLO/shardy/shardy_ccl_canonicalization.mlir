// REQUIRES: stablehlo
// RUN: ttmlir-opt --shardy-ccl-canonicalization %s | FileCheck %s

sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>

// all_reduce + multi-dim all_slice should be fused into
// reduce_scatter (on the matching axis) + residual all_slice (on the rest).
// CHECK-LABEL: func.func @reduce_scatter_fusion_multi_dim
// CHECK-NOT: sdy.all_reduce
// CHECK: sdy.reduce_scatter [{}, {}, {}, {"_axis_1"}]
// CHECK: sdy.all_slice [{}, {}, {"_axis_0"}, {}]
func.func @reduce_scatter_fusion_multi_dim(
    %arg0: tensor<4x1x32x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) -> tensor<4x1x32x2880xbf16> {
  %0 = sdy.all_reduce {"_axis_1"} %arg0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<4x1x32x2880xbf16>
  %1 = sdy.all_slice [{}, {}, {"_axis_0"}, {"_axis_1"}] %0 out_sharding=<@mesh, [{}, {}, {"_axis_0"}, {"_axis_1"}]> : tensor<4x1x32x2880xbf16>
  return %1 : tensor<4x1x32x2880xbf16>
}

// all_reduce + single-dim all_slice should be fused into a single
// reduce_scatter with no residual all_slice.
// CHECK-LABEL: func.func @reduce_scatter_fusion_single_dim
// CHECK-NOT: sdy.all_reduce
// CHECK-NOT: sdy.all_slice
// CHECK: sdy.reduce_scatter [{}, {}, {}, {"_axis_1"}]
// CHECK: return
func.func @reduce_scatter_fusion_single_dim(
    %arg0: tensor<4x1x32x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) -> tensor<4x1x32x2880xbf16> {
  %0 = sdy.all_reduce {"_axis_1"} %arg0 out_sharding=<@mesh, [{}, {}, {}, {}]> : tensor<4x1x32x2880xbf16>
  %1 = sdy.all_slice [{}, {}, {}, {"_axis_1"}] %0 out_sharding=<@mesh, [{}, {}, {}, {"_axis_1"}]> : tensor<4x1x32x2880xbf16>
  return %1 : tensor<4x1x32x2880xbf16>
}

// all_slice(all_gather(x)) with the same axes should cancel out to identity.
// The dead all_gather remains (no Pure trait) but the all_slice is eliminated.
// CHECK-LABEL: func.func @all_slice_cancels_all_gather
// CHECK: sdy.all_gather
// CHECK-NOT: sdy.all_slice
// CHECK: return %arg0
func.func @all_slice_cancels_all_gather(
    %arg0: tensor<16x1x720xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"_axis_1"}]>}) -> tensor<16x1x720xbf16> {
  %0 = sdy.all_gather [{}, {}, {"_axis_1"}] %arg0 out_sharding=<@mesh, [{}, {}, {}]> : tensor<16x1x720xbf16>
  %1 = sdy.all_slice [{}, {}, {"_axis_1"}] %0 out_sharding=<@mesh, [{}, {}, {"_axis_1"}]> : tensor<16x1x720xbf16>
  return %1 : tensor<16x1x720xbf16>
}
