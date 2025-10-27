// RUN: ttmlir-opt --ttcore-register-device --d2m-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// Use a non-identity view affine map to verify it is propagated to ViewLayoutAttr.
#layout_with_view = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
  undef, l1, sharded,
  index_map = (d0, d1, d2, d3) -> (d1, d0, d2, d3)
>

#layout_without_view = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
  undef, l1
>

func.func @propagate_view_map() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view> {
  %input = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_without_view>

  // CHECK: -> memref<{{.*}}, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #l1>
  %view = "d2m.view_layout"(%input) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_without_view>) -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>

  return %view : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>
}
