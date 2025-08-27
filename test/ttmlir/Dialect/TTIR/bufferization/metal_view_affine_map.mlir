// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// Use a non-identity view affine map to verify it is propagated to ViewLayoutAttr.
// Full memref rank here is 4: 2 grid dims + 2 shard dims.
#layout_with_view = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
  undef, l1,
  view_affine_map = (d0, d1, d2, d3) -> (d1, d0, d2, d3)
>

func.func @propagate_view_map() -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view> {
  %input = ttir.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>
  %storage = ttir.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>

  // After bufferization, the result should be a memref with a ttcore.view layout
  // that uses the same affine map provided in the MetalLayoutAttr.
  // Just check the result type arrow part contains our view map.
  // CHECK: -> memref<{{.*}}, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>, #l1>
  %stream = "ttir.stream_layout"(%input, %storage) : (tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>, tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>) -> tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>

  return %stream : tensor<1x1x2x4x!ttcore.tile<32x32, f32>, #layout_with_view>
}
