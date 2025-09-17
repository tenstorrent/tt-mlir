
#l1 = #ttnn.buffer_type<l1>

#ttnn_layout = #ttnn.ttnn_layout<
  (d0, d1) -> (d0, d1),
  <1x1, (d0, d1) -> (0, d0, d1)>,
  memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>
  >

#metal_layout = #ttcore.metal_layout<
  logical_shape = 32x32,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1
  >

module {
  func.func @test_no_bufferization(%arg0 : tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xf32, #ttnn_layout> {
    %0 = ttir.empty() : tensor<32x32xf32, #ttnn_layout>
    %1 = ttir.ttnn_metal_layout_cast %0 : tensor<32x32xf32, #ttnn_layout> -> tensor<32x32xf32, #metal_layout>
    %2 = ttir.ttnn_metal_layout_cast %1 : tensor<32x32xf32, #metal_layout> -> tensor<32x32xf32, #ttnn_layout>
    return %2 : tensor<32x32xf32, #ttnn_layout>
  }
}
