#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @model() -> tensor<32x32xf32, #ttnn_layout> {
    %0 = "ttnn.full"() <{shape = #ttnn.shape<32x32>, fill_value = 5.0 : f32}> : () -> tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
}
