#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, i32>, #dram>, <interleaved>>
module {
  func.func @model(%arg0: tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xi32, #ttnn_layout1> {
    %0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<i32>}> : (tensor<32x32xf32, #ttnn_layout>) -> tensor<32x32xi32, #ttnn_layout1>
    return %0 : tensor<32x32xi32, #ttnn_layout1>
  }
}
