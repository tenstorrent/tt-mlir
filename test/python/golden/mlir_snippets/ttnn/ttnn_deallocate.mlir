#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @model(%arg0: tensor<32x32xf32, #ttnn_layout>) -> () {
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<32x32xf32, #ttnn_layout>) -> ()
    return
  }
}
