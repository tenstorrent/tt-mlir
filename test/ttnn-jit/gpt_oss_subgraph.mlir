#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<4x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <4x8>, memref<1x12x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
module {
  func.func @gptoss_subgraph(%arg0: tensor<4x32x3072xbf16, #ttnn_layout>, %arg1: tensor<4x32x3072xbf16, #ttnn_layout>) -> tensor<4x32x3072xbf16, #ttnn_layout1> {
    %0 = "ttir.clamp_scalar"(%arg0) <{max = 7.000000e+00 : f32, min = -7.000000e+00 : f32}> : (tensor<4x32x3072xbf16, #ttnn_layout>) -> tensor<4x32x3072xbf16>
    %1 = "ttir.clamp_scalar"(%arg1) <{max = 7.000000e+00 : f32, min = -3.40282347E+38 : f32}> : (tensor<4x32x3072xbf16, #ttnn_layout>) -> tensor<4x32x3072xbf16>
    %2 = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 4, 32, 3072>}> : () -> tensor<4x32x3072xbf16>
    %3 = "ttir.add"(%0, %2) : (tensor<4x32x3072xbf16>, tensor<4x32x3072xbf16>) -> tensor<4x32x3072xbf16>
    %4 = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 4, 32, 3072>}> : () -> tensor<4x32x3072xbf16>
    %5 = "ttir.multiply"(%1, %4) : (tensor<4x32x3072xbf16>, tensor<4x32x3072xbf16>) -> tensor<4x32x3072xbf16>
    %6 = "ttir.sigmoid"(%5) : (tensor<4x32x3072xbf16>) -> tensor<4x32x3072xbf16>
    %7 = "ttir.multiply"(%1, %6) : (tensor<4x32x3072xbf16>, tensor<4x32x3072xbf16>) -> tensor<4x32x3072xbf16>
    %8 = "ttir.multiply"(%3, %7) : (tensor<4x32x3072xbf16>, tensor<4x32x3072xbf16>) -> tensor<4x32x3072xbf16>
    %9 = ttir.empty() : tensor<4x32x3072xbf16, #ttnn_layout1>
    %10 = ttir.to_layout %8, %9 : tensor<4x32x3072xbf16> into tensor<4x32x3072xbf16, #ttnn_layout1> -> tensor<4x32x3072xbf16, #ttnn_layout1>
    return %10 : tensor<4x32x3072xbf16, #ttnn_layout1>
  }
}
