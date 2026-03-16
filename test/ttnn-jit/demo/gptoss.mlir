#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<2x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>, exactGrid = true>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <2x8>, memref<1x12x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
module {
  func.func @gptoss_gateup_subgraph(%arg0: tensor<2x32x3072xbf16, #ttnn_layout>, %arg1: tensor<2x32x3072xbf16, #ttnn_layout>, %arg2: tensor<2x32x3072xbf16, #ttnn_layout>, %arg3: tensor<2x32x3072xbf16, #ttnn_layout>, %arg4: tensor<2x32x3072xbf16, #ttnn_layout>, %arg5: tensor<2x32x3072xbf16, #ttnn_layout>, %arg6: tensor<2x32x3072xbf16, #ttnn_layout>, %arg7: tensor<2x32x3072xbf16, #ttnn_layout>) -> tensor<2x32x3072xbf16, #ttnn_layout1> {
    %0 = "ttir.clamp_tensor"(%arg0, %arg2, %arg3) : (tensor<2x32x3072xbf16, #ttnn_layout>, tensor<2x32x3072xbf16, #ttnn_layout>, tensor<2x32x3072xbf16, #ttnn_layout>) -> tensor<2x32x3072xbf16>
    %1 = "ttir.add"(%0, %arg6) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16, #ttnn_layout>) -> tensor<2x32x3072xbf16>
    %2 = "ttir.clamp_tensor"(%arg1, %arg4, %arg5) : (tensor<2x32x3072xbf16, #ttnn_layout>, tensor<2x32x3072xbf16, #ttnn_layout>, tensor<2x32x3072xbf16, #ttnn_layout>) -> tensor<2x32x3072xbf16>
    %3 = "ttir.multiply"(%2, %arg7) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16, #ttnn_layout>) -> tensor<2x32x3072xbf16>
    %4 = "ttir.sigmoid"(%3) : (tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    %5 = "ttir.multiply"(%2, %4) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    %6 = "ttir.multiply"(%1, %5) : (tensor<2x32x3072xbf16>, tensor<2x32x3072xbf16>) -> tensor<2x32x3072xbf16>
    %7 = ttir.empty() : tensor<2x32x3072xbf16, #ttnn_layout1>
    %8 = ttir.to_layout %6, %7 : tensor<2x32x3072xbf16> into tensor<2x32x3072xbf16, #ttnn_layout1> -> tensor<2x32x3072xbf16, #ttnn_layout1>
    return %8 : tensor<2x32x3072xbf16, #ttnn_layout1>
  }
}
