#dram = #ttnn.buffer_type<dram>
#ttnn_layout_vals  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_idx   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xsi32, #dram>, <interleaved>>
#ttnn_layout_k     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<32xui32, #dram>, <interleaved>>
#ttnn_layout_p     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<32xbf16, #dram>, <interleaved>>
#ttnn_layout_out   = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<32xsi32, #dram>, <interleaved>>
module {
  func.func @model(
      %arg0: tensor<32x128xbf16, #ttnn_layout_vals>,
      %arg1: tensor<32x128xsi32, #ttnn_layout_idx>,
      %arg2: tensor<32xui32, #ttnn_layout_k>,
      %arg3: tensor<32xbf16, #ttnn_layout_p>,
      %arg4: tensor<32xbf16, #ttnn_layout_p>
  ) -> tensor<32xsi32, #ttnn_layout_out> {
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<32x128xbf16, #ttnn_layout_vals>,
           tensor<32x128xsi32, #ttnn_layout_idx>,
           tensor<32xui32, #ttnn_layout_k>,
           tensor<32xbf16, #ttnn_layout_p>,
           tensor<32xbf16, #ttnn_layout_p>) -> tensor<32xsi32, #ttnn_layout_out>
    return %0 : tensor<32xsi32, #ttnn_layout_out>
  }
}
