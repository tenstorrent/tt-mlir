// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  func.func @add_integer_broadcast(%arg0: tensor<1x1x1x128xsi32,#ttnn_layout1>, %arg1: tensor<1x1x128x128xsi32,#ttnn_layout>) -> tensor<1x1x128x128xbf16,#ttnn_layout2> {
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<1x1x1x128xbf16
    // CHECK-SAME: tensor<1x1x128x128xbf16
    // CHECK-SAME: tensor<1x1x128x128xbf16
    %0 = "ttnn.add"(%arg0, %arg1) <{output_dtype = #ttcore.supportedDataTypes<si32>}> : (tensor<1x1x1x128xsi32,#ttnn_layout1>, tensor<1x1x128x128xsi32,#ttnn_layout>) -> tensor<1x1x128x128xsi32,#ttnn_layout>
    %1 = "ttnn.to_layout"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x128x128xsi32, #ttnn_layout>) -> tensor<1x1x128x128xbf16, #ttnn_layout2>
    return %1 : tensor<1x1x128x128xbf16,#ttnn_layout2>
  }
}
