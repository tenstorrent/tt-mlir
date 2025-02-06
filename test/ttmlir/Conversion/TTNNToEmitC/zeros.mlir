// RUN: ttmlir-opt --convert-ttnn-to-emitc %s | FileCheck %s

#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1344 + d1 * 56 + d2, d3), <1x1>, memref<17472x42xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2048x128xf32, #system_memory>>

module  {
  func.func @zeros_4d_irregular_shapes() -> tensor<13x24x56x42xbf16, #ttnn_layout> {
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "ttnn::zeros"{{.*}} -> !emitc.opaque<"ttnn::Tensor">
    %0 = "ttnn.zeros"() <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, shape = #ttnn.shape<13x24x56x42>}> : () -> tensor<13x24x56x42xbf16, #ttnn_layout>
    return %0 : tensor<13x24x56x42xbf16, #ttnn_layout>
  }

  func.func @zeros_f32() -> tensor<32x64x128xf32, #ttnn_layout1> {
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "ttnn::zeros"{{.*}} -> !emitc.opaque<"ttnn::Tensor">
    %0 = "ttnn.zeros"() <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, shape = #ttnn.shape<32x64x128>}> : () -> tensor<32x64x128xf32, #ttnn_layout1>
    return %0 : tensor<32x64x128xf32, #ttnn_layout1>
  }
}
