// RUN: ttmlir-opt --ttcore-register-device --convert-ttir-to-ttnn -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Verify that ttir.to_layout lowers to the aggregate ttnn.to_tensor_spec op
// (and NOT ttnn.to_layout, which is the narrow tilize/untilize op that
// ttnn-decompose-layouts emits later and which survives to the backend).

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#device = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @to_layout_lowers_to_to_tensor_spec(%arg0: tensor<64x128xf32, #host>) -> tensor<64x128xbf16, #device> {
    // CHECK-LABEL: func.func @to_layout_lowers_to_to_tensor_spec
    // CHECK: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-NOT: "ttnn.to_layout"
    %0 = ttir.empty() : tensor<64x128xbf16, #device>
    %1 = ttir.to_layout %arg0, %0 : tensor<64x128xf32, #host> into tensor<64x128xbf16, #device> -> tensor<64x128xbf16, #device>
    return %1 : tensor<64x128xbf16, #device>
  }
}
