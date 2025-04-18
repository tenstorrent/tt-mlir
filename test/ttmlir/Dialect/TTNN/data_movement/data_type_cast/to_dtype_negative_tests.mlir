// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s
// Negative tests for to_dtype operation
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x32x32xf32, #system_memory>>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x32x32xbf16, #system_memory>>
// Verify that verification fails when dtype attribute is empty.
module {
  func.func @to_dtype_dtype_attribute_missmatch(%arg0: tensor<2x32x32xf32, #ttnn_layout_host_rm_f32>) -> tensor<2x32x32xbf16, #ttnn_layout_host_rm_bf16> {
    %0 = "ttnn.to_dtype"(%arg0) <{dtype = #tt.supportedDataTypes<f32>}> : (tensor<2x32x32xf32, #ttnn_layout_host_rm_f32>) -> tensor<2x32x32xbf16, #ttnn_layout_host_rm_bf16>
    // CHECK: error: 'ttnn.to_dtype' op Output tensor data type bf16 must match the data type of dtype attribute f32.
    return %0 : tensor<2x32x32xbf16, #ttnn_layout_host_rm_bf16>
  }
}
