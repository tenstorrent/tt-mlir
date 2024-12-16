// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttnn-modify-signatures-for-dylib %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  // CHECK: func.func @add(%arg0: tuple<[[TENSOR_A:.*>]], [[TENSOR_B:.*>]]>) -> tensor<32x32xbf16, #ttnn_layout> {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK-NEXT: %0 = tt.get_tuple_element %arg0[0] : (tuple<[[TENSOR_A]], [[TENSOR_B]]>) -> [[TENSOR_A]]
    // CHECK-NEXT: %1 = tt.get_tuple_element %arg0[1] : (tuple<[[TENSOR_A]], [[TENSOR_B]]>) -> [[TENSOR_B]]
    %0 = tensor.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
}

}
