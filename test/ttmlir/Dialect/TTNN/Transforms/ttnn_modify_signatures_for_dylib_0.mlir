// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttnn-tuplify-tensors -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // CHECK: func.func @add(%arg0: tuple<[[TENSOR_A:.*>]], [[TENSOR_B:.*>]]>) -> tuple<tensor<32x32xbf16, #ttnn_layout>> {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK-NEXT: %0 = ttcore.get_tuple_element %arg0[0] : (tuple<[[TENSOR_A]], [[TENSOR_B]]>) -> [[TENSOR_A]]
    // CHECK-NEXT: %1 = ttcore.get_tuple_element %arg0[1] : (tuple<[[TENSOR_A]], [[TENSOR_B]]>) -> [[TENSOR_B]]
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  // CHECK: func.func @multiple_returns(%arg0: tuple<[[TENSOR_A:.*>]], [[TENSOR_B:.*>]], [[TENSOR_C:.*>]]>) -> tuple<tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>> {
  func.func @multiple_returns(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    // CHECK-NEXT: %0 = ttcore.get_tuple_element %arg0[0] : (tuple<[[TENSOR_A]], [[TENSOR_B]], [[TENSOR_C]]>) -> [[TENSOR_A]]
    // CHECK-NEXT: %1 = ttcore.get_tuple_element %arg0[1] : (tuple<[[TENSOR_A]], [[TENSOR_B]], [[TENSOR_C]]>) -> [[TENSOR_B]]
    // CHECK-NEXT: %2 = ttcore.get_tuple_element %arg0[2] : (tuple<[[TENSOR_A]], [[TENSOR_B]], [[TENSOR_C]]>) -> [[TENSOR_C]]
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1, %3 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
