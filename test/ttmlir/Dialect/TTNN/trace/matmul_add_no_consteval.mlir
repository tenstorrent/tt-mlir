// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @matmul_with_bias_trace
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func @matmul_with_bias(
  func.func @matmul_with_bias(%arg0: tensor<64x32xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = ttnn.trace(%[[GET_DEVICE]], 0, false, @matmul_with_bias_trace_0, [%arg0, %arg1, %arg2])
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.matmul"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
