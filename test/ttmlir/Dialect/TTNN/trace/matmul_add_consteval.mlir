// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=true enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @matmul_with_bias_const_eval_0
  // CHECK: "ttnn.matmul"

  // CHECK-LABEL: func.func @matmul_with_bias_trace_0
  // CHECK: "ttnn.add"(%arg1, %arg0)

  // CHECK-LABEL: func.func @matmul_with_bias(
  func.func @matmul_with_bias(%arg0: tensor<64x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK: %[[LOAD_CACHED_RESULT:.+]] = ttcore.load_cached(@matmul_with_bias_const_eval_0, [%arg0, %arg1])
    // CHECK: %[[TRACE_RESULT:.+]] = ttnn.trace(%[[GET_DEVICE]], 0, false, @matmul_with_bias_trace_0, [%arg2, %[[LOAD_CACHED_RESULT]]])
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
