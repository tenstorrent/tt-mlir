// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=true enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @matmul_with_multiply_const_eval_0
  // CHECK: "ttnn.matmul"

  // CHECK-LABEL: func.func private @trace_0_matmul_with_multiply
  // CHECK: %[[TILED_ARG:.*]] = "ttnn.to_layout"(%arg0)
  // CHECK: "ttnn.multiply"(%arg1, %[[TILED_ARG]])

  // CHECK-LABEL: func.func private @run_and_capture_trace_0_matmul_with_multiply
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_matmul_with_multiply
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @matmul_with_multiply(
  func.func @matmul_with_multiply(%arg0: tensor<64x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK: %[[LOAD_CACHED_RESULT:.+]] = ttcore.load_cached(@matmul_with_multiply_const_eval_0, [%arg0, %arg1])
    // CHECK: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg2, %[[LOAD_CACHED_RESULT]]) <{capture_callee = @run_and_capture_trace_0_matmul_with_multiply, execute_callee = @execute_trace_0_matmul_with_multiply}>
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.matmul"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.multiply"(%1, %arg2, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
