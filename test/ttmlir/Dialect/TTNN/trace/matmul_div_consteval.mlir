// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=true enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @matmul_with_div_const_eval_0
  // CHECK: "ttnn.matmul"

  // CHECK-LABEL: func.func private @trace_0_matmul_with_div
  // CHECK: "ttnn.divide"(%arg1, %arg0)

  // CHECK-LABEL: func.func private @run_and_capture_trace_0_matmul_with_div
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_matmul_with_div
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @matmul_with_div(
  func.func @matmul_with_div(%arg0: tensor<64x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<64x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK: %[[LOAD_CACHED_RESULT:.+]] = ttcore.load_cached(@matmul_with_div_const_eval_0, [%arg0, %arg1])
    // CHECK: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg2, %[[LOAD_CACHED_RESULT]]) <{capture_callee = @run_and_capture_trace_0_matmul_with_div, execute_callee = @execute_trace_0_matmul_with_div}>
    // CHECK-NOT: "ttnn.divide"
    // CHECK-NOT: "ttnn.matmul"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.div"(%1, %arg2, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
