// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func private @trace_0_linear_with_add
  // CHECK: "ttnn.linear"
  // CHECK: "ttnn.add"(%0, %arg3)

  // CHECK-LABEL: func.func private @run_and_capture_trace_0_linear_with_add
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_linear_with_add
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @linear_with_add(
  func.func @linear_with_add(%arg0: tensor<64x32xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<64x64xbf16>, %arg3: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg0, %arg1, %arg2, %arg3) <{capture_callee = @run_and_capture_trace_0_linear_with_add, execute_callee = @execute_trace_0_linear_with_add}>
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.linear"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<64x64xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<64x32xbf16>, tensor<32x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = ttir.empty() : tensor<64x64xbf16>
    %3 = "ttir.add"(%1, %arg3, %2) : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
