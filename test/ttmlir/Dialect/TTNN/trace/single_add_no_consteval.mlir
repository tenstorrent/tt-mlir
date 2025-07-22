// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @trace_0_single_add
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func @run_and_capture_trace_0_single_add
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func @execute_trace_0_single_add
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @single_add(
  func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg0, %arg1) <{capture_callee = @run_and_capture_trace_0_single_add, execute_callee = @execute_trace_0_single_add}>
    // CHECK-NOT: "ttnn.add"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
