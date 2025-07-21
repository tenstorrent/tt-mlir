// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @trace_0_creation_ops
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func @run_and_capture_trace_0_creation_ops
  // CHECK-NOT: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func @execute_trace_0_creation_ops
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @creation_ops(
  func.func @creation_ops() -> tensor<4x4xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]]) <{capture_callee = @run_and_capture_trace_0_creation_ops, execute_callee = @execute_trace_0_creation_ops}>
    // CHECK-NOT: "ttnn.zeros"
    // CHECK-NOT: "ttnn.ones"
    // CHECK-NOT: "ttnn.arange"
    // CHECK-NOT: "ttnn.add"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>
    %1 = "ttir.ones"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>

    %2 = ttir.empty() : tensor<4x4xbf16>
    %3 = "ttir.add"(%0, %1, %2) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    %4 = ttir.empty() : tensor<4x4xbf16>
    %5 = "ttir.arange"() {start = 0 : si64, end = 4 : si64, step = 1 : si64, arange_dimension = 0 : i64} : () -> tensor<4x4xbf16>
    %6 = "ttir.add"(%3, %5, %4) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    return %6 : tensor<4x4xbf16>
  }
}
