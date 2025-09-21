// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func private @trace_0_creation_ops
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func private @run_and_capture_trace_0_creation_ops
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_creation_ops
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @creation_ops(
  func.func @creation_ops() -> tensor<4x4xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: "ttnn.zeros"
    // CHECK-NEXT: "ttnn.ones"
    // CHECK-NOT: "ttnn.add"
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %1, %2) <{capture_callee = @run_and_capture_trace_0_creation_ops, execute_callee = @execute_trace_0_creation_ops}>
    // CHECK: return %[[TRACE_RESULT]]
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 4>, dtype = i64}> : () -> tensor<4x4xbf16>
    %1 = "ttir.ones"() <{shape = array<i32: 4, 4>, dtype = bf16}> : () -> tensor<4x4xbf16>

    %2 = ttir.empty() : tensor<4x4xbf16>
    %3 = "ttir.add"(%0, %1, %2) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>

    return %3 : tensor<4x4xbf16>
  }
}
