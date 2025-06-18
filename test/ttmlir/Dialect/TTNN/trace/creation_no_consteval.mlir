// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @creation_ops_trace
  // CHECK: "ttnn.add"


  // CHECK-LABEL: func.func @creation_ops(
  func.func @creation_ops() -> tensor<4x4xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = ttnn.trace(%[[GET_DEVICE]], 0, false, @creation_ops_trace_0, [])
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
