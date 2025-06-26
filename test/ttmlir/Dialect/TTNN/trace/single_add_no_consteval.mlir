// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @single_add_trace
  // CHECK: "ttnn.add"

  // CHECK-LABEL: func.func @single_add(
  func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = ttnn.trace(%[[GET_DEVICE]], 0, false, @single_add_trace_0, [%arg0, %arg1])
    // CHECK-NOT: "ttnn.add"
    // CHECK: return %[[TRACE_RESULT]]
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
