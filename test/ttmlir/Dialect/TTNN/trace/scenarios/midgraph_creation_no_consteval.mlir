// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A creation op materialized in the *middle* of the graph (e.g. a mask/scalar
// emitted by SDPA decomposition) is not hoistable. It only depends on the
// device, so the trace-hoist pass sinks it above the trace region and treats it
// as a regular trace input, instead of failing with "Non-hoistable op found in
// the middle of hoistable ops".

module {
  // The mid-graph creation op is sunk out of the trace and passed in as an arg.
  // CHECK-LABEL: func.func private @trace_0_midgraph
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.add"
  // CHECK-NOT: "ttnn.ones"

  // CHECK-LABEL: func.func @midgraph(
  func.func @midgraph(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK: "ttnn.get_device"
    // The mid-graph creation op is hoisted above the trace op and staged to host.
    // CHECK: %[[ONES:.+]] = "ttnn.ones"
    // CHECK: %[[ONES_HOST:.+]] = "ttnn.from_device"(%[[ONES]])
    // CHECK: "ttnn.capture_or_execute_trace"({{.*}}%[[ONES_HOST]])
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.ones"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}
