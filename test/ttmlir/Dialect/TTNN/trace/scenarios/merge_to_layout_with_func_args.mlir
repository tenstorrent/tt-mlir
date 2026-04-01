// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=DIRECT

// Verify that mergeToLayoutOpsWithFuncArgs merges ToLayoutOps into function
// argument types for direct arg -> ToLayoutOp patterns. After merging, no
// ToLayoutOp should remain between function arguments and
// capture_or_execute_trace.

module {
  // DIRECT-LABEL: func.func @merge_to_layout_direct(
  func.func @merge_to_layout_direct(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // Verify that there are no to_layout ops between args and trace op.
    // The mergeToLayoutOpsWithFuncArgs function should merge the ToLayoutOps
    // (inserted by the trace hoist transform to move inputs to system memory)
    // into the function argument types.
    // DIRECT: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // DIRECT-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg0, %arg1)
    // DIRECT-NOT: "ttnn.to_layout"
    // DIRECT: return %[[TRACE_RESULT]]
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
