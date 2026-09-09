// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false enable-trace=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The traced computation stores its results into the output slot arguments
  // and returns nothing.
  // CHECK-LABEL: func.func private @trace_0_single_add
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>
  // CHECK-SAME: %arg1: tensor<32x32xbf16, #ttnn_layout>
  // CHECK-SAME: %arg2: tensor<32x32xbf16, #ttnn_layout>)
  // CHECK-NOT: ->
  // CHECK: %[[SUM:.+]] = "ttnn.add"(%arg1, %arg0)
  // CHECK: "ttnn.copy"(%[[SUM]], %arg2)
  // CHECK: return{{$}}

  // The allocation function is the only place the persistent slots are
  // allocated; device-resident arguments pass through as their own slots.
  // CHECK-LABEL: func.func private @allocate_slots_trace_0_single_add
  // CHECK-NOT: "ttnn.write_tensor"
  // CHECK-NOT: "ttnn.begin_trace_capture"
  // CHECK: %[[IN_SLOT:.+]] = "ttnn.empty"
  // CHECK: %[[OUT_SLOT:.+]] = "ttnn.empty"
  // CHECK: return %arg0, %[[IN_SLOT]], %[[OUT_SLOT]]

  // The capture function takes every slot as an argument and allocates
  // nothing itself.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_single_add
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout1>
  // CHECK-SAME: %arg1: tensor<32x32xbf16, #ttnn_layout>
  // CHECK-SAME: %arg2: tensor<32x32xbf16, #ttnn_layout>
  // CHECK-SAME: %arg3: tensor<32x32xbf16, #ttnn_layout>)
  // CHECK-SAME: -> tensor<ui32, #ttnn.trace_id>
  // CHECK-NOT: "ttnn.empty"
  // CHECK-NOT: "ttnn.copy"
  // CHECK: "ttnn.write_tensor"(%arg0, %arg2)
  // CHECK: call @trace_0_single_add(%arg1, %arg2, %arg3)
  // CHECK: %[[TRACE_ID:.+]] = "ttnn.begin_trace_capture"
  // CHECK: call @trace_0_single_add(%arg1, %arg2, %arg3)
  // CHECK: "ttnn.end_trace_capture"
  // CHECK: "ttnn.execute_trace"
  // CHECK: return %[[TRACE_ID]]

  // CHECK-LABEL: func.func private @execute_trace_0_single_add
  // CHECK: "ttnn.execute_trace"

  // CHECK-LABEL: func.func @single_add(
  func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // CHECK: %[[GET_DEVICE:.+]] = "ttnn.get_device"()
    // CHECK-NEXT: %[[TRACE_RESULT:.+]] = "ttnn.capture_or_execute_trace"(%[[GET_DEVICE]], %arg1, %arg0) <{allocate_slots_callee = @allocate_slots_trace_0_single_add, capture_callee = @run_and_capture_trace_0_single_add, execute_callee = @execute_trace_0_single_add, operandSegmentSizes = array<i32: 1, 2, 0>}>
    // CHECK-NOT: "ttnn.add"
    // CHECK: return %[[TRACE_RESULT]]
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
