// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttnn-trace-hoist-transform -o %t %s
// RUN: FileCheck %s --input-file=%t

// Global semaphores are pass-through handles: they are trace function arguments
// so device kernels can use them, but they have no persistent device slot. The
// capture function must therefore take them AFTER all of the slots, because the
// runtime forwards the slots as the flatbuffer program's tensor inputs and the
// semaphores separately as program semaphore_inputs.
//
// ttnn.create_global_semaphore carries TTCore_CreationOpTrait, so it is not
// hoisted and the semaphore becomes a trace input. This lets us pin the ordering
// without needing a CCL op or an OpModel build.

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // The trace body's arguments are [tensor inputs][output slots][semaphores].
  // CHECK-LABEL: func.func private @trace_0_main
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout>, %arg1: tensor<32x32xbf16, #ttnn_layout>, %arg2: tensor<32x32xbf16, #ttnn_layout>, %arg3: !ttnn.global_semaphore)
  // CHECK: "ttnn.reset_global_semaphore"
  // CHECK: "ttnn.copy"

  // A semaphore is not a slot, so the allocation function neither takes nor
  // returns one.
  // CHECK-LABEL: func.func private @allocate_slots_trace_0_main
  // CHECK-SAME: () -> (tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>, tensor<32x32xbf16, #ttnn_layout>)
  // CHECK-NOT: !ttnn.global_semaphore

  // The capture function's arguments are [host-staged inputs][input slots]
  // [output slots][semaphores].
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
  // CHECK-SAME: (%arg0: tensor<32x32xbf16, #ttnn_layout1>, %arg1: tensor<32x32xbf16, #ttnn_layout1>
  // CHECK-SAME: %arg2: tensor<32x32xbf16, #ttnn_layout>, %arg3: tensor<32x32xbf16, #ttnn_layout>
  // CHECK-SAME: %arg4: tensor<32x32xbf16, #ttnn_layout>, %arg5: !ttnn.global_semaphore)
  // CHECK-SAME: -> tensor<ui32, #ttnn.trace_id>
  // CHECK-NOT: "ttnn.empty"
  // CHECK-NOT: "ttnn.copy"
  // CHECK: "ttnn.write_tensor"(%arg0, %arg2)
  // CHECK: "ttnn.write_tensor"(%arg1, %arg3)
  // CHECK: call @trace_0_main(%arg2, %arg3, %arg4, %arg5)
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: call @trace_0_main(%arg2, %arg3, %arg4, %arg5)
  // CHECK: "ttnn.end_trace_capture"
  // CHECK: "ttnn.execute_trace"

  func.func @main(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %sem = "ttnn.create_global_semaphore"(%0) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore
    // The trace op forwards the semaphore in its own operand group (1 device,
    // 2 tensors, 1 semaphore).
    // CHECK: "ttnn.capture_or_execute_trace"
    // CHECK-SAME: allocate_slots_callee = @allocate_slots_trace_0_main
    // CHECK-SAME: capture_callee = @run_and_capture_trace_0_main
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 2, 1>
    %1 = "ttnn.add"(%arg0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    "ttnn.reset_global_semaphore"(%sem) <{value = 0 : ui32}> : (!ttnn.global_semaphore) -> ()
    %2 = "ttnn.multiply"(%1, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
    return %2 : tensor<32x32xbf16, #layout>
  }
}
