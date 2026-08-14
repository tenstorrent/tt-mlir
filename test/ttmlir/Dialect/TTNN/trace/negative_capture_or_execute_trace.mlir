// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for ttnn.capture_or_execute_trace op verifier.
//
// Capturing is split across two functions: `allocate_slots_callee` allocates the
// persistent trace input/output slots, and `capture_callee` takes those slots as
// arguments and captures against them. The trace function itself stores its
// results into the output slots and returns nothing. Each test below exercises a
// single error path in CaptureOrExecuteTraceOp::verify().

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 1: Slot allocation callee does not reference a function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op 'nonexistent_allocate' does not reference a function
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_allocate_callee_missing(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @nonexistent_allocate, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 2: Slot allocation function returns the wrong number of slots ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Slot allocation function 'allocate_slots_fn' must return 2 slots (1 input slots + 1 output slots), but returns 1
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_allocate_slot_count(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 3: Slot allocation output slot type does not match the op result ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Slot allocation function output slot 0 type mismatch
func.func private @trace_fn(%arg0: tensor<32x32xf32, #layout_f32>, %arg1: tensor<32x32xf32, #layout_f32>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xf32, #layout_f32>, tensor<32x32xf32, #layout_f32>) -> tensor<32x32xf32, #layout_f32>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xf32, #layout_f32>, tensor<32x32xf32, #layout_f32>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xf32, #layout_f32>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #layout_f32>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xf32, #layout_f32>
  return %1, %2 : tensor<32x32xf32, #layout_f32>, tensor<32x32xf32, #layout_f32>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xf32, #layout_f32>, %arg2: tensor<32x32xf32, #layout_f32>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xf32, #layout_f32>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xf32, #layout_f32>, tensor<32x32xf32, #layout_f32>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_allocate_output_slot_type(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 4: Capture callee does not reference a function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op 'nonexistent_capture' does not reference a function
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_callee_missing(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @nonexistent_capture, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 5: Capture function takes the wrong number of arguments ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Capture function 'capture_fn' must take 3 arguments (1 host-staged inputs + 2 slots + 0 semaphores), but has 2
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_arg_count(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 6: Capture function slot argument type does not match the allocated slot ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Capture function slot argument 0 type mismatch
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xf32, #layout_f32>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xf32, #layout_f32>) -> ()
  call @trace_fn(%arg2, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_slot_type(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 7: Capture function returns more than a trace id ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Capture function 'capture_fn' must return exactly one trace_id value, but returns 2
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2, %arg2 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_result_count(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 8: Capture function does not return a trace id tensor ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Capture function 'capture_fn' must return a trace_id tensor (scalar ui32 with TraceIdAttr encoding)
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %arg2 : tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_result_not_trace_id(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 9: No trace function (call op) found in capture function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op No trace function found in capture function
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_no_trace_func(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 10: Trace function returns results instead of storing into the output slots ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Trace function 'trace_fn' must return no results; it stores its outputs into the output slot arguments
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %0 : tensor<32x32xbf16, #layout>
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  %1 = call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_trace_func_returns_results(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 11: Device config mismatch between trace op and callee get_device ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Device configuration of get_device op in callee must match device configuration of trace op
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %d = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_device_mismatch(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 12: Execute callee does not reference a function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op 'nonexistent_execute' does not reference a function
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func @test_execute_callee_missing(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @nonexistent_execute, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 13: Execute function has the wrong number of arguments ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Execute function 'execute_fn' must take exactly one trace_id argument, but has 2
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>, %arg1: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_execute_arg_count(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 14: Execute function argument is not a trace_id tensor ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Execute function 'execute_fn' argument must be a trace_id tensor (scalar ui32 with TraceIdAttr encoding)
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.add"(%arg0, %arg0) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.copy"(%0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}
func.func private @allocate_slots_fn() -> (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  %2 = "ttnn.empty"(%0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>, %arg2: tensor<32x32xbf16, #layout>) -> tensor<ui32, #ttnn.trace_id> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.write_tensor"(%arg0, %arg1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%arg1, %arg2) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> ()
  %2 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %2) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %2 : tensor<ui32, #ttnn.trace_id>
}
func.func private @execute_fn(%arg0: tensor<32x32xbf16, #layout>) {
  return
}
func.func @test_execute_arg_type(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{allocate_slots_callee = @allocate_slots_fn, capture_callee = @capture_fn, execute_callee = @execute_fn, operandSegmentSizes = array<i32: 1, 1, 0>}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}
