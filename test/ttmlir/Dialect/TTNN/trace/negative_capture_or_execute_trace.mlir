// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for ttnn.capture_or_execute_trace op verifier.
// Each test case exercises a single error path in CaptureOrExecuteTraceOp::verify().

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 1: Capture callee does not reference a function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op 'nonexistent_capture' does not reference a function
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_capture_callee_missing(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @nonexistent_capture, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 2: Input argument count mismatch with capture function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Number of input arguments (2) does not match capture function 'capture_fn' input count (1)
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  return %arg0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_input_count_mismatch(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// --- Test 3: Input argument type mismatch with capture function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Input argument 1 type mismatch
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  return %arg0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xf32, #layout_f32>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xf32, #layout_f32>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %arg1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xf32, #layout_f32>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_input_type_mismatch(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  // Op passes bf16 but capture_fn expects f32 for arg1.
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 4: No trace function (call op) found in capture function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op No trace function found in capture function
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %1) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %1, %arg1 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_no_trace_function(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0, %arg1) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#layout_64x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// --- Test 5: Output result count mismatch with trace function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Number of output results (2) does not match trace function 'trace_fn' output count (1)
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  return %arg0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_output_count_mismatch(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<32x32xbf16, #layout>, tensor<64x64xbf16, #layout_64x64>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  // trace_fn returns 1 result, but op declares 2 outputs.
  %1, %2 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> (tensor<32x32xbf16, #layout>, tensor<64x64xbf16, #layout_64x64>)
  return %1, %2 : tensor<32x32xbf16, #layout>, tensor<64x64xbf16, #layout_64x64>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// --- Test 6: Output result type mismatch with trace function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Output result 0 type mismatch
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  return %arg0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_output_type_mismatch(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xf32, #layout_f32> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  // trace_fn returns bf16 but op declares f32 output.
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xf32, #layout_f32>
  return %1 : tensor<32x32xf32, #layout_f32>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 7: Trace function input argument not on device ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op All input arguments of trace function must be on device
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  call @trace_fn(%arg0) : (tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%arg0) : (tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %arg0, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_trace_arg_not_on_device(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 8: Device config mismatch between trace op and callee get_device ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Device configuration of get_device op in callee must match device configuration of trace op
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  // Trace function uses a different mesh shape (1x2 vs 1x1).
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
  %1 = "ttnn.add"(%arg0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>) {
  return
}
func.func @test_device_config_mismatch(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 9: Execute callee does not reference a function ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op 'nonexistent_execute' does not reference a function
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.add"(%arg0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func @test_execute_callee_missing(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @nonexistent_execute}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 10: Execute function has wrong number of arguments ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Execute function 'execute_fn' must take exactly one trace_id argument, but has 2 arguments
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.add"(%arg0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<ui32, #ttnn.trace_id>, %arg1: tensor<32x32xbf16, #layout>) {
  return
}
func.func @test_execute_wrong_arg_count(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 11: Execute function argument is not a trace_id tensor ---
// CHECK: error: 'ttnn.capture_or_execute_trace' op Execute function 'execute_fn' argument must be a trace_id tensor (scalar ui32 with TraceIdAttr encoding)
func.func private @trace_fn(%arg0: tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.add"(%arg0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  return %0 : tensor<32x32xbf16, #layout>
}
func.func private @capture_fn(%arg0: tensor<32x32xbf16, #host_layout>) -> (tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<32x32>}> : (!ttnn.device) -> tensor<32x32xbf16, #layout>
  "ttnn.write_tensor"(%arg0, %1) <{blocking = false, cq_id = 0 : ui32}> : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  %3 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  %4 = call @trace_fn(%1) : (tensor<32x32xbf16, #layout>) -> tensor<32x32xbf16, #layout>
  "ttnn.end_trace_capture"(%0, %3) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return %3, %1, %4 : tensor<ui32, #ttnn.trace_id>, tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #layout>
}
func.func private @execute_fn(%arg0: tensor<32x32xbf16, #layout>) {
  return
}
func.func @test_execute_wrong_arg_type(%arg0: tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.capture_or_execute_trace"(%0, %arg0) <{capture_callee = @capture_fn, execute_callee = @execute_fn}> : (!ttnn.device, tensor<32x32xbf16, #host_layout>) -> tensor<32x32xbf16, #layout>
  return %1 : tensor<32x32xbf16, #layout>
}
