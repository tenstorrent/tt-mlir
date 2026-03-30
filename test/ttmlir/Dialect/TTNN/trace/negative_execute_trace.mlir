// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for ttnn.execute_trace op verifier.

#dram = #ttnn.buffer_type<dram>

// --- Test 1: Invalid CQ ID ---
// CHECK: error: 'ttnn.execute_trace' op Invalid CQ ID 5
func.func @test_invalid_cq_id(%arg0: tensor<ui32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 5 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// --- Test 2: Trace ID is not a scalar tensor (rank != 0) ---
// CHECK: error: Trace ID must be a scalar
func.func @test_trace_id_not_scalar(%arg0: tensor<4xui32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 0 : ui32}> : (!ttnn.device, tensor<4xui32, #ttnn.trace_id>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>

// --- Test 3: Trace ID is not unsigned ---
// CHECK: error: Trace ID must be unsigned
func.func @test_trace_id_not_unsigned(%arg0: tensor<si32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 0 : ui32}> : (!ttnn.device, tensor<si32, #ttnn.trace_id>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>

// --- Test 4: Trace ID is not 32-bit ---
// CHECK: error: Trace ID must be 32-bit
func.func @test_trace_id_wrong_width(%arg0: tensor<ui64, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui64, #ttnn.trace_id>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>

// --- Test 5: Trace ID missing TraceIdAttr encoding ---
// CHECK: error: Trace ID must have the TraceIdAttr encoding
func.func @test_trace_id_no_encoding(%arg0: tensor<ui32>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>

// --- Test 6: Trace ID is not an integer type ---
// CHECK: error: Trace ID must be an integer
func.func @test_trace_id_not_integer(%arg0: tensor<f32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.execute_trace"(%0, %arg0) <{blocking = false, cq_id = 0 : ui32}> : (!ttnn.device, tensor<f32, #ttnn.trace_id>) -> ()
  return
}
