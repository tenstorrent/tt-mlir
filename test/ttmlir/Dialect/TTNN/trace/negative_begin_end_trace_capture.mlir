// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for ttnn.begin_trace_capture and ttnn.end_trace_capture op
// verifiers. Both ops share the same verification logic (CQ ID + trace ID
// tensor checks).

// ===== begin_trace_capture tests =====

// --- Test 1: begin_trace_capture with invalid CQ ID ---
// CHECK: error: 'ttnn.begin_trace_capture' op Invalid CQ ID 5
func.func @test_begin_invalid_cq_id() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 5 : ui32}> : (!ttnn.device) -> tensor<ui32, #ttnn.trace_id>
  "ttnn.end_trace_capture"(%0, %1) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return
}

// -----

// --- Test 2: end_trace_capture with invalid CQ ID ---
// CHECK: error: 'ttnn.end_trace_capture' op Invalid CQ ID 3
func.func @test_end_invalid_cq_id(%arg0: tensor<ui32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 3 : ui32}> : (!ttnn.device, tensor<ui32, #ttnn.trace_id>) -> ()
  return
}

// -----

// --- Test 3: begin_trace_capture trace ID not scalar ---
// CHECK: error: Trace ID must be a scalar
func.func @test_begin_trace_id_not_scalar() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<4xui32, #ttnn.trace_id>
  return
}

// -----

// --- Test 4: end_trace_capture trace ID not scalar ---
// CHECK: error: Trace ID must be a scalar
func.func @test_end_trace_id_not_scalar(%arg0: tensor<4xui32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<4xui32, #ttnn.trace_id>) -> ()
  return
}

// -----

// --- Test 5: begin_trace_capture trace ID not unsigned ---
// CHECK: error: Trace ID must be unsigned
func.func @test_begin_trace_id_not_unsigned() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<si32, #ttnn.trace_id>
  return
}

// -----

// --- Test 6: end_trace_capture trace ID not unsigned ---
// CHECK: error: Trace ID must be unsigned
func.func @test_end_trace_id_not_unsigned(%arg0: tensor<si32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<si32, #ttnn.trace_id>) -> ()
  return
}

// -----

// --- Test 7: begin_trace_capture trace ID wrong width ---
// CHECK: error: Trace ID must be 32-bit
func.func @test_begin_trace_id_wrong_width() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui64, #ttnn.trace_id>
  return
}

// -----

// --- Test 8: end_trace_capture trace ID wrong width ---
// CHECK: error: Trace ID must be 32-bit
func.func @test_end_trace_id_wrong_width(%arg0: tensor<ui64, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui64, #ttnn.trace_id>) -> ()
  return
}

// -----

// --- Test 9: begin_trace_capture trace ID missing encoding ---
// CHECK: error: Trace ID must have the TraceIdAttr encoding
func.func @test_begin_trace_id_no_encoding() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<ui32>
  return
}

// -----

// --- Test 10: end_trace_capture trace ID missing encoding ---
// CHECK: error: Trace ID must have the TraceIdAttr encoding
func.func @test_end_trace_id_no_encoding(%arg0: tensor<ui32>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<ui32>) -> ()
  return
}

// -----

// --- Test 11: begin_trace_capture trace ID not integer ---
// CHECK: error: Trace ID must be an integer
func.func @test_begin_trace_id_not_integer() {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.begin_trace_capture"(%0) <{cq_id = 0 : ui32}> : (!ttnn.device) -> tensor<f32, #ttnn.trace_id>
  return
}

// -----

// --- Test 12: end_trace_capture trace ID not integer ---
// CHECK: error: Trace ID must be an integer
func.func @test_end_trace_id_not_integer(%arg0: tensor<f32, #ttnn.trace_id>) {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  "ttnn.end_trace_capture"(%0, %arg0) <{cq_id = 0 : ui32}> : (!ttnn.device, tensor<f32, #ttnn.trace_id>) -> ()
  return
}
