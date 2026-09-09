// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttnn.copy op verifier.
// ttnn.copy exists so a trace can write its results into a caller-provided
// persistent slot. It deliberately rejects anything tt-metal's copy would refuse
// at runtime, so a mismatch is a compile error rather than a TT_FATAL raised in
// the middle of a trace capture window.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 1: Source tensor on host ---
// CHECK: error: 'ttnn.copy' op Source tensor must be on device memory
func.func @test_copy_src_on_host(%arg0: tensor<32x32xbf16, #host_layout>, %arg1: tensor<32x32xbf16, #layout>) {
  "ttnn.copy"(%arg0, %arg1) : (tensor<32x32xbf16, #host_layout>, tensor<32x32xbf16, #layout>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#host_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>

// --- Test 2: Destination tensor on host ---
// CHECK: error: 'ttnn.copy' op Destination tensor must be on device memory
func.func @test_copy_dst_on_host(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xbf16, #host_layout>) {
  "ttnn.copy"(%arg0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xbf16, #host_layout>) -> ()
  return
}

// -----

#dram = #ttnn.buffer_type<dram>
#layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#layout_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// --- Test 3: Mismatched types ---
// CHECK: error: 'ttnn.copy' op Source and destination tensor types must match
func.func @test_copy_type_mismatch(%arg0: tensor<32x32xbf16, #layout>, %arg1: tensor<32x32xf32, #layout_f32>) {
  "ttnn.copy"(%arg0, %arg1) : (tensor<32x32xbf16, #layout>, tensor<32x32xf32, #layout_f32>) -> ()
  return
}
