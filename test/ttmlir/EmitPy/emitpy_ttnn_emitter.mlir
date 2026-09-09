// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that EmitPyTTNNEmitter's argument emission logic.
//
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-opt %t.mlir --verify-each -o /dev/null
// RUN: ttmlir-translate --mlir-to-python %t.mlir | FileCheck %s

// Optional attribute conversion (MatmulProgramConfig, activation).
func.func @matmul(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x96xbf16>) -> tensor<64x96xbf16>
  return %0 : tensor<64x96xbf16>
}

// Absent optional operand (kv_input_tensor) and absent optional attribute
// (num_kv_heads).
func.func @split_qkv(%arg0: tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>) {
  %0, %1, %2 = "ttir.split_query_key_value_and_split_heads"(%arg0) <{num_heads = 16 : ui32, transpose_key = false}> : (tensor<2x32x3072xf32>) -> (tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>)
  return %0, %1, %2 : tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>, tensor<2x16x32x64xf32>
}

// Positional tensor operands plus converted dtype/memory_config kwargs.
func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// Non-MLIR attribute (approximate StringRef) and default vs explicit value.
func.func @gelu_bw_default(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %0 = "ttir.gelu_bw"(%arg0, %arg1) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

func.func @gelu_bw(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %0 = "ttir.gelu_bw"(%arg0, %arg1) <{approximate = "tanh"}> : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

// Explicit emit(std::nullopt, "dtype") for a keyword with no MLIR attribute.
func.func @remainder(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = "ttir.remainder"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

// CHECK-LABEL: def matmul(
// CHECK: program_config=None
// CHECK: activation=None

// CHECK-LABEL: def split_qkv(
// CHECK: ttnn.transformer.split_query_key_value_and_split_heads({{.*}}, None, num_heads=16, num_kv_heads=None, transpose_key=False

// CHECK-LABEL: def add(
// CHECK: ttnn.add({{.*}}, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.MemoryConfig

// CHECK-LABEL: def gelu_bw_default(
// CHECK: approximate="none"

// CHECK-LABEL: def gelu_bw(
// CHECK: approximate="tanh"

// CHECK-LABEL: def remainder(
// CHECK: dtype=None
