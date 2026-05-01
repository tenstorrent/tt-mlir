// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// ============================================================================
// Validation functions
// ============================================================================

// Verify CPU-hoisted functions call ttir_cpu.<op> directly and are placed
// right before their first caller.

// Binary arithmetic
// CHECK-LABEL: def cpu_hoisted_ttir_add_{{.*}}
// CHECK: ttir_cpu.add(
// CHECK-LABEL: def add_validation
// CHECK: cpu_hoisted_ttir_add_{{.*}}
// CHECK: ttnn.add(
// CHECK: ttnn.subtract(
func.func @add_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Binary division
// CHECK-LABEL: def cpu_hoisted_ttir_div_{{.*}}
// CHECK: ttir_cpu.div(
// CHECK-LABEL: def div_validation
// CHECK: cpu_hoisted_ttir_div_{{.*}}
// CHECK: ttnn.divide(
// CHECK: ttnn.subtract(
func.func @div_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.div"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.div"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Binary comparison
// CHECK-LABEL: def cpu_hoisted_ttir_eq_{{.*}}
// CHECK: ttir_cpu.eq(
// CHECK-LABEL: def eq_validation
// CHECK: cpu_hoisted_ttir_eq_{{.*}}
// CHECK: ttnn.eq(
func.func @eq_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xbf16> {
  %cpu_result = "ttir.eq"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xbf16>
  %device_result = "ttir.eq"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xbf16>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %diff : tensor<32x32xbf16>
}

// Binary logical
// CHECK-LABEL: def cpu_hoisted_ttir_logical_{{.*}}
// CHECK: ttir_cpu.logical_and(
// CHECK-LABEL: def logical_and_validation
// CHECK: cpu_hoisted_ttir_logical_{{.*}}
// CHECK: ttnn.logical_and(
func.func @logical_and_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.logical_and"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.logical_and"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Binary min/max
// CHECK-LABEL: def cpu_hoisted_ttir_maximum_{{.*}}
// CHECK: ttir_cpu.maximum(
// CHECK-LABEL: def maximum_validation
// CHECK: cpu_hoisted_ttir_maximum_{{.*}}
// CHECK: ttnn.maximum(
// CHECK: ttnn.subtract(
func.func @maximum_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.maximum"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.maximum"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Matrix multiplication
// CHECK-LABEL: def cpu_hoisted_ttir_matmul_{{.*}}
// CHECK: ttir_cpu.matmul(
// CHECK-LABEL: def matmul_validation
// CHECK: cpu_hoisted_ttir_matmul_{{.*}}
// CHECK: ttnn.matmul(
// CHECK: ttnn.subtract(
func.func @matmul_validation(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.matmul"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.matmul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary activation
// CHECK-LABEL: def cpu_hoisted_ttir_relu_{{.*}}
// CHECK: ttir_cpu.relu(
// CHECK-LABEL: def relu_validation
// CHECK: cpu_hoisted_ttir_relu_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
func.func @relu_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.relu"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.relu"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary trigonometric
// CHECK-LABEL: def cpu_hoisted_ttir_sin_{{.*}}
// CHECK: ttir_cpu.sin(
// CHECK-LABEL: def sin_validation
// CHECK: cpu_hoisted_ttir_sin_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SIN)]
func.func @sin_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.sin"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.sin"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def cpu_hoisted_ttir_asinh_{{.*}}
// CHECK: ttir_cpu.asinh(
// CHECK-LABEL: def asinh_validation
// CHECK: cpu_hoisted_ttir_asinh_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ASINH)]
func.func @asinh_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.asinh"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.asinh"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary exponential
// CHECK-LABEL: def cpu_hoisted_ttir_exp_{{.*}}
// CHECK: ttir_cpu.exp(
// CHECK-LABEL: def exp_validation
// CHECK: cpu_hoisted_ttir_exp_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP)]
func.func @exp_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.exp"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.exp"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary logarithmic
// CHECK-LABEL: def cpu_hoisted_ttir_log_{{.*}}
// CHECK: ttir_cpu.log(
// CHECK-LABEL: def log_validation
// CHECK: cpu_hoisted_ttir_log_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)]
func.func @log_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.log"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.log"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary rounding
// CHECK-LABEL: def cpu_hoisted_ttir_ceil_{{.*}}
// CHECK: ttir_cpu.ceil(
// CHECK-LABEL: def ceil_validation
// CHECK: cpu_hoisted_ttir_ceil_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.CEIL)]
func.func @ceil_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.ceil"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.ceil"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary sign/abs
// CHECK-LABEL: def cpu_hoisted_ttir_abs_{{.*}}
// CHECK: ttir_cpu.abs(
// CHECK-LABEL: def abs_validation
// CHECK: cpu_hoisted_ttir_abs_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
func.func @abs_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.abs"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.abs"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary reciprocal
// CHECK-LABEL: def cpu_hoisted_ttir_reciprocal_{{.*}}
// CHECK: ttir_cpu.reciprocal(
// CHECK-LABEL: def reciprocal_validation
// CHECK: cpu_hoisted_ttir_reciprocal_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RECIP)]
func.func @reciprocal_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.reciprocal"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.reciprocal"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Unary sqrt
// CHECK-LABEL: def cpu_hoisted_ttir_sqrt_{{.*}}
// CHECK: ttir_cpu.sqrt(
// CHECK-LABEL: def sqrt_validation
// CHECK: cpu_hoisted_ttir_sqrt_{{.*}}
// CHECK: ttnn.subtract(
// CHECK-SAME: input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT)]
func.func @sqrt_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.sqrt"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.sqrt"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Pooling
// CHECK-LABEL: def cpu_hoisted_ttir_max_{{.*}}
// CHECK: ttir_cpu.max_pool2d(
// CHECK-LABEL: def max_pool2d_validation
// CHECK: cpu_hoisted_ttir_max_{{.*}}
// CHECK: ttnn.max_pool2d(
// CHECK: ttnn.subtract(
func.func @max_pool2d_validation(%arg0: tensor<1x32x32x32xf32>) -> tensor<1x16x16x32xf32> {
  %cpu_result = "ttir.max_pool2d"(%arg0) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> {ttir.should_hoist} : (tensor<1x32x32x32xf32>) -> tensor<1x16x16x32xf32>
  %device_result = "ttir.max_pool2d"(%arg0) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x32x32x32xf32>) -> tensor<1x16x16x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<1x16x16x32xf32>, tensor<1x16x16x32xf32>) -> tensor<1x16x16x32xf32>
  return %diff : tensor<1x16x16x32xf32>
}

// Reduction
// CHECK-LABEL: def cpu_hoisted_ttir_sum_{{.*}}
// CHECK: ttir_cpu.sum(
// CHECK-LABEL: def sum_validation
// CHECK: cpu_hoisted_ttir_sum_{{.*}}
// CHECK: ttnn.sum(
// CHECK: ttnn.subtract(
func.func @sum_validation(%arg0: tensor<32x32xf32>) -> tensor<1x32xf32> {
  %cpu_result = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<1x32xf32>
  %device_result = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<32x32xf32>) -> tensor<1x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  return %diff : tensor<1x32xf32>
}

// Data manipulation - reshape
// CHECK-LABEL: def cpu_hoisted_ttir_reshape_{{.*}}
// CHECK: ttir_cpu.reshape(
// CHECK-LABEL: def reshape_validation
// CHECK: cpu_hoisted_ttir_reshape_{{.*}}
// CHECK: ttnn.reshape(
// CHECK: ttnn.subtract(
func.func @reshape_validation(%arg0: tensor<32x32xf32>) -> tensor<1024xf32> {
  %cpu_result = "ttir.reshape"(%arg0) <{shape = [1024 : i32]}> {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<1024xf32>
  %device_result = "ttir.reshape"(%arg0) <{shape = [1024 : i32]}> : (tensor<32x32xf32>) -> tensor<1024xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
  return %diff : tensor<1024xf32>
}

// Data manipulation - concat
// CHECK-LABEL: def cpu_hoisted_ttir_concat_{{.*}}
// CHECK: ttir_cpu.concat(
// CHECK-LABEL: def concat_validation
// CHECK: cpu_hoisted_ttir_concat_{{.*}}
// CHECK: ttnn.concat(
// CHECK: ttnn.subtract(
func.func @concat_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<64x32xf32> {
  %cpu_result = "ttir.concat"(%arg0, %arg1) <{dim = 0 : si32}> {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
  %device_result = "ttir.concat"(%arg0, %arg1) <{dim = 0 : si32}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<64x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %diff : tensor<64x32xf32>
}

// Data manipulation - permute
// CHECK-LABEL: def cpu_hoisted_ttir_permute_{{.*}}
// CHECK: ttir_cpu.permute(
// CHECK-LABEL: def permute_validation
// CHECK: cpu_hoisted_ttir_permute_{{.*}}
// CHECK: ttnn.permute(
// CHECK: ttnn.subtract(
func.func @permute_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// Softmax
// CHECK-LABEL: def cpu_hoisted_ttir_softmax_{{.*}}
// CHECK: ttir_cpu.softmax(
// CHECK-LABEL: def softmax_validation
// CHECK: cpu_hoisted_ttir_softmax_{{.*}}
// CHECK: ttnn.softmax(
// CHECK: ttnn.subtract(
func.func @softmax_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.softmax"(%arg0) <{dimension = 1 : si32}> : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// LayerNorm
// CHECK-LABEL: def cpu_hoisted_ttir_layer_{{.*}}
// CHECK: ttir_cpu.layer_norm(
// CHECK-LABEL: def layer_norm_validation
// CHECK: cpu_hoisted_ttir_layer_{{.*}}
// CHECK: ttnn.layer_norm(
// CHECK: ttnn.subtract(
func.func @layer_norm_validation(%arg0: tensor<32x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128xf32>) -> tensor<32x128xf32> {
  %cpu_result = "ttir.layer_norm"(%arg0, %arg1, %arg2) <{normalized_shape = array<i64: 128>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> {ttir.should_hoist} : (tensor<32x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
  %device_result = "ttir.layer_norm"(%arg0, %arg1, %arg2) <{normalized_shape = array<i64: 128>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<32x128xf32>, tensor<128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
  return %diff : tensor<32x128xf32>
}
