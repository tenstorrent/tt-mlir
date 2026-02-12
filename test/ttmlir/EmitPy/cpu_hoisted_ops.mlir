// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Verify CPU-hoisted functions are generated with golden_function calls

// Binary arithmetic
// CHECK-LABEL: def cpu_hoisted_ttir_add_{{.*}}
// CHECK: ttnn.add.golden_function

// Binary division
// CHECK-LABEL: def cpu_hoisted_ttir_div_{{.*}}
// CHECK: ttnn.divide.golden_function

// Binary comparison
// CHECK-LABEL: def cpu_hoisted_ttir_eq_{{.*}}
// CHECK: ttnn.eq.golden_function

// Binary logical
// CHECK-LABEL: def cpu_hoisted_ttir_logical_{{.*}}
// CHECK: ttnn.logical_and.golden_function

// Binary min/max
// CHECK-LABEL: def cpu_hoisted_ttir_maximum_{{.*}}
// CHECK: ttnn.maximum.golden_function

// Matrix multiplication
// CHECK-LABEL: def cpu_hoisted_ttir_matmul_{{.*}}
// CHECK: ttnn.matmul.golden_function

// Unary activation
// CHECK-LABEL: def cpu_hoisted_ttir_relu_{{.*}}
// CHECK: ttnn.relu.golden_function

// Unary trigonometric
// CHECK-LABEL: def cpu_hoisted_ttir_sin_{{.*}}
// CHECK: ttnn.sin.golden_function

// Unary exponential/logarithmic
// CHECK-LABEL: def cpu_hoisted_ttir_exp_{{.*}}
// CHECK: ttnn.exp.golden_function

// CHECK-LABEL: def cpu_hoisted_ttir_log_{{.*}}
// CHECK: ttnn.log.golden_function

// Unary rounding
// CHECK-LABEL: def cpu_hoisted_ttir_ceil_{{.*}}
// CHECK: ttnn.ceil.golden_function

// Unary sign/abs
// CHECK-LABEL: def cpu_hoisted_ttir_abs_{{.*}}
// CHECK: ttnn.abs.golden_function

// Unary reciprocal
// CHECK-LABEL: def cpu_hoisted_ttir_reciprocal_{{.*}}
// CHECK: ttnn.reciprocal.golden_function

// Unary sqrt
// CHECK-LABEL: def cpu_hoisted_ttir_sqrt_{{.*}}
// CHECK: ttnn.sqrt.golden_function

// Pooling
// CHECK-LABEL: def cpu_hoisted_ttir_max_{{.*}}
// CHECK: ttnn.max_pool2d.golden_function

// Reduction
// CHECK-LABEL: def cpu_hoisted_ttir_sum_{{.*}}
// CHECK: ttnn.sum.golden_function

// Data manipulation - reshape
// CHECK-LABEL: def cpu_hoisted_ttir_reshape_{{.*}}
// CHECK: ttnn.reshape.golden_function

// Data manipulation - concat
// CHECK-LABEL: def cpu_hoisted_ttir_concat_{{.*}}
// CHECK: ttnn.concat.golden_function

// Data manipulation - permute
// CHECK-LABEL: def cpu_hoisted_ttir_permute_{{.*}}
// CHECK: ttnn.permute.golden_function

// Softmax
// CHECK-LABEL: def cpu_hoisted_ttir_softmax_{{.*}}
// CHECK: ttnn.softmax.golden_function

// ============================================================================
// Validation functions
// ============================================================================

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

// CHECK-LABEL: def eq_validation
// CHECK: cpu_hoisted_ttir_eq_{{.*}}
// CHECK: ttnn.eq(
func.func @eq_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xbf16> {
  %cpu_result = "ttir.eq"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xbf16>
  %device_result = "ttir.eq"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xbf16>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %diff : tensor<32x32xbf16>
}

// CHECK-LABEL: def logical_and_validation
// CHECK: cpu_hoisted_ttir_logical_{{.*}}
// CHECK: ttnn.logical_and(
func.func @logical_and_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.logical_and"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.logical_and"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

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

// CHECK-LABEL: def relu_validation
// CHECK: cpu_hoisted_ttir_relu_{{.*}}
// CHECK: ttnn.relu(
// CHECK: ttnn.subtract(
func.func @relu_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.relu"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.relu"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def sin_validation
// CHECK: cpu_hoisted_ttir_sin_{{.*}}
// CHECK: ttnn.sin(
// CHECK: ttnn.subtract(
func.func @sin_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.sin"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.sin"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def exp_validation
// CHECK: cpu_hoisted_ttir_exp_{{.*}}
// CHECK: ttnn.exp(
// CHECK: ttnn.subtract(
func.func @exp_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.exp"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.exp"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def log_validation
// CHECK: cpu_hoisted_ttir_log_{{.*}}
// CHECK: ttnn.log(
// CHECK: ttnn.subtract(
func.func @log_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.log"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.log"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def ceil_validation
// CHECK: cpu_hoisted_ttir_ceil_{{.*}}
// CHECK: ttnn.ceil(
// CHECK: ttnn.subtract(
func.func @ceil_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.ceil"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.ceil"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def abs_validation
// CHECK: cpu_hoisted_ttir_abs_{{.*}}
// CHECK: ttnn.abs(
// CHECK: ttnn.subtract(
func.func @abs_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.abs"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.abs"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def reciprocal_validation
// CHECK: cpu_hoisted_ttir_reciprocal_{{.*}}
// CHECK: ttnn.reciprocal(
// CHECK: ttnn.subtract(
func.func @reciprocal_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.reciprocal"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.reciprocal"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

// CHECK-LABEL: def sqrt_validation
// CHECK: cpu_hoisted_ttir_sqrt_{{.*}}
// CHECK: ttnn.sqrt(
// CHECK: ttnn.subtract(
func.func @sqrt_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.sqrt"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.sqrt"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}

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
