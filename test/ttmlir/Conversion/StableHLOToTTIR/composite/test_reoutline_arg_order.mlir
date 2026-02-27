// REQUIRES: stablehlo
// RUN: ttmlir-opt --reoutline-composite %s | FileCheck %s

// Test that reoutlining preserves the original composite operand order
// when captures are defined in a different block position order.
//
// Setup: two external values are defined in block position order [%1, %3].
// The subtract op uses them as: stablehlo.subtract %3, %1
// with arg_operand_indices = [0, 1], meaning:
//   operand 0 (%3) was original composite arg 0
//   operand 1 (%1) was original composite arg 1
//
// Without fix: captures sorted by block pos = [%1, %3]
//   -> composite "test.subtract" %1, %3 (WRONG - swapped!)
// With fix: captures sorted by arg index = [%3, %1]
//   -> composite "test.subtract" %3, %1 (CORRECT)

module @CompositeArgOrder attributes {} {
  func.func @main(%arg0: tensor<32x16xf32>, %arg1: tensor<32x16xf32>) -> tensor<32x16xf32> {
    // First capture chain (defined first in block, from %arg0):
    %0 = stablehlo.reshape %arg0 : (tensor<32x16xf32>) -> tensor<1x32x16xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x32x16xf32>) -> tensor<32x16xf32>
    // Second capture chain (defined second in block, from %arg1):
    %2 = stablehlo.reshape %arg1 : (tensor<32x16xf32>) -> tensor<2x16x16xf32>
    %3 = stablehlo.reshape %2 : (tensor<2x16x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.subtract %3, %1 {reoutline.group = "composite_test.subtract.impl", reoutline.seed, reoutline.orig_name = "test.subtract", reoutline.comp_attrs = {}, reoutline.arg_operand_indices = array<i64: 0, 1>} : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
}

// Verify composite has operands in the original order (%3 first, %1 second).
// CHECK: [[V1:%[^ ]+]] = stablehlo.reshape{{.*}}: (tensor<1x32x16xf32>) -> tensor<32x16xf32>
// CHECK: [[V3:%[^ ]+]] = stablehlo.reshape{{.*}}: (tensor<2x16x16xf32>) -> tensor<32x16xf32>
// CHECK: stablehlo.composite "test.subtract" [[V3]], [[V1]]
// Verify the callee has subtract in canonical arg order (arg0 - arg1).
// CHECK: func.func private @outlined_composite_test.subtract.impl
// CHECK: stablehlo.subtract %arg0, %arg1
