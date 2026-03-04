// REQUIRES: stablehlo
// RUN: ttmlir-opt --reoutline-composite -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that reoutlining preserves the original composite operand order
// when captures are defined in a different block position order.

// CHECK-LABEL: func.func @main
module @CompositeArgOrder attributes {} {
  func.func @main(%arg0: tensor<32x16xf32>, %arg1: tensor<32x16xf32>) -> tensor<32x16xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<32x16xf32>) -> tensor<1x32x16xf32>
    // CHECK: [[V1:%[^ ]+]] = stablehlo.reshape{{.*}}(tensor<1x32x16xf32>) -> tensor<32x16xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x32x16xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.reshape %arg1 : (tensor<32x16xf32>) -> tensor<2x16x16xf32>
    // CHECK: [[V3:%[^ ]+]] = stablehlo.reshape{{.*}}(tensor<2x16x16xf32>) -> tensor<32x16xf32>
    %3 = stablehlo.reshape %2 : (tensor<2x16x16xf32>) -> tensor<32x16xf32>
    // CHECK: stablehlo.composite "test.subtract" [[V3]], [[V1]]
    %4 = stablehlo.subtract %3, %1 {reoutline.group = "composite_test.subtract.impl", reoutline.seed, reoutline.orig_name = "test.subtract", reoutline.comp_attrs = {}, reoutline.arg_operand_indices = array<i64: 0, 1>} : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
}
// CHECK: func.func private @outlined_composite_test.subtract.impl
// CHECK: stablehlo.subtract %arg0, %arg1
