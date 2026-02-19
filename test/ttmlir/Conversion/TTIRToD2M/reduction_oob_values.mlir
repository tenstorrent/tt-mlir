// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that reduction ops set the correct identity OOB fill values on their
// operand/output layouts, while eltwise ops remain undef.

// Sum operands/outputs should use 'zero' (identity for addition):
// CHECK-DAG: #[[SUM_INPUT:.*]] = #ttcore.metal_layout<logical_shape = 128x96,{{.*}}zero{{.*}}>
// CHECK-DAG: #[[SUM_OUTPUT:.*]] = #ttcore.metal_layout<logical_shape = 1x96,{{.*}}zero{{.*}}>

// Max operands/outputs should use 'neginf' (identity for max):
// CHECK-DAG: #[[MAX_INPUT:.*]] = #ttcore.metal_layout<logical_shape = 128x96,{{.*}}neginf{{.*}}>
// CHECK-DAG: #[[MAX_OUTPUT:.*]] = #ttcore.metal_layout<logical_shape = 128x1,{{.*}}neginf{{.*}}>

// Eltwise operands should still use 'undef':
// CHECK-DAG: #[[ELT_LAYOUT:.*]] = #ttcore.metal_layout<logical_shape = 128x96,{{.*}}undef{{.*}}>

module {
  // CHECK-LABEL: func @sum_reduce_R
  func.func @sum_reduce_R(%arg: tensor<128x96xf32>) -> tensor<1x96xf32> {
    // CHECK: d2m.empty{{.*}}#[[SUM_INPUT]]
    // CHECK: d2m.tile_reduce_sum
    %0 = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<1x96xf32>
    return %0 : tensor<1x96xf32>
  }

  // CHECK-LABEL: func @max_reduce_C
  func.func @max_reduce_C(%arg: tensor<128x96xf32>) -> tensor<128x1xf32> {
    // CHECK: d2m.empty{{.*}}#[[MAX_INPUT]]
    // CHECK: d2m.tile_reduce_max
    %0 = "ttir.max"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<128x1xf32>
    return %0 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @eltwise_add
  func.func @eltwise_add(%a: tensor<128x96xf32>, %b: tensor<128x96xf32>) -> tensor<128x96xf32> {
    // CHECK: d2m.empty{{.*}}#[[ELT_LAYOUT]]
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%a, %b) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    return %0 : tensor<128x96xf32>
  }
}
