// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize -o %t.grid %s

// Verify that reduction ops emit explicit masks with the correct identity OOB
// fill values when padding exists, while eltwise ops do not materialize masks.

module {
  // CHECK-LABEL: func @sum_reduce_R
  func.func @sum_reduce_R(%arg: tensor<128x96xf32>) -> tensor<1x96xf32> {
    // CHECK: d2m.tile_reduce_sum
    %0 = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<1x96xf32>
    return %0 : tensor<1x96xf32>
  }

  // CHECK-LABEL: func @max_reduce_C
  func.func @max_reduce_C(%arg: tensor<128x96xf32>) -> tensor<128x1xf32> {
    // CHECK: d2m.tile_reduce_max
    %0 = "ttir.max"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<128x1xf32>
    return %0 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @sum_reduce_unaligned
  func.func @sum_reduce_unaligned(%arg: tensor<127x97xf32>) -> tensor<1x97xf32> {
    // CHECK: %[[TO_LAYOUT:.*]] = d2m.to_layout %arg
    // CHECK: %[[MASK:.*]] = d2m.mask %[[TO_LAYOUT]], %{{.*}} logical_shape = [127, 97] fill_value = <zero>
    // CHECK: ins(%[[MASK]],
    // CHECK: d2m.tile_reduce_sum
    %0 = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<127x97xf32>) -> tensor<1x97xf32>
    return %0 : tensor<1x97xf32>
  }

  // CHECK-LABEL: func @max_reduce_unaligned
  func.func @max_reduce_unaligned(%arg: tensor<127x97xf32>) -> tensor<127x1xf32> {
    // CHECK: %[[TO_LAYOUT:.*]] = d2m.to_layout %arg
    // CHECK: %[[MASK:.*]] = d2m.mask %[[TO_LAYOUT]], %{{.*}} logical_shape = [127, 97] fill_value = <neginf>
    // CHECK: ins(%[[MASK]],
    // CHECK: d2m.tile_reduce_max
    %0 = "ttir.max"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<127x97xf32>) -> tensor<127x1xf32>
    return %0 : tensor<127x1xf32>
  }

  // Mean uses the same 'zero' OOB fill as sum (identity for averaging).
  // CHECK-LABEL: func @mean_reduce_C
  func.func @mean_reduce_C(%arg: tensor<128x96xf32>) -> tensor<128x1xf32> {
    // CHECK: d2m.tile_reduce_mean
    %0 = "ttir.mean"(%arg) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<128x1xf32>
    return %0 : tensor<128x1xf32>
  }

  // CHECK-LABEL: func @eltwise_add
  func.func @eltwise_add(%a: tensor<128x96xf32>, %b: tensor<128x96xf32>) -> tensor<128x96xf32> {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.tile_add
    %0 = "ttir.add"(%a, %b) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    return %0 : tensor<128x96xf32>
  }

  // Inner min is decomposed to neg→max→neg inside TTIRToD2M.
  // CHECK-LABEL: func @min_reduce_R
  // CHECK: d2m.tile_negative
  // CHECK: d2m.tile_reduce_max
  // CHECK: d2m.tile_negative
  func.func @min_reduce_R(%arg: tensor<128x96xf32>) -> tensor<1x96xf32> {
    %0 = "ttir.min"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<1x96xf32>
    return %0 : tensor<1x96xf32>
  }

  // Back-to-back ops: explicit masks preserve needed reduction identities
  // between generics when padding requires them.

  // CHECK-LABEL: func @add_then_sum
  // Eltwise -> reduction: aligned add result feeds the reduction directly.
  // CHECK: %[[ADD_RES:[0-9]+]] = d2m.generic
  // CHECK: %[[INIT:[0-9]+]] = d2m.generic
  // CHECK: d2m.generic
  // CHECK: ins(%[[ADD_RES]], %[[INIT]]
  // CHECK: d2m.tile_reduce_sum
  func.func @add_then_sum(%a: tensor<128x96xf32>, %b: tensor<128x96xf32>) -> tensor<1x96xf32> {
    %add = "ttir.add"(%a, %b) : (tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
    %sum = "ttir.sum"(%add) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<1x96xf32>
    return %sum : tensor<1x96xf32>
  }

  // CHECK-LABEL: func @sum_then_add
  // Reduction → eltwise: the reduction result still feeds directly into the
  // add, while the scalar/input side is materialized separately.
  // CHECK: d2m.tile_fill
  // CHECK: } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>,{{.*}}
  // CHECK: %[[SUM_RES:[0-9]+]] = d2m.generic
  // CHECK: d2m.tile_reduce_sum
  // CHECK: %[[ADD_RHS:[0-9]+]] = d2m.to_layout %arg1, %{{.*}}
  // CHECK: ins(%[[SUM_RES]], %[[ADD_RHS]]
  // CHECK: d2m.tile_add
  func.func @sum_then_add(%arg: tensor<128x96xf32>, %b: tensor<1x96xf32>) -> tensor<1x96xf32> {
    %sum = "ttir.sum"(%arg) <{dim_arg = [-2: i32], keep_dim = true}> : (tensor<128x96xf32>) -> tensor<1x96xf32>
    %add = "ttir.add"(%sum, %b) : (tensor<1x96xf32>, tensor<1x96xf32>) -> tensor<1x96xf32>
    return %add : tensor<1x96xf32>
  }
}
