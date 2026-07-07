// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies WhereOp::fold: a where with a constant-splat condition folds to the
// selected branch:
//   where(nonzero,  x, y) -> x
//   where(zeros, x, y) -> y
// The condition may come from ttir.ones/ttir.zeros/ttir.full or a splat
// ttir.constant (getConstantValue looks through all of them). This is the GLM
// 4.7 router group mask when n_group == 1, where the mask is statically all
// ones (see tenstorrent/tt-mlir#8928, tenstorrent/tt-xla#5427).

module {
  // where(ones, x, y) -> x
  func.func @where_ones_folds_to_true(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_ones_folds_to_true
    // CHECK-NOT: "ttir.where"
    // CHECK: return %arg0
    %cond = "ttir.ones"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xbf16>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // where(zeros, x, y) -> y
  func.func @where_zeros_folds_to_false(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_zeros_folds_to_false
    // CHECK-NOT: "ttir.where"
    // CHECK: return %arg1
    %cond = "ttir.zeros"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xbf16>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Splat ttir.constant condition (the GLM n_group==1 mask form) -> x.
  func.func @where_splat_constant_ones_folds_to_true(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_splat_constant_ones_folds_to_true
    // CHECK-NOT: "ttir.where"
    // CHECK: return %arg0
    %cond = "ttir.constant"() <{value = dense<true> : tensor<4x4xi1>}> : () -> tensor<4x4xi1>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Negative: a non-constant condition must NOT fold.
  func.func @where_dynamic_condition_not_folded(%cond: tensor<4x4xi1>, %x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_dynamic_condition_not_folded
    // CHECK: "ttir.where"
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // A non-zero constant condition (any value, not just 1) selects the
  // true branch, matching where's non-zero=true predicate semantics.
  func.func @where_constant_nonzero_folds_to_true(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_constant_nonzero_folds_to_true
    // CHECK-NOT: "ttir.where"
    // CHECK: return %arg0
    %cond = "ttir.full"() <{shape = array<i32: 4, 4>, fill_value = 5.0 : f32}> : () -> tensor<4x4xf32>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Integer-typed splat condition (all ones) -> x. Exercises the IntegerAttr
  // path of isConstantOne (getSplatValue -> isOneAttr), not just i1/float.
  func.func @where_int_splat_ones_folds_to_true(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_int_splat_ones_folds_to_true
    // CHECK-NOT: "ttir.where"
    // CHECK: return %arg0
    %cond = "ttir.constant"() <{value = dense<1> : tensor<4x4xi32>}> : () -> tensor<4x4xi32>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xi32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Constant-true but broadcasting condition: folding to x would change the
  // result type, so the guard declines the fold.
  func.func @where_broadcast_condition_not_folded(%x: tensor<1x4xf32>, %y: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK-LABEL: @where_broadcast_condition_not_folded
    // CHECK: "ttir.where"
    %cond = "ttir.ones"() <{shape = array<i32: 4, 4>}> : () -> tensor<4x4xbf16>
    %0 = "ttir.where"(%cond, %x, %y) : (tensor<4x4xbf16>, tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
