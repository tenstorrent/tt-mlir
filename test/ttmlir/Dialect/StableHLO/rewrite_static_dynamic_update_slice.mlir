// REQUIRES: stablehlo
// RUN: ttmlir-opt --rewrite-static-dynamic-update-slice %s | FileCheck %s

module {
  func.func @static_dynamic_slice_expr(%arg0: tensor<4x32xf32>) -> tensor<1x32xf32> {
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %start = stablehlo.subtract %c4, %c1 : tensor<i32>
    // CHECK-LABEL: func.func @static_dynamic_slice_expr
    // CHECK-NOT: stablehlo.dynamic_slice
    // CHECK: stablehlo.slice %arg0 [3:4, 0:32]
    // CHECK-NOT: stablehlo.dynamic_slice
    // CHECK: return
    %0 = stablehlo.dynamic_slice %arg0, %start, %c0, sizes = [1, 32] : (tensor<4x32xf32>, tensor<i32>, tensor<i32>) -> tensor<1x32xf32>
    return %0 : tensor<1x32xf32>
  }

  func.func @static_dynamic_slice_clamped(%arg0: tensor<4x32xf32>) -> tensor<2x16xf32> {
    %c_neg1 = stablehlo.constant dense<-1> : tensor<i32>
    %c99 = stablehlo.constant dense<99> : tensor<i32>
    // CHECK-LABEL: func.func @static_dynamic_slice_clamped
    // CHECK-NOT: stablehlo.dynamic_slice
    // CHECK: stablehlo.slice %arg0 [0:2, 16:32]
    // CHECK-NOT: stablehlo.dynamic_slice
    // CHECK: return
    %0 = stablehlo.dynamic_slice %arg0, %c_neg1, %c99, sizes = [2, 16] : (tensor<4x32xf32>, tensor<i32>, tensor<i32>) -> tensor<2x16xf32>
    return %0 : tensor<2x16xf32>
  }

  func.func @single_axis(%arg0: tensor<4x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<4x32xf32> {
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    // CHECK-LABEL: func.func @single_axis
    // CHECK-NOT: stablehlo.dynamic_update_slice
    // CHECK: stablehlo.slice %arg0 [0:1, 0:32]
    // CHECK: stablehlo.slice %arg0 [2:4, 0:32]
    // CHECK: stablehlo.concatenate {{.*}} dim = 0
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c1, %c0 : (tensor<4x32xf32>, tensor<1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  func.func @single_axis_expr(%arg0: tensor<4x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<4x32xf32> {
    %c4 = stablehlo.constant dense<4> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %start = stablehlo.subtract %c4, %c1 : tensor<i32>
    // CHECK-LABEL: func.func @single_axis_expr
    // CHECK-NOT: stablehlo.dynamic_update_slice
    // CHECK: stablehlo.slice %arg0 [0:3, 0:32]
    // CHECK: stablehlo.concatenate {{.*}} dim = 0
    // CHECK-NOT: stablehlo.dynamic_update_slice
    // CHECK: return
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %start, %c0 : (tensor<4x32xf32>, tensor<1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  func.func @single_axis_clamped(%arg0: tensor<4x32xf32>, %arg1: tensor<1x32xf32>) -> tensor<4x32xf32> {
    %c99 = stablehlo.constant dense<99> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    // CHECK-LABEL: func.func @single_axis_clamped
    // CHECK-NOT: stablehlo.dynamic_update_slice
    // CHECK: stablehlo.slice %arg0 [0:3, 0:32]
    // CHECK: stablehlo.concatenate {{.*}} dim = 0
    // CHECK-NOT: stablehlo.dynamic_update_slice
    // CHECK: return
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c99, %c0 : (tensor<4x32xf32>, tensor<1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  func.func @dynamic_start_kept(%arg0: tensor<4x32xf32>, %arg1: tensor<1x32xf32>, %arg2: tensor<i32>) -> tensor<4x32xf32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    // CHECK-LABEL: func.func @dynamic_start_kept
    // CHECK: stablehlo.dynamic_update_slice
    %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %c0 : (tensor<4x32xf32>, tensor<1x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }
}
