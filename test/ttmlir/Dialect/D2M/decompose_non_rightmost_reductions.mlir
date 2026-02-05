// RUN: ttmlir-opt --d2m-decompose-non-rightmost-reductions %s | FileCheck %s

// Test: Reduce on dimension 0 with keep_dim=false (the target use case)
// Input: 32x1x128x360 -> reduce(dim=0, keep_dim=false) -> 1x128x360
// CHECK-LABEL: @reduce_sum_dim0_no_keepdim
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK-SAME: -> tensor<1x128x360x32xbf16>
// CHECK: %[[R:.*]] = "ttir.sum"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK-SAME: -> tensor<1x128x360x1xbf16>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK-SAME: -> tensor<1x1x128x360xbf16>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]]) <{shape = [1 : i32, 128 : i32, 360 : i32]}>
// CHECK-SAME: -> tensor<1x128x360xbf16>
// CHECK: return %[[RESHAPE]]
func.func @reduce_sum_dim0_no_keepdim(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
  return %0 : tensor<1x128x360xbf16>
}

// Test: Reduce on dimension 0 with keep_dim=true
// Input: 32x1x128x360 -> reduce(dim=0, keep_dim=true) -> 1x1x128x360
// CHECK-LABEL: @reduce_sum_dim0_keepdim
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK-SAME: -> tensor<1x128x360x32xbf16>
// CHECK: %[[R:.*]] = "ttir.sum"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK-SAME: -> tensor<1x128x360x1xbf16>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK-SAME: -> tensor<1x1x128x360xbf16>
// CHECK: return %[[P1]]
func.func @reduce_sum_dim0_keepdim(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x1x128x360xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<32x1x128x360xbf16>) -> tensor<1x1x128x360xbf16>
  return %0 : tensor<1x1x128x360xbf16>
}

// Test: Reduce on dimension 1 (also non-rightmost for 4D tensor)
// Input: 32x16x128x360 -> reduce(dim=1, keep_dim=false) -> 32x128x360
// CHECK-LABEL: @reduce_sum_dim1_no_keepdim
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}>
// CHECK-SAME: -> tensor<32x128x360x16xbf16>
// CHECK: %[[R:.*]] = "ttir.sum"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK-SAME: -> tensor<32x128x360x1xbf16>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 0, 3, 1, 2>}>
// CHECK-SAME: -> tensor<32x1x128x360xbf16>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]]) <{shape = [32 : i32, 128 : i32, 360 : i32]}>
// CHECK-SAME: -> tensor<32x128x360xbf16>
// CHECK: return %[[RESHAPE]]
func.func @reduce_sum_dim1_no_keepdim(%arg0: tensor<32x16x128x360xbf16>) -> tensor<32x128x360xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x16x128x360xbf16>) -> tensor<32x128x360xbf16>
  return %0 : tensor<32x128x360xbf16>
}

// Test: Reduce on last dimension - should NOT be decomposed (rightmost)
// CHECK-LABEL: @reduce_sum_last_dim_no_decompose
// CHECK: %[[R:.*]] = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}>
// CHECK-SAME: -> tensor<32x1x128xbf16>
// CHECK-NOT: ttir.permute
// CHECK: return %[[R]]
func.func @reduce_sum_last_dim_no_decompose(%arg0: tensor<32x1x128x360xbf16>) -> tensor<32x1x128xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<32x1x128xbf16>
  return %0 : tensor<32x1x128xbf16>
}

// Test: Reduce on second-to-last dimension - should NOT be decomposed (rightmost 2)
// CHECK-LABEL: @reduce_sum_second_last_dim_no_decompose
// CHECK: %[[R:.*]] = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}>
// CHECK-SAME: -> tensor<32x1x360xbf16>
// CHECK-NOT: ttir.permute
// CHECK: return %[[R]]
func.func @reduce_sum_second_last_dim_no_decompose(%arg0: tensor<32x1x128x360xbf16>) -> tensor<32x1x360xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<32x1x360xbf16>
  return %0 : tensor<32x1x360xbf16>
}

// Test: Max reduction on dimension 0
// CHECK-LABEL: @reduce_max_dim0
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK: %[[R:.*]] = "ttir.max"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]])
// CHECK: return %[[RESHAPE]]
func.func @reduce_max_dim0(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
  %0 = "ttir.max"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
  return %0 : tensor<1x128x360xbf16>
}

// Test: Min reduction on dimension 0
// CHECK-LABEL: @reduce_min_dim0
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK: %[[R:.*]] = "ttir.min"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]])
// CHECK: return %[[RESHAPE]]
func.func @reduce_min_dim0(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
  %0 = "ttir.min"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
  return %0 : tensor<1x128x360xbf16>
}

// Test: Mean reduction on dimension 0
// CHECK-LABEL: @reduce_mean_dim0
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK: %[[R:.*]] = "ttir.mean"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]])
// CHECK: return %[[RESHAPE]]
func.func @reduce_mean_dim0(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
  %0 = "ttir.mean"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
  return %0 : tensor<1x128x360xbf16>
}

// Test: 3D tensor reduce on dim 0
// CHECK-LABEL: @reduce_3d_dim0
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 0>}>
// CHECK-SAME: -> tensor<128x360x32xf32>
// CHECK: %[[R:.*]] = "ttir.sum"(%[[P0]]) <{dim_arg = [2 : i32], keep_dim = true}>
// CHECK-SAME: -> tensor<128x360x1xf32>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 2, 0, 1>}>
// CHECK-SAME: -> tensor<1x128x360xf32>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]]) <{shape = [128 : i32, 360 : i32]}>
// CHECK-SAME: -> tensor<128x360xf32>
// CHECK: return %[[RESHAPE]]
func.func @reduce_3d_dim0(%arg0: tensor<32x128x360xf32>) -> tensor<128x360xf32> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<32x128x360xf32>) -> tensor<128x360xf32>
  return %0 : tensor<128x360xf32>
}

// Test: Negative dimension indexing (-4 on 4D tensor = dim 0)
// CHECK-LABEL: @reduce_negative_dim
// CHECK: %[[P0:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 3, 0>}>
// CHECK: %[[R:.*]] = "ttir.sum"(%[[P0]]) <{dim_arg = [3 : i32], keep_dim = true}>
// CHECK: %[[P1:.*]] = "ttir.permute"(%[[R]]) <{permutation = array<i64: 3, 0, 1, 2>}>
// CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[P1]])
// CHECK: return %[[RESHAPE]]
func.func @reduce_negative_dim(%arg0: tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16> {
  %0 = "ttir.sum"(%arg0) <{dim_arg = [-4 : i32], keep_dim = false}> : (tensor<32x1x128x360xbf16>) -> tensor<1x128x360xbf16>
  return %0 : tensor<1x128x360xbf16>
}
