// RUN: ttmlir-opt --ttir-fold-full-to-scalar %s | FileCheck %s

// FullOp with all broadcastable users should fold to tensor<1x1xf32>.
// CHECK-LABEL: func.func @fold_full_all_broadcastable
func.func @fold_full_all_broadcastable(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
  // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
  %0 = "ttir.full"() <{shape = array<i32: 64, 32>, fill_value = 1.0 : f32}> : () -> tensor<64x32xf32>
  // CHECK: "ttir.add"(%arg0, %[[FULL]])
  %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

// FullOp with mixed broadcastable and non-broadcastable users should NOT fold.
// CHECK-LABEL: func.func @no_fold_mixed_users
func.func @no_fold_mixed_users(%arg0: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xf32>) {
  // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 64, 32>}> : () -> tensor<64x32xf32>
  %0 = "ttir.full"() <{shape = array<i32: 64, 32>, fill_value = 2.0 : f32}> : () -> tensor<64x32xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1, %0 : tensor<64x32xf32>, tensor<64x32xf32>
}

// FullOp already volume-1 should NOT fold (no-op).
// CHECK-LABEL: func.func @no_fold_already_scalar
func.func @no_fold_already_scalar(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
  // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 3.000000e+00 : f32, shape = array<i32: 1>}> : () -> tensor<1xf32>
  %0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 3.0 : f32}> : () -> tensor<1xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<1xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

// ZerosOp with all broadcastable users should fold to tensor<1x1xf32>.
// CHECK-LABEL: func.func @fold_zeros_all_broadcastable
func.func @fold_zeros_all_broadcastable(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
  // CHECK: %[[ZEROS:.*]] = "ttir.zeros"() <{shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
  %0 = "ttir.zeros"() <{shape = array<i32: 64, 32>}> : () -> tensor<64x32xf32>
  // CHECK: "ttir.add"(%arg0, %[[ZEROS]])
  %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

// OnesOp with all broadcastable users should fold to tensor<1x1xf32>.
// CHECK-LABEL: func.func @fold_ones_all_broadcastable
func.func @fold_ones_all_broadcastable(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
  // CHECK: %[[ONES:.*]] = "ttir.ones"() <{shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
  %0 = "ttir.ones"() <{shape = array<i32: 64, 32>}> : () -> tensor<64x32xf32>
  // CHECK: "ttir.multiply"(%arg0, %[[ONES]])
  %1 = "ttir.multiply"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1 : tensor<64x32xf32>
}

// ZerosOp with mixed users should NOT fold.
// CHECK-LABEL: func.func @no_fold_zeros_mixed_users
func.func @no_fold_zeros_mixed_users(%arg0: tensor<64x32xf32>) -> (tensor<64x32xf32>, tensor<64x32xf32>) {
  // CHECK: %[[ZEROS:.*]] = "ttir.zeros"() <{shape = array<i32: 64, 32>}> : () -> tensor<64x32xf32>
  %0 = "ttir.zeros"() <{shape = array<i32: 64, 32>}> : () -> tensor<64x32xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  return %1, %0 : tensor<64x32xf32>, tensor<64x32xf32>
}

// FullOp is the shape carrier — the other operand is smaller.
// After folding, the consumer's implicit shape shrinks, so a broadcast
// is added on the consumer's output to restore the original shape.
// CHECK-LABEL: func.func @fold_full_shape_carrier
func.func @fold_full_shape_carrier(%arg0: tensor<1x32xf32>) -> tensor<64x32xf32> {
  // CHECK: %[[FULL:.*]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
  %0 = "ttir.full"() <{shape = array<i32: 64, 32>, fill_value = 1.0 : f32}> : () -> tensor<64x32xf32>
  // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[FULL]]) : (tensor<1x32xf32>, tensor<1x1xf32>) -> tensor<1x32xf32>
  %1 = "ttir.add"(%arg0, %0) : (tensor<1x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
  // CHECK: %[[BCAST:.*]] = "ttir.broadcast"(%[[ADD]]) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x32xf32>) -> tensor<64x32xf32>
  // CHECK: return %[[BCAST]]
  return %1 : tensor<64x32xf32>
}
