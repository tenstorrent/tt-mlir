// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // and(0, x) -> zeros(result_type) (absorbing)
  func.func @logical_and_zero_absorb(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_zero_absorb
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(x, 0) -> zeros(result_type) (absorbing, rhs)
  func.func @logical_and_zero_absorb_rhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_zero_absorb_rhs
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%arg0, %zero) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(integer zero, x) -> zeros(result_type)
  func.func @logical_and_zero_int(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0 : i32}> : () -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @logical_and_zero_int
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%zero, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  // and(zeros_op, x) -> zeros(result_type)
  func.func @logical_and_zeros_op(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_zeros_op
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(broadcast(reshape(full(0))), x) -> zeros(result_type) (look through layout ops)
  func.func @logical_and_broadcast_zero(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 1, 1>, fill_value = 0.000000e+00 : f32}> : () -> tensor<1x1xf32>
    %reshaped = "ttir.reshape"(%zero) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted = "ttir.broadcast"(%reshaped) <{broadcast_dimensions = array<i64: 64, 64>}> : (tensor<1x1xf32>) -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_broadcast_zero
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%broadcasted, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(nonzero, x) -> ones(result_type) (absorbing)
  func.func @logical_or_nonzero_absorb(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_nonzero_absorb
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(x, nonzero) -> ones(result_type) (absorbing, rhs)
  func.func @logical_or_nonzero_absorb_rhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_nonzero_absorb_rhs
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%arg0, %ones) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(5.0, x) -> ones(result_type) (non-one nonzero constant still folds correctly)
  func.func @logical_or_nonone_absorb(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %five = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 5.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_nonone_absorb
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%five, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(ones_op, x) -> ones(result_type) (named OnesOp)
  func.func @logical_or_ones_op(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_ones_op
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(broadcast(reshape(full(1))), x) -> ones(result_type) (look through layout ops)
  func.func @logical_or_broadcast_nonzero(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %ones = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %reshaped = "ttir.reshape"(%ones) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>) -> tensor<1x1xi32>
    %broadcasted = "ttir.broadcast"(%reshaped) <{broadcast_dimensions = array<i64: 64, 64>}> : (tensor<1x1xi32>) -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @logical_or_broadcast_nonzero
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%broadcasted, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  // Identity: and(nonzero, x) -> x for i1 (boolean-valued, no ne needed)
  func.func @logical_and_identity_i1(%arg0: tensor<64x64xi1>) -> tensor<64x64xi1> {
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xi1>
    // CHECK-LABEL: func.func @logical_and_identity_i1
    // CHECK-NOT: "ttir.logical_and"
    // CHECK-NOT: "ttir.ne"
    // CHECK: return %arg0
    %1 = "ttir.logical_and"(%ones, %arg0) : (tensor<64x64xi1>, tensor<64x64xi1>) -> tensor<64x64xi1>
    return %1 : tensor<64x64xi1>
  }

  // Identity: and(x, nonzero) -> x for i1 (boolean-valued, no ne needed)
  func.func @logical_and_identity_i1_rhs(%arg0: tensor<64x64xi1>) -> tensor<64x64xi1> {
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xi1>
    // CHECK-LABEL: func.func @logical_and_identity_i1_rhs
    // CHECK-NOT: "ttir.logical_and"
    // CHECK-NOT: "ttir.ne"
    // CHECK: return %arg0
    %1 = "ttir.logical_and"(%arg0, %ones) : (tensor<64x64xi1>, tensor<64x64xi1>) -> tensor<64x64xi1>
    return %1 : tensor<64x64xi1>
  }

  // Identity: or(zero, x) -> x for i1 (boolean-valued, no ne needed)
  func.func @logical_or_identity_i1(%arg0: tensor<64x64xi1>) -> tensor<64x64xi1> {
    %zero = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xi1>
    // CHECK-LABEL: func.func @logical_or_identity_i1
    // CHECK-NOT: "ttir.logical_or"
    // CHECK-NOT: "ttir.ne"
    // CHECK: return %arg0
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xi1>, tensor<64x64xi1>) -> tensor<64x64xi1>
    return %1 : tensor<64x64xi1>
  }

  // Identity: or(x, zero) -> x for i1 (boolean-valued, no ne needed)
  func.func @logical_or_identity_i1_rhs(%arg0: tensor<64x64xi1>) -> tensor<64x64xi1> {
    %zero = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xi1>
    // CHECK-LABEL: func.func @logical_or_identity_i1_rhs
    // CHECK-NOT: "ttir.logical_or"
    // CHECK-NOT: "ttir.ne"
    // CHECK: return %arg0
    %1 = "ttir.logical_or"(%arg0, %zero) : (tensor<64x64xi1>, tensor<64x64xi1>) -> tensor<64x64xi1>
    return %1 : tensor<64x64xi1>
  }

  // and(full(5), full(1)) -> full(1) (full(1) is boolean-valued, full(5) is nonzero identity)
  func.func @logical_and_full_one(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %five = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 5.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %one = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_full_one
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 1.000000e+00
    %1 = "ttir.logical_and"(%five, %one) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Both constant nonzero: and(full(5), full(3)) -> ones
  func.func @logical_and_both_nonzero(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %five = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 5.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %three = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 3.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_both_nonzero
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_and"(%five, %three) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Identity: and(nonzero, zeros_op) -> zeros_op (ZerosOp is boolean-valued)
  func.func @logical_and_zeros_rhs_identity(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    %zeros = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_zeros_rhs_identity
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.zeros"
    %1 = "ttir.logical_and"(%ones, %zeros) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Both constant zero: or(zero, zero) -> zeros
  func.func @logical_or_both_zero(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero1 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %zero2 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_both_zero
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 0.000000e+00
    // CHECK-NOT: "ttir.full"
    %1 = "ttir.logical_or"(%zero1, %zero2) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Identity: or(zero, ones_op) -> ones_op (OnesOp is boolean-valued)
  func.func @logical_or_ones_rhs_identity(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zeros = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_ones_rhs_identity
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.ones"
    %1 = "ttir.logical_or"(%zeros, %ones) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Identity: and(nonzero, ge_result) -> ge_result (comparison output is boolean)
  func.func @logical_and_identity_cmp_output(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %cmp = "ttir.ge"(%arg0, %arg1) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_identity_cmp_output
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: %[[CMP:.*]] = "ttir.ge"
    // CHECK: return %[[CMP]]
    %1 = "ttir.logical_and"(%ones, %cmp) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // Identity: or(zero, logical_and_result) -> logical_and_result (logical output is boolean)
  func.func @logical_or_identity_logical_output(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %land = "ttir.logical_and"(%arg0, %arg1) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_identity_logical_output
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: %[[LAND:.*]] = "ttir.logical_and"
    // CHECK: return %[[LAND]]
    %1 = "ttir.logical_or"(%zero, %land) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // No fold: and(nonzero, dynamic_f32) stays (cannot prove x is boolean-valued)
  func.func @logical_and_no_fold_dynamic(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_and_no_fold_dynamic
    // CHECK: "ttir.logical_and"
    %1 = "ttir.logical_and"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // No fold: or(zero, dynamic_f32) stays (cannot prove x is boolean-valued)
  func.func @logical_or_no_fold_dynamic(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_no_fold_dynamic
    // CHECK: "ttir.logical_or"
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // No fold: and(nonzero, dynamic_i32) stays
  func.func @logical_and_no_fold_dynamic_int(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %one = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1 : i32}> : () -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @logical_and_no_fold_dynamic_int
    // CHECK: "ttir.logical_and"
    %1 = "ttir.logical_and"(%one, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  // No fold: or(zero, dynamic_i32) stays
  func.func @logical_or_no_fold_dynamic_int(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0 : i32}> : () -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @logical_or_no_fold_dynamic_int
    // CHECK: "ttir.logical_or"
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }
}
