// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  // or(0, x) -> x
  func.func @logical_or_zero_lhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_or"
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(x, 0) -> x
  func.func @logical_or_zero_rhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_or"
    %1 = "ttir.logical_or"(%arg0, %zero) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(0, x) -> x with integer zero
  func.func @logical_or_zero_int(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0 : i32}> : () -> tensor<64x64xi32>
    // CHECK-NOT: "ttir.logical_or"
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  // and(nonzero, x) -> x
  func.func @logical_and_nonzero_lhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_and"
    %1 = "ttir.logical_and"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(x, nonzero) -> x
  func.func @logical_and_nonzero_rhs(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_and"
    %1 = "ttir.logical_and"(%arg0, %ones) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(nonzero_int, x) -> x
  func.func @logical_and_nonzero_int(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1 : i32}> : () -> tensor<64x64xi32>
    // CHECK-NOT: "ttir.logical_and"
    %1 = "ttir.logical_and"(%ones, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  // or(nonzero, x) -> nonzero (absorbing element)
  func.func @logical_or_nonzero_absorb(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_or"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 1.000000e+00
    %1 = "ttir.logical_or"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(0, x) -> 0 (absorbing element)
  func.func @logical_and_zero_absorb(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_and"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 0.000000e+00
    %1 = "ttir.logical_and"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(zeros_op, x) -> x (named ZerosOp)
  func.func @logical_or_zeros_op(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_or"
    %1 = "ttir.logical_or"(%zero, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(ones_op, x) -> x (named OnesOp)
  func.func @logical_and_ones_op(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %ones = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-NOT: "ttir.logical_and"
    %1 = "ttir.logical_and"(%ones, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // or(broadcast(reshape(full(0))), x) -> x (look through layout ops)
  func.func @logical_or_broadcast_chain(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %zero = "ttir.full"() <{shape = array<i32: 1, 1>, fill_value = 0.000000e+00 : f32}> : () -> tensor<1x1xf32>
    %reshaped = "ttir.reshape"(%zero) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted = "ttir.broadcast"(%reshaped) <{broadcast_dimensions = array<i64: 64, 64>}> : (tensor<1x1xf32>) -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @logical_or_broadcast_chain
    // CHECK-NOT: "ttir.logical_or"
    %1 = "ttir.logical_or"(%broadcasted, %arg0) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  // and(broadcast(reshape(full(1))), x) -> x (look through layout ops)
  func.func @logical_and_broadcast_chain(%arg0: tensor<64x64xi32>) -> tensor<64x64xi32> {
    %ones = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %reshaped = "ttir.reshape"(%ones) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xi32>) -> tensor<1x1xi32>
    %broadcasted = "ttir.broadcast"(%reshaped) <{broadcast_dimensions = array<i64: 64, 64>}> : (tensor<1x1xi32>) -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @logical_and_broadcast_chain
    // CHECK-NOT: "ttir.logical_and"
    %1 = "ttir.logical_and"(%broadcasted, %arg0) : (tensor<64x64xi32>, tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }
}
