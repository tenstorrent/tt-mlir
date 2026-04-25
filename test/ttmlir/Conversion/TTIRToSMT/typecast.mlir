// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// Truncation: i32 -> i8
// CHECK-LABEL: func.func @test_truncate
// CHECK-SAME: (%[[X:.*]]: !smt.bv<32>) -> !smt.bv<8>
func.func @test_truncate(%x: tensor<1xi32>) -> tensor<1xi8> {
  // CHECK: smt.bv.extract %[[X]] from 0 : (!smt.bv<32>) -> !smt.bv<8>
  %0 = "ttir.typecast"(%x) : (tensor<1xi32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// Zero-extend: i3 -> i8
// CHECK-LABEL: func.func @test_zero_extend
// CHECK-SAME: (%[[X:.*]]: !smt.bv<3>) -> !smt.bv<8>
func.func @test_zero_extend(%x: tensor<1xi3>) -> tensor<1xi8> {
  // CHECK: %[[ZERO:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<5>
  // CHECK: smt.bv.concat %[[ZERO]], %[[X]] : !smt.bv<5>, !smt.bv<3>
  %0 = "ttir.typecast"(%x) : (tensor<1xi3>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// Same width: identity
// CHECK-LABEL: func.func @test_same_width
// CHECK-SAME: (%[[X:.*]]: !smt.bv<8>) -> !smt.bv<8>
func.func @test_same_width(%x: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: return %[[X]]
  %0 = "ttir.typecast"(%x) : (tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// Boolean truncation: i8 -> i1
// CHECK-LABEL: func.func @test_truncate_bool
func.func @test_truncate_bool(%x: tensor<1xi8>) -> tensor<1xi1> {
  // CHECK: smt.bv.extract {{.*}} from 0 : (!smt.bv<8>) -> !smt.bv<1>
  %0 = "ttir.typecast"(%x) : (tensor<1xi8>) -> tensor<1xi1>
  return %0 : tensor<1xi1>
}
