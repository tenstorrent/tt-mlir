// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_where
// CHECK-SAME: (%[[C:.*]]: !smt.bv<1>, %[[T:.*]]: !smt.bv<8>, %[[F:.*]]: !smt.bv<8>) -> !smt.bv<8>
func.func @test_where(%cond: tensor<1xi1>, %t: tensor<1xi8>, %f: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: %[[ZERO:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  // CHECK: %[[BOOL:.*]] = smt.distinct %[[C]], %[[ZERO]] : !smt.bv<1>
  // CHECK: %[[R:.*]] = smt.ite %[[BOOL]], %[[T]], %[[F]] : !smt.bv<8>
  %0 = "ttir.where"(%cond, %t, %f) : (tensor<1xi1>, tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
  // CHECK: return %[[R]]
  return %0 : tensor<1xi8>
}

// Multi-bit condition (any non-zero is true)
// CHECK-LABEL: func.func @test_where_wide_cond
func.func @test_where_wide_cond(%cond: tensor<1xi8>, %t: tensor<1xi32>, %f: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: %[[ZERO:.*]] = smt.bv.constant #smt.bv<0> : !smt.bv<8>
  // CHECK: smt.distinct {{.*}}, %[[ZERO]] : !smt.bv<8>
  // CHECK: smt.ite {{.*}} : !smt.bv<32>
  %0 = "ttir.where"(%cond, %t, %f) : (tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
