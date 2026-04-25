// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_lshift
func.func @test_lshift(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: smt.bv.shl {{.*}} : !smt.bv<32>
  %0 = "ttir.logical_left_shift"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @test_rshift
func.func @test_rshift(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: smt.bv.lshr {{.*}} : !smt.bv<32>
  %0 = "ttir.logical_right_shift"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
