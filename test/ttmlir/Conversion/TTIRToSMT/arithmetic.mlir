// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_add
func.func @test_add(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: smt.bv.add {{.*}} : !smt.bv<32>
  %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: %[[NEG:.*]] = smt.bv.neg {{.*}} : !smt.bv<32>
  // CHECK: smt.bv.add {{.*}}, %[[NEG]] : !smt.bv<32>
  %0 = "ttir.subtract"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: smt.bv.mul {{.*}} : !smt.bv<32>
  %0 = "ttir.multiply"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
