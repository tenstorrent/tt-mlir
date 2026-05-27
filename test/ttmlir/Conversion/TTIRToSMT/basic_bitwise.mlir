// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_and
// CHECK-SAME: (%[[A:.*]]: !smt.bv<8>, %[[B:.*]]: !smt.bv<8>) -> !smt.bv<8>
func.func @test_and(%a: tensor<1xi8>, %b: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: %[[R:.*]] = smt.bv.and %[[A]], %[[B]] : !smt.bv<8>
  %0 = "ttir.bitwise_and"(%a, %b) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
  // CHECK: return %[[R]] : !smt.bv<8>
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: func.func @test_or
func.func @test_or(%a: tensor<1xi32>, %b: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK: smt.bv.or {{.*}} : !smt.bv<32>
  %0 = "ttir.bitwise_or"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @test_xor
func.func @test_xor(%a: tensor<1xi16>, %b: tensor<1xi16>) -> tensor<1xi16> {
  // CHECK: smt.bv.xor {{.*}} : !smt.bv<16>
  %0 = "ttir.bitwise_xor"(%a, %b) : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi16>
  return %0 : tensor<1xi16>
}

// CHECK-LABEL: func.func @test_not
func.func @test_not(%a: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK: smt.bv.not {{.*}} : !smt.bv<8>
  %0 = "ttir.bitwise_not"(%a) : (tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}
