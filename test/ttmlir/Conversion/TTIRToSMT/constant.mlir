// RUN: ttmlir-opt --convert-ttir-to-smt -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @test_const
func.func @test_const() -> tensor<1xi8> {
  // CHECK: %[[C:.*]] = smt.bv.constant #smt.bv<42> : !smt.bv<8>
  %0 = "ttir.constant"() {value = dense<42> : tensor<1xi8>} : () -> tensor<1xi8>
  // CHECK: return %[[C]]
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: func.func @test_const_zero
func.func @test_const_zero() -> tensor<1xi32> {
  // CHECK: smt.bv.constant #smt.bv<0> : !smt.bv<32>
  %0 = "ttir.constant"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: func.func @test_const_negative
func.func @test_const_negative() -> tensor<1xi8> {
  // CHECK: smt.bv.constant #smt.bv<-1> : !smt.bv<8>
  %0 = "ttir.constant"() {value = dense<-1> : tensor<1xi8>} : () -> tensor<1xi8>
  return %0 : tensor<1xi8>
}
