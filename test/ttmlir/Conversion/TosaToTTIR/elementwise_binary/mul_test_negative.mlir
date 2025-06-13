// RUN: not ttmlir-opt --convert-tosa-to-ttir %s 2>&1 | FileCheck %s
// Negative test for elementwise mul operation.


// Verify that a shift different from 0 raises an error.
module attributes {} {
  func.func @test_shifted_mul(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
    // CHECK: error: 'tosa.mul' op conversion does not support shifted multiply
    %shift = "tosa.const"() <{values = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.mul %arg0, %arg1, %shift : (tensor<13x21x3xi32>, tensor<13x21x3xi32>, tensor<1xi8>) -> tensor<13x21x3xi32>
    return %0 : tensor<13x21x3xi32>
  }
}
