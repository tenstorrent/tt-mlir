// RUN: tt-alchemist gen-tests -i %s -o %t --verbose
// RUN: ls %t/test_*.py | FileCheck %s
// CHECK: test_add.py
// CHECK: test_relu.py

module attributes {ttnn.device = #ttnn.device<0>} {
  func.func @main(%arg0: tensor<1x32x128x128xbf16>, %arg1: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %1 = "ttnn.relu"(%0) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    // Another add with different shape to test parametrization
    %arg2 = "ttnn.constant"() {value = dense<1.0> : tensor<1x64x256x256xbf16>} : () -> tensor<1x64x256x256xbf16>
    %arg3 = "ttnn.constant"() {value = dense<2.0> : tensor<1x64x256x256xbf16>} : () -> tensor<1x64x256x256xbf16>
    %2 = "ttnn.add"(%arg2, %arg3) : (tensor<1x64x256x256xbf16>, tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>

    // Another relu with different shape
    %3 = "ttnn.relu"(%2) : (tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>

    return %1 : tensor<1x32x128x128xbf16>
  }
}