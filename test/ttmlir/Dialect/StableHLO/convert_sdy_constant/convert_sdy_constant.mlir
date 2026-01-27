// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-sdy-const-to-stablehlo-const %s | FileCheck %s

// CHECK-LABEL: func.func @convert_sdy_const
func.func @convert_sdy_const() -> tensor<2x3xf32> {
  // CHECK: stablehlo.constant dense<1.000000e+00> : tensor<2x3xf32>
  // CHECK-NOT: sdy.constant
  %0 = sdy.constant dense<1.0> : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
