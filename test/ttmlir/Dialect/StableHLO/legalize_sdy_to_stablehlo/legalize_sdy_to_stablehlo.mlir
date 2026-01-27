// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-sdy-to-stablehlo %s | FileCheck %s

// CHECK-LABEL: func.func @legalize_sdy_constant
func.func @legalize_sdy_constant() -> tensor<2x3xf32> {
  // CHECK: stablehlo.constant dense<1.000000e+00> : tensor<2x3xf32>
  // CHECK-NOT: sdy.constant
  %0 = sdy.constant dense<1.0> : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
