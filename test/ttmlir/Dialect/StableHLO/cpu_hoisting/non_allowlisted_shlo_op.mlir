// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s -verify-diagnostics

module {
  func.func @dynamic_reshape_example() -> tensor<3x2xi64> {
    %operand = "stablehlo.constant"()
      {value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>}
      : () -> tensor<2x3xi64>

    %output_shape = "stablehlo.constant"()
      {value = dense<[3, 2]> : tensor<2xi64>}
      : () -> tensor<2xi64>

    // expected-error@+1 {{failed to legalize operation 'stablehlo.dynamic_reshape'}}
    %result = "stablehlo.dynamic_reshape"(%operand, %output_shape)
      : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>

    return %result : tensor<3x2xi64>
  }
}
