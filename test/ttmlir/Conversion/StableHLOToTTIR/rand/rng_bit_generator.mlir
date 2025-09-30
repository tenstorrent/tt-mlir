// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_rng_bit_generator attributes {} {
  func.func public @test_rng_bit_generator() -> (tensor<2xui64>, tensor<2x4xui32>) {
    %initial_state = stablehlo.constant dense<[42, 24]> : tensor<2xui64>
    %output_state, %values = stablehlo.rng_bit_generator %initial_state, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x4xui32>)
    // CHECK: ttir.rand
    // CHECK-SAME: high = 4.2949673E+9 : f32,
    // CHECK-SAME: low = 0.000000e+00 : f32,
    // CHECK-SAME: seed = 0 : ui32,
    // CHECK-SAME: size = [2 : i32, 4 : i32]
    // CHECK-SAME: -> tensor<2x4xf32>
    // CHECK: ttir.typecast
    // CHECK-SAME: -> tensor<2x4xui32>
    return %output_state, %values : tensor<2xui64>, tensor<2x4xui32>
  }
}
