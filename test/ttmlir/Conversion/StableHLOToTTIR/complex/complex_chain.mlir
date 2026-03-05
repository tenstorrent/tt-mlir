// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test chained complex ops: complex(real, imag) then real and imag extraction.
module {
  func.func @test_complex_real_imag_chain(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>, tensor<3xf32>) {
    %c = "stablehlo.complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>
    %r = "stablehlo.real"(%c) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    %i = "stablehlo.imag"(%c) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
    // CHECK: "ttir.reshape"(%arg0)
    // CHECK: "ttir.reshape"(%arg1)
    // CHECK: "ttir.concat"
    // CHECK: "ttir.slice_static"
    // CHECK: "ttir.slice_static"
    return %r, %i : tensor<3xf32>, tensor<3xf32>
  }

  // Test real/imag on a complex constant.
  func.func @test_complex_constant_real_imag() -> (tensor<16x8xf32>, tensor<16x8xf32>) {
    %cst = stablehlo.constant dense<(1.41421354, 0.000000e+00)> : tensor<16x8xcomplex<f32>>
    %r = "stablehlo.real"(%cst) : (tensor<16x8xcomplex<f32>>) -> tensor<16x8xf32>
    %i = "stablehlo.imag"(%cst) : (tensor<16x8xcomplex<f32>>) -> tensor<16x8xf32>
    // CHECK: "ttir.constant"
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
    // CHECK: "ttir.slice_static"
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 1 : i32]
    return %r, %i : tensor<16x8xf32>, tensor<16x8xf32>
  }
}
