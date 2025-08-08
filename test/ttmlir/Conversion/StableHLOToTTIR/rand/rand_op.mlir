// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_rand() -> tensor<8x4xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %c = stablehlo.constant dense<[8, 4]> : tensor<2xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = stablehlo.convert %cst_0 : (tensor<1xbf16>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.convert %cst : (tensor<1xbf16>) -> tensor<1xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: %{{[0-9]+}} = "ttir.rand"() <{dtype = bf16, high = 1.000000e+00 : f32, low = 0.000000e+00 : f32, seed = 0 : ui32, size = [8 : i32, 4 : i32]}> : () -> tensor<8x4xbf16>
    %4 = stablehlo.rng %1, %3, %c, distribution =  UNIFORM : (tensor<bf16>, tensor<bf16>, tensor<2xi64>) -> tensor<8x4xbf16>
    return %4 : tensor<8x4xbf16>
  }
}
