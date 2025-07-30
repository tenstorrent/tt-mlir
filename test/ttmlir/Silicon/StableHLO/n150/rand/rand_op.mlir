// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module {
  func.func @test_rand() -> tensor<8x4xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %c = stablehlo.constant dense<[8, 4]> : tensor<2xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = stablehlo.convert %cst_0 : (tensor<1xbf16>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.convert %cst : (tensor<1xbf16>) -> tensor<1xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK-LABEL: @test_rand
    // CHECK: "ttnn.rand"(%{{[0-9]+}})
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: high = 1.000000e+00 : f32,
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: low = 0.000000e+00 : f32,
    // CHECK-SAME: seed = 0 : ui32,
    // CHECK-SAME: size = #ttnn.shape<8x4>
    // CHECK-SAME: -> tensor<8x4xbf16,
    %4 = stablehlo.rng %1, %3, %c, distribution =  UNIFORM : (tensor<bf16>, tensor<bf16>, tensor<2xi64>) -> tensor<8x4xbf16>
    return %4 : tensor<8x4xbf16>
  }
}
