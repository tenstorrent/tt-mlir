// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s -o %t.mlir --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%"
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @test_clamp_tensor attributes {} {
  func.func public @test_clamp_tensor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_clamp_tensor(
    // CHECK: %{{[0-9]+}} = "ttnn.clamp_tensor"(%arg0, %arg1, %arg2)
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.clamp %arg1, %arg0, %arg2 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_clamp_tensor_constant(%arg0: tensor<32x32xbf16>, %arg1: tensor<bf16>) -> tensor<32x32xbf16> {
    // CHECK-LABEL: func.func public @test_clamp_tensor_constant(
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: %{{[0-9]+}} = "ttnn.clamp_tensor"(%arg0, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: tensor<32x32xbf16,
    // CHECK-SAME: tensor<32x32xbf16,
    // CHECK-SAME: tensor<32x32xbf16,
    // CHECK-SAME: -> tensor<32x32xbf16,
    %2 = stablehlo.clamp %1, %arg0, %arg1 : (tensor<bf16>, tensor<32x32xbf16>, tensor<bf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}
