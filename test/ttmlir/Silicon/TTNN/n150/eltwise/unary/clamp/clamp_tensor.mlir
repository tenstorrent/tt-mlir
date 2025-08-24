// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func public @test_clamp_tensor(%arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
  // CHECK-LABEL: func.func public @test_clamp_tensor(
  %0 = ttir.empty() : tensor<32x64xf32>
  // CHECK: %{{[0-9]+}} = "ttnn.clamp_tensor"(%arg0, %arg1, %arg2)
  // CHECK-SAME: tensor<32x64xf32,
  // CHECK-SAME: tensor<32x64xf32,
  // CHECK-SAME: tensor<32x64xf32,
  // CHECK-SAME: -> tensor<32x64xf32,
  %1 = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
  return %1 : tensor<32x64xf32>
}
