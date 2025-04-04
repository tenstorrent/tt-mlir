// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func public @test_clamp_tensor(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x64xbf16>, %arg2: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
  // CHECK-LABEL: func.func public @test_clamp_tensor(
  %0 = ttir.empty() : tensor<32x64xbf16>
  // CHECK: %{{[0-9]+}} = "ttnn.clamp_tensor"(%arg0, %arg1, %arg2)
  // CHECK-SAME: tensor<32x64xbf16,
  // CHECK-SAME: tensor<32x64xbf16,
  // CHECK-SAME: tensor<32x64xbf16,
  // CHECK-SAME: -> tensor<32x64xbf16,
  %1 = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xbf16>, tensor<32x64xbf16>, tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}
