// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @clamp(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %1 = "ttir.clamp_scalar"(%arg0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // CHECK: "ttnn.clamp_scalar"
  // CHECK-SAME: {max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}
  // CHECK-SAME: tensor<64x128xbf16
  // CHECK-SAME: -> tensor<64x128xbf16
  return %1 : tensor<64x128xbf16>
}

// Regression test for https://github.com/tenstorrent/tt-mlir/issues/7496.
// torch.clamp(min=1) lowers to clamp_tensor(min=1, max=INT64_MAX).
// INT64_MAX needs to get saturated to INT32_MAX.
func.func @clamp_i64_sentinel(%arg0: tensor<64x128xi64>) -> tensor<64x128xi64> {
  %min = "ttir.constant"() <{value = dense<1> : tensor<64x128xi64>}> : () -> tensor<64x128xi64>
  %max = "ttir.constant"() <{value = dense<9223372036854775807> : tensor<64x128xi64>}> : () -> tensor<64x128xi64>
  %1 = "ttir.clamp_tensor"(%arg0, %min, %max) : (tensor<64x128xi64>, tensor<64x128xi64>, tensor<64x128xi64>) -> tensor<64x128xi64>
  // CHECK: "ttnn.clamp_scalar"
  // CHECK-SAME: max = 2147483647 : i32
  // CHECK-SAME: min = 1 : i32
  // CHECK-SAME: tensor<64x128xsi32
  return %1 : tensor<64x128xi64>
}
