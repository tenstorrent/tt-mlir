// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @tan(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
  %0 = tensor.empty() : tensor<64x128xbf16>
  %1 = "ttir.tan"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
  // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.tan"(%arg0)
  // CHECK-SAME: (tensor<64x128xbf16, {{.*}}>)
  // CHECK-SAME: -> tensor<64x128xbf16, {{.*}}>
  return %1 : tensor<64x128xbf16>
  // CHECK: return %[[RETURN_VALUE]] : tensor<64x128xbf16, {{.*}}>
}
