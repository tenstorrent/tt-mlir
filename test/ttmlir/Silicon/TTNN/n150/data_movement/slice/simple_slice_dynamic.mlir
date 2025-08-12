// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  // Assuming the difference in values of %arg2 (ends) and %arg1 (begins) will match output shape.
  func.func @dynamic_slice(%arg0: tensor<4x32x32xbf16>, %arg1: tensor<3xui32>, %arg2: tensor<3xui32>) -> tensor<2x16x16xbf16> {
    %0 = ttir.empty() : tensor<2x16x16xbf16>
    // CHECK: = "ttnn.slice_dynamic"
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
  }

  // Using constant op to make sure ends and begins match output shape.
  func.func @dynamic_slice1(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xbf16>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %3 : tensor<2x16x16xbf16>
  }
}
