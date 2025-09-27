// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  // Using constant op to make sure ends and begins match output shape.
  func.func @dynamic_slice1(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xbf16>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %3 : tensor<2x16x16xbf16>
  }

  func.func @dynamic_slice_f32(%arg0: tensor<4x32x32xf32>) -> tensor<2x16x16xf32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xf32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xf32>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    return %3 : tensor<2x16x16xf32>
  }

  func.func @dynamic_slice_f16(%arg0: tensor<4x32x32xf16>) -> tensor<2x16x16xf16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xf16>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xf16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xf16>) -> tensor<2x16x16xf16>
    return %3 : tensor<2x16x16xf16>
  }

  func.func @dynamic_slice_uint32(%arg0: tensor<4x32x32xui32>) -> tensor<2x16x16xui32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xui32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xui32>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xui32>) -> tensor<2x16x16xui32>
    return %3 : tensor<2x16x16xui32>
  }

  func.func @dynamic_slice_uint16(%arg0: tensor<4x32x32xui16>) -> tensor<2x16x16xui16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xui16>
    // CHECK: = "ttnn.slice_dynamic"
    // CHECK-SAME: (tensor<4x32x32xui32
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xui16>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xui16>) -> tensor<2x16x16xui16>
    return %3 : tensor<2x16x16xui16>
  }

  func.func @dynamic_slice_int32(%arg0: tensor<4x32x32xi32>) -> tensor<2x16x16xi32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<2x16x16xi32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xi32>, tensor<3xui32>, tensor<3xui32>, tensor<2x16x16xi32>) -> tensor<2x16x16xi32>
    return %3 : tensor<2x16x16xi32>
  }

  func.func @dynamic_slice_f32_strided(%arg0: tensor<4x32x32xf32>) -> tensor<1x8x8xf32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<1x8x8xf32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [2: i32, 2: i32, 2: i32]}> : (tensor<4x32x32xf32>, tensor<3xui32>, tensor<3xui32>, tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
    return %3 : tensor<1x8x8xf32>
  }

  func.func @dynamic_slice_f16_strided(%arg0: tensor<4x32x32xf16>) -> tensor<1x8x8xf16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<1x8x8xf16>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [2: i32, 2: i32, 2: i32]}> : (tensor<4x32x32xf16>, tensor<3xui32>, tensor<3xui32>, tensor<1x8x8xf16>) -> tensor<1x8x8xf16>
    return %3 : tensor<1x8x8xf16>
  }

  func.func @dynamic_slice_uint32_strided(%arg0: tensor<4x32x32xui32>) -> tensor<1x8x8xui32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<1x8x8xui32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [2: i32, 2: i32, 2: i32]}> : (tensor<4x32x32xui32>, tensor<3xui32>, tensor<3xui32>, tensor<1x8x8xui32>) -> tensor<1x8x8xui32>
    return %3 : tensor<1x8x8xui32>
  }

  func.func @dynamic_slice_uint16_strided(%arg0: tensor<4x32x32xui16>) -> tensor<1x8x8xui16> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<1x8x8xui16>
    // CHECK: = "ttnn.slice_dynamic"
    // CHECK-SAME: (tensor<4x32x32xui32
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [2: i32, 2: i32, 2: i32]}> : (tensor<4x32x32xui16>, tensor<3xui32>, tensor<3xui32>, tensor<1x8x8xui16>) -> tensor<1x8x8xui16>
    return %3 : tensor<1x8x8xui16>
  }

  func.func @dynamic_slice_int32_strided(%arg0: tensor<4x32x32xi32>) -> tensor<1x8x8xi32> {
    %0 = "ttir.constant"() <{value = dense<1> : tensor<3xui32>}> : () -> tensor<3xui32>
    %1 = "ttir.constant"() <{value = dense<[3, 17, 17]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %2 = ttir.empty() : tensor<1x8x8xi32>
    // CHECK: = "ttnn.slice_dynamic"
    %3 = "ttir.slice_dynamic"(%arg0, %0, %1, %2) <{step = [2: i32, 2: i32, 2: i32]}> : (tensor<4x32x32xi32>, tensor<3xui32>, tensor<3xui32>, tensor<1x8x8xi32>) -> tensor<1x8x8xi32>
    return %3 : tensor<1x8x8xi32>
  }
}
