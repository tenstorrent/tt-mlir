// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @bitwise_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_and"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_and_scalars(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = ttir.empty() : tensor<i32>
    %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK: "ttnn.bitwise_and"
    // CHECK-SAME: tensor<si32
    // CHECK-SAME: tensor<si32
    // CHECK-SAME: -> tensor<si32
    return %1 : tensor<i32>
  }

  func.func @bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_or"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_xor"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_xor_i8(%arg0: tensor<64x128xi8>, %arg1: tensor<64x128xi8>) -> tensor<64x128xi8> {
    %0 = ttir.empty() : tensor<64x128xi8>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<64x128xi8>, tensor<64x128xi8>, tensor<64x128xi8>) -> tensor<64x128xi8>
    // CHECK: "ttnn.bitwise_xor"
    // CHECK-SAME: tensor<64x128xui32
    // CHECK-SAME: tensor<64x128xui32
    // CHECK-SAME: -> tensor<64x128xui32
    return %1 : tensor<64x128xi8>
  }
}
