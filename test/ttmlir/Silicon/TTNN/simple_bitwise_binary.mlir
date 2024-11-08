// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// TODO(kmitrovic): Enable when binary bitwise ops are supported and merge with unary into one file.
// UNSUPPORTED: true

module attributes {} {
  func.func @bitwise_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xi32, {{.*}}
    %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: {{.*}} "ttnn.bitwise_and"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xi32, {{.*}}
    return %1 : tensor<64x128xi32>
    // CHECK: return {{.*}} tensor<64x128xi32, {{.*}}
  }

  func.func @bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xi32, {{.*}}
    %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: {{.*}} "ttnn.bitwise_or"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xi32, {{.*}}
    return %1 : tensor<64x128xi32>
    // CHECK: return {{.*}} tensor<64x128xi32, {{.*}}
  }

  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xi32, {{.*}}
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: {{.*}} "ttnn.bitwise_xor"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xi32, {{.*}}
    return %1 : tensor<64x128xi32>
    // CHECK: return {{.*}} tensor<64x128xi32, {{.*}}
  }
}
