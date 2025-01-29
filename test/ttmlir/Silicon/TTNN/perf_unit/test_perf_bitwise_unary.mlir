// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// TODO(kmitrovic): Failing due to https://github.com/tenstorrent/tt-mlir/issues/1571
// UNSUPPORTED: true

module attributes {} {
  func.func @bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xi32, {{.*}}
    %1 = "ttir.bitwise_not"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: {{.*}} "ttnn.bitwise_not"({{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xi32, {{.*}}
    return %1 : tensor<64x128xi32>
    // CHECK: return {{.*}} tensor<64x128xi32, {{.*}}
  }
}
