// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @repeat(%arg0: tensor<1x32x32xf32>) -> tensor<32x32x32xf32> {
    // CHECK: "ttnn.repeat"
    // CHECK-SAME: repeat_dims = [32 : i32, 1 : i32, 1 : i32]
    %0 = tensor.empty() : tensor<32x32x32xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i32: 32, 1, 1>} : (tensor<1x32x32xf32>, tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
    return %1 : tensor<32x32x32xf32>
  }
}
