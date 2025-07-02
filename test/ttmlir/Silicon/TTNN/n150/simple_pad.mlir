// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @main(%arg0: tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32> {
    %0 = ttir.empty() : tensor<1x132x132x384xf32>
    // CHECK: ttnn.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 2, 2, 2, 2, 0, 0>
    %1 = "ttir.pad"(%arg0, %0) <{padding = array<i32: 0, 0, 2, 2, 2, 2, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<1x128x128x384xf32>, tensor<1x132x132x384xf32>) -> tensor<1x132x132x384xf32>
    return %1 : tensor<1x132x132x384xf32>
  }
}
