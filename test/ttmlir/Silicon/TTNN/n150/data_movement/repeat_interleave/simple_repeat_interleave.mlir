// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @repeat_interleave(%arg0: tensor<1x4x32x1xf32>) -> tensor<1x4x32x4xf32> {
    %0 = ttir.empty() : tensor<1x4x32x4xf32>
    // CHECK: "ttnn.repeat_interleave"
    // CHECK-SAME: dim = 3 : si32
    // CHECK-SAME: repeats = 4 : ui32
    %1 = "ttir.repeat_interleave"(%arg0, %0) <{repeats = 4 : ui32, dim = 3 : si32}> : (tensor<1x4x32x1xf32>, tensor<1x4x32x4xf32>) -> tensor<1x4x32x4xf32>
    return %1 : tensor<1x4x32x4xf32>
  }
}
