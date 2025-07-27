// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @forward(%arg0: tensor<4x32x32xbf16>) -> tensor<4x32x16xbf16> {
    %0 = ttir.empty() : tensor<4x32x16xbf16>
    // CHECK: = "ttnn.slice"
    %1 = "ttir.index"(%arg0, %0) <{dim = 2: i32, begin = 0: i32, end = 32: i32, step = 2: i32}> : (tensor<4x32x32xbf16>, tensor<4x32x16xbf16>) -> tensor<4x32x16xbf16>
    return %1 : tensor<4x32x16xbf16>
  }
}
