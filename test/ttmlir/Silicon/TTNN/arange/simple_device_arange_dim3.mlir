// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.arange"[[C:.*]]
    %0 = "ttir.arange"() <{start = 0: si64, end = 128: si64, step = 1: si64, arange_dimension = 3: i64}> : () -> tensor<1x1x32x128xbf16>
    %1 = tensor.empty() : tensor<1x1x32x128xbf16>
    %2 = "ttir.multiply"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    return %2 : tensor<1x1x32x128xbf16>
  }
}
