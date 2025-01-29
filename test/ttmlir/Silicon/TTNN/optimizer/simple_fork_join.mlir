// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=GreedyL1Interleaved" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// UNSUPPORTED: true
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = tensor.empty() : tensor<64x128xbf16>
    %3 = "ttir.add"(%arg2, %arg3, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %4 = tensor.empty() : tensor<64x128xbf16>
    %5 = "ttir.add"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %6 = tensor.empty() : tensor<64x128xbf16>
    %7 = "ttir.relu"(%5, %6) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %7 : tensor<64x128xbf16>
  }
}
