// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
// RUN: FileCheck %s --input-file %t.mlir

module {
  func.func @main(%arg0: tensor<1x1024x64xbf16>, %arg1: tensor<1x32x1024x64xbf16>, %arg2: tensor<1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16> {
    // CHECK: ttnn.reshape
    // CHECK: ttnn.reshape
    // CHECK: ttnn.rotary_embedding"
    %0 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %2 = "ttir.multiply"(%arg1, %1) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %3 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 32 : i32, 1024 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x32xbf16>
    %4 = "ttir.neg"(%3) : (tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x32xbf16>
    %5 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 1024 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x32xbf16>
    %6 = "ttir.concat"(%4, %5) <{dim = 3 : si32}> : (tensor<1x32x1024x32xbf16>, tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x64xbf16>
    %7 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %8 = "ttir.broadcast"(%7) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %9 = "ttir.multiply"(%6, %8) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %10 = "ttir.add"(%2, %9) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    return %10 : tensor<1x32x1024x64xbf16>
  }
}
