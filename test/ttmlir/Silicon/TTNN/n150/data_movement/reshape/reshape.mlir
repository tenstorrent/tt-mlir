// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn


func.func @main_cleaned(%arg0: tensor<1x1x50x50xbf16>, %arg1: tensor<1x50x25xbf16>, %arg2: tensor<1x50x25xbf16>) -> tensor<1x1x25x25xbf16> {
  %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x50x50xbf16>) -> tensor<1x1x50x50xbf16>
  %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 50 : i32, 50 : i32]}> : (tensor<1x1x50x50xbf16>) -> tensor<1x50x50xbf16>
  "ttnn.deallocate"(%0) <{force = false}> : (tensor<1x1x50x50xbf16>) -> ()
  %2 = "ttnn.matmul"(%1, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x50x50xbf16>, tensor<1x50x25xbf16>) -> tensor<1x50x25xbf16>
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x50x50xbf16>) -> ()
  %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 1 : i32, 50 : i32, 25 : i32]}> : (tensor<1x50x25xbf16>) -> tensor<1x1x50x25xbf16>
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x50x25xbf16>) -> ()
  %4 = "ttnn.permute"(%3) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x50x25xbf16>) -> tensor<1x1x25x50xbf16>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x50x25xbf16>) -> ()
  %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 25 : i32, 50 : i32]}> : (tensor<1x1x25x50xbf16>) -> tensor<1x25x50xbf16>
  "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x25x50xbf16>) -> ()
  %6 = "ttnn.matmul"(%5, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x25x50xbf16>, tensor<1x50x25xbf16>) -> tensor<1x25x25xbf16>
  "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x25x50xbf16>) -> ()
  %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 1 : i32, 25 : i32, 25 : i32]}> : (tensor<1x25x25xbf16>) -> tensor<1x1x25x25xbf16>
  "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x25x25xbf16>) -> ()
  return %7 : tensor<1x1x25x25xbf16>
}
