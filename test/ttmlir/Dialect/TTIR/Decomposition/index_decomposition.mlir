// RUN: ttmlir-opt --ttir-to-ttir-decomposition --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @index_negative_bounds_fold(%arg0: tensor<3x3x3xf32>) -> tensor<2x2x3xf32> {
    %0 = "ttir.index"(%arg0) <{begin = -2 : i32, dim = 0 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<3x3x3xf32>) -> tensor<2x3x3xf32>
    %1 = "ttir.index"(%0) <{begin = -2 : i32, dim = 1 : i32, end = 3 : i32, step = 1 : i32}> : (tensor<2x3x3xf32>) -> tensor<2x2x3xf32>
    // CHECK: %[[SLICE:.+]] = "ttir.slice_static"(%arg0) <{begins = [1 : i32, 1 : i32, 0 : i32], ends = [3 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32]}>
    return %1 : tensor<2x2x3xf32>
  }
}
