// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK: "ttnn.topk"
  func.func @main(%arg0: tensor<2x6xf32> {ttcore.argument_type = #ttcore.argument_type<input>} loc("p0.1")) -> (tensor<2x3xf32>, tensor<2x3xi64>) {
    %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 6 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<6xi32>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 6 : i32]}> : (tensor<6xi32>) -> tensor<1x6xi32>
    %values, %indices = "ttir.sort"(%arg0) <{descending = true, dim = 1 : si32, stable = false}> : (tensor<2x6xf32>) -> (tensor<2x6xf32>, tensor<2x6xi32>)
    %2 = "ttir.slice_static"(%values) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xf32>) -> tensor<2x3xf32>
    %3 = "ttir.slice_static"(%indices) <{begins = [0 : i32, 0 : i32], ends = [2 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2x6xi32>) -> tensor<2x3xi32>
    %4 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<2x3xi32>) -> tensor<2x3xi64>
    return %2, %4 : tensor<2x3xf32>, tensor<2x3xi64>
  }
}
