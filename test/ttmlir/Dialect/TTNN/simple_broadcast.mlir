// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"(%{{[0-9]+}})
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{dimension = [0, 1, 2]}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}

module @jit_broadcast attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x512xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.reshape"(%{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"([[VAL0]])
    %0 = tensor.empty() : tensor<1x1xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    %2 = tensor.empty() : tensor<512x512xf32>
    %3 = "ttir.broadcast"(%1, %2) <{dimension = [1]}> : (tensor<1x1xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %4 = tensor.empty() : tensor<512x512xf32>
    %5 = "ttir.maximum"(%3, %arg1, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    return %5 : tensor<512x512xf32>
  }
}
