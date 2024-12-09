// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module @jit_broadcast attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x512xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    // CHECK-NOT: %[[C:.*]] = "ttnn.broadcast"[[C:.*]]
    %0 = tensor.empty() : tensor<512x512xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{dimension = [1]}> : (tensor<1xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    %2 = tensor.empty() : tensor<512x512xf32>
    %3 = "ttir.maximum"(%1, %arg1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
    return %3 : tensor<512x512xf32>
}
