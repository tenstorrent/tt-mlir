// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @power(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}} -> tensor<64x128xf32, {{.*}}
    %1 = "ttir.power"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[POW:[0-9]+]] = "ttnn.pow"({{.*}}, {{.*}}, %[[EMPTY]]){{.*}} -> tensor<64x128xf32, {{.*}}
    return %1 : tensor<64x128xf32>
    // CHECK: return {{.*}} : tensor<64x128xf32, {{.*}}
  }
}
