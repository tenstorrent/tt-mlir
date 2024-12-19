// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.empty"(%{{[0-9]+}}) <{dtype = {{.*}}, layout = {{.*}}, memory_config = {{.*}}, <{{.*}}>>, shape = #ttnn.shape<[[TENSOR_SHAPE:[0-9]+x[0-9]+]]>}>
    %1 = "ttir.sign"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.sign"(%arg0, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>, tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}) -> tensor<[[TENSOR_SHAPE]]x{{.*}}, {{.*}}>
    return %1 : tensor<64x128xf32>
    // CHECK: return %{{[0-9]+}} : tensor<[[TENSOR_SHAPE]]xf32, {{.*}}>
  }
}
