//RUN: ttmlir-opt --convert-ttir-to-nvvm %s | FileCheck %s

module attributes {} {
  func.func @test_1d_add(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = ttir.empty() : tensor<10xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    //CHECK: gpu.launch_func
    //CHECK: gpu.module
    return %1 : tensor<10xf32>
  }
}
