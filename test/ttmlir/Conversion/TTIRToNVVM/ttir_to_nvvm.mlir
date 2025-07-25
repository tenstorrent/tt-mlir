//RUN: ttmlir-opt --convert-ttir-to-nvvm %s | FileCheck %s

module attributes {} {
  func.func @test_matmul(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>) -> tensor<1x512xf32> {
    %0 = ttir.empty() : tensor<1x512xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    //CHECK-NOT: gpu.launch_func
    //CHECK-NOT: gpu.module
    return %1 : tensor<1x512xf32>
  }
}
