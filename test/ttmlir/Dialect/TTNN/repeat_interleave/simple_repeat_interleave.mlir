// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
// CHECK: "ttnn.repeat_interleave"
module {
  func.func @test_repeat_interleave(%arg0: tensor<4xf32>) -> tensor<8xf32> {
    %0 = tensor.empty() : tensor<8xf32>
    %1 = "ttir.repeat_interleave"(%arg0, %0) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<4xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
