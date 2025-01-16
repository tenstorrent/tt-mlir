// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @test_repeat_interleave(%arg0: tensor<1x8x1x12x64xf32>) -> tensor<1x8x1x12x64xf32> {
    // CHECK: "ttnn.repeat_interleave"
    // CHECK-SAME: dim = 0 : si32
    // CHECK-SAME: repeats = 1 : ui32
    %0 = tensor.empty() : tensor<1x8x1x12x64xf32>
    %1 = "ttir.repeat_interleave"(%arg0, %0) {repeats = 1 : ui32, dim = 0 : si32} : (tensor<1x8x1x12x64xf32>, tensor<1x8x1x12x64xf32>) -> tensor<1x8x1x12x64xf32>
    return %1 : tensor<1x8x1x12x64xf32>
  }
}

module {
  func.func @test_repeat_interleave(%arg0: tensor<1x8x1x12x64xf32>) -> tensor<1x8x4x12x64xf32> {
    // CHECK: "ttnn.repeat_interleave"
    // CHECK-SAME: dim = 2 : si32
    // CHECK-SAME: repeats = 4 : ui32
    %0 = tensor.empty() : tensor<1x8x4x12x64xf32>
    %1 = "ttir.repeat_interleave"(%arg0, %0) {repeats = 4 : ui32, dim = 2 : si32} : (tensor<1x8x1x12x64xf32>, tensor<1x8x4x12x64xf32>) -> tensor<1x8x4x12x64xf32>
    return %1 : tensor<1x8x4x12x64xf32>
  }
}
