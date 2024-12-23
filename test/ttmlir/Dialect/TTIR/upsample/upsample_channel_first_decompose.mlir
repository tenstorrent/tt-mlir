// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module {
  func.func @upsample_channel_first(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x128x64xf32> {
    %0 = tensor.empty() : tensor<5x3x128x64xf32>
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttir.upsample"
    // CHECK-SAME: channel_last = true
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 4, 2>, channel_last = false}> : (tensor<5x3x32x32xf32>, tensor<5x3x128x64xf32>) -> tensor<5x3x128x64xf32>
    return %1 : tensor<5x3x128x64xf32>
  }
}
