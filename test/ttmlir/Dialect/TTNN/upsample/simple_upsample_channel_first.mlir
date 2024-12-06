// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module {
  func.func @forward(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x128x64xf32> {
    %0 = tensor.empty() : tensor<5x3x128x64xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.upsample"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 4, 2>, channel_last = false, operand_constraints = [#any_device, #any_device]}> : (tensor<5x3x32x32xf32>, tensor<5x3x128x64xf32>) -> tensor<5x3x128x64xf32>
    return %1 : tensor<5x3x128x64xf32>
  }
}
