// TODO (azecevic): #652
// XFAIL: *
// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module {
  func.func @upsample_channel_first(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x128x64xf32> {
    %0 = tensor.empty() : tensor<5x3x128x64xf32>
    // CHECK: "ttir.permute"
    // CHECK: "ttir.upsample"
    // CHECK-SAME: channel_last = true
    // CHECK: "ttir.permute"
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 4, 2>, channel_last = false, operand_constraints = [#any_device, #any_device]}> : (tensor<5x3x32x32xf32>, tensor<5x3x128x64xf32>) -> tensor<5x3x128x64xf32>
    return %2 : tensor<5x3x128x64xf32>
  }
}
