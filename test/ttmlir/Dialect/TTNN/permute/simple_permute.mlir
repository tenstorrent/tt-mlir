// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @permute(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x32x64x1xf32> {
    %0 = tensor.empty() : tensor<4x32x64x1xf32>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 2, 3, 0>
    // CHECK-SAME: tensor<1x4x32x64xf32
    // CHECK-SAME: tensor<4x32x64x1xf32
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>, operand_constraints = [#any_device, #any_device]}> : (tensor<1x4x32x64xf32>, tensor<4x32x64x1xf32>) -> tensor<4x32x64x1xf32>
    return %1 : tensor<4x32x64x1xf32>
  }
}