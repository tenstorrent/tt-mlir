// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module {
  func.func @upsample_scale_unifrom(%arg0: tensor<4x32x64x3xf32>) -> tensor<4x64x128x3xf32> {
    %0 = tensor.empty() : tensor<4x64x128x3xf32>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xf32
    // CHECK-SAME: tensor<4x64x128x3xf32
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 2 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<4x32x64x3xf32>, tensor<4x64x128x3xf32>) -> tensor<4x64x128x3xf32>
    return %1 : tensor<4x64x128x3xf32>
  }

  func.func @upsample_scale_nonunifrom(%arg0: tensor<4x32x64x3xf32>) -> tensor<4x64x64x3xf32> {
    %0 = tensor.empty() : tensor<4x64x64x3xf32>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xf32
    // CHECK-SAME: tensor<4x64x64x3xf32
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<4x32x64x3xf32>, tensor<4x64x64x3xf32>) -> tensor<4x64x64x3xf32>
    return %1 : tensor<4x64x64x3xf32>
  }
}
