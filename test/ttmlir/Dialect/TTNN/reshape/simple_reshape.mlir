// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x34xbf16>) -> tensor<2x4x32x34xbf16> {
    %0 = tensor.empty() : tensor<2x4x32x34xbf16>
    // CHECK: %[[C:.*]] = "ttnn.to_layout"(%[[C:.*]], %[[C:.*]])
    // CHECK: %[[VAL3:.*]] = "ttnn.to_layout"(%[[C:.*]], %[[C:.*]])
    // CHECK-NEXT: %[[VAL4:.*]] = "ttnn.empty"(%[[C:.*]])
    // CHECK-NEXT: %[[VAL5:.*]] = "ttnn.reshape"(%[[VAL3]], %[[VAL4]])
    // CHECK-NEXT: %[[C:.*]] = "ttnn.to_layout"(%[[VAL5]], %[[C:.*]])
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 34: i32] , operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<4x2x32x34xbf16>, tensor<2x4x32x34xbf16>) -> tensor<2x4x32x34xbf16>
    return %1 : tensor<2x4x32x34xbf16>
  }
}
