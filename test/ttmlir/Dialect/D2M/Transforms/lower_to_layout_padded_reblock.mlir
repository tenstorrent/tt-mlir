// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-fe-pipeline %s | FileCheck %s

// Regression coverage for a padded DRAM bounce buffer feeding an L1 tensor with
// a smaller logical volume. This used to assert in calculateReblockMap while
// lowering the DRAM->L1 materialization for the reshape.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  // CHECK-LABEL: func.func @reshape
  func.func @reshape(%arg0: tensor<12x32x64xi64>) -> tensor<2x3x2x32x64xi64> {
    %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 2 : i32, 32 : i32, 64 : i32]}> : (tensor<12x32x64xi64>) -> tensor<2x3x2x32x64xi64>
    return %0 : tensor<2x3x2x32x64xi64>
  }
}

// CHECK: d2m.to_device
// CHECK: d2m.view_layout
// CHECK-SAME: tensor<1x1x512x64xsi32
// CHECK-SAME: -> tensor<12x1x2x1x32x32xsi32
// CHECK: d2m.to_host
