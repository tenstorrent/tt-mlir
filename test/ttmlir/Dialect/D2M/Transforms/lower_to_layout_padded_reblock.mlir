// RUN: ttmlir-opt --ttcore-register-device --d2m-fe-pipeline %s | FileCheck %s

// Regression coverage for a virtual-grid reblock feeding an L1 tensor with the
// same logical volume. This used to assert in calculateReblockMap when the
// intermediate buffer was padded beyond the reshape volume.

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  // CHECK-LABEL: func.func @reshape
  func.func @reshape(%arg0: tensor<12x32x64xi64>) -> tensor<2x3x2x32x64xi64> {
    %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 2 : i32, 32 : i32, 64 : i32]}> : (tensor<12x32x64xi64>) -> tensor<2x3x2x32x64xi64>
    return %0 : tensor<2x3x2x32x64xi64>
  }
}

// CHECK: d2m.to_device
// CHECK: d2m.view_layout
// CHECK-SAME: memref<12x1x2x1x1x1x!ttcore.tile<32x32, si32>
// CHECK-SAME: -> memref<1x1x1x12x1x2x!ttcore.tile<32x32, si32>
// CHECK: d2m.view_layout
// CHECK-SAME: -> memref<2x3x2x1x2x1x1x1x1x1x!ttcore.tile<32x32, si32>
// CHECK: d2m.to_host
