// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m %s | FileCheck %s

// CHECK-LABEL: func.func @squeeze_leading_unit_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<64xi32>
// CHECK: %[[INPUT_LAYOUT:.*]] = d2m.to_layout %[[ARG0]]
// CHECK: %[[VIEW:.*]] = d2m.view_layout %[[INPUT_LAYOUT]]
// CHECK: %[[RESHAPED:.*]] = d2m.to_layout %[[VIEW]]
// CHECK: return %[[RESHAPED]] : tensor<64xi32>
func.func @squeeze_leading_unit_dim(%arg0: tensor<1x64xi32>) -> tensor<64xi32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32]}> : (tensor<1x64xi32>) -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// CHECK-LABEL: func.func @squeeze_middle_unit_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<18x1x128xf32>) -> tensor<18x128xf32>
// CHECK: %[[INPUT_LAYOUT:.*]] = d2m.to_layout %[[ARG0]]
// CHECK: %[[VIEW:.*]] = d2m.view_layout %[[INPUT_LAYOUT]]
// CHECK: %[[RESHAPED:.*]] = d2m.to_layout %[[VIEW]]
// CHECK: return %[[RESHAPED]] : tensor<18x128xf32>
func.func @squeeze_middle_unit_dim(%arg0: tensor<18x1x128xf32>) -> tensor<18x128xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [18 : i32, 128 : i32]}> : (tensor<18x1x128xf32>) -> tensor<18x128xf32>
  return %0 : tensor<18x128xf32>
}
