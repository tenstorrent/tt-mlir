// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_lhs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_rhs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_result = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Check whether both operands are explicitly broadcasted after the workaround has been applied
func.func @test_binary_implicit_broadcast_both_operands(%arg0: tensor<1x16x1xf32, #ttnn_layout_lhs>, %arg1: tensor<1x1x32xf32, #ttnn_layout_rhs>) -> tensor<1x16x32xf32, #ttnn_layout_result> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.remainder"(%arg0, %arg1) : (tensor<1x16x1xf32, #ttnn_layout_lhs>, tensor<1x1x32xf32, #ttnn_layout_rhs>) -> tensor<1x16x32xf32, #ttnn_layout_result>
  // CHECK: "ttnn.remainder"
  // CHECK-SAME: (tensor<1x16x32xf32, {{.*}}>, tensor<1x16x32xf32, {{.*}}>)
  // CHECK-SAME: -> tensor<1x16x32xf32, {{.*}}>
  return %1 : tensor<1x16x32xf32, #ttnn_layout_result>
}

#ttnn_layout_lhs_1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_rhs_1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_result_1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Check whether the lhs operand is explicitly broadcasted after the workaround has been applied
func.func @test_binary_implicit_broadcast_first_operands(%arg0: tensor<1x16x32xf32, #ttnn_layout_lhs_1>, %arg1: tensor<8x16x32xf32, #ttnn_layout_rhs_1>) -> tensor<8x16x32xf32, #ttnn_layout_result_1> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.remainder"(%arg0, %arg1) : (tensor<1x16x32xf32, #ttnn_layout_lhs_1>, tensor<8x16x32xf32, #ttnn_layout_rhs_1>) -> tensor<8x16x32xf32, #ttnn_layout_result_1>
  // CHECK: "ttnn.remainder"
  // CHECK-SAME: (tensor<8x16x32xf32, {{.*}}>, tensor<8x16x32xf32, {{.*}}>)
  // CHECK-SAME: -> tensor<8x16x32xf32, {{.*}}>
  return %1 : tensor<8x16x32xf32, #ttnn_layout_result_1>
}

#ttnn_layout_lhs_2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_rhs_2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_result_2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Check whether the rhs operand is explicitly broadcasted after the workaround has been applied
func.func @test_binary_implicit_broadcast_second_operands(%arg0: tensor<8x16x32xf32, #ttnn_layout_lhs_2>, %arg1: tensor<1x16x32xf32, #ttnn_layout_rhs_2>) -> tensor<8x16x32xf32, #ttnn_layout_result_2> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.remainder"(%arg0, %arg1) : (tensor<8x16x32xf32, #ttnn_layout_lhs_2>, tensor<1x16x32xf32, #ttnn_layout_rhs_2>) -> tensor<8x16x32xf32, #ttnn_layout_result_2>
  // CHECK: "ttnn.remainder"
  // CHECK-SAME: (tensor<8x16x32xf32, {{.*}}>, tensor<8x16x32xf32, {{.*}}>)
  // CHECK-SAME: -> tensor<8x16x32xf32, {{.*}}>
  return %1 : tensor<8x16x32xf32, #ttnn_layout_result_2>
}
