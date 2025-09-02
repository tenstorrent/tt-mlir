// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_lhs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_rhs = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_result = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

// Check whether the lhs operand is explicitly broadcasted after the workaround has been applied
func.func @test_subtract_to_add_and_neg_workaround(%arg0: tensor<1x16x32xf32, #ttnn_layout_lhs>, %arg1: tensor<8x16x32xf32, #ttnn_layout_rhs>) -> tensor<8x16x32xf32, #ttnn_layout_result> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.subtract"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x16x32xf32, #ttnn_layout_lhs>, tensor<8x16x32xf32, #ttnn_layout_rhs>) -> tensor<8x16x32xf32, #ttnn_layout_result>
  // CHECK: = "ttnn.neg"
  // CHECK: = "ttnn.add"
  // CHECK-NOT: = "ttnn.subtract"
  return %1 : tensor<8x16x32xf32, #ttnn_layout_result>
}
