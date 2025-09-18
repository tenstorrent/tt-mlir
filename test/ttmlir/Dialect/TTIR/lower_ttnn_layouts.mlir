// RUN: ttmlir-opt -ttir-to-ttir-generic -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Single-case test: block-sharded L1 tiled TTNN layout is lowered to an
// explicit ttir.ttnn_to_metal_layout_cast with a ttcore.metal_layout encoding.

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals =
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>

// 256x256 on 8x8
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x1
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x4
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<8x2x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 4x1
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x1>, memref<2x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 4x4
// #ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x4>, memref<2x2x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>


module {
// CHECK-LABEL: func.func @test_lower_block_sharded_l1
func.func @test_lower_block_sharded_l1(
  %arg0: tensor<32x32xf32, #ttnn_layout1>
) -> tensor<32x32xf32, #ttnn_layout1> {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<32x32xf32, #ttnn_layout> -> tensor<32x32xf32, #layout>

  // And the ttir.abs should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttir.abs"(%[[CAST]], %[[CAST]])
  // CHECK-SAME: : (tensor<32x32xf32, #layout>, tensor<32x32xf32, #layout>) -> ()
  %0 = ttir.empty() : tensor<32x32xf32, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0, %0)  : (tensor<32x32xf32, #ttnn_layout1>, tensor<32x32xf32, #ttnn_layout1>) -> (tensor<32x32xf32, #ttnn_layout1>)

  return %1 : tensor<32x32xf32, #ttnn_layout1>
  }
// CHECK-LABEL: func.func @test_lower_block_sharded_l1_2
func.func @test_lower_block_sharded_l1_2(
  %arg0: tensor<256x256xf32, #ttnn_layout2>
) -> tensor<256x256xf32, #ttnn_layout2> {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<256x256xf32, #ttnn_layout1> -> tensor<256x256xf32, #layout1>

  // And the ttir.abs should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttir.abs"(%[[CAST]], %[[CAST]])
  // CHECK-SAME: : (tensor<256x256xf32, #layout1>, tensor<256x256xf32, #layout1>) -> ()
  %0 = ttir.empty() : tensor<256x256xf32, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0, %0)  : (tensor<256x256xf32, #ttnn_layout2>, tensor<256x256xf32, #ttnn_layout2>) -> (tensor<256x256xf32, #ttnn_layout2>)

    return %1 : tensor<256x256xf32, #ttnn_layout2>
  }


// CHECK-LABEL: func.func @test_lower_block_sharded_l1_3
func.func @test_lower_block_sharded_l1_3(
  %arg0: tensor<256x256xf32, #ttnn_layout3>
) -> tensor<256x256xf32, #ttnn_layout3> {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<256x256xf32, #ttnn_layout2> -> tensor<256x256xf32, #layout2>

  // And the ttir.abs should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttir.abs"(%[[CAST]], %[[CAST]])
  // CHECK-SAME: : (tensor<256x256xf32, #layout2>, tensor<256x256xf32, #layout2>) -> ()
  %0 = ttir.empty() : tensor<256x256xf32, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0, %0)  : (tensor<256x256xf32, #ttnn_layout3>, tensor<256x256xf32, #ttnn_layout3>) -> (tensor<256x256xf32, #ttnn_layout3>)

  return %1 : tensor<256x256xf32, #ttnn_layout3>
  }
// CHECK-LABEL: func.func @test_lower_block_sharded_l1_4
func.func @test_lower_block_sharded_l1_4(
  %arg0: tensor<256x256xf32, #ttnn_layout4>
) -> tensor<256x256xf32, #ttnn_layout4> {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<256x256xf32, #ttnn_layout3> -> tensor<256x256xf32, #layout3>

  // And the ttir.abs should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttir.abs"(%[[CAST]], %[[CAST]])
  // CHECK-SAME: : (tensor<256x256xf32, #layout3>, tensor<256x256xf32, #layout3>) -> ()

  %0 = ttir.empty() : tensor<256x256xf32, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0, %0)  : (tensor<256x256xf32, #ttnn_layout4>, tensor<256x256xf32, #ttnn_layout4>) -> (tensor<256x256xf32, #ttnn_layout4>)

    return %1 : tensor<256x256xf32, #ttnn_layout4>
  }
// CHECK-LABEL: func.func @test_lower_block_sharded_l1_5
func.func @test_lower_block_sharded_l1_5(
  %arg0: tensor<256x256xf32, #ttnn_layout5>
) -> tensor<256x256xf32, #ttnn_layout5> {
  // Expect the pass to insert a single cast op converting the TTNN layout to a TTCore metal layout.
  // (Alias asserted above; ensure the cast uses it.)
  // CHECK: %[[CAST:.*]] = ttir.ttnn_to_metal_layout_cast %arg0
  // CHECK-SAME: : tensor<256x256xf32, #ttnn_layout4> -> tensor<256x256xf32, #layout4>

  // And the ttir.abs should consume that cast value for both operands with metal_layout-typed tensors.
  // CHECK: "ttir.abs"(%[[CAST]], %[[CAST]])
  // CHECK-SAME: : (tensor<256x256xf32, #layout4>, tensor<256x256xf32, #layout4>) -> ()

  %0 = ttir.empty() : tensor<256x256xf32, #ttnn_layout5>
  %1 = "ttir.abs"(%arg0, %0)  : (tensor<256x256xf32, #ttnn_layout5>, tensor<256x256xf32, #ttnn_layout5>) -> (tensor<256x256xf32, #ttnn_layout5>)

    return %1 : tensor<256x256xf32, #ttnn_layout5>
  }
}
