// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttir-generic="ttnn-mode=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals

// 256x256 on 8x8
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x1
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x4
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<8x2x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 4x1
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x1>, memref<2x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>


module {
  // CHECK-LABEL: func.func @test_lower_block_sharded_l1
func.func @test_lower_block_sharded_l1(
  %arg0: tensor<32x32xf32, #ttnn_layout>, %out: tensor<32x32xf32, #ttnn_layout>
) -> tensor<32x32xf32, #ttnn_layout> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[RESULT:.*]] = ttir.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: outs(%[[CAST1]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK-DAG: ttir.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<32x32xf32, #ttnn_layout>, tensor<32x32xf32, #ttnn_layout>) -> (tensor<32x32xf32, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<32x32xf32, #ttnn_layout>
    return %1 : tensor<32x32xf32, #ttnn_layout>
  }


  // CHECK-LABEL: func.func @test_lower_block_sharded_l1_1
func.func @test_lower_block_sharded_l1_1(
  %arg0: tensor<256x256xf32, #ttnn_layout1>, %out: tensor<256x256xf32, #ttnn_layout1>
) -> tensor<256x256xf32, #ttnn_layout1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout1> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<256x256xf32, #ttnn_layout1> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[RESULT:.*]] = ttir.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK: outs(%[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK-DAG: ttir.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout1> -> tensor<256x256xf32, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<256x256xf32, #ttnn_layout1>, tensor<256x256xf32, #ttnn_layout1>) -> (tensor<256x256xf32, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout1>
    return %1 : tensor<256x256xf32, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_2
func.func @test_lower_block_sharded_l1_2(
  %arg0: tensor<256x256xf32, #ttnn_layout2>, %out: tensor<256x256xf32, #ttnn_layout2>
) -> tensor<256x256xf32, #ttnn_layout2> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout2> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<256x256xf32, #ttnn_layout2> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[RESULT:.*]] = ttir.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK: outs(%[[CAST1]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK-DAG: ttir.tile_abs
  // CHECK: = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout1> -> tensor<256x256xf32, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<256x256xf32, #ttnn_layout2>, tensor<256x256xf32, #ttnn_layout2>) -> (tensor<256x256xf32, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout2>
    return %1 : tensor<256x256xf32, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_3
func.func @test_lower_block_sharded_l1_3(
  %arg0: tensor<256x256xf32, #ttnn_layout3>, %out: tensor<256x256xf32, #ttnn_layout3>
) -> tensor<256x256xf32, #ttnn_layout3> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout3> -> tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<256x256xf32, #ttnn_layout3> -> tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[RESULT:.*]] = ttir.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK: outs(%[[CAST1]] : tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK-DAG: ttir.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<256x256xf32, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<256x256xf32, #ttnn_layout3>, tensor<256x256xf32, #ttnn_layout3>) -> (tensor<256x256xf32, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout3>
    return %1 : tensor<256x256xf32, #ttnn_layout3>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_4
func.func @test_lower_block_sharded_l1_4(
  %arg0: tensor<256x256xf32, #ttnn_layout4>, %out: tensor<256x256xf32, #ttnn_layout4>
) -> tensor<256x256xf32, #ttnn_layout4> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout4> -> tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<256x256xf32, #ttnn_layout4> -> tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[RESULT:.*]] = ttir.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK: outs(%[[CAST1]] : tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK-DAG: ttir.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout1> -> tensor<256x256xf32, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<256x256xf32, #ttnn_layout4>, tensor<256x256xf32, #ttnn_layout4>) -> (tensor<256x256xf32, #ttnn_layout4>)
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout4>
    return %1 : tensor<256x256xf32, #ttnn_layout4>
  }
}
