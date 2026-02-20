// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection="ttnn-mode=true" --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals

// CHECK: #layout6 = #ttcore.metal_layout<logical_shape = 4x64x128, dim_alignments = 1x32x32, collapsed_intervals

// CHECK: #layout7 = #ttcore.metal_layout<logical_shape = 4x64x128, dim_alignments = 1x32x32, collapsed_intervals

// CHECK: #layout8 = #ttcore.metal_layout<logical_shape = 4x64x128, dim_alignments = 1x32x32, collapsed_intervals

// CHECK: #layout9 = #ttcore.metal_layout<logical_shape = 4x64x128, dim_alignments = 1x32x32, collapsed_intervals

// CHECK: #layout10 = #ttcore.metal_layout<logical_shape = 4x64x128, dim_alignments = 1x32x32, collapsed_intervals

// CHECK: #layout11 = #ttcore.metal_layout<logical_shape = 1x8x32x256, dim_alignments = 1x1x32x32, collapsed_intervals

// CHECK: #layout12 = #ttcore.metal_layout<logical_shape = 1x8x32x256, dim_alignments = 1x1x32x32, collapsed_intervals

// CHECK: #layout13 = #ttcore.metal_layout<logical_shape = 1x8x32x256, dim_alignments = 1x1x32x32, collapsed_intervals

// CHECK: #layout14 = #ttcore.metal_layout<logical_shape = 1x8x32x256, dim_alignments = 1x1x32x32, collapsed_intervals



// Block Sharded - Rank 2 layouts
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
// 256x256 on 8x8
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x1
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 1x4
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x4>, memref<8x2x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 256x256 on 4x1
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x1>, memref<2x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<l1>>, <block_sharded>>


// Block Sharded - Rank 3 layouts
// 4x64x128 on 2x2
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <2x2>, memref<4x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 4x64x128 on 1x1
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 4x64x128 on 4x1
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <4x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 4x64x128 on 1x4
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x4>, memref<8x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>

// Block Sharded - Rank 4 layouts
// 1x8x32x256 on 8x1
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 1x8x32x256 on 1x8
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x8>, memref<8x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 1x8x32x256 on 8x8
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>
// 1x8x32x256 on 1x1
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x8x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>>

module {
  // CHECK-LABEL: func.func @test_lower_block_sharded_l1
func.func @test_lower_block_sharded_l1(
  %arg0: tensor<32x32xf32, #ttnn_layout>, %out: tensor<32x32xf32, #ttnn_layout>
) -> tensor<32x32xf32, #ttnn_layout> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout1>
  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: outs(%[[EMPTY0]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout1>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<32x32xf32, #ttnn_layout>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<32x32xf32, #ttnn_layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout1> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> -> tensor<32x32xf32, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<32x32xf32, #ttnn_layout>) -> (tensor<32x32xf32>)
  %2 = ttir.empty() : tensor<32x32xf32, #ttnn_layout>
  %3 = ttir.to_layout %1, %2 : tensor<32x32xf32> into tensor<32x32xf32, #ttnn_layout> -> tensor<32x32xf32, #ttnn_layout>
  // CHECK: return %[[CAST2]] : tensor<32x32xf32, #ttnn_layout>
    return %3 : tensor<32x32xf32, #ttnn_layout>
  }


  // CHECK-LABEL: func.func @test_lower_block_sharded_l1_1
func.func @test_lower_block_sharded_l1_1(
  %arg0: tensor<256x256xf32, #ttnn_layout1>, %out: tensor<256x256xf32, #ttnn_layout1>
) -> tensor<256x256xf32, #ttnn_layout1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout1> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[CAST0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2> -> tensor<256x256xf32, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<256x256xf32, #ttnn_layout1>) -> (tensor<256x256xf32>)
  %2 = ttir.empty() : tensor<256x256xf32, #ttnn_layout1>
  %3 = ttir.to_layout %1, %2 : tensor<256x256xf32> into tensor<256x256xf32, #ttnn_layout1> -> tensor<256x256xf32, #ttnn_layout1>
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout1>
    return %3 : tensor<256x256xf32, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_2
func.func @test_lower_block_sharded_l1_2(
  %arg0: tensor<256x256xf32, #ttnn_layout2>, %out: tensor<256x256xf32, #ttnn_layout2>
) -> tensor<256x256xf32, #ttnn_layout2> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout2> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout3>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout3>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<256x256xf32, #ttnn_layout2>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<256x256xf32, #ttnn_layout2> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<256x256xf32, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0)  : (tensor<256x256xf32, #ttnn_layout2>) -> (tensor<256x256xf32>)
  %2 = ttir.empty() : tensor<256x256xf32, #ttnn_layout2>
  %3 = ttir.to_layout %1, %2 : tensor<256x256xf32> into tensor<256x256xf32, #ttnn_layout2> -> tensor<256x256xf32, #ttnn_layout2>
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout2>
    return %3 : tensor<256x256xf32, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_3
func.func @test_lower_block_sharded_l1_3(
  %arg0: tensor<256x256xf32, #ttnn_layout3>, %out: tensor<256x256xf32, #ttnn_layout3>
) -> tensor<256x256xf32, #ttnn_layout3> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout3> -> tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout4>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout4>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<256x256xf32, #ttnn_layout3>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<256x256xf32, #ttnn_layout3> -> tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2> into tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x4x8x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<256x256xf32, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0)  : (tensor<256x256xf32, #ttnn_layout3>) -> (tensor<256x256xf32>)
  %2 = ttir.empty() : tensor<256x256xf32, #ttnn_layout3>
  %3 = ttir.to_layout %1, %2 : tensor<256x256xf32> into tensor<256x256xf32, #ttnn_layout3> -> tensor<256x256xf32, #ttnn_layout3>
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout3>
    return %3 : tensor<256x256xf32, #ttnn_layout3>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_4
func.func @test_lower_block_sharded_l1_4(
  %arg0: tensor<256x256xf32, #ttnn_layout4>, %out: tensor<256x256xf32, #ttnn_layout4>
) -> tensor<256x256xf32, #ttnn_layout4> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<256x256xf32, #ttnn_layout4> -> tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout5>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout5>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<256x256xf32, #ttnn_layout4>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<256x256xf32, #ttnn_layout4> -> tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, f32>, #layout2> into tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<4x1x2x8x!ttcore.tile<32x32, f32>, #layout2> -> tensor<256x256xf32, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0)  : (tensor<256x256xf32, #ttnn_layout4>) -> (tensor<256x256xf32>)
  %2 = ttir.empty() : tensor<256x256xf32, #ttnn_layout4>
  %3 = ttir.to_layout %1, %2 : tensor<256x256xf32> into tensor<256x256xf32, #ttnn_layout4> -> tensor<256x256xf32, #ttnn_layout4>
  // CHECK: return %[[CAST2]] : tensor<256x256xf32, #ttnn_layout4>
    return %3 : tensor<256x256xf32, #ttnn_layout4>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_5
func.func @test_lower_block_sharded_l1_5(
  %arg0: tensor<4x64x128xbf16, #ttnn_layout5>, %out: tensor<4x64x128xbf16, #ttnn_layout5>
) -> tensor<4x64x128xbf16, #ttnn_layout5> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<4x64x128xbf16, #ttnn_layout5> -> tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout7>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x4>
  // CHECK: ins(%[[VIEW0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout7>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<4x64x128xbf16, #ttnn_layout5>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<4x64x128xbf16, #ttnn_layout5> -> tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6> into tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<2x2x4x2x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<4x64x128xbf16, #ttnn_layout5>

  %1 = "ttir.abs"(%arg0)  : (tensor<4x64x128xbf16, #ttnn_layout5>) -> (tensor<4x64x128xbf16>)
  %2 = ttir.empty() : tensor<4x64x128xbf16, #ttnn_layout5>
  %3 = ttir.to_layout %1, %2 : tensor<4x64x128xbf16> into tensor<4x64x128xbf16, #ttnn_layout5> -> tensor<4x64x128xbf16, #ttnn_layout5>
  // CHECK: return %[[CAST2]] : tensor<4x64x128xbf16, #ttnn_layout5>
  return %3 : tensor<4x64x128xbf16, #ttnn_layout5>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_6
func.func @test_lower_block_sharded_l1_6(
  %arg0: tensor<4x64x128xbf16, #ttnn_layout6>, %out: tensor<4x64x128xbf16, #ttnn_layout6>
) -> tensor<4x64x128xbf16, #ttnn_layout6> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<4x64x128xbf16, #ttnn_layout6> -> tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout8>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x4>
  // CHECK: ins(%[[VIEW0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout8>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<4x64x128xbf16, #ttnn_layout6>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<4x64x128xbf16, #ttnn_layout6> -> tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6> into tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x1x8x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<4x64x128xbf16, #ttnn_layout6>
  %1 = "ttir.abs"(%arg0)  : (tensor<4x64x128xbf16, #ttnn_layout6>) -> (tensor<4x64x128xbf16>)
  %2 = ttir.empty() : tensor<4x64x128xbf16, #ttnn_layout6>
  %3 = ttir.to_layout %1, %2 : tensor<4x64x128xbf16> into tensor<4x64x128xbf16, #ttnn_layout6> -> tensor<4x64x128xbf16, #ttnn_layout6>
  // CHECK: return %[[CAST2]] : tensor<4x64x128xbf16, #ttnn_layout6>
  return %3 : tensor<4x64x128xbf16, #ttnn_layout6>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_7
func.func @test_lower_block_sharded_l1_7(
  %arg0: tensor<4x64x128xbf16, #ttnn_layout7>, %out: tensor<4x64x128xbf16, #ttnn_layout7>
) -> tensor<4x64x128xbf16, #ttnn_layout7> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<4x64x128xbf16, #ttnn_layout7> -> tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout9>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x4>
  // CHECK: ins(%[[VIEW0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout9>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<4x64x128xbf16, #ttnn_layout7>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<4x64x128xbf16, #ttnn_layout7> -> tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6> into tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<4x1x2x4x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<4x64x128xbf16, #ttnn_layout7>
  %1 = "ttir.abs"(%arg0)  : (tensor<4x64x128xbf16, #ttnn_layout7>) -> (tensor<4x64x128xbf16>)
  %2 = ttir.empty() : tensor<4x64x128xbf16, #ttnn_layout7>
  %3 = ttir.to_layout %1, %2 : tensor<4x64x128xbf16> into tensor<4x64x128xbf16, #ttnn_layout7> -> tensor<4x64x128xbf16, #ttnn_layout7>
  // CHECK: return %[[CAST2]] : tensor<4x64x128xbf16, #ttnn_layout7>
  return %3 : tensor<4x64x128xbf16, #ttnn_layout7>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_8
func.func @test_lower_block_sharded_l1_8(
  %arg0: tensor<4x64x128xbf16, #ttnn_layout8>, %out: tensor<4x64x128xbf16, #ttnn_layout8>
) -> tensor<4x64x128xbf16, #ttnn_layout8> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<4x64x128xbf16, #ttnn_layout8> -> tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout10>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x4>
  // CHECK: ins(%[[VIEW0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout10>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<4x64x128xbf16, #ttnn_layout8>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<4x64x128xbf16, #ttnn_layout8> -> tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x4x1x1x!ttcore.tile<32x32, bf16>, #layout6> into tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x4x8x1x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<4x64x128xbf16, #ttnn_layout8>
  %1 = "ttir.abs"(%arg0)  : (tensor<4x64x128xbf16, #ttnn_layout8>) -> (tensor<4x64x128xbf16>)
  %2 = ttir.empty() : tensor<4x64x128xbf16, #ttnn_layout8>
  %3 = ttir.to_layout %1, %2 : tensor<4x64x128xbf16> into tensor<4x64x128xbf16, #ttnn_layout8> -> tensor<4x64x128xbf16, #ttnn_layout8>
  // CHECK: return %[[CAST2]] : tensor<4x64x128xbf16, #ttnn_layout8>
  return %3 : tensor<4x64x128xbf16, #ttnn_layout8>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_9
func.func @test_lower_block_sharded_l1_9(
  %arg0: tensor<1x8x32x256xbf16, #ttnn_layout9>, %out: tensor<1x8x32x256xbf16, #ttnn_layout9>
) -> tensor<1x8x32x256xbf16, #ttnn_layout9> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x8x32x256xbf16, #ttnn_layout9> -> tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout12>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout12>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<1x8x32x256xbf16, #ttnn_layout9>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<1x8x32x256xbf16, #ttnn_layout9> -> tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11> into tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<8x1x1x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x8x32x256xbf16, #ttnn_layout9>
  %1 = "ttir.abs"(%arg0)  : (tensor<1x8x32x256xbf16, #ttnn_layout9>) -> (tensor<1x8x32x256xbf16>)
  %2 = ttir.empty() : tensor<1x8x32x256xbf16, #ttnn_layout9>
  %3 = ttir.to_layout %1, %2 : tensor<1x8x32x256xbf16> into tensor<1x8x32x256xbf16, #ttnn_layout9> -> tensor<1x8x32x256xbf16, #ttnn_layout9>
  // CHECK: return %[[CAST2]] : tensor<1x8x32x256xbf16, #ttnn_layout9>
  return %3 : tensor<1x8x32x256xbf16, #ttnn_layout9>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_10
func.func @test_lower_block_sharded_l1_10(
  %arg0: tensor<1x8x32x256xbf16, #ttnn_layout10>, %out: tensor<1x8x32x256xbf16, #ttnn_layout10>
) -> tensor<1x8x32x256xbf16, #ttnn_layout10> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x8x32x256xbf16, #ttnn_layout10> -> tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout13>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout13>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<1x8x32x256xbf16, #ttnn_layout10>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<1x8x32x256xbf16, #ttnn_layout10> -> tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11> into tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x8x8x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x8x32x256xbf16, #ttnn_layout10>
  %1 = "ttir.abs"(%arg0)  : (tensor<1x8x32x256xbf16, #ttnn_layout10>) -> (tensor<1x8x32x256xbf16>)
  %2 = ttir.empty() : tensor<1x8x32x256xbf16, #ttnn_layout10>
  %3 = ttir.to_layout %1, %2 : tensor<1x8x32x256xbf16> into tensor<1x8x32x256xbf16, #ttnn_layout10> -> tensor<1x8x32x256xbf16, #ttnn_layout10>
  // CHECK: return %[[CAST2]] : tensor<1x8x32x256xbf16, #ttnn_layout10>
  return %3 : tensor<1x8x32x256xbf16, #ttnn_layout10>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_11
func.func @test_lower_block_sharded_l1_11(
  %arg0: tensor<1x8x32x256xbf16, #ttnn_layout11>, %out: tensor<1x8x32x256xbf16, #ttnn_layout11>
) -> tensor<1x8x32x256xbf16, #ttnn_layout11> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x8x32x256xbf16, #ttnn_layout11> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[CAST0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x8x32x256xbf16, #ttnn_layout11>
  %1 = "ttir.abs"(%arg0)  : (tensor<1x8x32x256xbf16, #ttnn_layout11>) -> (tensor<1x8x32x256xbf16>)
  %2 = ttir.empty() : tensor<1x8x32x256xbf16, #ttnn_layout11>
  %3 = ttir.to_layout %1, %2 : tensor<1x8x32x256xbf16> into tensor<1x8x32x256xbf16, #ttnn_layout11> -> tensor<1x8x32x256xbf16, #ttnn_layout11>
  // CHECK: return %[[CAST2]] : tensor<1x8x32x256xbf16, #ttnn_layout11>
  return %3 : tensor<1x8x32x256xbf16, #ttnn_layout11>
  }

// CHECK-LABEL: func.func @test_lower_block_sharded_l1_12
func.func @test_lower_block_sharded_l1_12(
  %arg0: tensor<1x8x32x256xbf16, #ttnn_layout12>, %out: tensor<1x8x32x256xbf16, #ttnn_layout12>
) -> tensor<1x8x32x256xbf16, #ttnn_layout12> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1x8x32x256xbf16, #ttnn_layout12> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout14>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout14>)
  // CHECK: outs(%[[EMPTY0]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<1x8x32x256xbf16, #ttnn_layout12>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<1x8x32x256xbf16, #ttnn_layout12> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<8x8x1x1x!ttcore.tile<32x32, bf16>, #layout11> into tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<1x8x32x256xbf16, #ttnn_layout12>
  %1 = "ttir.abs"(%arg0)  : (tensor<1x8x32x256xbf16, #ttnn_layout12>) -> (tensor<1x8x32x256xbf16>)
  %2 = ttir.empty() : tensor<1x8x32x256xbf16, #ttnn_layout12>
  %3 = ttir.to_layout %1, %2 : tensor<1x8x32x256xbf16> into tensor<1x8x32x256xbf16, #ttnn_layout12> -> tensor<1x8x32x256xbf16, #ttnn_layout12>
  // CHECK: return %[[CAST2]] : tensor<1x8x32x256xbf16, #ttnn_layout12>
  return %3 : tensor<1x8x32x256xbf16, #ttnn_layout12>
  }
}
