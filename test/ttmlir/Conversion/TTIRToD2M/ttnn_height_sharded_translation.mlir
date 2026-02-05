// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>


// CHECK: #layout = #ttcore.metal_layout<logical_shape = 384x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 2) mod 6, (d0 + d1) mod 2, d2, d3)>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 384x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = map(0)>
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 384x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (d0 * 2 + d2, 0, 0, 0)>
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 384x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 6) mod 2, (d0 + d1) mod 6, d2, d3)>
// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 384x64, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 2) mod 6, (d0 + d1) mod 2, d2, d3)>
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 384x64, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = map(0)>
// CHECK: #layout6 = #ttcore.metal_layout<logical_shape = 384x64, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (d0 * 2 + d2 + d1 floordiv 2, 0, 0, d1 mod 2)>
// CHECK: #layout7 = #ttcore.metal_layout<logical_shape = 384x64, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 6) mod 2, (d0 + d1) mod 6, d2, d3)>
// CHECK: #layout8 = #ttcore.metal_layout<logical_shape = 2x384x32, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 2) mod 6, (d0 + d1) mod 2, d2, d3)>
// CHECK: #layout9 = #ttcore.metal_layout<logical_shape = 2x384x32, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: index_map = map(0)>
// CHECK: #layout10 = #ttcore.metal_layout<logical_shape = 2x384x32, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> ((d0 * 3 + d2) floordiv 2, 0, (d0 * 3 + d2) mod 2, 0)>
// CHECK: #layout11 = #ttcore.metal_layout<logical_shape = 2x2x384x32, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> (((d0 + d1) floordiv 2) mod 6, (d0 + d1) mod 2, d2, d3)>
// CHECK: #layout12 = #ttcore.metal_layout<logical_shape = 2x2x384x32, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: index_map = map(0)>
// CHECK: #layout13 = #ttcore.metal_layout<logical_shape = 2x2x384x32, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> ((d0 * 6 + d2) floordiv 4, 0, (d0 * 6 + d2) mod 4, 0)>
// CHECK: #layout14 = #ttcore.metal_layout<logical_shape = 4096x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> ((d0 + d1) mod 8, 0, d2, d3)>
// CHECK: #layout15 = #ttcore.metal_layout<logical_shape = 4096x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = map(0)>
// CHECK: #layout16 = #ttcore.metal_layout<logical_shape = 4096x32, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: index_map = (d0, d1, d2, d3) -> ((d0 * 2 + d2) floordiv 16, 0, (d0 * 2 + d2) mod 16, 0)>


// Height Sharded - Rank 2 layouts
// 384x32 on 6x2
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x32 on 2x6
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x64 on 6x2
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x64 on 2x6
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>


// Height Sharded - Rank 3 layouts
// 2x384x32 on 6x2
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>

// Height Sharded - Rank 4 layouts
// 2x2x384x32 on 6x2
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <6x2>, memref<4x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>

// Height Sharded - Extreme aspect ratio
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>

module {
  // CHECK-LABEL: func.func @test_lower_height_sharded_l1
func.func @test_lower_height_sharded_l1(
  %arg0: tensor<384x32xbf16, #ttnn_layout>
) -> tensor<384x32xbf16, #ttnn_layout> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x32xbf16, #ttnn_layout> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x32xbf16, #ttnn_layout> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<6x1>
  // CHECK: ins(%[[VIEW2]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[VIEW1]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<384x32xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x32xbf16, #ttnn_layout>) -> (tensor<384x32xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<384x32xbf16, #ttnn_layout>
    return %1 : tensor<384x32xbf16, #ttnn_layout>
  }


  // CHECK-LABEL: func.func @test_lower_height_sharded_l1_1
func.func @test_lower_height_sharded_l1_1(
  %arg0: tensor<384x32xbf16, #ttnn_layout1>) -> tensor<384x32xbf16, #ttnn_layout1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x32xbf16, #ttnn_layout1> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x32xbf16, #ttnn_layout1> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<6x1>
  // CHECK: ins(%[[VIEW2]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[VIEW1]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<6x1x2x1x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<384x32xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x32xbf16, #ttnn_layout1>) -> (tensor<384x32xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<384x32xbf16, #ttnn_layout1>
    return %1 : tensor<384x32xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_2
func.func @test_lower_height_sharded_l1_2(
  %arg0: tensor<384x64xbf16, #ttnn_layout2>) -> tensor<384x64xbf16, #ttnn_layout2> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x64xbf16, #ttnn_layout2> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout4>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout4> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout5>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x64xbf16, #ttnn_layout2> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout4>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout4> -> tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout5> -> tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<6x2>
  // CHECK: ins(%[[VIEW2]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK: outs(%[[VIEW1]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<384x64xbf16, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x64xbf16, #ttnn_layout2>) -> (tensor<384x64xbf16, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<384x64xbf16, #ttnn_layout2>
    return %1 : tensor<384x64xbf16, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_3
func.func @test_lower_height_sharded_l1_3(
  %arg0: tensor<384x64xbf16, #ttnn_layout3>) -> tensor<384x64xbf16, #ttnn_layout3> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x64xbf16, #ttnn_layout3> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout7>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout7> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout5>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x64xbf16, #ttnn_layout3> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout7>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout7> -> tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout5> -> tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<6x2>
  // CHECK: ins(%[[VIEW2]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK: outs(%[[VIEW1]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<6x2x2x1x!ttcore.tile<32x32, bf16>, #layout6> -> tensor<384x64xbf16, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x64xbf16, #ttnn_layout3>) -> (tensor<384x64xbf16, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<384x64xbf16, #ttnn_layout3>
    return %1 : tensor<384x64xbf16, #ttnn_layout3>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_4
func.func @test_lower_height_sharded_l1_4(
  %arg0: tensor<2x384x32xbf16, #ttnn_layout4>) -> tensor<2x384x32xbf16, #ttnn_layout4> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x384x32xbf16, #ttnn_layout4> -> tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout8>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout8> -> tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout9>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x384x32xbf16, #ttnn_layout4> -> tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout8>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout8> -> tensor<8x1x3x1x!ttcore.tile<32x32, bf16>, #layout10>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout9> -> tensor<8x1x3x1x!ttcore.tile<32x32, bf16>, #layout10>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x1>
  // CHECK: ins(%[[VIEW2]] : tensor<8x1x3x1x!ttcore.tile<32x32, bf16>, #layout10>)
  // CHECK: outs(%[[VIEW1]] : tensor<8x1x3x1x!ttcore.tile<32x32, bf16>, #layout10>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x1x3x1x!ttcore.tile<32x32, bf16>, #layout10> -> tensor<2x384x32xbf16, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x384x32xbf16, #ttnn_layout4>) -> (tensor<2x384x32xbf16, #ttnn_layout4>)
  // CHECK: return %[[CAST2]] : tensor<2x384x32xbf16, #ttnn_layout4>
    return %1 : tensor<2x384x32xbf16, #ttnn_layout4>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_5
func.func @test_lower_height_sharded_l1_5(
  %arg0: tensor<2x2x384x32xbf16, #ttnn_layout5>) -> tensor<2x2x384x32xbf16, #ttnn_layout5> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x384x32xbf16, #ttnn_layout5> -> tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout12>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x2x384x32xbf16, #ttnn_layout5> -> tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout11>
  // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout11> -> tensor<8x1x6x1x!ttcore.tile<32x32, bf16>, #layout13>
  // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout12> -> tensor<8x1x6x1x!ttcore.tile<32x32, bf16>, #layout13>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x1>
  // CHECK: ins(%[[VIEW2]] : tensor<8x1x6x1x!ttcore.tile<32x32, bf16>, #layout13>)
  // CHECK: outs(%[[VIEW1]] : tensor<8x1x6x1x!ttcore.tile<32x32, bf16>, #layout13>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x1x6x1x!ttcore.tile<32x32, bf16>, #layout13> -> tensor<2x2x384x32xbf16, #ttnn_layout5>

  %1 = "ttir.abs"(%arg0)  : (tensor<2x2x384x32xbf16, #ttnn_layout5>) -> (tensor<2x2x384x32xbf16, #ttnn_layout5>)
  // CHECK: return %[[CAST2]] : tensor<2x2x384x32xbf16, #ttnn_layout5>
  return %1 : tensor<2x2x384x32xbf16, #ttnn_layout5>
  }

  // CHECK-LABEL: func.func @test_lower_height_sharded_l1_6
  func.func @test_lower_height_sharded_l1_6(
    %arg0: tensor<4096x32xbf16, #ttnn_layout6>) -> tensor<4096x32xbf16, #ttnn_layout6> {
    // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<4096x32xbf16, #ttnn_layout6> -> tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout14>
    // CHECK-DAG: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout14> -> tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout15>
    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<4096x32xbf16, #ttnn_layout6> -> tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout14>
    // CHECK-DAG: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] : tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout14> -> tensor<64x1x2x1x!ttcore.tile<32x32, bf16>, #layout16>
    // CHECK-DAG: %[[VIEW2:.*]] = d2m.view_layout %[[VIEW0]] : tensor<8x1x16x1x!ttcore.tile<32x32, bf16>, #layout15> -> tensor<64x1x2x1x!ttcore.tile<32x32, bf16>, #layout16>
    // CHECK: %[[RESULT:.*]] = d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<64x1, (d0, d1) ->
    // CHECK: ins(%[[VIEW2]] : tensor<64x1x2x1x!ttcore.tile<32x32, bf16>, #layout16>)
    // CHECK: outs(%[[VIEW1]] : tensor<64x1x2x1x!ttcore.tile<32x32, bf16>, #layout16>)
    // CHECK-DAG: d2m.tile_abs
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<64x1x2x1x!ttcore.tile<32x32, bf16>, #layout16> -> tensor<4096x32xbf16, #ttnn_layout6>

    %1 = "ttir.abs"(%arg0)  : (tensor<4096x32xbf16, #ttnn_layout6>) -> (tensor<4096x32xbf16, #ttnn_layout6>)
    // CHECK: return %[[CAST2]] : tensor<4096x32xbf16, #ttnn_layout6>
    return %1 : tensor<4096x32xbf16, #ttnn_layout6>
  }
}
