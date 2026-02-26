// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection="ttnn-mode=true" --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK-DAG: #layout{{[0-9]*}} = #ttcore.metal_layout<logical_shape = 32x384, dim_alignments = 32x32, collapsed_intervals
// CHECK-DAG-SAME: undef, l1, sharded>
// CHECK-DAG: #layout{{[0-9]*}} = #ttcore.metal_layout<logical_shape = 64x384, dim_alignments = 32x32, collapsed_intervals
// CHECK-DAG-SAME: undef, l1, sharded>
// CHECK-DAG: #layout{{[0-9]*}} = #ttcore.metal_layout<logical_shape = 2x32x384, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-DAG-SAME: undef, l1, sharded>
// CHECK-DAG: #layout{{[0-9]*}} = #ttcore.metal_layout<logical_shape = 2x2x32x384, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-DAG-SAME: undef, l1, sharded>


// Width Sharded - Rank 2 layouts
// 32x384 on 6x2
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>
// 32x384 on 2x6
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>
// 64x384 on 6x2
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>
// 64x384 on 2x6
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>


// Width Sharded - Rank 3 layouts
// 2x32x384 on 6x2
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>

// Width Sharded - Rank 4 layouts
// 2x2x32x384 on 6x2
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <6x2>, memref<4x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>

// Width Sharded - Extreme aspect ratio
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x8>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, exactGrid = true>


module {
  // CHECK-LABEL: func.func @test_lower_width_sharded_l1
func.func @test_lower_width_sharded_l1(
  %arg0: tensor<32x384xbf16, #ttnn_layout>) -> tensor<32x384xbf16, #ttnn_layout> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x384xbf16, #ttnn_layout> -> tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<32x384xbf16, #ttnn_layout> -> tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<32x384xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<32x384xbf16, #ttnn_layout>) -> (tensor<32x384xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<32x384xbf16, #ttnn_layout>
    return %3 : tensor<32x384xbf16, #ttnn_layout>
  }


  // CHECK-LABEL: func.func @test_lower_width_sharded_l1_1
func.func @test_lower_width_sharded_l1_1(
  %arg0: tensor<32x384xbf16, #ttnn_layout1>) -> tensor<32x384xbf16, #ttnn_layout1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x384xbf16, #ttnn_layout1> -> tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<32x384xbf16, #ttnn_layout1> -> tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<32x384xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<32x384xbf16, #ttnn_layout1>) -> (tensor<32x384xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<32x384xbf16, #ttnn_layout1>
    return %3 : tensor<32x384xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_width_sharded_l1_2
func.func @test_lower_width_sharded_l1_2(
  %arg0: tensor<64x384xbf16, #ttnn_layout2>) -> tensor<64x384xbf16, #ttnn_layout2> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x384xbf16, #ttnn_layout2> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<64x384xbf16, #ttnn_layout2> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<64x384xbf16, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0)  : (tensor<64x384xbf16, #ttnn_layout2>) -> (tensor<64x384xbf16, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<64x384xbf16, #ttnn_layout2>
    return %3 : tensor<64x384xbf16, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_width_sharded_l1_3
func.func @test_lower_width_sharded_l1_3(
  %arg0: tensor<64x384xbf16, #ttnn_layout3>
) -> tensor<64x384xbf16, #ttnn_layout3> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x384xbf16, #ttnn_layout3> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<64x384xbf16, #ttnn_layout3> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<64x384xbf16, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0)  : (tensor<64x384xbf16, #ttnn_layout3>) -> (tensor<64x384xbf16, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<64x384xbf16, #ttnn_layout3>
    return %3 : tensor<64x384xbf16, #ttnn_layout3>
  }

// CHECK-LABEL: func.func @test_lower_width_sharded_l1_4
func.func @test_lower_width_sharded_l1_4(
  %arg0: tensor<2x32x384xbf16, #ttnn_layout4>
) -> tensor<2x32x384xbf16, #ttnn_layout4> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x32x384xbf16, #ttnn_layout4> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x32x384xbf16, #ttnn_layout4> -> tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x2x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<2x32x384xbf16, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x32x384xbf16, #ttnn_layout4>) -> (tensor<2x32x384xbf16, #ttnn_layout4>)
  // CHECK: return %[[CAST2]] : tensor<2x32x384xbf16, #ttnn_layout4>
    return %3 : tensor<2x32x384xbf16, #ttnn_layout4>
  }

// CHECK-LABEL: func.func @test_lower_width_sharded_l1_5
func.func @test_lower_width_sharded_l1_5(
  %arg0: tensor<2x2x32x384xbf16, #ttnn_layout5>
) -> tensor<2x2x32x384xbf16, #ttnn_layout5> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x32x384xbf16, #ttnn_layout5> -> tensor<1x12x4x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x2x32x384xbf16, #ttnn_layout5> -> tensor<1x12x4x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x12
  // CHECK: ins(%[[CAST0]] : tensor<1x12x4x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<1x12x4x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x12x4x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}> -> tensor<2x2x32x384xbf16, #ttnn_layout5>

  %1 = "ttir.abs"(%arg0)  : (tensor<2x2x32x384xbf16, #ttnn_layout5>) -> (tensor<2x2x32x384xbf16>)
  %2 = ttir.empty() : tensor<2x2x32x384xbf16, #ttnn_layout5>
  %3 = ttir.to_layout %1, %2 : tensor<2x2x32x384xbf16> into tensor<2x2x32x384xbf16, #ttnn_layout5> -> tensor<2x2x32x384xbf16, #ttnn_layout5>
  // CHECK: return %[[CAST2]] : tensor<2x2x32x384xbf16, #ttnn_layout5>
  return %3 : tensor<2x2x32x384xbf16, #ttnn_layout5>
  }

  // CHECK-LABEL: func.func @test_lower_width_sharded_l1_6
  func.func @test_lower_width_sharded_l1_6(
    %arg0: tensor<32x4096xbf16, #ttnn_layout6>
  ) -> tensor<32x4096xbf16, #ttnn_layout6> {
    // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x4096xbf16, #ttnn_layout6> -> tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14>
    // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] : tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14> -> tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout15>
    // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<1x64x1x2x!ttcore.tile<32x32, bf16>, #layout16>
    // CHECK: %[[VIEW1:.*]] = d2m.view_layout %[[VIEW0]] : tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout15> -> tensor<1x64x1x2x!ttcore.tile<32x32, bf16>, #layout17>
    // CHECK: %[[RESULT:.*]] = d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x64, (d0, d1) ->
    // CHECK: ins(%[[VIEW1]] : tensor<1x64x1x2x!ttcore.tile<32x32, bf16>, #layout17>)
    // CHECK: outs(%[[EMPTY0]] : tensor<1x64x1x2x!ttcore.tile<32x32, bf16>, #layout16>)
    // CHECK-DAG: d2m.tile_abs
    // CHECK: %[[EMPTY1:.*]] = d2m.empty() : tensor<32x4096xbf16, #ttnn_layout6>
    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY1]] : tensor<32x4096xbf16, #ttnn_layout6> -> tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14>
    // CHECK: %[[LAYOUT:.*]] = d2m.to_layout %[[RESULT]], %[[CAST1]] : tensor<1x64x1x2x!ttcore.tile<32x32, bf16>, #layout16> into tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14> -> tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14>
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[LAYOUT]] : tensor<1x8x1x16x!ttcore.tile<32x32, bf16>, #layout14> -> tensor<32x4096xbf16, #ttnn_layout6>

    %1 = "ttir.abs"(%arg0)  : (tensor<32x4096xbf16, #ttnn_layout6>) -> (tensor<32x4096xbf16>)
    %2 = ttir.empty() : tensor<32x4096xbf16, #ttnn_layout6>
    %3 = ttir.to_layout %1, %2 : tensor<32x4096xbf16> into tensor<32x4096xbf16, #ttnn_layout6> -> tensor<32x4096xbf16, #ttnn_layout6>
    // CHECK: return %[[CAST2]] : tensor<32x4096xbf16, #ttnn_layout6>
    return %3 : tensor<32x4096xbf16, #ttnn_layout6>
  }
}
