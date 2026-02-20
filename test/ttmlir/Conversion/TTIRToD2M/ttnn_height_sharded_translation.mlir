// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>


// CHECK:#layout = #ttcore.metal_layout<logical_shape = 384x32, dim_alignments = 32x32, collapsed_intervals
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 384x64, dim_alignments = 32x32, collapsed_intervals
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 2x384x32, dim_alignments = 1x32x32, collapsed_intervals
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 2x2x384x32, dim_alignments = 1x1x32x32, collapsed_intervals


// Width Sharded - Rank 2 layouts
// 384x32 on 6x2
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x32 on 2x6
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x64 on 6x2
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
// 384x64 on 2x6
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x6>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>


// Width Sharded - Rank 3 layouts
// 2x384x32 on 6x2
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <6x2>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>

// Width Sharded - Rank 4 layouts
// 2x2x384x32 on 6x2
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <6x2>, memref<4x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>


module {
  // CHECK-LABEL: func.func @test_lower_height_sharded_l1
func.func @test_lower_height_sharded_l1(
  %arg0: tensor<384x32xbf16, #ttnn_layout>
) -> tensor<384x32xbf16, #ttnn_layout> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x32xbf16, #ttnn_layout> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x32xbf16, #ttnn_layout> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<384x32xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x32xbf16, #ttnn_layout>) -> (tensor<384x32xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<384x32xbf16, #ttnn_layout>
    return %1 : tensor<384x32xbf16, #ttnn_layout>
  }


  // CHECK-LABEL: func.func @test_lower_height_sharded_l1_1
func.func @test_lower_height_sharded_l1_1(
  %arg0: tensor<384x32xbf16, #ttnn_layout1>) -> tensor<384x32xbf16, #ttnn_layout1> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x32xbf16, #ttnn_layout1> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x32xbf16, #ttnn_layout1> -> tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<384x32xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x32xbf16, #ttnn_layout1>) -> (tensor<384x32xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<384x32xbf16, #ttnn_layout1>
    return %1 : tensor<384x32xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_2
func.func @test_lower_height_sharded_l1_2(
  %arg0: tensor<384x64xbf16, #ttnn_layout2>) -> tensor<384x64xbf16, #ttnn_layout2> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x64xbf16, #ttnn_layout2> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x64xbf16, #ttnn_layout2> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<384x64xbf16, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x64xbf16, #ttnn_layout2>) -> (tensor<384x64xbf16, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<384x64xbf16, #ttnn_layout2>
    return %1 : tensor<384x64xbf16, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_3
func.func @test_lower_height_sharded_l1_3(
  %arg0: tensor<384x64xbf16, #ttnn_layout3>) -> tensor<384x64xbf16, #ttnn_layout3> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<384x64xbf16, #ttnn_layout3> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<384x64xbf16, #ttnn_layout3> -> tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout{{.*}}>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x1x2x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<384x64xbf16, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0)  : (tensor<384x64xbf16, #ttnn_layout3>) -> (tensor<384x64xbf16, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<384x64xbf16, #ttnn_layout3>
    return %1 : tensor<384x64xbf16, #ttnn_layout3>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_4
func.func @test_lower_height_sharded_l1_4(
  %arg0: tensor<2x384x32xbf16, #ttnn_layout4>) -> tensor<2x384x32xbf16, #ttnn_layout4> {

  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x384x32xbf16, #ttnn_layout4> -> tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x384x32xbf16, #ttnn_layout4> -> tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x2x1x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<2x384x32xbf16, #ttnn_layout4>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x384x32xbf16, #ttnn_layout4>) -> (tensor<2x384x32xbf16, #ttnn_layout4>)
  // CHECK: return %[[CAST2]] : tensor<2x384x32xbf16, #ttnn_layout4>
    return %1 : tensor<2x384x32xbf16, #ttnn_layout4>
  }

// CHECK-LABEL: func.func @test_lower_height_sharded_l1_5
func.func @test_lower_height_sharded_l1_5(
  %arg0: tensor<2x2x384x32xbf16, #ttnn_layout5>) -> tensor<2x2x384x32xbf16, #ttnn_layout5> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x384x32xbf16, #ttnn_layout5> -> tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %0 : tensor<2x2x384x32xbf16, #ttnn_layout5> -> tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<12x1
  // CHECK: ins(%[[CAST0]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout3>)
  // CHECK: outs(%[[CAST1]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout3>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<12x1x4x1x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<2x2x384x32xbf16, #ttnn_layout5>

  %1 = "ttir.abs"(%arg0)  : (tensor<2x2x384x32xbf16, #ttnn_layout5>) -> (tensor<2x2x384x32xbf16, #ttnn_layout5>)
  // CHECK: return %[[CAST2]] : tensor<2x2x384x32xbf16, #ttnn_layout5>
  return %1 : tensor<2x2x384x32xbf16, #ttnn_layout5>
  }
}
