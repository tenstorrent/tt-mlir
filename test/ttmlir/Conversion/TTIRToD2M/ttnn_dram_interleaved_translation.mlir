// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved>
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 2x512x1024, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved>
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 2x2x256x1024, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved>
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 32x2880, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved>


// Interleaved - Rank 2 layouts
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Interleaved - Rank 3 layouts
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Interleaved - Rank 4 layouts
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 256 + d2, d3), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Interleaved - Rank 2 layout
// 32x2880 on 1x1
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x90x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>

module {

// CHECK-LABEL: func.func @test_lower_interleaved_dram
func.func @test_lower_interleaved_dram(
  %arg0: tensor<1024x1024xbf16, #ttnn_layout>, %out: tensor<1024x1024xbf16, #ttnn_layout>
) -> tensor<1024x1024xbf16, #ttnn_layout> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<1024x1024xbf16, #ttnn_layout>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout>)
  // CHECK: outs(%[[VIEW1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1024x1024xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<1024x1024xbf16, #ttnn_layout>) -> (tensor<1024x1024xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<1024x1024xbf16, #ttnn_layout>
  return %1 : tensor<1024x1024xbf16, #ttnn_layout>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_1
func.func @test_lower_interleaved_dram_1(
  %arg0: tensor<2x512x1024xbf16, #ttnn_layout1>, %out: tensor<2x512x1024xbf16, #ttnn_layout1>
) -> tensor<2x512x1024xbf16, #ttnn_layout1> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<2x512x1024xbf16, #ttnn_layout1>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>)
  // CHECK: outs(%[[VIEW1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>)
  // CHECK-DAG: d2m.tile_abs

  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<2x512x1024xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x512x1024xbf16, #ttnn_layout1>) -> (tensor<2x512x1024xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<2x512x1024xbf16, #ttnn_layout1>
  return %1 : tensor<2x512x1024xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_2
func.func @test_lower_interleaved_dram_2(
  %arg0: tensor<2x2x256x1024xbf16, #ttnn_layout2>, %out: tensor<2x2x256x1024xbf16, #ttnn_layout2>
) -> tensor<2x2x256x1024xbf16, #ttnn_layout2> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout2>
  // CHECK: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] remapping = #map{{.*}} : tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[VIEW0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[VIEW1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<2x2x256x1024xbf16, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x2x256x1024xbf16, #ttnn_layout2>) -> (tensor<2x2x256x1024xbf16, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  return %1 : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  }

// CHECK-LABEL: func.func @test_lower_dram_interleaved_32x2880
func.func @test_lower_dram_interleaved_32x2880(
  %arg0: tensor<32x2880xbf16, #ttnn_layout3>, %out: tensor<32x2880xbf16, #ttnn_layout3>
) -> tensor<32x2880xbf16, #ttnn_layout3> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<32x2880xbf16, #ttnn_layout3>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x2880xbf16, #ttnn_layout3> -> tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[VIEW0:.*]] = d2m.view_layout %[[CAST0]] remapping = #map{{.*}} : tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]> -> tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<32x2880xbf16, #ttnn_layout3> -> tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>
  // CHECK: %[[VIEW1:.*]] = d2m.view_layout %[[CAST1]] remapping = #map{{.*}} : tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]> -> tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[VIEW0]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>)
  // CHECK: outs(%[[VIEW1]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]> -> tensor<32x2880xbf16, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0) : (tensor<32x2880xbf16, #ttnn_layout3>) -> (tensor<32x2880xbf16, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<32x2880xbf16, #ttnn_layout3>
  return %1 : tensor<32x2880xbf16, #ttnn_layout3>
  }
}
