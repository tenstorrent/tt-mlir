// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved
// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: l1
// CHECK: #layout2 = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) -> ((d0 * 4 + d2) floordiv 32, (d1 * 4 + d3) floordiv 32, (d0 * 4 + d2) mod 32, (d1 * 4 + d3) mod 32)>

// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 2x512x1024, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved
// CHECK: #layout4 = #ttcore.metal_layout<logical_shape = 2x512x1024, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: l1
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 2x512x1024, dim_alignments = 1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) -> ((d0 * 4 + d2) floordiv 32, (d1 * 4 + d3) floordiv 32, (d0 * 4 + d2) mod 32, (d1 * 4 + d3) mod 32)>

// CHECK: #layout6 = #ttcore.metal_layout<logical_shape = 2x2x256x1024, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved
// CHECK: #layout7 = #ttcore.metal_layout<logical_shape = 2x2x256x1024, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: l1
// CHECK: #layout8 = #ttcore.metal_layout<logical_shape = 2x2x256x1024, dim_alignments = 1x1x32x32, collapsed_intervals
// CHECK-SAME: dram, interleaved, index_map = (d0, d1, d2, d3) -> ((d0 * 4 + d2) floordiv 32, (d1 * 4 + d3) floordiv 32, (d0 * 4 + d2) mod 32, (d1 * 4 + d3) mod 32)>


// Interleaved - Rank 2 layouts
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Interleaved - Rank 3 layouts
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
// Interleaved - Rank 4 layouts
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 256 + d2, d3), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>

module {

// CHECK-LABEL: func.func @test_lower_interleaved_dram
func.func @test_lower_interleaved_dram(
  %arg0: tensor<1024x1024xbf16, #ttnn_layout>, %out: tensor<1024x1024xbf16, #ttnn_layout>
) -> tensor<1024x1024xbf16, #ttnn_layout> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2> -> tensor<1024x1024xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<1024x1024xbf16, #ttnn_layout>, tensor<1024x1024xbf16, #ttnn_layout>) -> (tensor<1024x1024xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<1024x1024xbf16, #ttnn_layout>
  return %1 : tensor<1024x1024xbf16, #ttnn_layout>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_1
func.func @test_lower_interleaved_dram_1(
  %arg0: tensor<2x512x1024xbf16, #ttnn_layout1>, %out: tensor<2x512x1024xbf16, #ttnn_layout1>
) -> tensor<2x512x1024xbf16, #ttnn_layout1> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout4>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout3>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout4>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout5>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout3>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout4>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout3>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout4>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout5>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout5>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout5>)
  // CHECK-DAG: d2m.tile_abs

  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout5> -> tensor<2x512x1024xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<2x512x1024xbf16, #ttnn_layout1>, tensor<2x512x1024xbf16, #ttnn_layout1>) -> (tensor<2x512x1024xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<2x512x1024xbf16, #ttnn_layout1>
  return %1 : tensor<2x512x1024xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_2
func.func @test_lower_interleaved_dram_2(
  %arg0: tensor<2x2x256x1024xbf16, #ttnn_layout2>, %out: tensor<2x2x256x1024xbf16, #ttnn_layout2>
) -> tensor<2x2x256x1024xbf16, #ttnn_layout2> {
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout7>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout6>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout7>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout8>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %arg1 : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout6>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout7>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #layout6>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout7>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout8>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout8>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout8>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout8> -> tensor<2x2x256x1024xbf16, #ttnn_layout2>
  %1 = "ttir.abs"(%arg0, %out)  : (tensor<2x2x256x1024xbf16, #ttnn_layout2>, tensor<2x2x256x1024xbf16, #ttnn_layout2>) -> (tensor<2x2x256x1024xbf16, #ttnn_layout2>)
  // CHECK: return %[[CAST2]] : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  return %1 : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  }
}
