// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1 = #ttnn.buffer_type<l1>

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
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R2_DRAM_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STORAGE_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R2_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STREAM_LAYOUT:layout[0-9]*]]>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<1024x1024xbf16, #ttnn_layout> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R2_DRAM_LAYOUT]]>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STORAGE_LAYOUT]]>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R2_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STREAM_LAYOUT]]>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STREAM_LAYOUT]]>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STREAM_LAYOUT]]>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R2_STREAM_LAYOUT]]> -> tensor<1024x1024xbf16, #ttnn_layout>
  %1 = "ttir.abs"(%arg0)  : (tensor<1024x1024xbf16, #ttnn_layout>) -> (tensor<1024x1024xbf16, #ttnn_layout>)
  // CHECK: return %[[CAST2]] : tensor<1024x1024xbf16, #ttnn_layout>
  return %1 : tensor<1024x1024xbf16, #ttnn_layout>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_1
func.func @test_lower_interleaved_dram_1(
  %arg0: tensor<2x512x1024xbf16, #ttnn_layout1>, %out: tensor<2x512x1024xbf16, #ttnn_layout1>
) -> tensor<2x512x1024xbf16, #ttnn_layout1> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<2x512x1024xbf16, #ttnn_layout1>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R3_DRAM_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STORAGE_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R3_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STREAM_LAYOUT:layout[0-9]*]]>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<2x512x1024xbf16, #ttnn_layout1> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R3_DRAM_LAYOUT]]>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STORAGE_LAYOUT]]>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R3_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STREAM_LAYOUT]]>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STREAM_LAYOUT]]>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STREAM_LAYOUT]]>)
  // CHECK-DAG: d2m.tile_abs

  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R3_STREAM_LAYOUT]]> -> tensor<2x512x1024xbf16, #ttnn_layout1>
  %1 = "ttir.abs"(%arg0)  : (tensor<2x512x1024xbf16, #ttnn_layout1>) -> (tensor<2x512x1024xbf16, #ttnn_layout1>)
  // CHECK: return %[[CAST2]] : tensor<2x512x1024xbf16, #ttnn_layout1>
  return %1 : tensor<2x512x1024xbf16, #ttnn_layout1>
  }

// CHECK-LABEL: func.func @test_lower_interleaved_dram_2
func.func @test_lower_interleaved_dram_2(
  %arg0: tensor<2x2x256x1024xbf16, #ttnn_layout2>, %out: tensor<2x2x256x1024xbf16, #ttnn_layout2>
) -> tensor<2x2x256x1024xbf16, #ttnn_layout2> {
  // CHECK: %[[EMPTY0:.*]] = d2m.empty() : tensor<2x2x256x1024xbf16, #ttnn_layout2>
  // CHECK: %[[CAST0:.*]] = ttir.ttnn_metal_layout_cast %arg0 : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R4_DRAM_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STORAGE_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R4_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STREAM_LAYOUT:layout[0-9]*]]>

  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<2x2x256x1024xbf16, #ttnn_layout2> -> tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R4_DRAM_LAYOUT]]>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STORAGE_LAYOUT]]>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x32x32x!ttcore.tile<32x32, bf16>, #[[R4_DRAM_LAYOUT]]>, tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STORAGE_LAYOUT]]>) -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STREAM_LAYOUT]]>

  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STREAM_LAYOUT]]>)
  // CHECK: outs(%[[STREAM1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STREAM_LAYOUT]]>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #[[R4_STREAM_LAYOUT]]> -> tensor<2x2x256x1024xbf16, #ttnn_layout2>
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
  // CHECK: %[[STORAGE0:.*]] = d2m.empty() : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STORAGE_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[STREAM0:.*]] = "d2m.stream_layout"(%[[CAST0]], %[[STORAGE0]]) : (tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>, tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STORAGE_LAYOUT]]>) -> tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STREAM_LAYOUT:layout[0-9]*]]>
  // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY0]] : tensor<32x2880xbf16, #ttnn_layout3> -> tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>
  // CHECK: %[[STORAGE1:.*]] = d2m.empty() : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STORAGE_LAYOUT]]>
  // CHECK: %[[STREAM1:.*]] = "d2m.stream_layout"(%[[CAST1]], %[[STORAGE1]]) : (tensor<1x1x1x90x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>, tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STORAGE_LAYOUT]]>) -> tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STREAM_LAYOUT]]>
  // CHECK: %[[RESULT:.*]] = d2m.generic{{.*}}
  // CHECK: ins(%[[STREAM0]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STREAM_LAYOUT]]>)
  // CHECK: outs(%[[STREAM1]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STREAM_LAYOUT]]>)
  // CHECK-DAG: d2m.tile_abs
  // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[RESULT]] : tensor<1x6x1x15x!ttcore.tile<32x32, bf16>, #[[DRAM_STREAM_LAYOUT]]> -> tensor<32x2880xbf16, #ttnn_layout3>
  %1 = "ttir.abs"(%arg0) : (tensor<32x2880xbf16, #ttnn_layout3>) -> (tensor<32x2880xbf16, #ttnn_layout3>)
  // CHECK: return %[[CAST2]] : tensor<32x2880xbf16, #ttnn_layout3>
  return %1 : tensor<32x2880xbf16, #ttnn_layout3>
  }
}
