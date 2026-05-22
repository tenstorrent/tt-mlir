// RUN: ttmlir-opt --split-input-file %s | FileCheck %s

// Verify that DRAM-sharded TTNN layouts parse, verify, and round-trip cleanly.

// Width-sharded across 12 DRAM banks (Wormhole-style dram grid).

#dram = #ttnn.buffer_type<dram>
#dram_width_sharded = #ttnn.ttnn_layout<
    (d0, d1) -> (d0, d1),
    <1x12>,
    memref<1x1x!ttcore.tile<32x32, f32>, #dram>,
    <width_sharded>,
    core_ranges = <[#ttnn.core_range<(0, 0), (11, 0)>]>>

// CHECK: #ttnn.ttnn_layout<{{.*}}<1x12>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (11,0)>]>>
// CHECK-LABEL: func.func @forward_width_sharded
module {
  func.func @forward_width_sharded(%arg0: tensor<32x384xf32, #dram_width_sharded>) -> tensor<32x384xf32, #dram_width_sharded> {
    %1 = "ttnn.relu"(%arg0) : (tensor<32x384xf32, #dram_width_sharded>) -> tensor<32x384xf32, #dram_width_sharded>
    return %1 : tensor<32x384xf32, #dram_width_sharded>
  }
}

// -----

// Height-sharded across 8 DRAM banks.  The DRAM bank grid is a single row
// `[1, N]`, so the CRS is `(0,0) -> (7,0)`.  BlockSharded is rejected for
// DRAM (covered by ttnnlayoutattr_tests_negative.mlir).

#dram = #ttnn.buffer_type<dram>
#dram_height_sharded = #ttnn.ttnn_layout<
    (d0, d1) -> (d0, d1),
    <8x1>,
    memref<1x1x!ttcore.tile<32x32, f32>, #dram>,
    <height_sharded>,
    core_ranges = <[#ttnn.core_range<(0, 0), (7, 0)>]>>

// CHECK: #ttnn.ttnn_layout<{{.*}}<8x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
// CHECK-LABEL: func.func @forward_height_sharded
module {
  func.func @forward_height_sharded(%arg0: tensor<256x32xf32, #dram_height_sharded>) -> tensor<256x32xf32, #dram_height_sharded> {
    %1 = "ttnn.relu"(%arg0) : (tensor<256x32xf32, #dram_height_sharded>) -> tensor<256x32xf32, #dram_height_sharded>
    return %1 : tensor<256x32xf32, #dram_height_sharded>
  }
}

// -----

// MemoryConfigAttr accepts a DRAM-sharded combination on ttnn.to_memory_config.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x12x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x12>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (11, 0)>]>>

// CHECK-DAG: #[[L1_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x1>, memref<1x12x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
// CHECK-DAG: #[[DRAM_WS_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}<1x12>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (11,0)>]>>
// CHECK-LABEL: func.func @forward_to_dram_sharded
// CHECK: ttnn.to_memory_config
// CHECK-SAME: (tensor<32x384xf32, #[[L1_LAYOUT]]>) -> tensor<32x384xf32, #[[DRAM_WS_LAYOUT]]>
module {
  func.func @forward_to_dram_sharded(%arg0: tensor<32x384xf32, #l1_layout>) -> tensor<32x384xf32, #dram_layout> {
    %1 = "ttnn.to_memory_config"(%arg0) : (tensor<32x384xf32, #l1_layout>) -> tensor<32x384xf32, #dram_layout>
    return %1 : tensor<32x384xf32, #dram_layout>
  }
}
