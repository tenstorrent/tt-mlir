// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for to_memory_config operation

// Verify that verification fails when buffer type is DRAM-Interleaved and
// grid shape is not 1x1.
#dram = #ttnn.buffer_type<dram>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#system_memory = #ttnn.buffer_type<system_memory>
#system_memory_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x3x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout> {
    // CHECK: error: expected (1, 1) grid shape for dram buffer type with interleaved memory layout, got (8, 8)
    %1 = "ttnn.relu"(%arg0) : (tensor<32x32xf32, #dram_layout>) -> tensor<32x32xf32, #dram_layout>
    return %1 : tensor<32x32xf32, #dram_layout>
  }
}

// -----

// Verify that verification fails when buffer type is system memory and grid shape is not 1x1.
#system_memory = #ttnn.buffer_type<system_memory>
#system_memory_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <3x4>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>>
module {
  func.func @forward(%arg0: tensor<32x32xf32, #system_memory_layout>) -> tensor<32x32xf32, #system_memory_layout> {
    // CHECK: error: expected (1, 1) grid shape for system_memory buffer type, got (3, 4)
    %1 = "ttnn.relu"(%arg0) : (tensor<32x32xf32, #system_memory_layout>) -> tensor<32x32xf32, #system_memory_layout>
    return %1 : tensor<32x32xf32, #system_memory_layout>
  }
}

// -----

// Verify that DRAM + BlockSharded is rejected.
#dram = #ttnn.buffer_type<dram>
#dram_block_sharded = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <2x4>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (3, 1)>]>>
module {
  func.func @forward(%arg0: tensor<128x256xf32, #dram_block_sharded>) -> tensor<128x256xf32, #dram_block_sharded> {
    // CHECK: error: BlockSharded layout is not supported for DRAM buffer type; use WidthSharded, HeightSharded or Interleaved
    %1 = "ttnn.relu"(%arg0) : (tensor<128x256xf32, #dram_block_sharded>) -> tensor<128x256xf32, #dram_block_sharded>
    return %1 : tensor<128x256xf32, #dram_block_sharded>
  }
}
