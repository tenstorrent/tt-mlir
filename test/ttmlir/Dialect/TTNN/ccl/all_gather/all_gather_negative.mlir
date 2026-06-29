// RUN: not ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
// Unit tests for ttnn all_gather op

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @all_gather_negative_invalid_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %1 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32, num_links = 1 : ui32}> : (tensor<4096x16384xf32, #ttnn_layout1>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.all_gather' op Invalid gather dimension for all reduce op. Gather dimension must be >= to input tensor rank or < -input tensor rank, got gather_dim = 2

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @all_gather_negative_invalid_negative_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %1 = "ttnn.all_gather"(%arg0) <{all_gather_dim = -3 : si32, cluster_axis = 1 : ui32, num_links = 1 : ui32}> : (tensor<4096x16384xf32, #ttnn_layout1>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.all_gather' op Invalid gather dimension for all reduce op. Gather dimension must be >= to input tensor rank or < -input tensor rank, got gather_dim = -3

// -----

// Reproduces the layout mismatch from the data-parallel gemma4 failure: a
// ROW_MAJOR ui32 operand (a typecast'd token-id input) feeding an all_gather
// whose result is TILE. all_gather is pure data movement and cannot change
// layout, so this used to slip through compilation and abort at runtime with
// "Layout mismatch, expected TILE, got ROW_MAJOR". The verifier now rejects it.
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xui32, #dram>, <interleaved>>
#ttnn_layout_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module @all_gather_negative_layout_mismatch attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1xui32, #ttnn_layout_rm>) -> (tensor<1x8xui32, #ttnn_layout_tile> {}) {
    %1 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32, num_links = 1 : ui32}> : (tensor<1x1xui32, #ttnn_layout_rm>) -> tensor<1x8xui32, #ttnn_layout_tile>
    return %1 : tensor<1x8xui32, #ttnn_layout_tile>
  }
}
// CHECK: error: 'ttnn.all_gather' op Input and output must have the same layout, but got ROW_MAJOR input and TILE output
