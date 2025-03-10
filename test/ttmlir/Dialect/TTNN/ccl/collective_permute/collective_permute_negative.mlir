// RUN: not ttmlir-opt --split-input-file --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device="force-reload=true" %s 2>&1 | FileCheck %s
// Unit tests for ttnn collective_permute op

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[0]> : tensor<1xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op The rank of source target pairs must be 2, got rank = 1

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 1], [0, 2]]> : tensor<2x2xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 2], [1, 2]]> : tensor<2x2xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs
