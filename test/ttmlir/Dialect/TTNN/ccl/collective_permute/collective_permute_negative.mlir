// RUN: not ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
// Unit tests for ttnn collective_permute op

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[0]> : tensor<1xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op The rank of source target pairs must be 2, got rank = 1

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 1], [0, 2]]> : tensor<2x2xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @collective_permute_invalid_source_target_pair_rank attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.collective_permute"(%arg0, %0) <{source_target_pairs = dense<[[0, 2], [1, 2]]> : tensor<2x2xi64>}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.collective_permute' op There are duplicate 'src' or 'dest' devices in source target pairs
