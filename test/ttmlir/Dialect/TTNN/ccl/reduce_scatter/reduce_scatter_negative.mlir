// RUN: not ttmlir-opt --split-input-file --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device="force-reload=true" %s 2>&1 | FileCheck %s
// Unit tests for ttnn reduce_scatter op

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_mean attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<mean>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_std attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<std>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_var attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<var>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, num_links = 1 : ui32, scatter_dim = 2 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid scatter dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = 2

// -----

#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32, tt.device = #device} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #tt.reduce_type<sum>, num_links = 1 : ui32, scatter_dim = -3 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !tt.device<#device>) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid scatter dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = -3
