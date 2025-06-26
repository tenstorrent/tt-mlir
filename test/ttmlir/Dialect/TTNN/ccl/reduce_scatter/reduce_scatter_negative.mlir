// RUN: not ttmlir-opt --split-input-file --ttcore-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
// Unit tests for ttnn reduce_scatter op

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_mean attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<mean>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_std attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<std>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_reduce_type_var attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<var>, num_links = 1 : ui32, scatter_dim = 1 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, num_links = 1 : ui32, scatter_dim = 2 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid scatter dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = 2

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @reduce_scatter_negative_invalid_dim attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4096x16384xf32, #ttnn_layout1>) -> (tensor<4096x16384xf32, #ttnn_layout1> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, num_links = 1 : ui32, scatter_dim = -3 : si32}> : (tensor<4096x16384xf32, #ttnn_layout1>, !ttnn.device) -> tensor<4096x16384xf32, #ttnn_layout1>
    return %1 : tensor<4096x16384xf32, #ttnn_layout1>
  }
}
// CHECK: error: 'ttnn.reduce_scatter' op Invalid scatter dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = -3
