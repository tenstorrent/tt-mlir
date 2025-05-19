// RUN: not ttmlir-opt --split-input-file --tt-register-device="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s
// Unit tests for ttnn all_to_all op


#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_split_dimension_range_1 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = -1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op splitDim must be in the range [0, rank(operands)]

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_split_dimension_range_2 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = 2 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op splitDim must be in the range [0, rank(operands)]

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_split_count_not_divisible attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 3 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op splitDim size must be divisible by splitCount

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_concat_dimension_range_1 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = -1 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op concatDim must be in the range [0, rank(operands)]

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_concat_dimension_range_1 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 2 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op concatDim must be in the range [0, rank(operands)]

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_split_count_range attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 0 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op splitCount must be a positive integer

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_output_type_1 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<512x32xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<512x32xf32, #ttnn_layout>
    return %1 : tensor<512x32xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op output type mismatch: expected type='tensor<128x128xf32,
// CHECK-SAME: output type='tensor<512x32xf32,

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_output_type_2 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xf32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 0 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xf32, #ttnn_layout>
    return %1 : tensor<128x128xf32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op output type mismatch: expected type='tensor<256x64xf32,
// CHECK-SAME: output type='tensor<128x128xf32,


// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @all_to_all_output_type_4 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 2 : i32} {
  func.func public @main(%arg0: tensor<128x128xf32, #ttnn_layout>) -> (tensor<128x128xi32, #ttnn_layout> {}) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32, #ttnn_layout>, !ttnn.device) -> tensor<128x128xi32, #ttnn_layout>
    return %1 : tensor<128x128xi32, #ttnn_layout>
  }
}
// CHECK: error: 'ttnn.all_to_all' op output type mismatch: expected type='tensor<128x128xf32,
// CHECK-SAME: output type='tensor<128x128xi32,
