// RUN: ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize --canonicalize %s | FileCheck %s

#indices_layout = #ttcore.metal_layout<logical_shape = 2x4, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#weight_layout = #ttcore.metal_layout<logical_shape = 8x16, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#output_layout = #ttcore.metal_layout<logical_shape = 2x4x16, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, sharded>
#indices_ui32_layout = #ttcore.metal_layout<logical_shape = 2x3, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#weight_bf16_layout = #ttcore.metal_layout<logical_shape = 16x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#output_bf16_layout = #ttcore.metal_layout<logical_shape = 2x3x8, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, sharded>
#indices_i32_layout = #ttcore.metal_layout<logical_shape = 3x1, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#weight_i32_layout = #ttcore.metal_layout<logical_shape = 16x1, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#output_i32_layout = #ttcore.metal_layout<logical_shape = 3x1x1, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @embedding_result_bufferizes_in_place
func.func @embedding_result_bufferizes_in_place(
    %indices: tensor<1x1x2x4xi32, #indices_layout>,
    %weight: tensor<1x1x8x16xf32, #weight_layout>) -> tensor<1x1x8x16xf32, #output_layout> {
  // CHECK: %[[OUTPUT:.*]] = memref.alloc() : memref<1x1x8x16xf32
  %output = d2m.empty() : tensor<1x1x8x16xf32, #output_layout>
  // CHECK: d2m.generic
  // CHECK: outs(%[[OUTPUT]] : memref<1x1x8x16xf32
  %result = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
      ins(%indices, %weight : tensor<1x1x2x4xi32, #indices_layout>, tensor<1x1x8x16xf32, #weight_layout>)
      outs(%output : tensor<1x1x8x16xf32, #output_layout>)
   {
    // CHECK: %[[INDEX_SCRATCH:.*]] = memref.alloc() : memref<1x1024xi32
    // CHECK: %[[ROW_SCRATCH:.*]] = memref.alloc() : memref<1x1024xf32
    // CHECK: d2m.indexed_row_copy {{.*}}, {{.*}}, %[[OUTPUT]] scratch %[[INDEX_SCRATCH]], %[[ROW_SCRATCH]]<8, 16>
    // CHECK-SAME: {indicesShape = array<i64: 2, 4>}
    // CHECK-SAME: : memref
    %embed = d2m.embedding %indices, %weight, %output<8, 16> {indicesShape = array<i64: 2, 4>} : tensor<1x1x2x4xi32, #indices_layout>, tensor<1x1x8x16xf32, #weight_layout>, tensor<1x1x8x16xf32, #output_layout> -> tensor<1x1x8x16xf32, #output_layout>
    d2m.yield %embed : (tensor<1x1x8x16xf32, #output_layout>)
  } : tensor<1x1x8x16xf32, #output_layout>
  return %result : tensor<1x1x8x16xf32, #output_layout>
}

// CHECK-LABEL: func.func @embedding_bf16_table_ui32_indices
func.func @embedding_bf16_table_ui32_indices(
    %indices: tensor<1x1x2x3xui32, #indices_ui32_layout>,
    %weight: tensor<1x1x16x8xbf16, #weight_bf16_layout>) -> tensor<1x1x6x8xbf16, #output_bf16_layout> {
  %output = d2m.empty() : tensor<1x1x6x8xbf16, #output_bf16_layout>
  %result = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
      ins(%indices, %weight : tensor<1x1x2x3xui32, #indices_ui32_layout>, tensor<1x1x16x8xbf16, #weight_bf16_layout>)
      outs(%output : tensor<1x1x6x8xbf16, #output_bf16_layout>)
   {
    // CHECK: memref.alloc() : memref<1x1024xui32
    // CHECK: memref.alloc() : memref<1x1024xbf16
    // CHECK: d2m.indexed_row_copy {{.*}} scratch {{.*}}<6, 8>
    %embed = d2m.embedding %indices, %weight, %output<6, 8> {indicesShape = array<i64: 2, 3>} : tensor<1x1x2x3xui32, #indices_ui32_layout>, tensor<1x1x16x8xbf16, #weight_bf16_layout>, tensor<1x1x6x8xbf16, #output_bf16_layout> -> tensor<1x1x6x8xbf16, #output_bf16_layout>
    d2m.yield %embed : (tensor<1x1x6x8xbf16, #output_bf16_layout>)
  } : tensor<1x1x6x8xbf16, #output_bf16_layout>
  return %result : tensor<1x1x6x8xbf16, #output_bf16_layout>
}

// CHECK-LABEL: func.func @embedding_i32_table_ui32_indices
func.func @embedding_i32_table_ui32_indices(
    %indices: tensor<1x1x3x1xui32, #indices_i32_layout>,
    %weight: tensor<1x1x16x1xi32, #weight_i32_layout>) -> tensor<1x1x3x1xi32, #output_i32_layout> {
  %output = d2m.empty() : tensor<1x1x3x1xi32, #output_i32_layout>
  %result = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
      ins(%indices, %weight : tensor<1x1x3x1xui32, #indices_i32_layout>, tensor<1x1x16x1xi32, #weight_i32_layout>)
      outs(%output : tensor<1x1x3x1xi32, #output_i32_layout>)
   {
    // CHECK: memref.alloc() : memref<1x1024xui32
    // CHECK: memref.alloc() : memref<1x1024xi32
    // CHECK: d2m.indexed_row_copy {{.*}} scratch {{.*}}<3, 1>
    %embed = d2m.embedding %indices, %weight, %output<3, 1> {indicesShape = array<i64: 3, 1>} : tensor<1x1x3x1xui32, #indices_i32_layout>, tensor<1x1x16x1xi32, #weight_i32_layout>, tensor<1x1x3x1xi32, #output_i32_layout> -> tensor<1x1x3x1xi32, #output_i32_layout>
    d2m.yield %embed : (tensor<1x1x3x1xi32, #output_i32_layout>)
  } : tensor<1x1x3x1xi32, #output_i32_layout>
  return %result : tensor<1x1x3x1xi32, #output_i32_layout>
}
