// RUN: ttmlir-opt --split-input-file --canonicalize %s | FileCheck %s

#indices_layout = #ttcore.metal_layout<logical_shape = 2x4, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#weight_layout = #ttcore.metal_layout<logical_shape = 8x16, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#output_layout = #ttcore.metal_layout<logical_shape = 2x4x16, dim_alignments = 1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, l1, sharded>

module {
  // CHECK-LABEL: func.func @tensor_embedding_result
  func.func @tensor_embedding_result(
      %indices: tensor<1x1x2x4xi32, #indices_layout>,
      %weight: tensor<1x1x8x16xf32, #weight_layout>,
      %output: tensor<1x1x8x16xf32, #output_layout>) -> tensor<1x1x8x16xf32, #output_layout> {
    // CHECK: %[[GENERIC:.*]] = d2m.generic
    // CHECK: %[[EMBED:.*]] = d2m.embedding {{.*}}<8, 16>
    // CHECK-SAME: {indicesShape = array<i64: 2, 4>}
    // CHECK-SAME: -> tensor<1x1x8x16xf32
    // CHECK: d2m.yield %[[EMBED]] : (tensor<1x1x8x16xf32
    %result = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%indices, %weight : tensor<1x1x2x4xi32, #indices_layout>, tensor<1x1x8x16xf32, #weight_layout>)
        outs(%output : tensor<1x1x8x16xf32, #output_layout>)
     {
      %embed = d2m.embedding %indices, %weight, %output<8, 16> {indicesShape = array<i64: 2, 4>} : tensor<1x1x2x4xi32, #indices_layout>, tensor<1x1x8x16xf32, #weight_layout>, tensor<1x1x8x16xf32, #output_layout> -> tensor<1x1x8x16xf32, #output_layout>
      d2m.yield %embed : (tensor<1x1x8x16xf32, #output_layout>)
    } : tensor<1x1x8x16xf32, #output_layout>
    return %result : tensor<1x1x8x16xf32, #output_layout>
  }
}

// -----

#l1 = #ttcore.memory_space<l1>

module {
  // CHECK-LABEL: func.func @memref_indexed_row_copy_explicit
  func.func @memref_indexed_row_copy_explicit(
      %indices: memref<1x1x2x4xi32, #l1>,
      %weight: memref<1x1x8x16xf32, #l1>,
      %output: memref<1x1x8x16xf32, #l1>,
      %index_scratch: memref<1x1024xi32, #l1>,
      %row_scratch: memref<1x1024xf32, #l1>) {
    // CHECK: d2m.indexed_row_copy {{.*}} scratch {{.*}}<8, 16>
    // CHECK-SAME: {indicesShape = array<i64: 2, 4>}
    d2m.indexed_row_copy %indices, %weight, %output scratch %index_scratch, %row_scratch<8, 16> {indicesShape = array<i64: 2, 4>} : memref<1x1x2x4xi32, #l1>, memref<1x1x8x16xf32, #l1>, memref<1x1x8x16xf32, #l1>, memref<1x1024xi32, #l1>, memref<1x1024xf32, #l1>
    return
  }
}
