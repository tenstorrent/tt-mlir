// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

// CHECK-LABEL: func.func @multi_output_generic_bufferization
func.func @multi_output_generic_bufferization(
    %input: tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
    -> (tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
        tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>) {
  // CHECK-DAG: %[[OUT0:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %out0 = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
  // CHECK-DAG: %[[OUT1:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %out1 = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK-NEXT: ins(%{{.*}} : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
  // CHECK-NEXT: outs(%[[OUT0]], %[[OUT1]] : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
  %results:2 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<2x2>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]
    }
    ins(%input : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
    outs(%out0, %out1 : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
                       tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>) {
  ^unified0:
    %i = d2m.block_index(0) : index
    %j = d2m.block_index(1) : index

    // CHECK: %[[IN_SHARD:.*]] = memref.alloc
    %in_shard = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    %loaded = d2m.remote_load %in_shard %input[%i, %j]
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

    // CHECK: %[[TMP0:.*]] = memref.alloc
    %tmp0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    // CHECK: d2m.local_copy %[[IN_SHARD]], %[[TMP0]]
    %copied0 = d2m.local_copy %loaded, %tmp0 indexing_maps = [#map, #map]
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

    // CHECK: %[[TMP1:.*]] = memref.alloc
    %tmp1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    // CHECK: d2m.local_copy %[[IN_SHARD]], %[[TMP1]]
    %copied1 = d2m.local_copy %loaded, %tmp1 indexing_maps = [#map, #map]
      : tensor<1x1x!ttcore.tile<32x32, f32>>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
      -> tensor<1x1x!ttcore.tile<32x32, f32>>

    // CHECK: d2m.remote_store %[[OUT0]]
    %store0 = d2m.remote_store %out0[%i, %j] %copied0
      : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
      -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.remote_store %[[OUT1]]
    %store1 = d2m.remote_store %out1[%i, %j] %copied1
      : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
        tensor<1x1x!ttcore.tile<32x32, f32>>
      -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

    d2m.yield %store0, %store1
      : (tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
         tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
      tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: return %[[OUT0]], %[[OUT1]]
  return %results#0, %results#1
    : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>,
      tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
}
