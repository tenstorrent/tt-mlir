// RUN: ttmlir-opt --split-input-file --canonicalize %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline %s

// Test for d2m.create_global_semaphore and global semaphore operands in d2m.generic.
// This tests:
// 1. Creating a global semaphore with d2m.create_global_semaphore
// 2. Passing the global semaphore as a capture operand to d2m.generic
// 3. Using d2m.semaphore_set and d2m.semaphore_wait inside the generic region

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Layout for 2x2 grid with 2x2 shard shape
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#sem_layout = #ttcore.metal_layout<logical_shape = 8x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

// CHECK-LABEL: func.func @generic_with_global_semaphore
func.func @generic_with_global_semaphore(
  %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
  %output = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

  // Create a global semaphore with initial value 0
  // CHECK: d2m.empty
  // CHECK: d2m.create_global_semaphore
  %sem_backing = d2m.empty() : tensor<8x8x1x1xui32, #sem_layout>
  %sem = d2m.create_global_semaphore(%sem_backing) {value = 0 : ui32}
    : tensor<8x8x1x1xui32, #sem_layout> -> !d2m.global_semaphore

  // Wrap input and output in identity stream_layout operations
  %input_storage = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  %arg0_stream = "d2m.stream_layout"(%arg0, %input_storage)
    : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
       tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  %output_storage = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  %output_stream = "d2m.stream_layout"(%output, %output_storage)
    : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
       tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK-SAME: captures(%{{.*}} : !d2m.global_semaphore)
  // CHECK: d2m.semaphore_set
  // CHECK: d2m.semaphore_wait
  %result = "d2m.generic"(%arg0_stream, %output_stream, %sem) <{
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    operandSegmentSizes = array<i32: 1, 1, 1>,
    threads = [#d2m.thread<unified>]
  }> ({
  ^unified0(%cb_in: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>,
            %cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Get core coordinates for semaphore operations
    %core0 = d2m.core_index(0) : index
    %core1 = d2m.core_index(1) : index

    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c2 step %c1 {
        %buffer_in = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
        %buffer_out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

        // Load input shard
        %loaded = d2m.remote_load %buffer_in %arg0_stream[%i, %j]
          : tensor<2x2x!ttcore.tile<32x32, f32>>,
            tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<2x2x!ttcore.tile<32x32, f32>>

        // Compute abs via linalg.generic
        %abs_result = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%loaded : tensor<2x2x!ttcore.tile<32x32, f32>>)
        outs(%buffer_out : tensor<2x2x!ttcore.tile<32x32, f32>>) {
        ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
          %abs = "d2m.tile_abs"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %abs : !ttcore.tile<32x32, f32>
        } -> tensor<2x2x!ttcore.tile<32x32, f32>>

        // Store result shard
        %stored = d2m.remote_store %output_stream[%i, %j] %abs_result
          : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
            tensor<2x2x!ttcore.tile<32x32, f32>>
          -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      }
    }

    // Use global semaphore for synchronization:
    // Set semaphore to signal completion (multicast to self with 1x1 shape)
    d2m.semaphore_set %sem, %c1, core[%core0, %core1] mcast[%c1, %c1] : !d2m.global_semaphore

    // Wait for semaphore to reach expected value and reset to 0
    d2m.semaphore_wait %sem, %c1 reset %c0 : !d2m.global_semaphore

    d2m.yield %output_stream : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
  }) : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
        tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
        !d2m.global_semaphore)
     -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

  return %result : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
}
