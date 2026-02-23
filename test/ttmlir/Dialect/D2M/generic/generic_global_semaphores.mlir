// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline %s | FileCheck %s

// Test for d2m.create_global_semaphore and global semaphore operands in d2m.generic.
// This tests:
// 1. Creating a global semaphore with d2m.create_global_semaphore
// 2. Passing the global semaphore as a runtime arg operand to d2m.generic
// 3. Using d2m.semaphore_set and d2m.semaphore_wait with the global semaphore inside the generic region
// 4. Ensuring that the liveness analysis is working for the global semaphore buffer.

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#sem_layout = #ttcore.metal_layout<logical_shape = 8x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @generic_with_global_semaphore
  func.func @generic_with_global_semaphore(
    %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  ) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %output = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

    // Create a global semaphore with initial value 0
    // CHECK: %[[SEM_BACKING:.*]] = "ttmetal.create_buffer"() <{address = [[SEM_ADDRESS:[0-9]+]] : i64}> : () -> memref<8x8x1x1xui32, #ttcore.shard<4x4, 1>, #l1>
    %sem_backing = d2m.empty() : tensor<8x8x1x1xui32, #sem_layout>

    // CHECK: "ttmetal.create_global_semaphore"() <{address = [[SEM_ADDRESS]] : i64, core_range = #ttmetal.core_range<0x0, 8x8>, initial_value = 0 : ui32}> : () -> !ttmetal.global_semaphore
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

    // CHECK: ttmetal.enqueue_program
    // CHECK-SAME: #ttmetal.kernel_args< ct_args = [{{.*}}<global_semaphore[0]>{{.*}}]
    // The below check is to ensure that the semaphore buffer is deallocated after the generic operation (i.e liveness analysis is working).
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
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index

      scf.for %i = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c2 step %c1 {
          %buffer_in = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          %buffer_out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          %loaded = d2m.remote_load %buffer_in %arg0_stream[%i, %j]
            : tensor<2x2x!ttcore.tile<32x32, f32>>,
              tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
            -> tensor<2x2x!ttcore.tile<32x32, f32>>
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
          %stored = d2m.remote_store %output_stream[%i, %j] %abs_result
            : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
              tensor<2x2x!ttcore.tile<32x32, f32>>
            -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        }
      }

      d2m.semaphore_wait %sem, %c1 : !d2m.global_semaphore

      d2m.yield %output_stream : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }) : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
          tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
          !d2m.global_semaphore)
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

    // CHECK: "ttmetal.reset_global_semaphore"
    d2m.reset_global_semaphore(%sem) {value = 0 : ui32} : !d2m.global_semaphore

    // CHECK: "ttmetal.deallocate_buffer"(%[[SEM_BACKING]])

    return %result : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

    // CHECK: func.func {{.*}} {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [{{.*}}<arg_type = global_semaphore, operand_index = {{[0-9]+}}>{{.*}}]>, {{.*}}}
    // CHECK: "noc_semaphore_wait"
  }
}
