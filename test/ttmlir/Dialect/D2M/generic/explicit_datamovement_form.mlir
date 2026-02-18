// RUN: ttmlir-opt --split-input-file --canonicalize %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-ttmetal-me-pipeline --ttir-to-ttmetal-be-pipeline %s

// Test for explicit datamovement form of d2m.generic with abs operation.
// The explicit datamovement form has empty block_factors, indexing_maps, and iterator_types.
// Users manually control loops via scf.for inside the region body.
//
// Note: tensor.empty ops are hoisted out of loops to ensure allocator liveness
// analysis works correctly (it expects ops to be in the same block).

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// Layout for 2x3 grid with 2x2 shard shape (64x96 logical shape in tiles)
#layout = #ttcore.metal_layout<logical_shape = 64x96, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

// CHECK-LABEL: func.func @explicit_datamovement_abs
func.func @explicit_datamovement_abs(
  %arg0: tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout> {
  %output = d2m.empty() : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>

  // Wrap input and output in identity stream_layout operations
  // CHECK-DAG: %[[INPUT_STREAM:.*]] = "d2m.stream_layout"(%arg0, %{{.*}}) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
  %input_storage = d2m.empty() : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
  %arg0_stream = "d2m.stream_layout"(%arg0, %input_storage) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
  // CHECK-DAG: %[[OUTPUT_STREAM:.*]] = "d2m.stream_layout"(%{{.*}}, %{{.*}}) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
  %output_storage = d2m.empty() : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
  %output_stream = "d2m.stream_layout"(%output, %output_storage) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK-SAME: block_factors = []
  // CHECK-SAME: grid = #ttcore.grid<1x1>
  // CHECK-SAME: indexing_maps = []
  // CHECK-SAME: iterator_types = []
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: d2m.remote_load
  // CHECK: linalg.generic
  // CHECK: d2m.tile_abs
  // CHECK: d2m.remote_store
  %result = "d2m.generic"(%arg0_stream, %output_stream) <{
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    operandSegmentSizes = array<i32: 1, 1, 0>,
    threads = [#d2m.thread<unified>]
  }> ({
  ^unified0(%cb_in: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>,
            %cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    // Define loop bounds: M=2, N=3 (grid dimensions)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index  // M - grid dim 0
    %c3 = arith.constant 3 : index  // N - grid dim 1

    // Nested scf.for loops iterating over grid dimensions
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c3 step %c1 {
        // Hoist tensor.empty out of loops for allocator liveness analysis
        %buffer_in = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
        %buffer_out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

        // Load input shard at grid position [i, j]
        %loaded = d2m.remote_load %buffer_in %arg0_stream[%i, %j]
          : tensor<2x2x!ttcore.tile<32x32, f32>>,
            tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
          -> tensor<2x2x!ttcore.tile<32x32, f32>>

        // Compute abs via linalg.generic with d2m.tile_abs
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

        // Store result shard at grid position [i, j]
        %stored = d2m.remote_store %output_stream[%i, %j] %abs_result
          : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>,
            tensor<2x2x!ttcore.tile<32x32, f32>>
          -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
      }
    }
    d2m.yield %output_stream : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>)
  }) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>

  return %result : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout>
}
