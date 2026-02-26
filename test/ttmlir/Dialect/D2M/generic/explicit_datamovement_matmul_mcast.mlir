// RUN: ttmlir-opt --split-input-file --canonicalize %s | FileCheck %s

// Test for explicit datamovement form of d2m.generic with matmul operation
// using low-level multicast remote_load (mcore/mshape).
//
// Matmul: C[M,N] = sum_K A[M,K] * B[K,N]
// Output grid: 2x4, K reduction blocks: 3
// A: grid 2x3, shard 2x2 tiles (logical 4x6 tiles)
// B: grid 3x4, shard 2x2 tiles (logical 6x8 tiles)
// C: grid 2x4, shard 2x2 tiles (logical 4x8 tiles)
//
// Multicast pattern:
//   A at [i,k] -> multicast along columns: mcore[0,0] mshape[1,4]
//   B at [k,j] -> multicast along rows: mcore[0,0] mshape[2,1]
//
// Each k iteration loads A/B, computes a partial matmul, and stores C.
// The inner linalg.generic reduction handles the tile-level contraction.

#l1_ = #ttcore.memory_space<l1>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

#layout_a = #ttcore.metal_layout<logical_shape = 64x96, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_b = #ttcore.metal_layout<logical_shape = 96x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_c = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @explicit_datamovement_matmul_mcast
func.func @explicit_datamovement_matmul_mcast(
  %arg0: tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>,
  %arg1: tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>
) -> tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c> {
  %output = d2m.empty() : tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>

  // Wrap inputs and output in stream_layout operations
  %input_a_storage = d2m.empty() : tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>
  %arg0_stream = "d2m.stream_layout"(%arg0, %input_a_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>, tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>) -> tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>

  %input_b_storage = d2m.empty() : tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>
  %arg1_stream = "d2m.stream_layout"(%arg1, %input_b_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>, tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>) -> tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>

  %output_storage = d2m.empty() : tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>
  %output_stream = "d2m.stream_layout"(%output, %output_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>, tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>) -> tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>

  // CHECK: d2m.generic
  // CHECK-SAME: block_factors = []
  // CHECK-SAME: grid = #ttcore.grid<1x1>
  // CHECK-SAME: indexing_maps = []
  // CHECK-SAME: iterator_types = []
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: d2m.remote_load {{.*}} mcore
  // CHECK: d2m.remote_load {{.*}} mcore
  // CHECK: linalg.generic
  // CHECK: d2m.tile_matmul
  // CHECK: d2m.remote_store
  %result = "d2m.generic"(%arg0_stream, %arg1_stream, %output_stream) <{
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    operandSegmentSizes = array<i32: 2, 1>,
    threads = [#d2m.thread<unified>]
  }> ({
  ^unified0(%cb_a: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>,
            %cb_b: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>,
            %cb_c: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index  // M - output grid dim 0
    %c3 = arith.constant 3 : index  // K - reduction blocks
    %c4 = arith.constant 4 : index  // N - output grid dim 1

    // Nested scf.for loops: M rows x N columns of output grid, K reduction
    scf.for %i = %c0 to %c2 step %c1 {
      scf.for %j = %c0 to %c4 step %c1 {
        scf.for %k = %c0 to %c3 step %c1 {
          %a_buf = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          %b_buf = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          %c_buf = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

          // Load A shard at [i, k] with multicast along columns (j direction)
          // Core 0 gathers and multicasts to a 1x4 region
          %a_loaded = d2m.remote_load %a_buf %arg0_stream[%i, %k] mcore[%c0, %c0] mshape[%c1, %c4]
            : tensor<2x2x!ttcore.tile<32x32, f32>>,
              tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>
            -> tensor<2x2x!ttcore.tile<32x32, f32>>

          // Load B shard at [k, j] with multicast along rows (i direction)
          // Core 0 gathers and multicasts to a 2x1 region
          %b_loaded = d2m.remote_load %b_buf %arg1_stream[%k, %j] mcore[%c0, %c0] mshape[%c2, %c1]
            : tensor<2x2x!ttcore.tile<32x32, f32>>,
              tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>
            -> tensor<2x2x!ttcore.tile<32x32, f32>>

          // Compute tile matmul: C += A * B (tile-level reduction in linalg)
          %matmul_result = linalg.generic {
            indexing_maps = [#mapL, #mapR, #mapO],
            iterator_types = ["parallel", "parallel", "reduction"]
          }
          ins(%a_loaded, %b_loaded : tensor<2x2x!ttcore.tile<32x32, f32>>,
                                     tensor<2x2x!ttcore.tile<32x32, f32>>)
          outs(%c_buf : tensor<2x2x!ttcore.tile<32x32, f32>>) {
          ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>,
               %c: !ttcore.tile<32x32, f32>):
            %mm = "d2m.tile_matmul"(%a, %b, %c)
              : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>,
                 !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %mm : !ttcore.tile<32x32, f32>
          } -> tensor<2x2x!ttcore.tile<32x32, f32>>

          // Store result shard at grid position [i, j]
          %stored = d2m.remote_store %output_stream[%i, %j] %matmul_result
            : tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>,
              tensor<2x2x!ttcore.tile<32x32, f32>>
            -> tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>
        }
      }
    }
    d2m.yield %output_stream : (tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>)
  }) : (tensor<2x3x2x2x!ttcore.tile<32x32, f32>, #layout_a>, tensor<3x4x2x2x!ttcore.tile<32x32, f32>, #layout_b>, tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>) -> tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>

  return %result : tensor<2x4x2x2x!ttcore.tile<32x32, f32>, #layout_c>
}
