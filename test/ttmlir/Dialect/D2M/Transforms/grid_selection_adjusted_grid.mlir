// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout2d = #ttcore.metal_layout<logical_shape = 32x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#parallel = #ttcore.iterator_type<parallel>

module attributes {ttcore.device = #any_device} {
  func.func @test_adjusted_grid(%arg0: tensor<32x256xf32>, %arg1: tensor<32x256xf32>) -> tensor<32x256xf32> {
    %0 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x256xf32> into tensor<1x1x32x256xf32, #layout2d> -> tensor<1x1x32x256xf32, #layout2d>

    %2 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %3 = d2m.to_layout %arg1, %2 : tensor<32x256xf32> into tensor<1x1x32x256xf32, #layout2d> -> tensor<1x1x32x256xf32, #layout2d>

    %4 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %5 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, 0)>], // second grid dimension does not participate in output so it is collapsed to 1x1
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]
    }
    ins(%1, %3 : tensor<1x1x32x256xf32, #layout2d>, tensor<1x1x32x256xf32, #layout2d>)
    outs(%4 : tensor<1x1x32x256xf32, #layout2d>) {
    ^unified0(%cb0: !d2m.cb<tensor<32x256xf32>>, %cb1: !d2m.cb<tensor<32x256xf32>>, %cb2: !d2m.cb<tensor<32x256xf32>>):
      %6 = tensor.empty() : tensor<32x256xf32>
      d2m.yield %6 : (tensor<32x256xf32>)
    } : tensor<1x1x32x256xf32, #layout2d>

    %7 = d2m.empty() : tensor<32x256xf32>
    %8 = d2m.to_layout %5, %7 : tensor<1x1x32x256xf32, #layout2d> into tensor<32x256xf32> -> tensor<32x256xf32>
    return %8 : tensor<32x256xf32>
  }
}

// CHECK-LABEL: func.func @test_adjusted_grid
// Inputs are first expanded to 1x8 by updateToLayoutOps, then viewed to 1x1.
// CHECK: %[[IN0_TO:.*]] = d2m.to_layout
// CHECK: %[[IN0_1X1:.*]] = d2m.view_layout %[[IN0_TO]]
// CHECK: %[[IN1_TO:.*]] = d2m.to_layout
// CHECK: %[[IN1_1X1:.*]] = d2m.view_layout %[[IN1_TO]]
// CHECK-DAG: %[[OUT_BASE:.*]] = d2m.empty() : tensor<1x8x32x32xf32, {{.*}}>
// withParallelization reblocks inputs back to 1x8 for compute and output to 1x1.
// CHECK: %[[IN0_1X8:.*]] = d2m.view_layout %[[IN0_1X1]] {{.*}} : tensor<1x1x32x256xf32, {{.*}}> -> tensor<1x8x32x32xf32, {{.*}}>
// CHECK: %[[IN1_1X8:.*]] = d2m.view_layout %[[IN1_1X1]] {{.*}} : tensor<1x1x32x256xf32, {{.*}}> -> tensor<1x8x32x32xf32, {{.*}}>
// CHECK: %[[OUT_1X1:.*]] = d2m.view_layout %[[OUT_BASE]] {{.*}} : tensor<1x8x32x32xf32, {{.*}}> -> tensor<1x1x32x256xf32, {{.*}}>
// The requested grid is adjusted to match output-derived shape (1x1), with reduction carried by block factor 8.
// CHECK: %[[GEN:.*]] = d2m.generic {block_factors = [1, 8], grid = #ttcore.grid<1x1>, indexing_maps = [{{.*}}], iterator_types = [#parallel, #parallel]
// CHECK: ins(%[[IN0_1X8]], %[[IN1_1X8]] : tensor<1x8x32x32xf32, {{.*}}>, tensor<1x8x32x32xf32, {{.*}}>)
// CHECK: outs(%[[OUT_1X1]] : tensor<1x1x32x256xf32, {{.*}}>)
