// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout2d = #ttcore.metal_layout<logical_shape = 32x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#parallel = #ttcore.iterator_type<parallel>

module attributes {ttcore.device = #any_device} {
  func.func @test_operand_view_insert(%arg0: tensor<32x256xf32>, %arg1: tensor<32x256xf32>) -> tensor<32x256xf32> {
    %0 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x256xf32> into tensor<1x1x32x256xf32, #layout2d> -> tensor<1x1x32x256xf32, #layout2d>

    %2 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %3 = d2m.to_layout %arg1, %2 : tensor<32x256xf32> into tensor<1x1x32x256xf32, #layout2d> -> tensor<1x1x32x256xf32, #layout2d>

    %4 = d2m.empty() : tensor<1x1x32x256xf32, #layout2d>
    %5 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>, // derived operand grid for operand 0 is 1x8, a view will be inserted.
        affine_map<(d0, d1) -> (d0, 0)>, // operand 1 does not vary along the second dimension, so derived grid is still 1x1
        affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]
    }
    ins(%1, %3 : tensor<1x1x32x256xf32, #layout2d>, tensor<1x1x32x256xf32, #layout2d>)
    outs(%4 : tensor<1x1x32x256xf32, #layout2d>) {
    ^unified0:
      %6 = tensor.empty() : tensor<32x256xf32>
      d2m.yield %6 : (tensor<32x256xf32>)
    } : tensor<1x1x32x256xf32, #layout2d>

    %7 = d2m.empty() : tensor<32x256xf32>
    %8 = d2m.to_layout %5, %7 : tensor<1x1x32x256xf32, #layout2d> into tensor<32x256xf32> -> tensor<32x256xf32>
    return %8 : tensor<32x256xf32>
  }
}

// CHECK-LABEL: func.func @test_operand_view_insert
// CHECK: %[[IN0:.*]] = d2m.to_layout
// for the first operand (1x1) inserted by updateToLayoutOps:
// CHECK: %[[IN0_VIEW:.*]] = d2m.view_layout %[[IN0]]
// for the second operand (1x1) inserted by updateToLayoutOps:
// CHECK: %[[IN1:.*]] = d2m.to_layout
// CHECK: %[[IN1_VIEW:.*]] = d2m.view_layout %[[IN1]]
// CHECK: %[[OUT:.*]] = d2m.empty() : tensor<1x8x32x32xf32
// for the first operand (back to 1x8) inserted by withParallelization:
// CHECK: %[[REBLOCK:.*]] = d2m.view_layout %[[IN0_VIEW]] {{.*}} : tensor<1x1x32x256xf32, {{.*}}> -> tensor<1x8x32x32xf32, {{.*}}>
// CHECK: %[[GEN:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x8>
// CHECK: ins(%[[REBLOCK]], %[[IN1_VIEW]] : tensor<1x8x32x32xf32, {{.*}}>, tensor<1x1x32x256xf32, {{.*}}>)
