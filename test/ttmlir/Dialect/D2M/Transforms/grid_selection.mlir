// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --split-input-file %s | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --split-input-file %s | FileCheck %s --check-prefix=CHECK-AFTER
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

module attributes {ttcore.device = #any_device} {
  func.func @test_grid_selection(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection
    // Verify TTIRToD2M creates 1x1 grids
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.generic {{{.*}}grid = #ttcore.grid<1x1>

    // CHECK-AFTER-LABEL: func.func @test_grid_selection
    // Verify D2MGridSelection optimizes to 8x8 grids
    // CHECK-AFTER: d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>

    %0 = "ttir.exp"(%arg0) : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }

  func.func @test_update_empty() -> (tensor<256x256xf32>) {
    // CHECK-BEFORE-LABEL: func.func @test_update_empty
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-AFTER-LABEL: func.func @test_update_empty
    // Verify D2MGridSelection optimizes the empty 8x8 grids
    // CHECK-AFTER: d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    %0 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    }
    ins() outs(%0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>)  {
    ^compute0(%cb_out: !d2m.cb<tensor<8x8x!ttcore.tile<32x32, f32>>>):
      %out = d2m.reserve %cb_out : <tensor<8x8x!ttcore.tile<32x32, f32>>> -> tensor<8x8x!ttcore.tile<32x32, f32>>
      d2m.yield %out : (tensor<8x8x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

    %2 = d2m.empty() : tensor<256x256xf32>
    %3 = d2m.to_layout %1, %2 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> into tensor<256x256xf32> -> tensor<256x256xf32>

    return %3 : tensor<256x256xf32>
  }
}
