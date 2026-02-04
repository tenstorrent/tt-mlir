// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m %s | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-grid-selection="override-device-shape=10,13" --canonicalize %s | FileCheck %s --check-prefix=CHECK-AFTER

// Test grid selection for Blackhole with tensor that fits onto entire grid (10x13)
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<10x13, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>
  func.func @test_grid_selection_bh_eltwise(%arg0: tensor<320x416xf32>) -> tensor<320x416xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection_bh_eltwise
    // Verify TTIRToD2M creates 1x1 grids
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.to_layout %arg0, %{{.*}} : tensor<320x416xf32> into tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.generic {{{.*}}grid = #ttcore.grid<1x1>

    // CHECK-AFTER-LABEL: func.func @test_grid_selection_bh_eltwise
    // Verify D2MGridSelection optimizes to 10x13 grids
    // CHECK-AFTER: d2m.empty() : tensor<10x13x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.to_layout %arg0, %{{.*}} : tensor<320x416xf32> into tensor<10x13x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<10x13>

    %0 = "ttir.exp"(%arg0) : (tensor<320x416xf32>) -> tensor<320x416xf32>
    return %0 : tensor<320x416xf32>
  }
}

// -----

// Test grid selection for Blackhole matmul grid without constraining dims
// CHECK-AFTER: #[[LAYOUT_MATMUL:.*]] = #ttcore.metal_layout<logical_shape = 320x320, dim_alignments = 320x320,  {{.*}} index_map = map(0)>
// CHECK-AFTER: #[[LAYOUT_MATMUL_2:.*]] = #ttcore.metal_layout<logical_shape = 320x416, dim_alignments = 320x416, {{.*}} index_map = map(0)>
#layout_matmul = #ttcore.metal_layout<logical_shape = 320x320, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout_matmul_2 = #ttcore.metal_layout<logical_shape = 320x416, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<10x13, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 8, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 8) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>
  func.func @test_grid_selection_bh_matmul(%arg0: tensor<320x320xf32>, %arg1: tensor<320x416xf32>) -> tensor<320x416xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection_bh_matmul
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x10x!ttcore.tile<32x32, f32>
    // CHECK-AFTER-LABEL: func.func @test_grid_selection_bh_matmul
    // Verify D2MGridSelection optimizes the empty 10x13 grids
    // CHECK-AFTER: d2m.empty() : tensor<10x10x1x1x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL]]>
    %0 = d2m.empty() : tensor<1x1x10x10x!ttcore.tile<32x32, f32>, #layout_matmul>

    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.empty() : tensor<10x13x1x1x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL_2]]>
    %1 = d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2>

    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.empty() : tensor<10x13x1x1x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL_2]]>
    %2 = d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2>

    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<10x13>
    %3 = d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<unified>]
    }
    ins(%0, %1 : tensor<1x1x10x10x!ttcore.tile<32x32, f32>, #layout_matmul>, tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2>)
    outs(%2 : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2>)  {
    ^unified0(%cb0: !d2m.cb<tensor<10x10x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<10x13x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<10x13x!ttcore.tile<32x32, f32>>>):
      %out = tensor.empty() : tensor<10x13x!ttcore.tile<32x32, f32>>
      d2m.yield %out : (tensor<10x13x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2>

    %4 = d2m.empty() : tensor<320x416xf32>
    %5 = d2m.to_layout %3, %4 : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_2> into tensor<320x416xf32> -> tensor<320x416xf32>

    return %5 : tensor<320x416xf32>
  }
}

// -----
// Test grid selection for Blackhole matmul grid with constraining dims
// CHECK-AFTER: #[[LAYOUT_MATMUL_CONSTRAINED_1:.*]] = #ttcore.metal_layout<logical_shape = 320x416, dim_alignments = 320x320,  {{.*}} index_map = map(0)>
// CHECK-AFTER: #[[LAYOUT_MATMUL_CONSTRAINED_2:.*]] = #ttcore.metal_layout<logical_shape = 416x416, dim_alignments = 320x416, {{.*}} index_map = map(0)>
// CHECK-AFTER: #[[LAYOUT_MATMUL_CONSTRAINED_3:.*]] = #ttcore.metal_layout<logical_shape = 320x416, dim_alignments = 320x416, {{.*}} index_map = map(0)>
#layout_matmul_constrained_1 = #ttcore.metal_layout<logical_shape = 320x416, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout_matmul_constrained_2 = #ttcore.metal_layout<logical_shape = 416x416, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

module {
  func.func @test_grid_selection_bh_matmul_constrained_dims(%arg0: tensor<320x416xf32>, %arg1: tensor<416x416xf32>) -> tensor<320x416xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection_bh_matmul_constrained_dims
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-AFTER-LABEL: func.func @test_grid_selection_bh_matmul_constrained_dims
    // Verify D2MGridSelection optimizes the empty 10x13 grids
    // CHECK-AFTER: d2m.empty() : tensor<10x10x1x2x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL_CONSTRAINED_1]]>
    %0 = d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1>

    // CHECK-BEFORE: d2m.empty() : tensor<1x1x13x13x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.empty() : tensor<10x13x2x1x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL_CONSTRAINED_2]]>
    %1 = d2m.empty() : tensor<1x1x13x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_2>

    // CHECK-BEFORE: d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.empty() : tensor<10x13x1x1x!ttcore.tile<32x32, f32>, #[[LAYOUT_MATMUL_CONSTRAINED_3]]>
    %2 = d2m.empty() : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1>

    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<10x13>
    %3 = d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<unified>]
    }
    ins(%0, %1 : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1>, tensor<1x1x13x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_2>)
    outs(%2 : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1>)  {
    ^unified0(%cb0: !d2m.cb<tensor<10x13x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<13x13x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<10x13x!ttcore.tile<32x32, f32>>>):
      %out = tensor.empty() : tensor<10x13x!ttcore.tile<32x32, f32>>
      d2m.yield %out : (tensor<10x13x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1>

    %4 = d2m.empty() : tensor<320x416xf32>
    %5 = d2m.to_layout %3, %4 : tensor<1x1x10x13x!ttcore.tile<32x32, f32>, #layout_matmul_constrained_1> into tensor<320x416xf32> -> tensor<320x416xf32>

    return %5 : tensor<320x416xf32>
  }
}
