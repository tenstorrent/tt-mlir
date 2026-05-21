// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection --canonicalize %s | FileCheck %s

module {
  // Loop-space grid normalization must keep a broadcasted compute generic on
  // the shared loop-space grid while selecting an ND virtual grid.
  // CHECK-LABEL: func.func @broadcast_loop_virtual_grid
  func.func @broadcast_loop_virtual_grid(
      %arg0: tensor<1x1x32x32xf32>,
      %arg1: tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32> {
    // CHECK: d2m.empty() {virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}} : tensor<1x64x1x1x1x1x1x1x!ttcore.tile<32x32, f32>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x64x1x1
    // CHECK-SAME: indexing_maps =
    // CHECK: "d2m.tile_add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x32x32xf32>, tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32>
    return %0 : tensor<1x64x32x32xf32>
  }

  // CHECK-LABEL: func.func @reduction_loop_grid_placeable
  func.func @reduction_loop_grid_placeable(
      %arg0: tensor<3x8x64x128xbf16>) -> tensor<3x1x64x128xbf16> {
    // CHECK-NOT: grid = #ttcore.grid<3x8x2x4
    // CHECK: d2m.generic {{.*}}grid = #ttcore.grid<3x1x2x4
    // CHECK-SAME: iterator_types = [#parallel, #reduction, #parallel, #parallel]
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<3x8x64x128xbf16>) -> tensor<3x1x64x128xbf16>
    return %0 : tensor<3x1x64x128xbf16>
  }

  // Candidate grids must be materializable through the finalized physical
  // layout grid selected for their producers. This case now keeps the full
  // legal 16x3 virtual grid without padding the producer to an incompatible
  // physical shape.
  // CHECK-LABEL: func.func @implicit_bcast_inner_2d_grid_materialization
  // CHECK: d2m.empty() {{.*}} : tensor<16x3x1x1x!ttcore.tile<32x32, f32>
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<16x3
  func.func @implicit_bcast_inner_2d_grid_materialization(
      %arg0: tensor<416x96xf32>,
      %arg1: tensor<416x1xf32>,
      %arg2: tensor<1x96xf32>,
      %arg3: tensor<1x1xf32>) -> tensor<416x96xf32> {
    %0 = "ttir.add"(%arg3, %arg2) : (tensor<1x1xf32>, tensor<1x96xf32>) -> tensor<1x96xf32>
    %1 = "ttir.add"(%0, %arg3) : (tensor<1x96xf32>, tensor<1x1xf32>) -> tensor<1x96xf32>
    %2 = "ttir.subtract"(%arg1, %arg3) : (tensor<416x1xf32>, tensor<1x1xf32>) -> tensor<416x1xf32>
    %3 = "ttir.subtract"(%arg3, %2) : (tensor<1x1xf32>, tensor<416x1xf32>) -> tensor<416x1xf32>
    %4 = "ttir.add"(%3, %1) : (tensor<416x1xf32>, tensor<1x96xf32>) -> tensor<416x96xf32>
    %5 = "ttir.add"(%4, %arg3) : (tensor<416x96xf32>, tensor<1x1xf32>) -> tensor<416x96xf32>
    return %5 : tensor<416x96xf32>
  }

  // Reduction loop factors do not consume additional physical cores. For large
  // matmuls, K-splitting must survive normalization as a block factor so the
  // operand staging grids do not collapse to skinny 8x1 / 1x8 layouts that
  // require huge per-core tilize buffers.
  // CHECK-LABEL: func.func @matmul_reduction_factor_not_physical_grid
  // CHECK: d2m.empty() : tensor<8x8x2x4x!ttcore.tile<32x32, f32>
  // CHECK: d2m.empty() : tensor<8x8x4x8x!ttcore.tile<32x32, f32>
  // CHECK: d2m.generic
  // CHECK-SAME: block_factors = [1, 1, 8]
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  func.func @matmul_reduction_factor_not_physical_grid(
      %arg0: tensor<512x1024xf32>,
      %arg1: tensor<1024x2048xf32>) -> tensor<512x2048xf32> {
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<512x1024xf32>, tensor<1024x2048xf32>) -> tensor<512x2048xf32>
    return %0 : tensor<512x2048xf32>
  }
}
