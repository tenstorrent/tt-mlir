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
    // CHECK: d2m.generic {{.*}}grid = #ttcore.grid<1x1x2x4
    // CHECK-SAME: iterator_types = [#parallel, #reduction, #parallel, #parallel]
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<3x8x64x128xbf16>) -> tensor<3x1x64x128xbf16>
    return %0 : tensor<3x1x64x128xbf16>
  }
}
